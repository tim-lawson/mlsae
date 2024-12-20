import functools
import weakref

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from mlsae.model import MLSAETransformer, TopK, TopKSAE


def get_model_repo_id(model: MLSAETransformer, transformer: bool) -> str:
    # NOTE: This is a hack. At the moment, we only distinguish between models trained on
    # a single layer and models trained on all layers.
    layers = None if len(model.layers) > 1 else model.layers
    return get_repo_id(
        model_name=model.model_name,
        expansion_factor=model.expansion_factor,
        k=model.k,
        tuned_lens=model.tuned_lens,
        transformer=transformer,
        layers=layers,
    )


def get_repo_id(
    model_name: str,
    expansion_factor: int,
    k: int,
    tuned_lens: bool,
    transformer: bool,
    layers: list[int] | None = None,
) -> str:
    """
    Get the repo_id that corresponds to the specified hyperparameters.
    You should probably change this!
    """
    model_name = model_name.split("/")[-1]
    repo_id = f"tim-lawson/mlsae-{model_name}-x{expansion_factor}-k{k}"
    if tuned_lens:
        repo_id += "-lens"
    if transformer:
        repo_id += "-tfm"
    if layers is not None:
        repo_id = repo_id.replace("mlsae", "sae")
        repo_id += f"-layers-{''.join(map(str, layers))}"
    return repo_id


def get_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def normalize(x: torch.Tensor, dim: int = 0, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
    return x / torch.max(norm, eps * torch.ones_like(norm))


# Copied from https://stackoverflow.com/a/33672499/23543959
def cache_method(*lru_args, **lru_kwargs):
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(self, *args, **kwargs):
            self_weak = weakref.ref(self)

            @functools.wraps(func)
            @functools.lru_cache(*lru_args, **lru_kwargs)
            def cached_method(*args, **kwargs):
                return func(self_weak(), *args, **kwargs)

            setattr(self, func.__name__, cached_method)
            return cached_method(*args, **kwargs)

        return wrapped_func

    return decorator


def load_single_layer(
    model_name: str,
    layer: int,
    device: torch.device,
    expansion_factor: int = 64,
    k: int = 32,
    tuned_lens: bool = False,
) -> MLSAETransformer:
    # NOTE: This is a hack. We want to feed an SAE trained at layer i with the input
    # activations from every layer. So, we:
    #   1. Load the multi-layer SAE and underlying transformer
    model_repo_id = get_repo_id(model_name, expansion_factor, k, tuned_lens, True)
    print("model repo_id:", model_repo_id)
    model = MLSAETransformer.from_pretrained(model_repo_id)
    model = model.to(device)

    #   2. Load the layer-specific SAE only
    autoencoder_repo_id = get_repo_id(
        model_name, expansion_factor, k, tuned_lens, False, [layer]
    )
    print("autoencoder repo_id:", autoencoder_repo_id)
    autoencoder = TopKSAE.from_pretrained(
        autoencoder_repo_id,
        # TODO: These should be taken from config.json
        n_inputs=model.n_inputs,
        n_latents=model.n_latents,
        k=model.k,
        dead_steps_threshold=model.dead_steps_threshold,
    )
    autoencoder = autoencoder.to(device)

    #   3. Replace the SAE in the multi-layer model with the layer-specific one
    model.autoencoder = autoencoder

    # Optional: check the hyperparameters match
    assert model.n_inputs == autoencoder.n_inputs
    assert model.n_latents == autoencoder.n_latents
    assert model.k == autoencoder.k
    assert model.dead_steps_threshold == autoencoder.dead_steps_threshold
    assert model.dead_threshold == autoencoder.dead_threshold
    assert model.auxk == autoencoder.auxk

    model.standardize = model.autoencoder.standardize

    return model


# NOTE: This is also a hack. We want the input activations to be normalized
# independently for each layer. So, we feed them to the SAE one layer at a time
# and combine the results. UPDATE: Turns out, this is equivalent to the forward method.
def forward_single_layer(
    model: MLSAETransformer, tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, TopK]:
    standardize = model.autoencoder.standardize
    inputs = model.forward_lens(model.transformer.forward(tokens))

    # topk, recons, _, _, _ = model.forward(tokens)
    # return inputs, recons, topk

    recons = torch.empty(inputs.shape, device=model.device)
    topk = TopK(
        values=torch.empty(
            (model.n_layers, model.batch_size, model.max_length, model.k),
            device=model.device,
        ),
        indices=torch.empty(
            (model.n_layers, model.batch_size, model.max_length, model.k),
            device=model.device,
            dtype=torch.long,
        ),
    )
    for layer in range(model.n_layers):
        model.autoencoder.standardize = True
        if layer == model.n_layers - 1:
            model.autoencoder.standardize = False
        topk_, recons_, _, _, _ = model.autoencoder.forward(inputs[layer])
        recons[layer] = recons_
        topk.indices[layer] = topk_.indices
        topk.values[layer] = topk_.values
    model.autoencoder.standardize = standardize
    return inputs, recons, topk


def get_input_ids(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, prompt: str
) -> torch.LongTensor:
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids = tokenizer.encode(prompt)
    return torch.LongTensor(
        [input_ids + [pad_token_id] * (tokenizer.model_max_length - len(input_ids))]
    )
