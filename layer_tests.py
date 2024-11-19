import os
from pprint import pprint

import pandas as pd
import torch
from torch.utils.data import DataLoader

from mlsae.model import DataConfig, MLSAETransformer, TopK, TopKSAE, get_test_dataloader
from mlsae.trainer import RunConfig, initialize
from mlsae.utils import get_device, get_repo_id

pythia_70m = "EleutherAI/pythia-70m-deduped"
pythia_160m = "EleutherAI/pythia-160m-deduped"
# pythia_410m = "EleutherAI/pythia-410m-deduped"
# pythia_1b = "EleutherAI/pythia-1b-deduped"

layers = {
    pythia_70m: range(6),
    pythia_160m: range(12),
    # pythia_410m: range(24),
    # pythia_1b: range(16),
}

config = RunConfig(data=DataConfig(max_tokens=10_000_000))


def test(model_name: str, layer: int):
    initialize(config.seed)
    device = get_device()

    # NOTE: This is a hack. We want to feed an SAE trained at layer i with the input
    # activations from every layer. So, we:
    #
    #   1. load the multi-layer SAE + transformer harness
    model_repo_id = get_repo_id(model_name, 64, 32, False, True)
    model = MLSAETransformer.from_pretrained(model_repo_id)
    model = model.to(device)

    #   2. load the layer-specific SAE
    autoencoder_repo_id = get_repo_id(model_name, 64, 32, False, False, [layer])
    autoencoder = TopKSAE.from_pretrained(
        autoencoder_repo_id,
        # TODO: not sure why these aren't taken from config.json
        n_inputs=model.n_inputs,
        n_latents=model.n_latents,
        k=model.k,
        dead_steps_threshold=model.dead_steps_threshold,
    )
    autoencoder = autoencoder.to(device)

    #   3. replace the autoencoder with the layer-specific one
    model.autoencoder = autoencoder

    print(model.layers)

    dataloader = get_test_dataloader(
        model.model_name,
        config.data.max_length,
        config.data.batch_size,
        config.data.num_workers or 1,
    )

    # output = test_lightning(model, dataloader)
    output = test_manual(model, dataloader, device)
    output = {k: v.item() for k, v in output.items()}
    pprint(output)

    filename_repo_id = get_repo_id(model_name, 64, 32, False, True, [layer])
    filename = f"test_{filename_repo_id.split('/')[-1]}.csv"
    pd.DataFrame(output, index=[0]).to_csv(os.path.join("out", filename), index=False)


def test_manual(
    model: MLSAETransformer, dataloader: DataLoader[torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    for i, batch in enumerate(dataloader):
        if i >= config.data.max_steps:
            break

        tokens: torch.Tensor = batch["input_ids"].to(device)
        inputs = model.forward_lens(model.transformer.forward(tokens))

        # NOTE: This is also a hack. We want the input activations to be normalized
        # independently for each layer. So, we feed them to the SAE one layer at a time
        # and combine the results.
        recons = torch.empty(inputs.shape, device=device)
        topk = TopK(
            torch.empty(
                (model.n_layers, model.batch_size, model.max_length, model.k),
                device=device,
            ),
            torch.empty(
                (model.n_layers, model.batch_size, model.max_length, model.k),
                device=device,
            ),
        )
        for layer in range(model.n_layers):
            topk_, recons_, _, _, _ = model.autoencoder.forward(inputs[layer])
            recons[layer] = recons_
            topk.indices[layer] = topk_.indices
            topk.values[layer] = topk_.values

        model.train_metrics.forward(
            inputs=inputs,
            indices=topk.indices,
            values=topk.values,
            recons=recons,
        )

        recons = model.inverse_lens(recons)

        model.forward_at_layer(inputs, recons, tokens)
        model.val_metrics.forward(
            loss_true=model.loss_true,
            loss_pred=model.loss_pred,
            logits_true=model.logits_true,
            logits_pred=model.logits_pred,
        )
        model.mse_loss.forward(inputs=inputs, recons=recons)

    return {
        **model.train_metrics.compute(),
        **model.val_metrics.compute(),
        "mse_loss": model.mse_loss.compute(),
        "aux_loss": model.aux_loss.compute(),
        "loss": model.mse_loss.compute() + model.aux_loss.compute(),
    }


def main() -> None:
    for model_name in [pythia_70m, pythia_160m]:
        for layer in layers[model_name]:
            try:
                test(model_name, layer)
            except Exception as e:
                print(e)


if __name__ == "__main__":
    main()
