import pytest
import torch
from transformers import (
    AutoTokenizer,
)
from transformers.models.llama.configuration_llama import LlamaConfig

from mlsae.model.transformers.llama import LlamaTransformer
from mlsae.model.transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaModel,
)
from mlsae.utils import get_input_ids

atol = 1e-2


@pytest.mark.slow()
@torch.no_grad()
def test_hidden_states() -> None:
    model_name = "meta-llama/Llama-3.2-1B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = get_input_ids(tokenizer, "The quick brown fox jumps over the lazy dog.")

    llama: LlamaModel = LlamaModel.from_pretrained(model_name)  # type: ignore
    config: LlamaConfig = llama.config  # type: ignore

    # Skip the final layer norm when collecting hidden states
    hidden_states = torch.stack(
        llama.forward(
            input_ids, output_hidden_states=True, skip_final_layer_norm=True
        ).hidden_states[1:]  # type: ignore
    )

    # We usually skip special tokens, but we may as well compare them
    my_llama = LlamaTransformer(
        model_name,
        config.max_position_embeddings,
        batch_size=1,
        skip_special_tokens=False,
    )
    my_hidden_states = my_llama.hidden_states(input_ids)

    for layer in range(len(hidden_states)):
        assert torch.allclose(
            hidden_states[layer],
            my_hidden_states[layer],
            atol=atol,
        )


@pytest.mark.slow()
@torch.no_grad()
def test_forward_at_layer() -> None:
    model_name = "meta-llama/Llama-3.2-1B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = get_input_ids(tokenizer, "The quick brown fox jumps over the lazy dog.")

    llama: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(model_name)  # type: ignore
    config: LlamaConfig = llama.config  # type: ignore

    # Skip the final layer norm when collecting hidden states
    hidden_states = torch.stack(
        llama.forward(
            input_ids,
            output_hidden_states=True,
            skip_final_layer_norm=True,
        ).hidden_states[1:]  # type: ignore
    )

    # Don't skip the final layer norm when computing the loss/logits
    output = llama.forward(input_ids, labels=input_ids)
    loss: torch.Tensor = output.loss  # type: ignore
    logits = output.logits  # type: ignore

    # We usually skip special tokens, but we may as well compare them
    my_llama = LlamaTransformer(
        model_name, config.n_positions, batch_size=1, skip_special_tokens=False
    )

    for layer in range(config.n_layer):
        my_loss, my_logits = my_llama.forward_at_layer(
            inputs_embeds=hidden_states,
            start_at_layer=layer,
            return_type="both",
            tokens=input_ids,
        )
        assert torch.allclose(my_loss, loss, atol=atol)
        assert torch.allclose(my_logits, logits, atol=atol)
