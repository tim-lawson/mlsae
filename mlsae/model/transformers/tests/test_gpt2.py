import torch
from transformers import AutoTokenizer

from mlsae.model.transformers.gpt2 import GPT2Transformer
from mlsae.model.transformers.models.gpt2.modeling_gpt2 import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Model,
)
from mlsae.utils import get_input_ids

atol = 1e-2


@torch.no_grad()
def test_hidden_states() -> None:
    model_name = "openai-community/gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = get_input_ids(tokenizer, "The quick brown fox jumps over the lazy dog.")

    gpt2: GPT2Model = GPT2Model.from_pretrained(model_name)  # type: ignore
    config: GPT2Config = gpt2.config  # type: ignore

    # Skip the final layer norm when collecting hidden states
    hidden_states = torch.stack(
        gpt2.forward(
            input_ids, output_hidden_states=True, skip_final_layer_norm=True
        ).hidden_states[1:]  # type: ignore
    )

    # We usually skip special tokens, but we may as well compare them
    my_gpt2 = GPT2Transformer(
        model_name, config.n_positions, batch_size=1, skip_special_tokens=False
    )
    my_hidden_states = my_gpt2.hidden_states(input_ids)

    for layer in range(len(hidden_states)):
        assert torch.allclose(
            hidden_states[layer],
            my_hidden_states[layer],
            atol=atol,
        )


@torch.no_grad()
def test_forward_at_layer() -> None:
    model_name = "openai-community/gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = get_input_ids(tokenizer, "The quick brown fox jumps over the lazy dog.")

    gpt2: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(model_name)  # type: ignore
    config: GPT2Config = gpt2.config  # type: ignore

    # Skip the final layer norm when collecting hidden states
    hidden_states = torch.stack(
        gpt2.forward(
            input_ids,
            output_hidden_states=True,
            skip_final_layer_norm=True,
        ).hidden_states[1:]  # type: ignore
    )

    # Don't skip the final layer norm when computing the loss/logits
    output = gpt2.forward(input_ids, labels=input_ids)
    loss: torch.Tensor = output.loss  # type: ignore
    logits = output.logits  # type: ignore

    # We usually skip special tokens, but we may as well compare them
    my_gpt2 = GPT2Transformer(
        model_name, config.n_positions, batch_size=1, skip_special_tokens=False
    )

    for layer in range(config.n_layer):
        my_loss, my_logits = my_gpt2.forward_at_layer(
            inputs_embeds=hidden_states,
            start_at_layer=layer,
            return_type="both",
            tokens=input_ids,
        )
        assert torch.allclose(my_loss, loss, atol=atol)
        assert torch.allclose(my_logits, logits, atol=atol)
