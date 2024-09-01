from typing import Literal, overload

import torch
from jaxtyping import Bool, Float, Int
from torch import FloatTensor, Tensor
from torch.nn import CrossEntropyLoss, Module
from transformers import (
    AutoTokenizer,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.models.gpt_neox.modeling_gpt_neox import (
    _prepare_4d_causal_attention_mask,
)


class Transformer(Module):
    def __init__(
        self,
        model_name: str,
        max_length: int,
        batch_size: int,
        skip_special_tokens: bool = True,
        # TODO: Check this works for non-consecutive layers
        layers: list[int] | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        """
        Args:
            model_name (str): The name of a pretrained GPTNeoXForCausalLM model.

            max_length (int): The maximum length of a tokenized input sequence.

            batch_size (int): The number of sequences in a batch.

            skip_special_tokens (bool): Whether to ignore special tokens.
                Defaults to True.

            layers (list[int] | None): The layers to train on.
                If None, all layers are trained on.
                Defaults to None.

            device (torch.device | str): The device to use.
                Defaults to "cpu".
        """

        super().__init__()

        self.model_name = model_name
        self.model: GPTNeoXForCausalLM = GPTNeoXForCausalLM.from_pretrained(model_name)  # type: ignore
        self.model.eval()

        self.batch_size = batch_size
        self.max_length = max_length

        self.config: GPTNeoXConfig = self.model.config  # type: ignore
        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = (
            AutoTokenizer.from_pretrained(model_name)
        )

        if layers is not None:
            assert all(0 <= i < self.config.num_hidden_layers for i in layers)
            self.layers = layers
        else:
            self.layers = list(range(self.config.num_hidden_layers))

        self.n_layers = len(self.layers)
        self.skip_special_tokens = skip_special_tokens

        # NOTE: We know all inputs have the same shape, so we can pre-compute these
        self.position_ids = torch.arange(
            0, self.max_length, dtype=torch.long, device=device
        ).unsqueeze(0)

        self.attention_mask: Tensor = _prepare_4d_causal_attention_mask(
            attention_mask=None,
            input_shape=(batch_size, max_length),
            # We only actually need the device and dtype of the inputs
            inputs_embeds=torch.empty(0, dtype=torch.float32, device=device),
            past_key_values_length=0,
        )  # type: ignore

        self.loss = CrossEntropyLoss()

    @torch.no_grad()
    def forward(
        self, tokens: Int[Tensor, "batch pos"]
    ) -> Float[Tensor, "n_layers batch pos d_model"]:
        """
        Returns the residual stream activation vectors from the specified layers.

        Args:
            tokens (Int[Tensor, "batch pos"]): The input tokens.

        Returns:
            out (Float[Tensor, "n_layers batch pos d_model"]): The residual stream
                activation vectors from the specified layers.
        """

        hidden_states = self.hidden_states(tokens)
        out = torch.stack([hidden_states[i] for i in self.layers])
        if self.skip_special_tokens:
            out[:, ~self._mask_special_tokens(tokens), :] = 0.0
        return out

    @torch.no_grad()
    def hidden_states(
        self, tokens: Int[Tensor, "batch pos"]
    ) -> list[Float[Tensor, "batch pos d_model"]]:
        """
        Return the hidden states.

        Similar to the `run_with_hooks` method of `HookedTransformer` in the
        TransformerLens library.

        Args:
            tokens (Int[Tensor, "batch pos"]): The input tokens.

        Returns:
            out (list[Float[Tensor, "batch pos d_model"]]): The hidden states.
        """

        position_ids = self._position_ids(tokens)
        attention_mask = self._attention_mask(tokens)

        out: list[Float[Tensor, "batch pos d_model"]] = []

        # NOTE: GPTNeoXModel includes the input embeddings in hidden_states, we don't
        inputs_embeds = self.model.gpt_neox.embed_in(tokens)
        hidden_states = self.model.gpt_neox.emb_dropout(inputs_embeds)

        for layer in self.model.gpt_neox.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]
            out = out + [hidden_states]

        # NOTE: Skip the final layer norm
        return out

    @overload
    def forward_at_layer(
        self,
        input: Float[Tensor, "batch pos d_model"],
        start_at_layer: int,
        return_type: Literal["loss"],
        tokens: Int[Tensor, "batch pos"] | None = None,
    ) -> Float[Tensor, ""]: ...

    @overload
    def forward_at_layer(
        self,
        input: Float[Tensor, "batch pos d_model"],
        start_at_layer: int,
        return_type: Literal["logits"],
        tokens: Int[Tensor, "batch pos"] | None = None,
    ) -> Float[Tensor, "batch pos d_vocab"]: ...

    @overload
    def forward_at_layer(
        self,
        input: Float[Tensor, "batch pos d_model"],
        start_at_layer: int,
        return_type: Literal["both"],
        tokens: Int[Tensor, "batch pos"] | None = None,
    ) -> tuple[Float[Tensor, ""], Float[Tensor, "batch pos d_vocab"]]: ...

    @torch.no_grad()
    def forward_at_layer(
        self,
        input: Float[Tensor, "batch pos d_model"],
        start_at_layer: int,
        return_type: Literal["loss", "logits", "both"] = "both",
        tokens: Int[Tensor, "batch pos"] | None = None,
    ) -> (
        Float[Tensor, ""]
        | Float[Tensor, "batch pos d_vocab"]
        | tuple[Float[Tensor, ""], Float[Tensor, "batch pos d_vocab"]]
    ):
        """
        Return the cross-entropy loss and/or logits, starting from the specified layer.

        The input tokens are needed to compute the loss.

        Also similar to the TransformerLens API.

        Args:
            inputs (Float[torch.Tensor, "layer batch pos n_inputs"]): The residual
                stream activations at the specified layer.

            start_at_layer (int): The layer at which to start the forward pass.

            return_type (Literal["loss", "logits", "both"]): Whether to return the
                cross-entropy loss and/or logits.

            tokens (Int[torch.Tensor, "batch pos"] | None): If the return_type is
                "loss" or "both", the input tokens, otherwise None.

        Returns:
            The cross-entropy loss and/or logits.
        """

        if return_type in ["loss", "both"] and tokens is None:
            raise ValueError("The input tokens are needed to compute the loss.")

        if tokens is None:
            tokens = torch.zeros(input.shape[0], input.shape[1], device=input.device)

        # Move tensors to the correct device, which we don't know in the constructor
        self.position_ids = self._position_ids(tokens)
        self.attention_mask = self._attention_mask(tokens)

        # Get the hidden states at the specified layer
        hidden_states = input[start_at_layer, ...]

        for i, layer in enumerate(self.model.gpt_neox.layers):
            # Skip layers before the specified layer
            if start_at_layer >= i:
                continue

            hidden_states = layer(
                hidden_states,
                attention_mask=self.attention_mask,
                position_ids=self.position_ids,
            )[0]

        # TODO: These are not equivalent!
        # hidden_states = self.model.gpt_neox.final_layer_norm(hidden_states)
        hidden_states = layer_norm(
            hidden_states,
            self.model.gpt_neox.final_layer_norm.weight,
            self.model.gpt_neox.final_layer_norm.bias,
            eps=self.model.gpt_neox.final_layer_norm.eps,
        )
        logits: FloatTensor = self.model.embed_out(hidden_states)

        if return_type == "logits":
            return logits

        # Shift to evaluate next-token predictions
        shifted = logits[:, :-1, :].contiguous()

        labels = tokens.to(logits.device)[:, 1:].contiguous()  # type: ignore

        loss = self.loss(shifted.view(-1, shifted.size(-1)), labels.view(-1))

        if return_type == "loss":
            return loss

        return loss, logits

    @torch.no_grad()
    def _mask_special_tokens(
        self, tokens: Int[Tensor, "batch pos"]
    ) -> Bool[Tensor, "batch pos"]:
        """Mask out special tokens (zero the activations)."""

        mask = torch.ones_like(tokens, dtype=torch.bool, device=tokens.device)

        if not self.skip_special_tokens or self.tokenizer is None:
            return mask

        if self.tokenizer.eos_token_id is not None:
            mask = mask & torch.ne(tokens, self.tokenizer.eos_token_id)
        if self.tokenizer.pad_token_id is not None:
            mask = mask & torch.ne(tokens, self.tokenizer.pad_token_id)
        if self.tokenizer.bos_token_id is not None:
            mask = mask & torch.ne(tokens, self.tokenizer.bos_token_id)

        return mask

    def _position_ids(self, tokens: Int[Tensor, "batch pos"]) -> Tensor:
        if tokens.shape != (self.batch_size, self.max_length):
            return torch.arange(
                0, tokens.shape[1], dtype=torch.long, device=tokens.device
            ).unsqueeze(0)

        return self.position_ids.to(device=tokens.device)

    def _attention_mask(self, tokens: Int[Tensor, "batch pos"]) -> Tensor:
        if tokens.shape != (self.batch_size, self.max_length):
            return _prepare_4d_causal_attention_mask(
                attention_mask=None,
                input_shape=tokens.shape,
                inputs_embeds=torch.empty(0, dtype=torch.float32, device=tokens.device),
                past_key_values_length=0,
            )  # type: ignore

        return self.attention_mask.to(device=tokens.device)


def layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> Tensor:
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    y = (x - mean) / torch.sqrt(var + eps)
    return y * weight + bias
