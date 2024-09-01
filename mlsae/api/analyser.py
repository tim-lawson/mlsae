"""
A helper class to analyse a pretrained MLSAE.
Based on https://github.com/callummcdougall/sae_vis.
"""

from typing import NamedTuple

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from loguru import logger
from pydantic import BaseModel
from torch import Tensor

from mlsae.api.models import (
    LatentActivations,
    LayerHistograms,
    Logit,
    LogitChanges,
    MaxLogits,
    Token,
)
from mlsae.model import MLSAETransformer
from mlsae.model.decoder import scatter_topk
from mlsae.utils import cache_method


class DefaultParams(BaseModel):
    bins: int = 64
    """The number of equal-width bins for histograms."""

    num_tokens: int = 16
    """The number of logits to return for each token position."""


class Output(NamedTuple):
    """A thin wrapper around the outputs of the autoencoder forward pass."""

    tokens: Int[Tensor, "pos"]
    inputs: Float[Tensor, "n_layers pos n_inputs"]
    latents: Float[Tensor, "n_layers pos n_latents"]
    recons: Float[Tensor, "n_layers pos n_inputs"]
    metrics: dict[str, Float[Tensor, ""]]


class LogitsProbs(NamedTuple):
    logits: Float[Tensor, "pos d_vocab"]
    probs: Float[Tensor, "pos d_vocab"]


class Analyser:
    def __init__(
        self,
        repo_id: str,
        device: torch.device | str = "cpu",
        default_params: DefaultParams | None = None,
    ) -> None:
        logger.info(f"repo_id: {repo_id}")
        self.model = MLSAETransformer.from_pretrained(repo_id).to(device)
        self.model.requires_grad_(False)

        self.autoencoder = self.model.autoencoder
        self.transformer = self.model.transformer
        self.transformer.skip_special_tokens = False
        self.tokenizer = self.model.transformer.tokenizer

        self.default_params = default_params or DefaultParams()

    def params(self) -> dict:
        return self.model.hparams_initial

    def _convert_text_to_ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def _convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        tokens: list[str] = self.tokenizer.convert_ids_to_tokens(
            ids, skip_special_tokens=False
        )  # type: ignore
        return [replace_special(token) for token in tokens]

    def _convert_text_to_batch(self, text: str) -> Int[Tensor, "batch pos"]:
        return torch.tensor(self._convert_text_to_ids(text)).unsqueeze(0)

    @cache_method()
    def _forward(self, text: str) -> Output:
        """Forward pass through the transformer and autoencoder."""

        tokens = self._convert_text_to_batch(text).to(self.transformer.model.device)
        inputs = self.transformer.forward(tokens)
        topk, recons, auxk, auxk_recons, dead = self.autoencoder.forward(inputs)
        latents = scatter_topk(topk, self.model.n_latents)

        metrics = self.model.train_metrics.forward(
            inputs=inputs,
            indices=topk.indices,
            values=topk.values,
            recons=recons,
        )

        return Output(
            tokens=tokens.squeeze(),
            inputs=inputs.squeeze(),
            latents=latents.squeeze(),
            recons=recons.squeeze(),
            metrics=metrics,
        )

    def prompt_tokens(self, prompt: str) -> list[Token]:
        """Tokenize the specified prompt."""

        ids = self._convert_text_to_ids(prompt)
        return [
            Token(id=id, token=token, pos=pos)
            for pos, (id, token) in enumerate(
                zip(ids, self._convert_ids_to_tokens(ids), strict=False)
            )
        ]

    def prompt_metrics(self, prompt: str) -> dict[str, float]:
        """Find the metric values for the specified prompt."""

        return {k: v.item() for k, v in self._forward(prompt).metrics.items()}

    def prompt_latent_activations(self, prompt: str) -> LatentActivations:
        """Find the latent activations for the specified prompt."""

        latents = self._forward(prompt).latents
        max = einops.reduce(
            latents, "n_layers pos n_latents -> n_layers pos", reduction="max"
        )
        return LatentActivations(values=latents.tolist(), max=max.tolist())

    def prompt_layer_histograms(
        self, prompt: str, bins: int | None = None
    ) -> LayerHistograms:
        """
        Find layer-wise histograms of the latent activations for the specified prompt.
        """

        bins = bins or self.default_params.bins

        latents = self._forward(prompt).latents
        max = latents.max().item()

        values: list[list[int]] = []
        edges = [round(edge, 3) for edge in torch.linspace(0, max, bins + 1).tolist()]
        for layer in latents:
            values.append(torch.histc(layer, bins=bins, min=0, max=max).tolist())
            layer = layer.cpu().detach()

        return LayerHistograms(values=values, edges=edges)

    def prompt_logits_input(
        self, prompt: str, num_tokens: int | None = None
    ) -> MaxLogits:
        """Find the maximum logits of the transformer for the specified prompt."""

        num_tokens = num_tokens or self.default_params.num_tokens

        return self._prompt_max_logits(
            self._logit_probs(self._forward(prompt).inputs), num_tokens
        )

    def prompt_logits_recon(
        self, prompt: str, layer: int, num_logits: int | None = None
    ) -> tuple[MaxLogits, LogitChanges]:
        """
        Find the maximum logits and changes in logits when the activations of the
        transformer at the specified layer are reconstructed from the autoencoder
        latents for the specified prompt.
        """

        num_logits = num_logits or self.default_params.num_tokens

        tokens, inputs, latents, recons, metrics = self._forward(prompt)

        before = self._logit_probs(inputs, layer)
        after = self._logit_probs(recons, layer)

        return (
            self._prompt_max_logits(after, num_logits),
            self._prompt_logit_changes(before, after, num_logits),
        )

    def prompt_logits_steer(
        self,
        prompt: str,
        latent: int,
        layer: int,
        factor: float = -1,
        num_logits: int | None = None,
    ) -> tuple[MaxLogits, LogitChanges]:
        """
        Find the maximum logits and changes in logits when the activations of the
        transformer at the specified layer are steered by the specified autoencoder
        latent for the specified prompt.
        """

        num_logits = num_logits or self.default_params.num_tokens

        inputs = self._forward(prompt).inputs
        steered = inputs + factor * self.autoencoder.decoder.weight[:, latent]

        before = self._logit_probs(inputs, layer)
        after = self._logit_probs(steered, layer)

        return (
            self._prompt_max_logits(after, num_logits),
            self._prompt_logit_changes(before, after, num_logits),
        )

    def _logit_probs(
        self, x: Float[Tensor, "n_layers pos n_inputs"], layer: int | None = None
    ) -> LogitsProbs:
        """
        Find the logits and softmax-normalized probabilities when the specified
        activations are passed through the transformer at the specified layer.
        """

        start_at_layer = layer or self.transformer.model.config.num_hidden_layers - 1

        logits = self.transformer.forward_at_layer(
            x.unsqueeze(1), start_at_layer, return_type="logits"
        ).squeeze()

        return LogitsProbs(logits=logits, probs=F.softmax(logits, dim=-1))

    def _prompt_max_logits(self, after: LogitsProbs, k: int) -> MaxLogits:
        """Find the k largest logits and softmax-normalized probabilities."""

        def pos_max_logits(pos: int) -> list[Logit]:
            topk = torch.topk(after.logits[pos], k)

            ids = topk.indices.tolist()
            tokens = self._convert_ids_to_tokens(ids)
            logits = topk.values.tolist()
            probs = torch.gather(after.probs[pos], 0, topk.indices).tolist()

            return [
                Logit(id=id, token=token or "", logit=logit, prob=prob)
                for id, token, logit, prob in zip(
                    ids, tokens, logits, probs, strict=False
                )
            ]

        return MaxLogits(
            max=[pos_max_logits(pos) for pos in range(after.logits.size(0))]
        )

    def _prompt_logit_changes(
        self, before: LogitsProbs, after: LogitsProbs, k: int
    ) -> LogitChanges:
        """Find the k largest positive and negative changes in the specified logits."""

        change = after.logits - before.logits

        def pos_logit_changes(pos: int, largest: bool) -> list[Logit]:
            topk = torch.topk(change[pos], k, largest=largest)
            ids = topk.indices.tolist()
            tokens = self._convert_ids_to_tokens(ids)
            logits = topk.values.tolist()
            return [
                Logit(id=id, token=token or "", logit=logit)
                for id, token, logit in zip(ids, tokens, logits, strict=False)
            ]

        max, min = [], []
        for pos in range(after.logits.size(0)):
            max.append(pos_logit_changes(pos, True))
            min.append(pos_logit_changes(pos, False))
        return LogitChanges(max=max, min=min)


# Based on https://github.com/callummcdougall/sae_vis/blob/eee2ac65737a63f1442416d1206f9ad23ffb9e07/sae_vis/utils_fns.py#L199-L271.
def replace_special(token: str) -> str:
    """Replace special tokenization characters with human-readable equivalents."""

    for k, v in {
        "âĢĶ": "—",
        "âĢĵ": "–",
        "âĢĭ": "",
        "âĢľ": '"',
        "âĢĿ": '"',
        "âĢĺ": "'",
        "âĢĻ": "'",
        "Ġ": " ",
        "Ċ": "\n",
        "ĉ": "\t",
    }.items():
        token = token.replace(k, v) if token is not None else ""
    return token
