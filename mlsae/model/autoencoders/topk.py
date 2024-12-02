# Based on https://github.com/openai/sparse_autoencoder/blob/4965b941e9eb590b00b253a2c406db1e1b193942/sparse_autoencoder/train.py

from typing import NamedTuple

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch.nn import Linear, Module, Parameter

from mlsae.model.decoder import decode
from mlsae.model.types import Stats, TopK
from mlsae.model_card import model_card_template

from .utils import standardize, unit_norm_decoder


class TopKSAEOut(NamedTuple):
    """The output of the autoencoder forward pass."""

    topk: TopK
    """The k largest latents."""

    recons: torch.Tensor
    """The reconstructions from the k largest latents."""

    auxk: TopK | None
    """If auxk is not None, the auxk largest dead latents."""

    auxk_recons: torch.Tensor | None
    """If auxk is not None, the reconstructions from the auxk largest dead latents."""

    dead: torch.Tensor
    """The fraction of dead latents."""


class TopKSAE(
    Module,
    PyTorchModelHubMixin,
    model_card_template=model_card_template(False),
    license="mit",
    language="en",
    library_name="mlsae",
    repo_url="https://github.com/tim-lawson/mlsae",
    tags=["arxiv:2409.04185"],
):
    last_nonzero: torch.Tensor
    """The number of steps since the latents have activated."""

    def __init__(
        self,
        n_inputs: int,
        n_latents: int,
        k: int,
        dead_steps_threshold: int,
        dead_threshold: float = 1e-3,
        # TODO: Make this optional and default to a power of 2 close to d_model / 2.
        auxk: int | None = 256,
        standardize: bool = True,
    ) -> None:
        """
        Args:
            n_inputs (int): The number of inputs.

            n_latents (int): The number of latents.

            k (int): The number of largest latents to keep.

            dead_steps_threshold (int): The number of steps after which a latent is
                flagged as dead during training.

            dead_threshold (float): The threshold for a latent to be considered
                activated. Defaults to 1e-3.

            auxk (int | None): The number of dead latents with which to model the
                reconstruction error. Defaults to 256.

            standardize (bool): Whether to standardize the inputs. Defaults to True.
        """

        super().__init__()

        self.n_inputs = n_inputs
        self.n_latents = n_latents
        self.k = k
        self.auxk = auxk
        self.dead_steps_threshold = dead_steps_threshold
        self.dead_threshold = dead_threshold
        self.standardize = standardize

        self.encoder = Linear(n_inputs, n_latents, bias=False)
        self.decoder = Linear(n_latents, n_inputs, bias=False)
        self.pre_encoder_bias = Parameter(torch.zeros(n_inputs))

        self.register_buffer("last_nonzero", torch.zeros(n_latents, dtype=torch.long))

        self.decoder.weight.data = self.encoder.weight.data.T.clone()
        self.decoder.weight.data = self.decoder.weight.data.T.contiguous().T
        unit_norm_decoder(self.decoder)

    def encode(
        self, inputs: torch.Tensor
    ) -> tuple[TopK, TopK | None, Stats | None, torch.Tensor]:
        stats = None
        if self.standardize:
            inputs, stats = standardize(inputs)

        # Keep a reference to the latents before the TopK activation function
        latents = self.encoder.forward(inputs - self.pre_encoder_bias)

        # Find the k largest latents
        topk = TopK(*torch.topk(latents, k=self.k, sorted=False))

        # Update the number of steps since the latents have activated
        last_nonzero = torch.zeros_like(self.last_nonzero, device=inputs.device)
        last_nonzero.scatter_add_(
            dim=0,
            index=topk.indices.reshape(-1),
            src=(topk.values > self.dead_threshold).to(last_nonzero.dtype).reshape(-1),
        )
        self.last_nonzero *= 1 - last_nonzero.clamp(max=1)
        self.last_nonzero += 1

        # Mask the latents flagged as dead during training
        dead_mask = self.last_nonzero >= self.dead_steps_threshold
        latents.data *= dead_mask  # in-place to save memory

        # Compute the fraction of dead latents
        dead = torch.sum(dead_mask, dtype=torch.float32).detach() / self.n_latents

        # If auxk is not None, find the auxk largest dead latents
        auxk = None
        if self.auxk is not None:
            auxk = TopK(*torch.topk(latents, k=self.auxk, sorted=False))

        return topk, auxk, stats, dead

    def decode(self, topk: TopK, stats: Stats | None = None) -> torch.Tensor:
        recons = decode(topk, self.decoder.weight) + self.pre_encoder_bias
        if stats is not None:
            recons = recons * stats.std + stats.mean
        return recons

    def forward(self, inputs: torch.Tensor) -> TopKSAEOut:
        topk, auxk, stats, dead = self.encode(inputs)

        # Apply ReLU to ensure the k largest latents are non-negative
        values = torch.relu(topk.values)
        topk = TopK(values, topk.indices)
        recons = self.decode(topk, stats)

        auxk_recons = None
        if auxk is not None:
            auxk_values = torch.relu(auxk.values)
            auxk = TopK(auxk_values, auxk.indices)
            auxk_recons = self.decode(auxk)

        return TopKSAEOut(topk, recons, auxk, auxk_recons, dead)
