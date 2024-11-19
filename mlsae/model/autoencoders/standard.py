from typing import NamedTuple

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch.nn import Linear, Module, Parameter

from mlsae.model.types import Stats

from .utils import standardize, unit_norm_decoder


class SAEOut(NamedTuple):
    """The output of the autoencoder forward pass."""

    latents: torch.Tensor
    """The latents."""

    recons: torch.Tensor
    """The reconstructions."""

    dead: torch.Tensor
    """The fraction of dead latents."""


# TODO: This is equivalent to TopK SAE with k = n_latents and auxk = None.
class SAE(Module, PyTorchModelHubMixin):
    last_nonzero: torch.Tensor
    """The number of steps since the latents have activated."""

    def __init__(
        self,
        n_inputs: int,
        n_latents: int,
        dead_steps_threshold: int,
        dead_threshold: float = 1e-3,
        standardize: bool = True,
    ) -> None:
        """
        Args:
            n_inputs (int): The number of inputs.

            n_latents(int): The number of latents.

            dead_steps_threshold (int): The number of steps after which a latent is
                flagged as dead during training.

            dead_threshold (float): The threshold for a latent to be considered
                activated. Defaults to 1e-3.

            standardize (bool): Whether to standardize the inputs. Defaults to True.
        """
        super().__init__()

        self.n_inputs = n_inputs
        self.n_latents = n_latents
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
    ) -> tuple[torch.Tensor, Stats | None, torch.Tensor]:
        stats = None
        if self.standardize:
            inputs, stats = standardize(inputs)

        latents = self.encoder.forward(inputs - self.pre_encoder_bias)

        # Find the k largest latents (purely to maximize consistency with TopKSAE)
        topk = torch.topk(latents, self.n_latents, sorted=False)

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

        # Compute the fraction of dead latents
        dead = torch.sum(dead_mask, dtype=torch.float32).detach() / self.n_latents

        return latents, stats, dead

    def decode(self, latents: torch.Tensor, stats: Stats | None = None) -> torch.Tensor:
        recons = (latents @ self.decoder.weight.T) + self.pre_encoder_bias
        if stats is not None:
            recons = recons * stats.std + stats.mean
        return recons

    def forward(self, inputs: torch.Tensor) -> SAEOut:
        latents, stats, dead = self.encode(inputs)
        latents = torch.relu(latents)
        recons = self.decode(latents, stats)
        return SAEOut(latents, recons, dead)
