# Based on https://github.com/openai/sparse_autoencoder/blob/4965b941e9eb590b00b253a2c406db1e1b193942/sparse_autoencoder/train.py

from typing import NamedTuple

import einops
import torch
from huggingface_hub import PyTorchModelHubMixin
from jaxtyping import Float
from torch.nn import Linear, Module, ModuleList, Parameter

from mlsae.model.decoder import decode
from mlsae.model.types import Stats, TopK


class EncoderOutput(NamedTuple):
    """The output of the encoder forward pass."""

    topk: TopK
    """The k largest latents."""

    auxk: TopK | None
    """If auxk is not None, the auxk largest dead latents."""

    stats: Stats | None
    """If normalize is True, the mean and standard deviation of the inputs."""

    dead: Float[torch.Tensor, ""]
    """The fraction of dead latents."""


class AutoencoderOutput(NamedTuple):
    """The output of the autoencoder forward pass."""

    topk: TopK
    """The k largest latents."""

    recons: Float[torch.Tensor, "layer batch pos n_inputs"]
    """The reconstructions from the k largest latents."""

    auxk: TopK | None
    """If auxk is not None, the auxk largest dead latents."""

    auxk_recons: Float[torch.Tensor, "layer batch pos n_inputs"] | None
    """If auxk is not None, the reconstructions from the auxk largest dead latents."""

    dead: Float[torch.Tensor, ""]
    """The fraction of dead latents."""


class MLSAE(
    Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/tim-lawson/mlsae",
    language="en",
    library_name="mlsae",
    license="mit",
):
    """
    Multi-Layer Sparse Autoencoder (MLSAE) PyTorch module.

    References:

    - [Gao et al., 2024. Scaling and evaluating sparse autoencoders.](https://cdn.openai.com/papers/sparse-autoencoders.pdf)
    - [Bricken et al., 2023. Towards Monosemanticity.](https://transformer-circuits.pub/2023/monosemantic-features)
    """

    last_nonzero: Float[torch.Tensor, "n_latents"]
    """The number of steps since the latents have activated."""

    def __init__(
        self,
        layers: list[int],
        n_inputs: int,
        n_latents: int,
        k: int,
        dead_steps_threshold: int,
        dead_threshold: float = 1e-3,
        # TODO: Make this optional and default to a power of 2 close to d_model / 2.
        auxk: int | None = 256,
        standardize: bool = True,
        lens: bool = False,
    ) -> None:
        """
        Args:
            layers (list[int] | None): The layers to train on.

            n_inputs (int): The number of inputs.

            n_latents(int): The number of latents.

            k (int): The number of largest latents to keep.

            dead_steps_threshold (int): The number of steps after which a latent is
                flagged as dead during training.

            dead_threshold (float): The threshold for a latent to be considered
                activated. Defaults to 1e-3.

            auxk (int | None): The number of dead latents with which to model the
                reconstruction error. Defaults to 256.

            standardize (bool): Whether to standardize the inputs. Defaults to True.

            lens (bool): Whether to learn a layer-specific transform before/after the
                encoder/decoder. Defaults to False.
        """

        super().__init__()

        self.n_inputs = n_inputs
        self.n_latents = n_latents
        self.k = k
        self.auxk = auxk
        self.dead_steps_threshold = dead_steps_threshold
        self.dead_threshold = dead_threshold
        self.standardize = standardize
        self.lens = lens
        self.layers = layers

        self.encoder = Linear(n_inputs, n_latents, bias=False)
        self.decoder = Linear(n_latents, n_inputs, bias=False)
        self.pre_encoder_bias = Parameter(torch.zeros(n_inputs))

        self.register_buffer("last_nonzero", torch.zeros(n_latents, dtype=torch.long))

        self.decoder.weight.data = self.encoder.weight.data.T.clone()
        self.decoder.weight.data = self.decoder.weight.data.T.contiguous().T
        unit_norm_decoder(self.decoder)

        if self.lens:
            self.encoder_lens = ModuleList(
                [Linear(n_inputs, n_inputs, bias=False) for _ in self.layers]
            )
            self.decoder_lens = ModuleList(
                [Linear(n_inputs, n_inputs, bias=False) for _ in self.layers]
            )

    def encode(
        self, inputs: Float[torch.Tensor, "layer batch pos n_inputs"]
    ) -> EncoderOutput:
        stats = None
        if self.standardize:
            inputs, stats = standardize(inputs)

        if self.lens:
            # Apply layer-specific transforms before the encoder
            for i, _layer in enumerate(self.layers):
                inputs[i, ...] = self.encoder_lens[i](inputs[i, ...])

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

        return EncoderOutput(topk, auxk, stats, dead)

    def decode(
        self, topk: TopK, stats: Stats | None = None
    ) -> Float[torch.Tensor, "layer batch pos n_inputs"]:
        recons = decode(topk, self.decoder.weight) + self.pre_encoder_bias

        if self.lens:
            # Apply layer-specific transforms after the decoder
            for i, _layer in enumerate(self.layers):
                recons[i, ...] = self.decoder_lens[i](recons[i, ...])

        if stats is not None:
            recons = recons * stats.std + stats.mean

        return recons

    def forward(
        self, inputs: Float[torch.Tensor, "layer batch pos n_inputs"]
    ) -> AutoencoderOutput:
        topk, auxk, stats, dead = self.encode(inputs)

        # Apply ReLU to ensure the k largest latents are non-negative
        values = torch.relu(topk.values)
        topk = TopK(values, topk.indices)
        recons = self.decode(topk, stats)

        if auxk is not None:
            auxk_values = torch.relu(auxk.values)
            auxk = TopK(auxk_values, auxk.indices)
            auxk_recons = self.decode(auxk)

        return AutoencoderOutput(topk, recons, auxk, auxk_recons, dead)


def unit_norm_decoder(decoder: Linear) -> None:
    """Unit-normalize the decoder weight vectors."""

    decoder.weight.data /= decoder.weight.data.norm(dim=0)


# TODO: Use kernels.triton_add_mul_ if it's available
@torch.no_grad()
def unit_norm_decoder_gradient(decoder: Linear) -> None:
    """
    Remove the component of the gradient parallel to the decoder weight vectors.
    Assumes that the decoder weight vectors are unit-normalized.
    NOTE: Without `@torch.no_grad()`, this causes a memory leak!
    """

    assert decoder.weight.grad is not None
    scalar = einops.einsum(
        decoder.weight.grad,
        decoder.weight,
        "... n_latents n_inputs, ... n_latents n_inputs -> ... n_inputs",
    )
    vector = einops.einsum(
        scalar,
        decoder.weight,
        "... n_inputs, ... n_latents n_inputs -> ... n_latents n_inputs",
    )
    decoder.weight.grad -= vector


def standardize(
    x: Float[torch.Tensor, "... n_inputs"], eps: float = 1e-5
) -> tuple[Float[torch.Tensor, "... n_inputs"], Stats]:
    """Standardize the inputs to zero mean and unit variance."""

    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, Stats(mu, std)
