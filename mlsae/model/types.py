from typing import NamedTuple

import torch
from jaxtyping import Float, Int


class TopK(NamedTuple):
    """The k largest latents. Wraps 'torch.return_types.topk'."""

    values: Float[torch.Tensor, "layer batch pos k"]
    """The values of the k largest latents."""

    indices: Int[torch.Tensor, "layer batch pos k"]
    """The indices of the k largest latents."""


class Stats(NamedTuple):
    """Used to standardize the input activation vectors."""

    mean: torch.Tensor
    std: torch.Tensor
