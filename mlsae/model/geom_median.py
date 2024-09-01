# Based on https://github.com/EleutherAI/sae/blob/19d95a401e9d17dbf7d6fb0fa7a91081f1b0d01f/sae/utils.py

import einops
import torch
from jaxtyping import Float


@torch.no_grad()
def geometric_median(
    points: Float[torch.Tensor, "layer batch pos n_inputs"],
    max_iter: int = 100,
    tol: float = 1e-5,
) -> Float[torch.Tensor, "n_inputs"]:
    """
    Compute the geometric median of the points along the last axis.

    Used to initialize the pre-encoder bias.

    Args:
        points (Float[torch.Tensor, "layer batch pos n_inputs"]): The points from
            which to compute the geometric median.

        max_iter (int): The maximum number of iterations. Defaults to 100.

        tol (float): The tolerance for early stopping. Defaults to 1e-5.

    Returns:
        out (Float[torch.Tensor, "n_inputs"]): The geometric median of the points along
            the last axis.
    """

    points = einops.rearrange(
        points, "layer batch pos n_inputs -> (layer batch pos) n_inputs"
    )
    curr = points.mean(dim=0)
    prev = torch.zeros_like(curr)
    weights = torch.ones(len(points), device=points.device)
    for _ in range(max_iter):
        prev = curr
        weights = 1 / torch.norm(points - curr, dim=1)
        weights /= weights.sum()
        curr = (weights.unsqueeze(1) * points).sum(dim=0)
        if torch.norm(curr - prev) < tol:
            break
    return curr
