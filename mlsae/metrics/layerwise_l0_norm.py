import torch
from jaxtyping import Float, Int64
from torchmetrics import Metric


class LayerwiseL0Norm(Metric):
    """
    L0 norm (sparsity). Average count of nonzero latent activations.

    Fixed at k (the number of largest latents to keep) during training.
    """

    is_differentiable = False
    full_state_update = False

    layer_nonzero: Float[torch.Tensor, "n_layers"]
    """Layerwise count of nonzero latent activations."""

    tokens: Int64[torch.Tensor, ""]
    """Count of tokens."""

    def __init__(self, n_layers: int, dead_threshold: float) -> None:
        super().__init__()
        self.add_state(
            "layer_nonzero",
            torch.zeros(n_layers, dtype=torch.float),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "tokens", torch.zeros(1, dtype=torch.int64), dist_reduce_fx="sum"
        )
        self.dead_threshold = dead_threshold

    @torch.no_grad()
    def update(
        self, values: Float[torch.Tensor, "n_layers batch pos k"], **kwargs
    ) -> None:
        self.layer_nonzero.add_(torch.sum(values > self.dead_threshold, dim=(1, 2, 3)))
        self.tokens.add_(values.shape[1] * values.shape[2])

    @torch.no_grad()
    def compute(self) -> Float[torch.Tensor, "n_layers"]:
        return self.layer_nonzero / self.tokens
