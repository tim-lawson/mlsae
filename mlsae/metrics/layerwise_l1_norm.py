import torch
from jaxtyping import Float, Int64
from torchmetrics import Metric


class LayerwiseL1Norm(Metric):
    """L1 norm. Average sum of absolute latent activations."""

    is_differentiable = False
    full_state_update = False

    layer_abs: Float[torch.Tensor, "n_layers"]
    """Layerwise sum of absolute latent activations."""

    tokens: Int64[torch.Tensor, ""]
    """Layerwise count of tokens."""

    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.add_state(
            "layer_abs", torch.zeros(n_layers, dtype=torch.float), dist_reduce_fx="sum"
        )
        self.add_state(
            "tokens", torch.zeros(1, dtype=torch.int64), dist_reduce_fx="sum"
        )

    @torch.no_grad()
    def update(
        self, values: Float[torch.Tensor, "n_layers batch pos k"], **kwargs
    ) -> None:
        self.layer_abs.add_(torch.sum(torch.abs(values), dim=(1, 2, 3)))
        self.tokens.add_(values.shape[1] * values.shape[2])

    @torch.no_grad()
    def compute(self) -> Float[torch.Tensor, "n_layers"]:
        return self.layer_abs / self.tokens
