import torch
from jaxtyping import Float
from torchmetrics import Metric


class LayerwiseLossDelta(Metric):
    """
    Downstream loss (replace the inputs by the reconstruction during the forward pass).

    The average delta between the cross-entropy loss for the inputs and reconstructions.
    """

    is_differentiable = False
    full_state_update = False

    layer_delta_loss: Float[torch.Tensor, "n_layers"]
    """Layerwise sum of deltas between cross-entropy losses."""

    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.add_state(
            "layer_delta_loss", default=torch.zeros(n_layers), dist_reduce_fx="mean"
        )

    @torch.no_grad()
    def update(
        self,
        loss_true: Float[torch.Tensor, "n_layers"],
        loss_pred: Float[torch.Tensor, "n_layers"],
        **kwargs,
    ) -> None:
        self.layer_delta_loss.add_(loss_pred - loss_true)

    @torch.no_grad()
    def compute(self) -> Float[torch.Tensor, "n_layers"]:
        return self.layer_delta_loss
