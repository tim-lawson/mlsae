import torch
from jaxtyping import Float
from torchmetrics import Metric


class LayerwiseMSE(Metric):
    """Mean squared error (MSE) or L2 reconstruction loss."""

    is_differentiable = True
    full_state_update = False

    layer_mse: Float[torch.Tensor, "n_layers"]
    """Layerwise mean of MSEs between inputs and reconstructions."""

    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.add_state(
            "layer_mse", torch.zeros(n_layers, dtype=torch.float), dist_reduce_fx="mean"
        )

    @torch.no_grad()
    def update(
        self,
        inputs: Float[torch.Tensor, "n_layers batch pos n_inputs"],
        recons: Float[torch.Tensor, "n_layers batch pos n_inputs"],
        **kwargs,
    ) -> None:
        self.layer_mse.add_(torch.mean((recons - inputs).pow(2), dim=(1, 2, 3)))

    @torch.no_grad()
    def compute(self) -> Float[torch.Tensor, "n_layers"]:
        return self.layer_mse / self.update_count
