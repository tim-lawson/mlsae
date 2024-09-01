import torch
from jaxtyping import Float
from torchmetrics import Metric


class MSELoss(Metric):
    """
    The average FVU of the main model `e = inputs - recons`, where `recons` is the
    reconstruction using the top-k latents.

    Equivalent to normalized MSE in Gao et al. [2024], except we compute the variance
    per batch instead of once at the beginning of training.
    """

    is_differentiable = True
    full_state_update = False

    layer_mse: Float[torch.Tensor, "n_layers"]
    """Layerwise sum of MSEs between the inputs and reconstructions."""

    layer_var: Float[torch.Tensor, "n_layers"]
    """Layerwise sum of variances of the inputs."""

    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.add_state(
            "layer_mse", torch.zeros(n_layers, dtype=torch.float), dist_reduce_fx="sum"
        )
        self.add_state(
            "layer_var", torch.zeros(n_layers, dtype=torch.float), dist_reduce_fx="sum"
        )

    def update(
        self,
        inputs: Float[torch.Tensor, "n_layers batch pos n_inputs"],
        recons: Float[torch.Tensor, "n_layers batch pos n_inputs"],
        **kwargs,
    ) -> None:
        self.layer_mse.add_(torch.mean((recons - inputs).pow(2), dim=(1, 2, 3)))
        self.layer_var.add_(torch.var(inputs, dim=(1, 2, 3)))

    def compute(self) -> Float[torch.Tensor, ""]:
        return (self.layer_mse / self.layer_var).mean()
