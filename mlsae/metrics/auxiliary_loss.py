import torch
from jaxtyping import Float
from torchmetrics import Metric


class AuxiliaryLoss(Metric):
    """
    The auxiliary loss (AuxK) models the reconstruction error using the top-`k_aux` dead
    latents (typically `d_model // 2`) [Gao et al., 2024].
    Latents are flagged as dead during training if they have not activated for
    some predetermined number of tokens (typically 10 million).

    Then, given the reconstruction error of the main model `e = inputs - recons`, we
    define the auxiliary loss as the MSE between `e` and the reconstruction using the
    top `k_aux` dead latents.
    We compute the MSE normalization per token, because the scale of the error changes
    throughout training.
    """

    is_differentiable = True
    full_state_update = False

    auxk_coef: float
    """Coefficient of the auxiliary loss."""

    auxk_mse: Float[torch.Tensor, ""]
    """Sum of MSEs between reconstruction errors and top-`k_aux` reconstructions."""

    def __init__(self, auxk_coef: float) -> None:
        super().__init__()
        self.auxk_coef = auxk_coef
        self.add_state(
            "auxk_mse",
            torch.zeros(1, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        inputs: Float[torch.Tensor, "n_layers batch pos n_inputs"],
        recons: Float[torch.Tensor, "n_layers batch pos n_inputs"],
        auxk_recons: Float[torch.Tensor, "n_layers batch pos n_inputs"] | None,
        **kwargs,
    ) -> None:
        if auxk_recons is not None:
            error = inputs - recons
            self.auxk_mse.add_(
                (error - auxk_recons).pow(2).mean()
                / (error - torch.mean(error, dim=3, keepdim=True)).pow(2).mean()
            )

    def compute(self) -> Float[torch.Tensor, ""]:
        return self.auxk_coef * self.auxk_mse.nan_to_num(0)
