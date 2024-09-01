import torch
import torch.nn.functional as F
from jaxtyping import Float
from torchmetrics import Metric


class LayerwiseLogitMSE(Metric):
    """
    Downstream loss (replace the inputs by the reconstruction during the forward pass).

    The MSE between the logits for the inputs and reconstructions.
    """

    is_differentiable = False
    full_state_update = False

    layer_logit_mse: Float[torch.Tensor, "n_layers"]
    """Layerwise sum of MSEs between logits."""

    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.add_state(
            "layer_logit_mse",
            default=torch.zeros(n_layers),
            dist_reduce_fx="mean",
        )

    @torch.no_grad()
    def update(
        self,
        logits_true: Float[torch.Tensor, "n_layers batch pos d_vocab"],
        logits_pred: Float[torch.Tensor, "n_layers batch pos d_vocab"],
        **kwargs,
    ) -> None:
        # NOTE: Iterate over layers to reduce memory usage.
        for layer in range(self.n_layers):
            self.layer_logit_mse[layer].add_(
                F.mse_loss(logits_true[layer], logits_pred[layer])
            )

    @torch.no_grad()
    def compute(self) -> Float[torch.Tensor, "n_layers"]:
        return self.layer_logit_mse
