import torch
from jaxtyping import Float, Int
from torchmetrics import Metric


class DeadLatents(Metric):
    """
    Estimate the fraction of dead latents from the number of tokens activated by each
    latent and the number of tokens elapsed in a training step.

    Note that we consider a latent live if it is activated *at any layer*.
    """

    is_differentiable = False
    full_state_update = False

    latent_tokens: Float[torch.Tensor, "n_latents"]
    """Count of tokens activated by each latent."""

    tokens: Int[torch.Tensor, ""]
    """Count of tokens."""

    def __init__(self, n_latents: int, dead_tokens_threshold: float) -> None:
        super().__init__()
        self.n_latents = n_latents
        self.dead_tokens_threshold = dead_tokens_threshold
        self.add_state(
            "latent_tokens",
            torch.zeros(n_latents, dtype=torch.float),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "tokens", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum"
        )

    @torch.no_grad()
    def update(
        self, indices: Int[torch.Tensor, "n_layers batch pos k"], **kwargs
    ) -> None:
        self.latent_tokens.add_(
            torch.bincount(indices.reshape(-1), minlength=self.n_latents)
        )
        self.tokens += indices.shape[1] * indices.shape[2]

    @torch.no_grad()
    def compute(self) -> torch.Tensor:
        return (
            torch.sum(
                self.latent_tokens < self.tokens / self.dead_tokens_threshold,
                dtype=torch.float,
            )
            / self.n_latents
        )
