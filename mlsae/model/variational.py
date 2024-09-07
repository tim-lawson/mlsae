import math
from dataclasses import dataclass
from typing import cast

import einops
import torch
import torch.nn.functional as F
import wandb
from huggingface_hub import PyTorchModelHubMixin
from jaxtyping import Float, Int
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from simple_parsing import Serializable, parse
from torch.nn import Linear, Module, Softplus

from mlsae.model import DataModule
from mlsae.model.transformer import Transformer
from mlsae.trainer.config import RunConfig, initialize


@dataclass
class VSAEOutput:
    z_dist: torch.distributions.Distribution
    z_sample: torch.Tensor
    x_recon: torch.Tensor
    loss: torch.Tensor
    loss_recon: torch.Tensor
    loss_kl: torch.Tensor


class VSAE(Module, PyTorchModelHubMixin):
    def __init__(self, n_inputs: int, n_latents: int) -> None:
        super().__init__()

        self.n_inputs = n_inputs
        self.n_latents = n_latents

        self.encoder = Linear(n_inputs, 2 * n_latents)
        self.decoder = Linear(n_latents, n_inputs, bias=False)
        self.softplus = Softplus()

    def encode(self, x: torch.Tensor) -> torch.distributions.Distribution:
        x = einops.rearrange(x, "l b p f -> (l b p) f")
        x = self.encoder.forward(x)

        mu, logvar = torch.chunk(x, 2, dim=-1)
        scale = self.softplus.forward(logvar) + 1e-8
        scale_tril = torch.diag_embed(scale)

        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)

    def reparameterize(self, dist: torch.distributions.Distribution):
        return dist.rsample()

    def decode(self, z: torch.Tensor):
        return self.decoder.forward(z)

    def forward(self, x: torch.Tensor) -> VSAEOutput:
        dist = self.encode(x)
        z = self.reparameterize(dist)
        recon_x = self.decode(z)

        loss_recon = F.mse_loss(recon_x, x, reduction="none").sum(-1).mean()
        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(z.shape[-1], device=z.device)
            .unsqueeze(0)
            .expand(z.shape[0], -1, -1),
        )
        loss_kl = torch.distributions.kl.kl_divergence(dist, std_normal).mean()
        loss = loss_recon + loss_kl

        return VSAEOutput(
            z_dist=dist,
            z_sample=z,
            x_recon=recon_x,
            loss=loss,
            loss_recon=loss_recon,
            loss_kl=loss_kl,
        )


@dataclass
class VSAEConfig(Serializable):
    """The autoencoder configuration."""

    expansion_factor: int = 16
    """The ratio of the number of latents to the number of inputs."""

    lr: float = 1e-4
    """The learning rate."""

    skip_special_tokens: bool = True
    """Whether to ignore special tokens."""


class VSAETransformer(LightningModule, PyTorchModelHubMixin):
    def __init__(
        self,
        model_name: str = "EleutherAI/pythia-70m-deduped",
        layers: list[int] | None = None,
        expansion_factor: int = 16,
        lr: float = 1e-4,
        skip_special_tokens: bool = True,
        max_length: int = 2048,
        batch_size: int = 1,
        accumulate_grad_batches: int = 64,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.expansion_factor = expansion_factor
        self.lr = lr
        self.skip_special_tokens = skip_special_tokens
        self.max_length = max_length
        self.batch_size = batch_size
        self.accumulate_grad_batches = accumulate_grad_batches

        self.transformer = Transformer(
            self.model_name,
            self.max_length,
            self.batch_size,
            self.skip_special_tokens,
            layers=layers,
            device=self.device,
        )
        self.transformer.requires_grad_(False)

        self.layers = self.transformer.layers
        self.n_layers = self.transformer.n_layers
        self.n_inputs = self.transformer.config.hidden_size
        self.n_latents = self.n_inputs * self.expansion_factor

        self.vsae: VSAE = VSAE(self.n_inputs, self.n_latents)  # type: ignore

    def forward(self, tokens: Int[torch.Tensor, "batch pos"]) -> VSAEOutput:
        return self.vsae.forward(self.transformer.forward(tokens))

    def training_step(
        self, batch: dict[str, Int[torch.Tensor, "batch pos"]], batch_idx: int
    ) -> Float[torch.Tensor, ""]:
        output = self.vsae.forward(self.transformer.forward(batch["input_ids"]))
        self.log_dict(
            {
                "loss/total": output.loss,
                "loss/recon": output.loss_recon,
                "loss/kl": output.loss_kl,
            }
        )
        return output.loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.vsae.parameters(), lr=self.lr, eps=6.25e-10)  # type: ignore


def train(config: RunConfig) -> None:
    initialize(config.seed)

    data = DataModule(config.model_name, config=config.data)

    model: VSAETransformer = VSAETransformer(
        config.model_name,
        config.layers,
        config.autoencoder.expansion_factor,
        config.autoencoder.lr,
        config.autoencoder.skip_special_tokens,
        config.data.max_length,
        config.data.batch_size,
        config.trainer.accumulate_grad_batches,
    )  # type: ignore

    wandb.login()  # type: ignore

    trainer = Trainer(
        precision=cast(_PRECISION_INPUT, config.trainer.precision),
        logger=WandbLogger(
            name=config.run,
            save_dir="wandb_logs",
            project=config.project,
            log_model=True,
        ),
        max_steps=config.trainer.max_steps
        or math.ceil(config.data.max_steps / config.trainer.accumulate_grad_batches),
        limit_val_batches=config.trainer.limit_val_batches,
        val_check_interval=config.trainer.val_check_interval,
        log_every_n_steps=config.trainer.log_every_n_steps,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        deterministic=True,
        default_root_dir=config.trainer.default_root_dir,
    )

    trainer.fit(model, datamodule=data, ckpt_path=config.trainer.checkpoint_path)

    wandb.finish()  # type: ignore


if __name__ == "__main__":
    train(parse(RunConfig))
