from dataclasses import dataclass
from functools import partial

import torch
import wandb
import wandb.plot
from huggingface_hub import PyTorchModelHubMixin
from jaxtyping import Float, Int
from lightning.pytorch import LightningModule
from simple_parsing import Serializable
from torchmetrics import MetricCollection

from mlsae.metrics import (
    AuxiliaryLoss,
    DeadLatents,
    LayerwiseFVU,
    LayerwiseL1Norm,
    LayerwiseLogitKLDiv,
    LayerwiseLogitMSE,
    LayerwiseLossDelta,
    LayerwiseMSE,
    LayerwiseWrapper,
    MSELoss,
)
from mlsae.model.autoencoder import (
    MLSAE,
    AutoencoderOutput,
    unit_norm_decoder,
    unit_norm_decoder_gradient,
)
from mlsae.model.geom_median import geometric_median
from mlsae.model.transformer import Transformer


@dataclass
class MLSAEConfig(Serializable):
    """The autoencoder configuration."""

    dead_tokens_threshold: int = 10_000_000
    """The number of tokens after which a latent is flagged as dead during training."""

    expansion_factor: int = 16
    """The ratio of the number of latents to the number of inputs."""

    k: int = 32
    """The number of largest latents to keep."""

    # TODO: Make this optional and default to a power of 2 close to d_model / 2.
    auxk: int | None = 256
    """The number of dead latents with which to model the reconstruction error."""

    auxk_coef: float | None = 1 / 32
    """The coefficient of the auxiliary loss."""

    dead_threshold: float = 1e-3
    """The threshold activation for a latent to be considered activated."""

    # TODO: Make this optional and default to the scaling law from Gao et al [2024].
    lr: float = 1e-4
    """The learning rate."""

    standardize: bool = True
    """Whether to standardize the inputs."""

    skip_special_tokens: bool = True
    """Whether to ignore special tokens."""

    lens: bool = False
    """Whether to learn a layer-specific transform before/after the encoder/decoder."""


class MLSAETransformer(
    LightningModule,
    PyTorchModelHubMixin,
    repo_url="https://github.com/tim-lawson/mlsae",
    language="en",
    library_name="mlsae",
    license="mit",
):
    loss_true: Float[torch.Tensor, "n_layers"]
    loss_pred: Float[torch.Tensor, "n_layers"]
    logits_true: Float[torch.Tensor, "n_layers pos d_vocab"]
    logits_pred: Float[torch.Tensor, "n_layers pos d_vocab"]

    def __init__(
        self,
        model_name: str = "EleutherAI/pythia-70m-deduped",
        # TODO: Check this works for non-consecutive layers
        layers: list[int] | None = None,
        expansion_factor: int = 16,
        k: int = 32,
        auxk: int | None = 256,
        auxk_coef: float | None = 1 / 32,
        dead_tokens_threshold: int = 10_000_000,
        dead_threshold: float = 1e-3,
        lr: float = 1e-4,
        standardize: bool = True,
        skip_special_tokens: bool = True,
        max_length: int = 2048,
        batch_size: int = 1,
        accumulate_grad_batches: int = 64,
        lens: bool = False,
        # NOTE: These are only used for loading pretrained models
        dead_steps_threshold: int | None = None,
    ) -> None:
        """
        Multi-Layer Sparse Autoencoder (MLSAE) PyTorch Lightning module.
        Includes the underlying transformer.

        References:

        - [Gao et al., 2024. Scaling and evaluating sparse autoencoders.](https://cdn.openai.com/papers/sparse-autoencoders.pdf)
        - [Bricken et al., 2023. Towards Monosemanticity.](https://transformer-circuits.pub/2023/monosemantic-features)

        Args:
            model_name (str): The name of a pretrained GPTNeoXForCausalLM model.

            layers (list[int] | None): The layers to train on.
                If None, all layers are trained on. Defaults to None.

            expansion_factor (int): The ratio of the number of latents to the number of
                inputs. Defaults to 16.

            k (int): The number of largest latents to keep. Defaults to 32.

            auxk (int | None): The number of dead latents with which to model the
                reconstruction error. Defaults to 256.

            auxk_coef (float | None): The coefficient of the auxiliary loss.
                Defaults to 1 / 32.

            dead_tokens_threshold (int): The number of tokens after which a latent is
                flagged as dead during training. Defaults to 10 million.

            dead_threshold (float): The threshold for a latent to be considered
                activated. Defaults to 1e-3.

            lr (float): The learning rate. Defaults to 1e-4.

            standardize (bool): Whether to standardize the inputs. Defaults to True.

            skip_special_tokens (bool): Whether to ignore special tokens.
                Defaults to True.

            max_length (int): The maximum length of a tokenized input sequence.
                Defaults to 2048.

            batch_size (int): The number of sequences in a batch. Defaults to 1.

            accumulate_grad_batches (int): The number of batches over which to
                accumulate gradients. Defaults to 64.
        """

        super().__init__()

        self.model_name = model_name
        self.expansion_factor = expansion_factor
        self.k = k
        self.auxk = auxk
        self.auxk_coef = auxk_coef
        self.dead_tokens_threshold = dead_tokens_threshold
        self.dead_threshold = dead_threshold
        self.lr = lr
        self.standardize = standardize
        self.skip_special_tokens = skip_special_tokens
        self.max_length = max_length
        self.batch_size = batch_size
        self.accumulate_grad_batches = accumulate_grad_batches
        self.lens = lens

        # Set the number of steps after which a latent is flagged as dead from the
        # number of tokens per batch and the number of batches per step.
        self.dead_steps_threshold = (
            dead_steps_threshold
            or self.dead_tokens_threshold
            // (self.batch_size * self.max_length * self.accumulate_grad_batches)
        )

        self.transformer = Transformer(
            self.model_name,
            self.max_length,
            self.batch_size,
            self.skip_special_tokens,
            layers=layers,
            device=self.device,
        )
        self.transformer.eval()

        self.layers = self.transformer.layers
        self.n_layers = self.transformer.n_layers
        self.n_inputs = self.transformer.config.hidden_size
        self.n_latents = self.n_inputs * self.expansion_factor

        self.save_hyperparameters(ignore=["autoencoder", "transformer"])

        self.autoencoder: MLSAE = MLSAE(
            self.layers,
            self.n_inputs,
            self.n_latents,
            self.k,
            self.dead_steps_threshold,
            self.dead_threshold,
            self.auxk,
            self.standardize,
            self.lens,
        )  # type: ignore

        self.mse_loss = MSELoss(self.n_layers)
        self.aux_loss = AuxiliaryLoss(self.auxk_coef or 0.0)

        wrap = partial(
            LayerwiseWrapper,
            labels=[f"layer_{layer}" for layer in self.transformer.layers],
        )

        self.train_metrics = MetricCollection(
            {
                "dead/rel": DeadLatents(self.n_latents, self.dead_tokens_threshold),
                "l1": wrap(LayerwiseL1Norm(self.n_layers), prefix="l1/"),
                "mse": wrap(LayerwiseMSE(self.n_layers), prefix="mse/"),
                "fvu": wrap(LayerwiseFVU(self.n_layers), prefix="fvu/"),
            },
            prefix="train/",
        )

        self.val_metrics = MetricCollection(
            {
                "loss/delta": wrap(
                    LayerwiseLossDelta(self.n_layers), prefix="loss/delta/"
                ),
                "logit/mse": wrap(
                    LayerwiseLogitMSE(self.n_layers), prefix="logit/mse/"
                ),
                "logit/kldiv": wrap(
                    LayerwiseLogitKLDiv(self.n_layers), prefix="logit/kldiv/"
                ),
            },
            prefix="val/",
        )

        logits = (
            self.n_layers,
            self.transformer.batch_size,
            self.transformer.max_length,
            self.transformer.config.vocab_size,
        )
        self.register_buffer("loss_true", torch.zeros(self.n_layers))
        self.register_buffer("loss_pred", torch.zeros(self.n_layers))
        self.register_buffer("logits_true", torch.zeros(logits))
        self.register_buffer("logits_pred", torch.zeros(logits))

    def forward(self, tokens: Int[torch.Tensor, "batch pos"]) -> AutoencoderOutput:
        return self.autoencoder.forward(self.transformer.forward(tokens))

    def training_step(
        self, batch: dict[str, Int[torch.Tensor, "batch pos"]], batch_idx: int
    ) -> Float[torch.Tensor, ""]:
        inputs = self.transformer.forward(batch["input_ids"])

        if batch_idx == 0:
            self.autoencoder.pre_encoder_bias.data = geometric_median(inputs)

        topk, recons, auxk, auxk_recons, dead = self.autoencoder.forward(inputs)

        train_metrics = self.train_metrics.forward(
            inputs=inputs,
            indices=topk.indices,
            values=topk.values,
            recons=recons,
        )

        mse_loss = self.mse_loss.forward(inputs=inputs, recons=recons)
        aux_loss = self.aux_loss.forward(
            inputs=inputs, recons=recons, auxk_recons=auxk_recons
        )
        loss = mse_loss + aux_loss

        self.log_dict(
            {
                **train_metrics,
                "loss/total": loss,
                "loss/mse": mse_loss,
                "loss/auxk": aux_loss,
                "train/dead/abs": dead,
            }
        )

        return loss

    def forward_at_layer(
        self,
        inputs: Float[torch.Tensor, "n_layers batch pos n_inputs"],
        recons: Float[torch.Tensor, "n_layers batch pos n_inputs"],
        tokens: Int[torch.Tensor, "batch pos"],
    ) -> None:
        for layer in range(self.n_layers):
            loss, logits = self.transformer.forward_at_layer(
                inputs, layer, return_type="both", tokens=tokens
            )
            self.loss_true[layer] = loss
            self.logits_true[layer] = logits

            loss, logits = self.transformer.forward_at_layer(
                recons, layer, return_type="both", tokens=tokens
            )
            self.loss_pred[layer] = loss
            self.logits_pred[layer] = logits

    @torch.no_grad()
    def validation_step(self, batch: dict[str, Int[torch.Tensor, "batch pos"]]) -> None:
        tokens = batch["input_ids"]
        inputs = self.transformer.forward(tokens)
        topk, recons, auxk, auxk_recons, dead = self.autoencoder.forward(inputs)

        self.forward_at_layer(inputs, recons, tokens)
        val_metrics = self.val_metrics.forward(
            loss_true=self.loss_true,
            loss_pred=self.loss_pred,
            logits_true=self.logits_true,
            logits_pred=self.logits_pred,
        )

        self.log_dict(val_metrics)

    @torch.no_grad()
    def test_step(self, batch: dict[str, Int[torch.Tensor, "batch pos"]]) -> None:
        tokens = batch["input_ids"]
        inputs = self.transformer.forward(tokens)
        topk, recons, auxk, auxk_recons, dead = self.autoencoder.forward(inputs)

        train_metrics = self.train_metrics.forward(
            inputs=inputs,
            indices=topk.indices,
            values=topk.values,
            recons=recons,
        )

        self.forward_at_layer(inputs, recons, tokens)
        val_metrics = self.val_metrics.forward(
            loss_true=self.loss_true,
            loss_pred=self.loss_pred,
            logits_true=self.logits_true,
            logits_pred=self.logits_pred,
        )

        mse_loss = self.mse_loss.forward(inputs=inputs, recons=recons)
        aux_loss = self.aux_loss.forward(
            inputs=inputs, recons=recons, auxk_recons=auxk_recons
        )
        loss = mse_loss + aux_loss

        self.log_dict(
            {
                **train_metrics,
                **val_metrics,
                "loss/total": loss,
                "loss/mse": mse_loss,
                "loss/auxk": aux_loss,
            }
        )

    def on_after_backward(self) -> None:
        unit_norm_decoder(self.autoencoder.decoder)
        unit_norm_decoder_gradient(self.autoencoder.decoder)

    def on_train_end(self) -> None:
        del self.loss_true
        del self.loss_pred
        del self.logits_true
        del self.logits_pred
        del self.autoencoder.last_nonzero

    def configure_optimizers(self):
        return torch.optim.Adam(  # type: ignore
            self.autoencoder.parameters(), lr=self.lr, eps=6.25e-10
        )

    def _log_latent_histograms(
        self, values: Float[torch.Tensor, "layer batch pos k"]
    ) -> None:
        for layer in range(self.n_layers):
            title = f"latent/layer_{layer}"
            table = wandb.Table(
                # Convert 3-d tensor (batch, pos, n_latents) to 2-d array [[x], ...]
                data=values[layer].detach().cpu().numpy().reshape(-1, 1),
                columns=["latent"],
            )
            wandb.log({title: wandb.plot.histogram(table, "latent", title=title)})
