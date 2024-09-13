import os
from dataclasses import dataclass

from huggingface_hub import HfApi
from loguru import logger
from simple_parsing import Serializable, parse

from mlsae.model import MLSAETransformer
from mlsae.utils import get_repo_id


@dataclass
class Config(Serializable):
    ckpt_path: str
    """The path to a model checkpoint."""

    repo_id: str | None = None
    """The repo_id to use for the model. If None, a repo_id will be generated."""


if __name__ == "__main__":
    config = parse(Config)
    api = HfApi()

    logger.info(f"loading from: {config.ckpt_path}")
    model = MLSAETransformer.load_from_checkpoint(config.ckpt_path)

    # NOTE: Remove the buffers, if we haven't already. This saves A LOT of space!
    if hasattr(model, "loss_true"):
        del model.loss_true
    if hasattr(model, "loss_pred"):
        del model.loss_pred
    if hasattr(model, "logits_true"):
        del model.logits_true
    if hasattr(model, "logits_pred"):
        del model.logits_pred
    if hasattr(model.autoencoder, "last_nonzero"):
        del model.autoencoder.last_nonzero

    # The PyTorch Lightning module, which includes the underlying transformer.
    repo_id = (
        f"{config.repo_id}-tfm"
        if config.repo_id is not None
        else get_repo_id(
            model_name=model.model_name,
            expansion_factor=model.expansion_factor,
            k=model.k,
            transformer=True,
            tuned_lens=model.tuned_lens,
        )
    )

    save_dir = f"models/{repo_id}"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(
        save_directory=save_dir,
        repo_id=repo_id,
        push_to_hub=True,
    )

    # The PyTorch autoencoder module, which is much smaller.
    repo_id = (
        config.repo_id
        if config.repo_id is not None
        else get_repo_id(
            model_name=model.model_name,
            expansion_factor=model.expansion_factor,
            k=model.k,
            tuned_lens=model.tuned_lens,
        )
    )
    save_dir = f"models/{repo_id}"
    os.makedirs(save_dir, exist_ok=True)
    model.autoencoder.save_pretrained(
        save_directory=save_dir,
        repo_id=repo_id,
        push_to_hub=True,
    )
