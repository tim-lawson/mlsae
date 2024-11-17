import os

from loguru import logger

from mlsae.model import MLSAETransformer
from mlsae.utils import get_repo_id


def find_ckpt_paths(
    ckpt_dir: str = "wandb_logs/lightning_logs", step: int = 7616
) -> list[str]:
    paths: list[str] = []
    for root, _, files in os.walk(ckpt_dir):
        for file in files:
            if file.endswith(f"step={step}.ckpt"):
                paths.append(os.path.join(root, file))
    return paths


def upload_models(ckpt_path: str, repo_id: str | None = None) -> None:
    logger.info(f"loading from: {ckpt_path}")
    model = MLSAETransformer.load_from_checkpoint(ckpt_path)

    # Remove the buffers, if we haven't already. This saves A LOT of space!
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
    repo_id = f"{repo_id}-tfm" if repo_id is not None else get_repo_id(model, True)
    save_dir = f"models/{repo_id}"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_directory=save_dir, repo_id=repo_id, push_to_hub=True)

    # The PyTorch autoencoder module, which is much smaller.
    repo_id = repo_id if repo_id is not None else get_repo_id(model, False)
    save_dir = f"models/{repo_id}"
    os.makedirs(save_dir, exist_ok=True)
    model.autoencoder.save_pretrained(
        save_directory=save_dir, repo_id=repo_id, push_to_hub=True
    )


if __name__ == "__main__":
    for path in find_ckpt_paths():
        upload_models(path)
