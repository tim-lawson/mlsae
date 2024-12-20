import os

from mlsae.model import MLSAETransformer
from mlsae.utils import get_model_repo_id


def find_ckpt_paths(
    ckpt_dir: str = "wandb_logs/lightning_logs", step: int = 7616
) -> list[str]:
    paths: list[str] = []
    for root, _, files in os.walk(ckpt_dir):
        for file in files:
            if file.endswith(f"step={step}.ckpt"):
                paths.append(os.path.join(root, file))
    return paths


def upload_models(ckpt_path: str, dry_run: bool) -> None:
    print(f"loading from: {ckpt_path}")
    model = MLSAETransformer.load_from_checkpoint(ckpt_path, strict=False)

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
    repo_id_tfm = get_model_repo_id(model, True)
    print("repo_id (transformer):", repo_id_tfm)
    save_dir_tfm = f"models/{repo_id_tfm}"

    if not dry_run:
        os.makedirs(save_dir_tfm, exist_ok=True)
        model.save_pretrained(
            save_directory=save_dir_tfm,
            repo_id=repo_id_tfm,
            push_to_hub=True,
            model_card_kwargs=dict(
                model_id=repo_id_tfm,
                base_model=model.model_name,
                model_name=model.model_name,
                expansion_factor=model.expansion_factor,
                k=model.k,
            ),
        )

    # The PyTorch autoencoder module, which is much smaller.
    repo_id = get_model_repo_id(model, False)
    print("repo_id (autoencoder):", repo_id)
    save_dir = f"models/{repo_id}"

    if not dry_run:
        os.makedirs(save_dir, exist_ok=True)
        model.autoencoder.save_pretrained(
            save_directory=save_dir,
            repo_id=repo_id,
            push_to_hub=True,
            model_card_kwargs=dict(
                model_id=repo_id,
                base_model=model.model_name,
                model_name=model.model_name,
                expansion_factor=model.expansion_factor,
                k=model.k,
            ),
        )


if __name__ == "__main__":
    dry_run = False

    for path in find_ckpt_paths(step=7616):
        upload_models(path, dry_run)

    # NOTE: GPT-2 has max_length 1024, so we have twice as many steps for 1B tokens.
    for path in find_ckpt_paths(step=15232):
        upload_models(path, dry_run)
