import os
from typing import cast

import pandas as pd
import torch
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

from mlsae.model import DataConfig, MLSAETransformer
from mlsae.model.data import get_test_dataloader
from mlsae.model.types import TopK
from mlsae.trainer import RunConfig
from mlsae.trainer.config import initialize
from mlsae.utils import get_device

pythia_70m = "EleutherAI/pythia-70m-deduped"
pythia_160m = "EleutherAI/pythia-160m-deduped"
# pythia_410m = "EleutherAI/pythia-410m-deduped"
# pythia_1b = "EleutherAI/pythia-1b-deduped"

layers = {
    pythia_70m: range(6),
    pythia_160m: range(12),
    # pythia_410m: range(24),
    # pythia_1b: range(16),
}

config = RunConfig(data=DataConfig(max_tokens=1_000_000))


def test(repo_id: str, layer: int):
    device = get_device()
    initialize(config.seed)

    model = MLSAETransformer.from_pretrained(repo_id)
    model.requires_grad_(False)
    model.layers = [layer]
    model = model.to(device)

    dataloader = get_test_dataloader(
        model.model_name,
        config.data.max_length,
        config.data.batch_size,
        config.data.num_workers or 1,
    )

    # output = test_lightning(model, dataloader)
    output = test_manual(model, dataloader, device)
    output = {k: v.item() for k, v in output.items()}
    print(output)

    del model

    filename = f"test_{repo_id.split('/')[-1]}.csv"
    pd.DataFrame(output, index=[0]).to_csv(os.path.join("out", filename), index=False)


def test_manual(
    model: MLSAETransformer, dataloader: DataLoader[torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    for i, batch in enumerate(dataloader):
        if i >= config.data.max_steps:
            break

        tokens: torch.Tensor = batch["input_ids"].to(device)
        inputs = model.forward_lens(model.transformer.forward(tokens))

        # TODO: forgive me, o lord
        recons = torch.empty(inputs.shape, device=device)
        topk = TopK(
            torch.empty(
                (model.n_layers, model.batch_size, model.max_length, model.k),
                device=device,
            ),
            torch.empty(
                (model.n_layers, model.batch_size, model.max_length, model.k),
                device=device,
            ),
        )
        for layer in range(inputs.shape[0]):
            topk_, recons_, _, _, _ = model.autoencoder.forward(inputs[layer])
            recons[layer] = recons_
            topk.indices[layer] = topk_.indices
            topk.values[layer] = topk_.values

        model.train_metrics.forward(
            inputs=inputs,
            indices=topk.indices,
            values=topk.values,
            recons=recons,
        )

        recons = model.inverse_lens(recons)

        model.forward_at_layer(inputs, recons, tokens)
        model.val_metrics.forward(
            loss_true=model.loss_true,
            loss_pred=model.loss_pred,
            logits_true=model.logits_true,
            logits_pred=model.logits_pred,
        )
        model.mse_loss.forward(inputs=inputs, recons=recons)

    return {
        **model.train_metrics.compute(),
        **model.val_metrics.compute(),
        "mse_loss": model.mse_loss.compute(),
        "aux_loss": model.aux_loss.compute(),
        "loss": model.mse_loss.compute() + model.aux_loss.compute(),
    }


# TODO: I don't trust this.
def test_lightning(model: MLSAETransformer, dataloader: DataLoader[torch.Tensor]):
    trainer = Trainer(
        precision=cast(_PRECISION_INPUT, config.trainer.precision),
        limit_test_batches=config.data.max_steps,
        deterministic=True,
    )
    return trainer.test(model, dataloaders=dataloader)


def main() -> None:
    for repo_id, layer in [
        ("tim-lawson/sae-pythia-70m-deduped-x64-k32-tfm-layers-0", 0),
        ("tim-lawson/sae-pythia-70m-deduped-x64-k32-tfm-layers-1", 1),
        ("tim-lawson/sae-pythia-70m-deduped-x64-k32-tfm-layers-2", 2),
        ("tim-lawson/sae-pythia-70m-deduped-x64-k32-tfm-layers-3", 3),
        ("tim-lawson/sae-pythia-70m-deduped-x64-k32-tfm-layers-4", 4),
        ("tim-lawson/sae-pythia-70m-deduped-x64-k32-tfm-layers-5", 5),
    ]:
        test(repo_id, layer)


if __name__ == "__main__":
    main()
