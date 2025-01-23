import os
from pprint import pprint

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mlsae.model import DataConfig, MLSAETransformer, get_test_dataloader
from mlsae.trainer import RunConfig, initialize
from mlsae.utils import forward_single_layer, get_device, get_repo_id, load_single_layer

pythia_70m = "EleutherAI/pythia-70m-deduped"
pythia_160m = "EleutherAI/pythia-160m-deduped"
pythia_410m = "EleutherAI/pythia-410m-deduped"
# pythia_1b = "EleutherAI/pythia-1b-deduped"

layers = {
    pythia_70m: range(6),
    pythia_160m: range(12),
    pythia_410m: range(24),
    # pythia_1b: range(16),
}

config = RunConfig(data=DataConfig(max_tokens=1_000_000))


def test(model_name: str, layer: int, tuned_lens: bool):
    initialize(config.seed)
    device = get_device()

    model = load_single_layer(model_name, layer, device)

    dataloader = get_test_dataloader(
        model.model_name,
        config.data.max_length,
        config.data.batch_size,
        config.data.num_workers or 1,
    )

    output = test_manual(model, dataloader, device)
    output = {k: v.item() for k, v in output.items()}
    pprint(output)

    filename_repo_id = get_repo_id(model_name, 64, 32, tuned_lens, True, [layer])
    filename = f"test_{filename_repo_id.split('/')[-1]}.csv"
    pd.DataFrame(output, index=[0]).to_csv(os.path.join("out", filename), index=False)


def test_manual(
    model: MLSAETransformer, dataloader: DataLoader[torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    def compute() -> dict[str, torch.Tensor]:
        return {
            **model.train_metrics.compute(),
            **model.val_metrics.compute(),
            "loss/mse": model.mse_loss.compute(),
            "loss/auxk": model.aux_loss.compute(),
            "loss/total": model.mse_loss.compute() + model.aux_loss.compute(),
        }

    pbar = tqdm(total=config.data.max_steps)
    for i, batch in enumerate(dataloader):
        if i >= config.data.max_steps:
            break

        tokens: torch.Tensor = batch["input_ids"].to(device)
        inputs, recons, topk = forward_single_layer(model, tokens)

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
            # logits_true=model.logits_true,
            # logits_pred=model.logits_pred,
        )

        model.mse_loss.forward(inputs=inputs, recons=recons)

        # pbar.write(str(compute()))
        pbar.update(1)

    return compute()


def main() -> None:
    for model_name in [pythia_70m, pythia_160m, pythia_410m]:
        for layer in layers[model_name]:
            test(model_name, layer, False)

    for model_name in [pythia_70m]:
        for layer in layers[model_name]:
            test(model_name, layer, True)


if __name__ == "__main__":
    main()
