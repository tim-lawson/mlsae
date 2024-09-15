import json
import os
import sqlite3
from collections.abc import Generator
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import NamedTuple

import einops
import torch
from datasets import Dataset, DatasetDict, load_dataset
from loguru import logger
from simple_parsing import Serializable, field, parse
from tqdm import tqdm

from mlsae.model import DataConfig, MLSAETransformer
from mlsae.model.data import get_train_dataloader
from mlsae.trainer import initialize
from mlsae.utils import get_device


@dataclass
class Config(Serializable):
    repo_id: str
    """
    The name of a pretrained autoencoder and transformer from HuggingFace, or the path
    to a directory that contains them.
    """

    data: DataConfig = field(default_factory=DataConfig)
    """The data configuration. Remember to set max_tokens to a reasonable value!"""

    seed: int = 42
    """The seed for global random state."""

    n_examples: int = 32
    """The number of examples to keep for each latent and layer."""

    n_tokens: int = 4
    """The number of tokens to include either side of the maximum activation."""

    delete_every_n_steps: int = 10
    """The number of steps between deleting examples not in the top n_examples."""

    push_to_hub: bool = True
    """Whether to push the dataset to HuggingFace."""


class Example(NamedTuple):
    latent: int
    layer: int
    token_id: int
    token: str
    act: float
    token_ids: list[int]
    tokens: list[str]
    acts: list[float]

    def serialize(self) -> tuple[int | str | float, ...]:
        return (
            self.latent,
            self.layer,
            self.token_id,
            self.token,
            self.act,
            json.dumps(self.token_ids),
            json.dumps(self.tokens),
            json.dumps(self.acts),
        )

    @staticmethod
    def from_row(row: tuple) -> "Example":
        return Example(
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            json.loads(row[5]),
            json.loads(row[6]),
            json.loads(row[7]),
        )

    @staticmethod
    def from_dict(data: dict) -> "Example":
        return Example(
            data["latent"],
            data["layer"],
            data["token_id"],
            data["token"],
            data["act"],
            json.loads(data["token_ids"]),
            json.loads(data["tokens"]),
            json.loads(data["acts"]),
        )


class Examples:
    conn: sqlite3.Connection

    def __init__(self, repo_id: str) -> None:
        repo_id = Examples.repo_id(repo_id)
        filename = Examples.filename(repo_id)

        if Path(filename).exists():
            logger.info(f"connecting to database: {filename}")
            self.conn = sqlite3.connect(filename)

        else:
            logger.info(f"loading dataset: {repo_id}")
            dataset = load_dataset(repo_id)
            assert isinstance(dataset, DatasetDict)
            dataset = dataset["train"]

            logger.info(f"creating database: {filename}")
            self.conn, cursor = create_db(filename)
            batch = []
            for i, example in tqdm(enumerate(dataset.to_list())):
                batch.append(Example.from_dict(example))
                if i % 1000 == 0:
                    insert_examples(cursor, batch)
                    self.conn.commit()
                    batch = []
            insert_examples(cursor, batch)
            self.conn.commit()

    def get(self, layer: int, latent: int) -> list[Example]:
        return select_examples(self.conn.cursor(), latent, layer)

    @staticmethod
    def repo_id(repo_id: str) -> str:
        if repo_id.endswith("-examples"):
            return repo_id
        if repo_id.endswith("-tfm"):
            return repo_id.replace("-tfm", "-examples")
        return repo_id + "-examples"

    @staticmethod
    def filename(repo_id: str) -> str:
        os.makedirs("out", exist_ok=True)
        return os.path.join("out", f"{Examples.repo_id(repo_id).replace('/', '-')}.db")


def get_examples(
    model: MLSAETransformer,
    batch: dict[str, torch.Tensor],
    n_tokens: int,
    dead_threshold: float = 1e-3,
    device: torch.device | str = "cpu",
) -> Generator[Example, None, None]:
    batch_tokens = batch["input_ids"].to(device)
    inputs = model.transformer.forward(batch_tokens)
    topk = model.autoencoder.encode(inputs).topk

    batch_tokens = einops.rearrange(batch_tokens, "b s -> (b s)")
    batch_acts = einops.rearrange(topk.values, "l b s k -> l (b s) k").half()
    batch_latents = einops.rearrange(topk.indices, "l b s k -> l (b s) k")

    layers, positions, indices = torch.where(batch_acts > dead_threshold)
    for layer, pos, k, latent in torch.stack(
        [layers, positions, indices, batch_latents[layers, positions, indices]], dim=1
    ).tolist():
        token_id = int(batch_tokens[pos].item())
        token = model.transformer.tokenizer.decode(token_id)
        act = batch_acts[layer, pos, k].item()

        token_ids = batch_tokens[pos - n_tokens : pos + n_tokens].tolist()
        tokens: list[str] = model.transformer.tokenizer.convert_ids_to_tokens(token_ids)  # type: ignore
        acts = (
            torch.where(
                batch_latents[layer, pos - n_tokens : pos + n_tokens] == latent,
                batch_acts[layer, pos - n_tokens : pos + n_tokens],
                0.0,
            )
            .sum(dim=-1)
            .tolist()
        )

        yield Example(latent, layer, token_id, token, act, token_ids, tokens, acts)


def create_db(database: str | PathLike) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS examples (
            id INTEGER PRIMARY KEY,
            latent INTEGER,
            layer INTEGER,
            token_id INTEGER,
            token TEXT,
            act REAL,
            token_ids JSON,
            tokens JSON,
            acts JSON
        )
        """,
    )
    conn.commit()
    return conn, cursor


def insert_examples(cursor: sqlite3.Cursor, examples: list[Example]) -> None:
    cursor.executemany(
        """
        INSERT INTO examples (
            latent,
            layer,
            token_id,
            token,
            act,
            token_ids,
            tokens,
            acts
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [example.serialize() for example in examples],
    )


def delete_examples(cursor: sqlite3.Cursor, n_examples: int) -> None:
    cursor.execute(
        """
        DELETE FROM examples
        WHERE id IN (
            SELECT id
            FROM (
                SELECT id,
                    ROW_NUMBER() OVER (
                        PARTITION BY latent, layer
                        ORDER BY act DESC
                    ) as rank
                FROM examples
            )
            WHERE rank > ?
        );
        """,
        (n_examples,),
    )


def select_examples(cursor: sqlite3.Cursor, latent: int, layer: int) -> list[Example]:
    cursor.execute(
        """
        SELECT latent,
        layer,
        token_id,
        token,
        act,
        token_ids,
        tokens,
        acts
        FROM examples
        WHERE layer = ?
        AND latent = ?
        ORDER BY act DESC
        """,
        (layer, latent),
    )
    return [Example.from_row(row) for row in cursor.fetchall()]


@torch.no_grad()
def save_examples(config: Config, device: torch.device | str = "cpu") -> None:
    model = MLSAETransformer.from_pretrained(config.repo_id).to(device)

    dataloader = get_train_dataloader(
        config.data.path,
        model.model_name,
        config.data.max_length,
        config.data.batch_size,
    )

    conn, cursor = create_db(Examples.filename(config.repo_id))
    batch: dict[str, torch.Tensor]
    for i, batch in tqdm(enumerate(dataloader), total=config.data.max_steps):
        examples = list(get_examples(model, batch, config.n_tokens, device=device))
        insert_examples(cursor, examples)
        if i > config.data.max_steps:
            break
        if i % config.delete_every_n_steps == 0:
            delete_examples(cursor, config.n_examples)
            conn.commit()
    delete_examples(cursor, config.n_examples)
    conn.commit()

    if config.push_to_hub:
        repo_id = Examples.repo_id(config.repo_id)
        dataset = Dataset.from_sql("SELECT * FROM examples", conn)
        assert isinstance(dataset, Dataset)
        dataset.push_to_hub(repo_id, commit_description=config.dumps_json())


if __name__ == "__main__":
    device = get_device()
    config = parse(Config)
    initialize(config.seed)
    save_examples(config, device)
