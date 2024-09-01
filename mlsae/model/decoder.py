import os

import torch
from jaxtyping import Float
from loguru import logger
from torch import Tensor

from mlsae.model.types import TopK


# NOTE: Avoid this where possible to save memory!
def scatter_topk(topk: TopK, n_latents: int) -> Float[Tensor, "... n_latents"]:
    """
    Scatter the k largest latents into a new tensor of shape (..., n_latents).

    Args:
        topk (TopK): The k largest latents.

        n_latents (int): The number of latents.

    Returns:
        out (Float[Tensor, "... n_latents"]): The k largest latents.
    """

    # ... n_latents
    buffer = topk.values.new_zeros((*topk.indices.shape[:-1], n_latents))
    # ... k -> ... n_latents
    return buffer.scatter_(dim=-1, index=topk.indices, src=topk.values)


# Based on https://github.com/EleutherAI/sae/blob/19d95a401e9d17dbf7d6fb0fa7a91081f1b0d01f/sae/utils.py
def decode_triton(topk: TopK, weight: Tensor) -> Tensor:
    shape = topk.indices.shape[:-1]
    k = topk.indices.shape[-1]
    n_inputs, n_latents = weight.shape

    indices_flat = topk.indices.view(-1, k)
    values_flat = topk.values.view(-1, k)

    output: Tensor = TritonDecoderAutograd.apply(indices_flat, values_flat, weight)  # type: ignore

    return output.view(*shape, n_inputs)


def decode_cuda(topk: TopK, weight: Tensor, chunk_size: int = 1024) -> Tensor:
    shape = topk.indices.shape[:-1]
    k = topk.indices.shape[-1]
    n_inputs, n_latents = weight.shape

    indices_flat = topk.indices.view(-1, k)
    values_flat = topk.values.view(-1, k)

    batch_size = indices_flat.shape[0]

    output = torch.zeros(
        batch_size, n_inputs, device=topk.values.device, dtype=topk.values.dtype
    )

    for i in range(0, batch_size, chunk_size):
        indices_chunk = indices_flat[i : i + chunk_size]
        values_chunk = values_flat[i : i + chunk_size]

        chunk_sparse = torch.sparse_coo_tensor(
            indices=torch.cat(
                [
                    torch.arange(
                        indices_chunk.shape[0], device=indices_chunk.device
                    ).repeat_interleave(k),
                    indices_chunk.flatten(),
                ]
            ).view(2, -1),
            values=values_chunk.flatten(),
            size=(indices_chunk.shape[0], n_latents),
        )

        chunk_output = torch.sparse.mm(chunk_sparse, weight.t())

        output[i : i + chunk_size] = chunk_output

    return output.view(*shape, n_inputs)


# NOTE: 'sparse_coo_tensor' isn't supported yet for the MPS backend
def decode_mps(topk: TopK, weight: Tensor, chunk_size: int = 1024) -> Tensor:
    shape = topk.indices.shape[:-1]
    k = topk.indices.shape[-1]
    n_inputs, n_latents = weight.shape

    indices_flat = topk.indices.view(-1, k)
    values_flat = topk.values.view(-1, k)

    batch_size = indices_flat.shape[0]

    output = torch.zeros(
        batch_size, n_inputs, device=topk.values.device, dtype=topk.values.dtype
    )

    for i in range(0, batch_size, chunk_size):
        indices_chunk = indices_flat[i : i + chunk_size]
        values_chunk = values_flat[i : i + chunk_size]

        weight_mask = weight[:, indices_chunk.view(-1)].view(
            n_inputs, indices_chunk.shape[0], k
        )

        output_chunk = torch.bmm(
            values_chunk.unsqueeze(1), weight_mask.permute(1, 2, 0)
        ).squeeze(1)

        output[i : i + chunk_size] = output_chunk

    return output.view(*shape, n_inputs)


def decode(topk: TopK, weight: Tensor) -> Tensor:
    """
    Sparse decoder implementation.

    Args:
        topk (TopK): The k largest latents.

        weight (Float[Tensor, "n_inputs n_latents"]): The decoder weight matrix.

    Returns:
        out (Float[Tensor, "... n_inputs"]): The reconstructions.
    """
    ...


try:
    from .kernels import TritonDecoderAutograd
except ImportError:
    logger.info("Triton not found")
    if torch.backends.mps.is_available():
        logger.info("MPS backend, using 'bmm' decoder")
        decode = decode_mps
    else:
        logger.info("CPU/CUDA backend, using 'sparse_coo_tensor' decoder")
        decode = decode_cuda
else:
    logger.info("Triton found")
    if os.environ.get("USE_TRITON", "1") == "1":
        logger.info("Triton enabled, using Triton decoder")
        decode = decode_triton
    else:
        logger.info("Triton disabled, using 'sparse_coo_tensor' decoder")
        decode = decode_cuda
