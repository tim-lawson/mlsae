import torch

from mlsae.model.autoencoders import SAE, TopKSAE
from mlsae.model.decoder import scatter_topk


@torch.no_grad()
def test_autoencoders() -> None:
    n_inputs = 512
    n_latents = 64 * n_inputs
    dead_steps_threshold = 10_000_000
    k = n_latents

    sae: SAE = SAE(n_inputs, n_latents, dead_steps_threshold)  # type: ignore

    topk_sae: TopKSAE = TopKSAE(n_inputs, n_latents, k, dead_steps_threshold, auxk=None)  # type: ignore
    topk_sae.encoder.weight.data = sae.encoder.weight.data
    topk_sae.decoder.weight.data = sae.decoder.weight.data
    topk_sae.pre_encoder_bias.data = sae.pre_encoder_bias.data

    inputs = torch.rand(1, n_inputs)

    sae_latents, sae_recons, sae_dead = sae.forward(inputs)
    topk_sae_topk, topk_sae_recons, _, _, topk_sae_dead = topk_sae.forward(inputs)
    topk_sae_latents = scatter_topk(topk_sae_topk, n_latents)

    assert torch.allclose(sae_latents, topk_sae_latents, atol=1e-3)
    assert torch.allclose(sae_recons, topk_sae_recons, atol=1e-3)
    assert torch.allclose(sae_dead, topk_sae_dead, atol=1e-3)
