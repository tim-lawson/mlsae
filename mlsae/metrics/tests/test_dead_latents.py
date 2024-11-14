import torch

from mlsae.metrics import DeadLatents


def test_dead_latents() -> None:
    metric = DeadLatents(4, 4)

    metric.update(indices=torch.tensor([[[[0], [0]]], [[[0], [0]]]]))
    assert metric.tokens == 2
    assert torch.allclose(metric.latent_tokens, torch.tensor([4.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(metric.compute(), torch.tensor(0.75))

    metric.update(indices=torch.tensor([[[[0], [0]]], [[[1], [1]]]]))
    assert metric.tokens == 4
    assert torch.allclose(metric.latent_tokens, torch.tensor([6.0, 2.0, 0.0, 0.0]))
    assert torch.allclose(metric.compute(), torch.tensor(0.5))

    metric.update(indices=torch.tensor([[[[1], [1]]], [[[2], [2]]]]))
    assert metric.tokens == 6
    assert torch.allclose(metric.latent_tokens, torch.tensor([6.0, 4.0, 2.0, 0.0]))
    assert torch.allclose(metric.compute(), torch.tensor(0.25))

    metric.update(indices=torch.tensor([[[[2], [2]]], [[[3], [3]]]]))
    assert metric.tokens == 8
    assert torch.allclose(metric.latent_tokens, torch.tensor([6.0, 4.0, 4.0, 2.0]))
    assert torch.allclose(metric.compute(), torch.tensor(0.0))

    metric.update(indices=torch.tensor([[[[3], [3]]], [[[0], [0]]]]))
    assert metric.tokens == 10
    assert torch.allclose(metric.latent_tokens, torch.tensor([8.0, 4.0, 4.0, 4.0]))
    assert torch.allclose(metric.compute(), torch.tensor(0.0))

    metric.update(indices=torch.tensor([[[[0], [0]]], [[[0], [0]]]]))
    assert metric.tokens == 12
    assert torch.allclose(metric.latent_tokens, torch.tensor([12.0, 4.0, 4.0, 4.0]))
    assert torch.allclose(metric.compute(), torch.tensor(0.0))

    metric.update(indices=torch.tensor([[[[0], [0]]], [[[0], [0]]]]))
    assert metric.tokens == 14
    assert torch.allclose(metric.latent_tokens, torch.tensor([16.0, 4.0, 4.0, 4.0]))
    assert torch.allclose(metric.compute(), torch.tensor(0.0))

    metric.update(indices=torch.tensor([[[[0], [0]]], [[[0], [0]]]]))
    assert metric.tokens == 16
    assert torch.allclose(metric.latent_tokens, torch.tensor([20.0, 4.0, 4.0, 4.0]))
    assert torch.allclose(metric.compute(), torch.tensor(0.0))

    metric.update(indices=torch.tensor([[[[0], [0]]], [[[0], [0]]]]))
    assert metric.tokens == 18
    assert torch.allclose(metric.latent_tokens, torch.tensor([24.0, 4.0, 4.0, 4.0]))
    assert torch.allclose(metric.compute(), torch.tensor(0.75))
