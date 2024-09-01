import pytest
import torch
from jaxtyping import Float

from mlsae.metrics import MSELoss

n_layers = 6
shape = (n_layers, 1, 2048, 512)

generator = torch.Generator()
generator.manual_seed(42)

normal_zeros = torch.normal(torch.ones(*shape), std=1, generator=generator)
normal_ones = torch.normal(torch.zeros(*shape), std=1, generator=generator)


@pytest.mark.parametrize(
    ("n_layers", "inputs", "recons", "expected"),
    [
        pytest.param(
            n_layers,
            normal_zeros,
            normal_zeros,
            torch.tensor(0.0),
            id="both 0",
        ),
        pytest.param(
            n_layers,
            normal_ones,
            normal_ones,
            torch.tensor(0.0),
            id="both 1",
        ),
        pytest.param(
            n_layers,
            normal_zeros,
            torch.ones(*shape),
            torch.tensor(1.0),
            id="0 and 1",
        ),
        pytest.param(
            n_layers,
            normal_ones,
            torch.zeros(*shape),
            torch.tensor(1.0),
            id="1 and 0",
        ),
    ],
)
def test_loss_mse(
    n_layers: int,
    inputs: Float[torch.Tensor, "n_layers batch pos n_inputs"],
    recons: Float[torch.Tensor, "n_layers batch pos n_inputs"],
    expected: Float[torch.Tensor, "n_layers"],
) -> None:
    metric = MSELoss(n_layers)

    metric.update(inputs=inputs, recons=recons)
    assert torch.allclose(metric.compute(), expected, atol=1e-2)

    metric.update(inputs=inputs, recons=recons)
    metric.update(inputs=inputs, recons=recons)
    assert torch.allclose(metric.compute(), expected, atol=1e-2)
