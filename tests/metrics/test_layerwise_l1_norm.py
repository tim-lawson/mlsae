import pytest
import torch
from jaxtyping import Float

from mlsae.metrics import LayerwiseL1Norm

n_layers = 6
shape = (n_layers, 1, 2048, 32)


@pytest.mark.parametrize(
    ("n_layers", "values", "expected"),
    [
        pytest.param(
            n_layers, torch.zeros(*shape), torch.zeros(n_layers), id="all zero"
        ),
        pytest.param(
            n_layers, torch.ones(*shape), torch.ones(n_layers) * 32.0, id="all +1"
        ),
        pytest.param(
            n_layers, torch.ones(*shape) * -1, torch.ones(n_layers) * 32.0, id="all -1"
        ),
    ],
)
def test_layerwise_l1_norm(
    n_layers: int,
    values: Float[torch.Tensor, "n_layers batch pos k"],
    expected: Float[torch.Tensor, "n_layers"],
) -> None:
    metric = LayerwiseL1Norm(n_layers)

    metric.update(values=values)
    assert torch.allclose(metric.compute(), expected)

    metric.update(values=values)
    metric.update(values=values)
    assert torch.allclose(metric.compute(), expected)
