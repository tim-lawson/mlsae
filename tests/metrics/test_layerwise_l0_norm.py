import pytest
import torch
from jaxtyping import Float

from mlsae.metrics import LayerwiseL0Norm

n_layers = 6
shape = (n_layers, 1, 2048, 32)


@pytest.mark.parametrize(
    ("n_layers", "dead_threshold", "values", "expected"),
    [
        pytest.param(
            n_layers,
            1e-3,
            torch.zeros(*shape),
            torch.zeros(n_layers),
            id="all zero",
        ),
        pytest.param(
            n_layers,
            1e-3,
            torch.ones(*shape) * 1e-4,
            torch.zeros(n_layers),
            id="below threshold",
        ),
        pytest.param(
            n_layers,
            1e-3,
            torch.ones(*shape),
            torch.ones(n_layers) * 32.0,
            id="above threshold",
        ),
    ],
)
def test_layerwise_l0_norm(
    n_layers: int,
    dead_threshold: float,
    values: Float[torch.Tensor, "n_layers batch pos k"],
    expected: Float[torch.Tensor, "n_layers"],
) -> None:
    metric = LayerwiseL0Norm(n_layers, dead_threshold)

    metric.update(values=values)
    assert torch.allclose(metric.compute(), expected)

    metric.update(values=values)
    metric.update(values=values)
    assert torch.allclose(metric.compute(), expected)
