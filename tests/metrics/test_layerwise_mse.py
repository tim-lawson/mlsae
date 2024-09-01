import pytest
import torch
from jaxtyping import Float

from mlsae.metrics import LayerwiseMSE

n_layers = 6
shape = (n_layers, 1, 2048, 512)


@pytest.mark.parametrize(
    ("n_layers", "inputs", "recons", "expected"),
    [
        pytest.param(
            n_layers,
            torch.zeros(*shape),
            torch.zeros(*shape),
            torch.zeros(n_layers),
            id="both 0",
        ),
        pytest.param(
            n_layers,
            torch.ones(*shape),
            torch.ones(*shape),
            torch.zeros(n_layers),
            id="both 1",
        ),
        pytest.param(
            n_layers,
            torch.zeros(*shape),
            torch.ones(*shape),
            torch.ones(n_layers),
            id="0 and 1",
        ),
        pytest.param(
            n_layers,
            torch.ones(*shape),
            torch.zeros(*shape),
            torch.ones(n_layers),
            id="1 and 0",
        ),
    ],
)
def test_layerwise_mse(
    n_layers: int,
    inputs: Float[torch.Tensor, "n_layers batch pos n_inputs"],
    recons: Float[torch.Tensor, "n_layers batch pos n_inputs"],
    expected: Float[torch.Tensor, "n_layers"],
) -> None:
    metric = LayerwiseMSE(n_layers)

    metric.update(inputs=inputs, recons=recons)
    assert torch.allclose(metric.compute(), expected)

    metric.update(inputs=inputs, recons=recons)
    metric.update(inputs=inputs, recons=recons)
    assert torch.allclose(metric.compute(), expected)
