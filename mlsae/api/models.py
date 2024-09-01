from pydantic import BaseModel


class Token(BaseModel):
    id: int
    """The token id."""
    token: str
    """The token string."""
    pos: int
    """The token position."""


class Logit(BaseModel):
    id: int
    """The token id."""
    token: str
    """The token string."""
    logit: float
    """The logit value."""
    prob: float | None = None
    """The softmax-normalized logit value."""


class MaxLogits(BaseModel):
    max: list[list[Logit]]
    """The maximum logit values for each token position."""


class LogitChanges(BaseModel):
    max: list[list[Logit]]
    """The maximum changes in logit values for each token position."""
    min: list[list[Logit]]
    """The minimum changes in logit values for each token position."""


class LatentActivations(BaseModel):
    values: list[list[list[float]]]
    """The latent activations for each layer, position, and latent dimension."""
    max: list[list[float]]
    """The maximum latent activations for each layer and token position."""


class LayerHistograms(BaseModel):
    values: list[list[int]]
    """The histogram values for each layer."""
    edges: list[float]
    """The histogram edges across all layers."""
