from dataclasses import dataclass

import orjson
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from simple_parsing import Serializable, parse

from mlsae.trainer import initialize
from mlsae.utils import get_device

from .analyser import Analyser
from .models import (
    Example,
    LatentActivations,
    LayerHistograms,
    LogitChanges,
    MaxLogits,
    Token,
)


@dataclass
class Config(Serializable):
    repo_id: str
    """
    The name of a pretrained autoencoder and transformer from HuggingFace, or the path
    to a directory that contains them.
    """


config = parse(Config)
analyser = Analyser(repo_id=config.repo_id, device=get_device())


class ORJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content) -> bytes:
        return orjson.dumps(content)


app = FastAPI(
    docs_url="/api/py/docs",
    openapi_url="/api/py/openapi.json",
    default_response_class=ORJSONResponse,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/py/params")
async def params() -> dict:
    return analyser.params()


class ExamplesRequest(BaseModel):
    layer: int
    latent: int


@app.post("/api/py/examples")
async def examples(body: ExamplesRequest) -> list[Example]:
    return analyser.examples(body.layer, body.latent)


class PromptRequest(BaseModel):
    prompt: str


@app.post("/api/py/prompt/tokens")
async def prompt_tokens(body: PromptRequest) -> list[Token]:
    return analyser.prompt_tokens(body.prompt)


@app.post("/api/py/prompt/metrics")
async def prompt_metrics(body: PromptRequest) -> dict[str, float]:
    return analyser.prompt_metrics(body.prompt)


@app.post("/api/py/prompt/latent-activations")
async def prompt_latent_activations(body: PromptRequest) -> LatentActivations:
    return analyser.prompt_latent_activations(body.prompt)


@app.post("/api/py/prompt/layer-histograms")
async def prompt_layer_histograms(body: PromptRequest) -> LayerHistograms:
    return analyser.prompt_layer_histograms(body.prompt)


@app.post("/api/py/prompt/logits-input")
async def prompt_logits_input(body: PromptRequest) -> MaxLogits:
    return analyser.prompt_logits_input(body.prompt)


class PromptLogitsReconRequest(BaseModel):
    prompt: str
    layer: int


@app.post("/api/py/prompt/logits-recon")
async def prompt_logits_recon(
    body: PromptLogitsReconRequest,
) -> tuple[MaxLogits, LogitChanges]:
    return analyser.prompt_logits_recon(body.prompt, body.layer)


class PromptLogitsSteerRequest(BaseModel):
    prompt: str
    latent: int
    layer: int
    factor: float


@app.post("/api/py/prompt/logits-steer")
async def prompt_logits_steer(
    body: PromptLogitsSteerRequest,
) -> tuple[MaxLogits, LogitChanges]:
    return analyser.prompt_logits_steer(
        body.prompt, body.latent, body.layer, body.factor
    )


if __name__ == "__main__":
    import uvicorn

    initialize(42)
    uvicorn.run(app, port=8001)
