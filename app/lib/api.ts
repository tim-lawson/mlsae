import { z } from "zod";
import {
  latentActivationsSchema,
  layerHistogramsSchema,
  logitChangesSchema,
  maxLogitsSchema,
  tokenSchema,
} from "./models";

const isDev = process.env.NODE_ENV === "development";

const API_URL = isDev ? "http://localhost:3000/api/py" : "";

export async function getParameters() {
  const response = await fetch(`${API_URL}/params`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  });
  const data: unknown = await response.json();
  return z.record(z.string(), z.any()).parse(data);
}

export async function getPromptTokens(prompt: string) {
  const response = await fetch(`${API_URL}/prompt/tokens`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ prompt }),
  });
  const data: unknown = await response.json();
  return z.array(tokenSchema).parse(data);
}

export async function getPromptLatentActivations(prompt: string) {
  const response = await fetch(`${API_URL}/prompt/latent-activations`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ prompt }),
  });
  const data: unknown = await response.json();
  return latentActivationsSchema.parse(data);
}

export async function getPromptLayerHistograms(prompt: string) {
  const response = await fetch(`${API_URL}/prompt/layer-histograms`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ prompt }),
  });
  const data: unknown = await response.json();
  return layerHistogramsSchema.parse(data);
}

export async function getPromptLogitsInput(prompt: string) {
  const response = await fetch(`${API_URL}/prompt/logits-input`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ prompt }),
  });
  const data: unknown = await response.json();
  return maxLogitsSchema.parse(data);
}

export async function getPromptLogitsRecon(prompt: string, layer: number) {
  const response = await fetch(`${API_URL}/prompt/logits-recon`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ prompt, layer }),
  });
  const data: unknown = await response.json();
  return z.tuple([maxLogitsSchema, logitChangesSchema]).parse(data);
}

export async function getPromptLogitsSteer(
  prompt: string,
  latent: number,
  layer: number,
  factor: number,
) {
  const response = await fetch(`${API_URL}/prompt/logits-steer`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ prompt, latent, layer, factor }),
  });
  const data: unknown = await response.json();
  return z.tuple([maxLogitsSchema, logitChangesSchema]).parse(data);
}
