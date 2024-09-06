// See `mlsae/api/models.py` for the corresponding Python dataclasses.

import { z } from "zod";

export const tokenSchema = z.object({
  id: z.number().int(),
  token: z.string(),
  pos: z.number().int(),
});

const logitSchema = z.object({
  id: z.number().int(),
  token: z.string(),
  logit: z.number(),
  prob: z.number().nullable(),
});

export const maxLogitsSchema = z.object({
  max: z.array(z.array(logitSchema)),
});

export const logitChangesSchema = z.object({
  max: z.array(z.array(logitSchema)),
  min: z.array(z.array(logitSchema)),
});

export const latentActivationsSchema = z.object({
  values: z.array(z.array(z.array(z.number()))),
  max: z.array(z.array(z.number())),
});

export const layerHistogramsSchema = z.object({
  values: z.array(z.array(z.number().int())),
  edges: z.array(z.number()),
});

export type TokenType = z.infer<typeof tokenSchema>;
export type LogitType = z.infer<typeof logitSchema>;
export type MaxLogitsType = z.infer<typeof maxLogitsSchema>;
export type LatentActivationsType = z.infer<typeof latentActivationsSchema>;
export type LayerHistogramsType = z.infer<typeof layerHistogramsSchema>;