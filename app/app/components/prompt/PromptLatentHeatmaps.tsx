"use client";

import React from "react";
import ColorSpan from "~/app/components/ColorSpan";
import { Select } from "~/app/use-select";
import { Card, CardContent } from "~/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import { LatentActivationsType } from "~/lib/models";
import { cn } from "~/lib/utils";

export default function LatentHeatmapComponent({
  latentActivations,
  threshold,
  stateLatent,
  stateLayer,
  statePosition,
  perToken = true,
}: {
  latentActivations: LatentActivationsType;
  threshold: number;
  stateLatent: Select<number>;
  stateLayer: Select<number>;
  statePosition: Select<number>;
  perToken?: boolean;
}) {
  const nLayers = latentActivations.max.length;
  const layers = Array.from({ length: nLayers }).map((_, layer) => layer);

  const latentHeatmaps = React.useMemo(() => {
    return getLatentHeatmaps(
      latentActivations,
      threshold,
      statePosition.clicked,
      perToken,
    );
  }, [latentActivations, threshold, statePosition.clicked, perToken]);

  return (
    <Card>
      <CardContent className="pt-4">
        <Table className="border-collapse" cellSpacing={0} cellPadding={0}>
          <TableHeader>
            <TableRow>
              <TableHead className="w-16">Latent</TableHead>
              <TableHead className="w-32 pr-4 text-right">Mean Layer</TableHead>
              {layers.map((layer) => (
                <TableHead key={layer} className="text-center">
                  Layer {layer}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody className="font-mono text-xs">
            {latentHeatmaps.map((latentHeatmap) => {
              return (
                <TableRow
                  key={latentHeatmap.latent}
                  className="border-0"
                  onClick={() => stateLatent.onClick(latentHeatmap.latent)}
                  onMouseEnter={() =>
                    stateLatent.onMouseEnter(latentHeatmap.latent)
                  }
                  onMouseLeave={stateLatent.onMouseLeave}
                >
                  <TableCell className="font-normal">
                    {latentHeatmap.latent}
                  </TableCell>
                  <TableCell className="pr-4 text-right">
                    {latentHeatmap.layer_mean.toFixed(2)}
                  </TableCell>
                  {layers.map((layer) => {
                    const latentHeatmapLayer = latentHeatmap.layers.find(
                      (heatmapLayer) => heatmapLayer.layer === layer,
                    );
                    const absolute = latentHeatmapLayer?.absolute ?? 0;
                    const relative = latentHeatmapLayer?.relative ?? 0;
                    const string = absolute.toFixed(3);
                    return (
                      <TableCell key={layer} className="p-0 h-6">
                        <ColorSpan
                          key={layer}
                          className={cn(
                            "h-full flex items-center justify-center",
                            layer === stateLayer.active &&
                              "bg-slate-200/50 dark:bg-slate-800/50",
                          )}
                          opacity={relative}
                          onClick={() => stateLayer.onClick(layer)}
                          onMouseEnter={() => stateLayer.onMouseEnter(layer)}
                          onMouseLeave={stateLayer.onMouseLeave}
                          title={string}
                        >
                          {absolute > 0 ? (
                            <span className="text-xs font-mono">{string}</span>
                          ) : null}
                        </ColorSpan>
                      </TableCell>
                    );
                  })}
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

interface LatentHeatmap {
  latent: number;
  layers: {
    layer: number;
    absolute: number;
    relative: number;
  }[];
  layer_mean: number;
}

function getLatentHeatmaps(
  latentActivations: LatentActivationsType,
  threshold: number,
  position: number,
  perToken = true,
) {
  const latentMap: Record<number, LatentHeatmap["layers"]> = {};
  const nLayers = latentActivations.max.length;
  const nLatents = latentActivations.values[0][0].length;

  for (let layer = 0; layer < nLayers; layer++) {
    for (let latent = 0; latent < nLatents; latent++) {
      let absolute: number;
      let relative: number;
      if (perToken) {
        absolute = latentActivations.values[layer][position][latent];
        relative = absolute / latentActivations.max[layer][position];
      } else {
        absolute = latentActivations.values[layer].reduce((total, values) => {
          return total + values[latent];
        }, 0);
        relative =
          absolute /
          latentActivations.max[layer].reduce((total, value) => {
            return total + value;
          }, 0);
      }
      if (relative > threshold) {
        if (latentMap[latent] === undefined) {
          latentMap[latent] = [];
        }
        latentMap[latent].push({
          layer,
          absolute,
          relative,
        });
      }
    }
  }

  const latentActivationTotal: Record<number, number> = {};
  const latentActivationLayerTotal: Record<number, number> = {};
  for (const key of Object.keys(latentMap)) {
    const latent = Number(key);
    for (let layer = 0; layer < nLayers; layer++) {
      const activation = latentActivations.values[layer][position][latent];
      if (activation > 0) {
        if (latentActivationTotal[latent] === undefined) {
          latentActivationTotal[latent] = 0;
        }
        latentActivationTotal[latent] += activation;

        if (latentActivationLayerTotal[latent] === undefined) {
          latentActivationLayerTotal[latent] = 0;
        }
        latentActivationLayerTotal[latent] += activation * layer;
      }
    }
  }

  let latentHeatmaps: LatentHeatmap[] = Object.entries(latentMap).map(
    ([key, layers]) => {
      const latent = Number(key);
      return {
        latent,
        layers,
        layer_mean:
          latentActivationLayerTotal[latent] / latentActivationTotal[latent],
      };
    },
  );

  latentHeatmaps.sort((a, b) => {
    const latentCenter = a.layer_mean - b.layer_mean;
    const activationTotal =
      b.layers.reduce((total, { absolute }) => total + absolute, 0) -
      a.layers.reduce((total, { absolute }) => total + absolute, 0);
    return latentCenter !== 0 ? latentCenter : activationTotal;
  });

  return latentHeatmaps;
}
