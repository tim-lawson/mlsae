"use client";

import { scaleSymlog } from "d3-scale";
import React from "react";
import { Area, AreaChart, XAxis, YAxis } from "recharts";
import useSWR from "swr";
import colors from "tailwindcss/colors";
import { useDarkMode } from "usehooks-ts";
import { Select } from "~/app/use-select";
import { Card, CardContent } from "~/components/ui/card";
import { ChartConfig, ChartContainer } from "~/components/ui/chart";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import { getPromptLayerHistograms } from "~/lib/api";
import { toSigned } from "~/lib/format";
import { LayerHistogramsType } from "~/lib/models";
import { cn } from "~/lib/utils";

const symlog = scaleSymlog();
const chartConfig = {
  count: {},
} satisfies ChartConfig;
const chartMargin = { top: 0, left: 0, right: 0, bottom: 0 } as const;

function chartData(histograms: LayerHistogramsType, layer: number) {
  return histograms.edges
    .map((edge, index) => ({
      edge,
      value: histograms.values[layer][index],
    }))
    .slice(0, -1);
}

export default function LayerHistogramsComponent({
  prompt,
  stateLayer,
}: {
  prompt: string;
  stateLayer: Select<number>;
}) {
  const { data: histograms } = useSWR(
    ["prompt/layer-histograms", prompt],
    ([_key, prompt]) => getPromptLayerHistograms(prompt),
    {
      keepPreviousData: true,
    },
  );

  const { isDarkMode } = useDarkMode();

  if (!histograms) {
    return null;
  }

  const nLayers = histograms.values.length;
  const layers = Array.from({ length: nLayers }, (_, i) => i);

  return (
    <Card>
      <CardContent className="pt-4">
        <Table className="table-fixed">
          <TableHeader>
            <TableRow>
              <TableHead className="w-16">Layer</TableHead>
              <TableHead className="text-left">Histogram</TableHead>
            </TableRow>
          </TableHeader>
        </Table>
        <TableBody className="text-xs font-mono">
          {layers.map((layer) => {
            const isActive = layer == stateLayer.active;
            const color = isDarkMode
              ? isActive
                ? colors.orange[800]
                : colors.slate[100]
              : isActive
                ? colors.orange[400]
                : colors.slate[900];
            return (
              <TableRow
                key={layer}
                onClick={() => stateLayer.onClick(layer)}
                onMouseEnter={() => stateLayer.onMouseEnter(layer)}
                onMouseLeave={stateLayer.onMouseLeave}
              >
                <TableCell className="w-16">{layer}</TableCell>
                <TableCell>
                  <ChartContainer
                    config={chartConfig}
                    className={cn(
                      "h-14 w-[calc(100vw-8rem)]",
                      isActive && "bg-slate-100 dark:bg-slate-900",
                    )}
                  >
                    <AreaChart
                      accessibilityLayer
                      data={chartData(histograms, layer)}
                      margin={chartMargin}
                    >
                      <Area
                        type="step"
                        dataKey="value"
                        dot={false}
                        activeDot={false}
                        stroke="none"
                        fill={color}
                        fillOpacity={1}
                        isAnimationActive={false}
                      />
                      <XAxis hide tickFormatter={toSigned} />
                      <YAxis hide scale={symlog} />
                    </AreaChart>
                  </ChartContainer>
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </CardContent>
    </Card>
  );
}
