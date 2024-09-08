"use client";

import React from "react";
import useSWR from "swr";
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
import { getExamples } from "~/lib/api";
import { escapeWhitespace } from "~/lib/format";

export default function LatentExamplesComponent({
  className,
  stateLatent,
  stateLayer,
}: {
  className?: string;
  stateLatent: Select<number>;
  stateLayer: Select<number>;
}) {
  const { data: examples } = useSWR(
    ["examples", stateLatent.clicked, stateLayer.clicked],
    ([_key, latent, layer]) => getExamples(latent, layer),
    {
      keepPreviousData: true,
    },
  );

  if (examples === undefined || examples.length === 0) {
    return null;
  }

  return (
    <Card className={className}>
      <CardContent className="pt-4">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Example</TableHead>
              <TableHead className="w-24 text-right">Activation</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {examples.map((example, index) => {
              return (
                <TableRow key={index} className="border-0 font-mono text-xs">
                  <TableCell className="max-w-96 truncate">
                    {example.tokens.map((token, position) => {
                      const value = example.acts[position];
                      return (
                        <ColorSpan key={position} opacity={value / example.act}>
                          {escapeWhitespace(token)}
                        </ColorSpan>
                      );
                    })}
                  </TableCell>
                  <TableCell className="pl-2 text-right">
                    {example.act.toFixed(3)}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
