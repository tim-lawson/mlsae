"use client";

import useSWR from "swr";
import LogitsTable from "~/app/components/LogitsTable";
import { Select } from "~/app/use-select";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { getPromptLogitsSteer } from "~/lib/api";

export default function PromptLogitsSteered({
  prompt,
  stateLatent,
  stateLayer,
  statePosition,
  factor,
  className,
}: {
  prompt: string;
  stateLatent: Select<number>;
  stateLayer: Select<number>;
  statePosition: Select<number>;
  factor: number;
  className?: string;
}) {
  const { data } = useSWR(
    [
      "prompt/logits-steer",
      prompt,
      stateLatent.clicked,
      stateLayer.clicked,
      factor,
    ],
    ([_key, prompt, latent, layer]) =>
      getPromptLogitsSteer(prompt, latent, layer, factor),
    {
      keepPreviousData: false,
    },
  );

  const [values, changes] = data ?? [];
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>
          Steered by latent {stateLatent.clicked} at layer {stateLayer.clicked}
        </CardTitle>
      </CardHeader>
      <CardContent className="flex gap-4">
        <LogitsTable
          caption="Max logits"
          data={values?.max[statePosition.clicked]}
          color="bg-red-400/50 dark:bg-red-800/50"
        />
        <LogitsTable
          caption="Max delta"
          data={changes?.max[statePosition.clicked]}
          color="bg-red-400/50 dark:bg-red-800/50"
        />
        <LogitsTable
          caption="Min delta"
          data={changes?.min[statePosition.clicked]}
          color="bg-indigo-400/50 dark:bg-indigo-800/50"
        />
      </CardContent>
    </Card>
  );
}
