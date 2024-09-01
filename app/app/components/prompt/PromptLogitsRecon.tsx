"use client";

import useSWR from "swr";
import LogitsTable from "~/app/components/LogitsTable";
import { Select } from "~/app/use-select";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { getPromptLogitsRecon as getPromptLogitsRecon } from "~/lib/api";

export default function PromptLogitsRecon({
  prompt,
  stateLayer,
  statePosition,
  className,
}: {
  prompt: string;
  stateLayer: Select<number>;
  statePosition: Select<number>;
  className?: string;
}) {
  const { data } = useSWR(
    ["prompt/logits-recon", prompt, stateLayer.clicked],
    ([_key, prompt, layer]) => getPromptLogitsRecon(prompt, layer),
    {
      keepPreviousData: false,
    },
  );

  const [values, changes] = data ?? [];
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>Reconstruction at layer {stateLayer.clicked}</CardTitle>
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
