"use client";

import LogitsTable from "~/app/components/LogitsTable";
import { Select } from "~/app/use-select";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { MaxLogitsType } from "~/lib/models";

export default function PromptLogitsInput({
  values,
  statePosition,
  className,
}: {
  values: MaxLogitsType;
  statePosition: Select<number>;
  className?: string;
}) {
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>Input activations</CardTitle>
      </CardHeader>
      <CardContent>
        <LogitsTable
          caption="Max logits"
          data={values.max[statePosition.active]}
          color="bg-red-400/50 dark:bg-red-800/50"
        />
      </CardContent>
    </Card>
  );
}
