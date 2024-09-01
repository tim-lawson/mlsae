"use client";

import ColorSpan from "~/app/components/ColorSpan";
import { Select } from "~/app/use-select";
import { Card, CardContent } from "~/components/ui/card";
import { escapeWhitespace } from "~/lib/format";
import { LatentActivationsType, TokenType } from "~/lib/models";
import { cn } from "~/lib/utils";

export default function PromptActivationsComponent({
  className,
  tokens,
  latentActivations,
  stateLatent,
  stateLayer,
  statePosition,
}: {
  className?: string;
  tokens: TokenType[];
  latentActivations: LatentActivationsType;
  stateLatent: Select<number>;
  stateLayer: Select<number>;
  statePosition: Select<number>;
}) {
  return (
    <Card className={className}>
      <CardContent className="pt-4">
        <div className="flex flex-wrap font-mono text-xs">
          {tokens.map((token, index) => {
            const absolute =
              latentActivations.values[stateLayer.active][token.pos][
                stateLatent.active
              ];

            const relative =
              absolute / latentActivations.max[stateLayer.active][token.pos];

            return (
              <ColorSpan
                key={token.pos}
                className={cn(
                  token.pos === statePosition.active &&
                    "z-40 ring ring-slate-900 dark:ring-slate-100",
                )}
                opacity={relative}
                onClick={() => statePosition.onClick(index)}
                onMouseEnter={() => statePosition.onMouseEnter(index)}
                onMouseLeave={statePosition.onMouseLeave}
              >
                {escapeWhitespace(token.token)}
              </ColorSpan>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}
