"use client";

import React from "react";
import Controls from "~/app/components/Controls";
import PromptActivations from "~/app/components/prompt/PromptActivations";
import PromptLatentHeatmaps from "~/app/components/prompt/PromptLatentHeatmaps";
import PromptLayerHistograms from "~/app/components/prompt/PromptLayerHistograms";
import PromptLogitsInput from "~/app/components/prompt/PromptLogitsInput";
import PromptLogitsRecon from "~/app/components/prompt/PromptLogitsRecon";
import PromptLogitsSteer from "~/app/components/prompt/PromptLogitsSteer";
import useSelect from "~/app/use-select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "~/components/ui/tabs";
import type {
  LatentActivationsType,
  MaxLogitsType,
  TokenType,
} from "~/lib/models";

export default function Page({
  parameters,
  prompt,
  tokens,
  latentActivations,
  logitsSource,
}: {
  parameters: any;
  prompt: string;
  tokens: TokenType[];
  latentActivations: LatentActivationsType;
  logitsSource: MaxLogitsType;
}) {
  const nLatents = latentActivations.values[0][0].length;
  const nLayers = latentActivations.max.length;
  const nPositions = tokens.length;

  const stateLatent = useSelect(0);
  const stateLayer = useSelect(0);
  const statePosition = useSelect(0);

  const [threshold, setThreshold] = React.useState(0.05);
  const [factor, setFactor] = React.useState(0);

  return (
    <div className="flex flex-col gap-4 overflow-hidden">
      <Controls
        nLatents={nLatents}
        nLayers={nLayers}
        nPositions={nPositions}
        stateLatent={stateLatent}
        stateLayer={stateLayer}
        statePosition={statePosition}
        threshold={threshold}
        onChangeThreshold={setThreshold}
        factor={factor}
        onChangeFactor={setFactor}
      />
      <PromptActivations
        className="h-32 overflow-y-auto"
        tokens={tokens}
        latentActivations={latentActivations}
        stateLatent={stateLatent}
        stateLayer={stateLayer}
        statePosition={statePosition}
      />
      <Tabs defaultValue="histogram">
        <TabsList className="">
          <TabsTrigger value="histogram">Histograms</TabsTrigger>
          <TabsTrigger value="heatmap">Heatmaps</TabsTrigger>
          <TabsTrigger value="logit">Logits</TabsTrigger>
        </TabsList>
        <TabsContent
          value="histogram"
          className="mt-2 max-h-[calc(100vh-20.5rem)] overflow-y-auto"
        >
          <PromptLayerHistograms prompt={prompt} stateLayer={stateLayer} />
        </TabsContent>
        <TabsContent
          value="heatmap"
          className="mt-2 max-h-[calc(100vh-20.5rem)] overflow-y-auto"
        >
          <PromptLatentHeatmaps
            latentActivations={latentActivations}
            stateLatent={stateLatent}
            stateLayer={stateLayer}
            statePosition={statePosition}
            threshold={threshold}
          />
        </TabsContent>
        <TabsContent
          value="logit"
          className="mt-2 max-h-[calc(100vh-20.5rem)] overflow-y-auto grid grid-cols-5 gap-4"
        >
          <PromptLogitsInput
            className="col-span-1"
            values={logitsSource}
            statePosition={statePosition}
          />
          <PromptLogitsRecon
            className="col-span-2"
            prompt={prompt}
            stateLayer={stateLayer}
            statePosition={statePosition}
          />
          <PromptLogitsSteer
            className="col-span-2"
            prompt={prompt}
            stateLatent={stateLatent}
            stateLayer={stateLayer}
            statePosition={statePosition}
            factor={factor}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
}
