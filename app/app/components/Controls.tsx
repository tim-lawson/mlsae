"use client";

import React from "react";
import { Select } from "~/app/use-select";
import { Input } from "~/components/ui/input";
import { Label } from "~/components/ui/label";

export default function Controls({
  nLayers,
  nLatents,
  nPositions,
  stateLayer,
  stateLatent,
  statePosition,
  threshold,
  onChangeThreshold,
  factor,
  onChangeFactor,
}: {
  nLayers: number;
  nLatents: number;
  nPositions: number;
  stateLayer: Select<number>;
  stateLatent: Select<number>;
  statePosition: Select<number>;
  threshold: number;
  onChangeThreshold: (value: number) => void;
  factor: number;
  onChangeFactor: (value: number) => void;
}) {
  const onChangeLayer = (event: React.ChangeEvent<HTMLInputElement>) => {
    stateLayer.onClick(event.target.valueAsNumber);
  };

  const onChangeLatent = (event: React.ChangeEvent<HTMLInputElement>) => {
    stateLatent.onClick(event.target.valueAsNumber);
  };

  const onChangePosition = (event: React.ChangeEvent<HTMLInputElement>) => {
    statePosition.onClick(event.target.valueAsNumber);
  };

  return (
    <div className="h-10 px-4 flex items-center justify-stretch gap-4">
      <div className="flex items-center gap-2">
        <Label htmlFor="layer">Layer</Label>
        <Input
          id="layer"
          type="number"
          value={stateLayer.active}
          onChange={onChangeLayer}
          min={0}
          max={nLayers - 1}
          step={1}
          className="h-8"
        />
      </div>
      <div className="flex items-center gap-2">
        <Label htmlFor="latent">Latent</Label>
        <Input
          id="latent"
          type="number"
          value={stateLatent.active}
          onChange={onChangeLatent}
          min={0}
          max={nLatents - 1}
          step={1}
          className="h-8"
        />
      </div>
      <div className="flex items-center gap-2">
        <Label htmlFor="position">Position</Label>
        <Input
          id="position"
          type="number"
          value={statePosition.active}
          onChange={onChangePosition}
          min={0}
          max={nPositions - 1}
          step={1}
          className="h-8"
        />
      </div>
      <div className="flex items-center gap-2">
        <Label htmlFor="threshold" className="text-nowrap">
          Threshold
        </Label>
        <Input
          id="threshold"
          type="number"
          value={threshold}
          onChange={(event) => onChangeThreshold(event.target.valueAsNumber)}
          min={0.0}
          max={0.95}
          step={0.05}
          className="h-8"
        />
      </div>
      <div className="flex items-center gap-2">
        <Label htmlFor="factor" className="text-nowrap">
          Steering factor
        </Label>
        <Input
          id="factor"
          type="number"
          value={factor}
          onChange={(event) => onChangeFactor(event.target.valueAsNumber)}
          min={-10}
          max={10}
          step={0.5}
          className="h-8"
        />
      </div>
    </div>
  );
}
