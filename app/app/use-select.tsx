import * as React from "react";

export default function useSelect<T>(initialState: T) {
  const [clicked, setClicked] = React.useState<T>(initialState);
  const onClick = (value: T) => setClicked(value);

  const [hovered, setHovered] = React.useState<T | null>(null);
  const onMouseEnter = (value: T) => setHovered(value);
  const onMouseLeave = () => setHovered(null);

  const active = hovered !== null ? hovered : clicked;

  return {
    active,
    clicked,
    onClick,
    onMouseEnter,
    onMouseLeave,
  };
}

export type Select<T> = ReturnType<typeof useSelect<T>>;
