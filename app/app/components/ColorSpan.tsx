import { cn } from "~/lib/utils";

export default function ColorSpan({
  children,
  opacity,
  className,
  color = "bg-orange-400 dark:bg-orange-800",
  style = {},
  ...props
}: {
  children?: React.ReactNode;
  opacity: number;
  className?: string | boolean | undefined;
  color?: string;
  style?: React.CSSProperties;
} & React.HTMLAttributes<HTMLSpanElement>) {
  return (
    <span {...props} className={cn("relative", className)} style={style}>
      <div
        className={cn("z-0 absolute top-0 left-0 bottom-0 right-0", color)}
        style={{ opacity }}
      />
      <span className="relative z-10 px-0.5 select-none">{children}</span>
    </span>
  );
}
