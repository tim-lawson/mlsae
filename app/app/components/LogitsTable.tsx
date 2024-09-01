import ColorSpan from "~/app/components/ColorSpan";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableRow,
} from "~/components/ui/table";
import { escapeWhitespace, toSigned, toUnitInterval } from "~/lib/format";
import { LogitType } from "~/lib/models";

export default function LogitsTable({
  caption,
  data,
  color,
}: {
  caption: string;
  data: LogitType[] | undefined;
  color: string;
}) {
  const hasProbability = data?.[0].prob !== null;
  return (
    <Table className="caption-top">
      <TableCaption className="h-8 m-0 pb-2 flex items-center text-left font-medium leading-none tracking-wide uppercase">
        {caption}
      </TableCaption>
      <TableBody className="font-mono text-xs">
        {data?.map((logit) => (
          <TableRow key={logit.id} className="border-b-0">
            <TableCell className="px-0 py-0.5 max-w-12 truncate text-left">
              <ColorSpan
                opacity={toUnitInterval(Math.abs(logit.logit))}
                color={color}
              >
                {escapeWhitespace(logit.token)}
              </ColorSpan>
            </TableCell>
            {!hasProbability && (
              <TableCell className="px-0 py-0.5 text-right">
                {toSigned(logit.logit)}
              </TableCell>
            )}
            {hasProbability && (
              <TableCell className="px-0 py-0.5 text-right">
                {logit.prob?.toFixed(3)}
              </TableCell>
            )}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
