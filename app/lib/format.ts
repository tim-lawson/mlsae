export const toUnitInterval = (value: number) => 1 / (1 + Math.exp(-value));

const fmtSigned = Intl.NumberFormat("en-US", {
  signDisplay: "always",
  minimumFractionDigits: 3,
});

export const toSigned = (value: number) => fmtSigned.format(value);

const REGEX_SPACE =
  /[\u0020\u00A0\u2000-\u2009\u200a​\u200b\u200c\u200d\ufeff​\u202f\u205f​\u3000\u1680​\u180e]/g;

const REGEX_TAB = /[\u0009\u000b\u000c]/g;

const REGEX_NEWLINE = /[\u000a\u000d\u0085]/g;

export function escapeWhitespace(value: string): string {
  return value
    .replace(REGEX_SPACE, "␣")
    .replace(REGEX_TAB, "⇥")
    .replace(REGEX_NEWLINE, "↵");
}
