import { Inter } from "next/font/google";
import "./globals.css";
import { cn } from "~/lib/utils";

const inter = Inter({
  subsets: ["latin", "latin-ext"],
});

export default async function Layout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="text-primary">
      <body
        className={cn(
          inter.className,
          "bg-slate-100 dark:bg-slate-900 text-slate-950 dark:text-slate-200 h-screen overflow-hidden",
        )}
      >
        <header
          className={cn(
            "sticky top-0 z-50 h-12 flex items-center px-6 text-xl font-medium tracking-wide",
            "bg-slate-800 dark:bg-slate-950 text-slate-100 dark:text-slate-50",
          )}
        >
          Residual Stream Analysis with Multi-Layer SAEs
        </header>
        <main className="p-4 pb-0">{children}</main>
      </body>
    </html>
  );
}
