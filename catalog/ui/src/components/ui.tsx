// Small building blocks in the AeroAudio style: buttons, panels, chips, selects, status pills.

import type { ButtonHTMLAttributes, ReactNode, SelectHTMLAttributes } from "react";
import { Link } from "react-router-dom";

export function cx(...c: (string | false | null | undefined)[]): string {
  return c.filter(Boolean).join(" ");
}

type Variant = "primary" | "outline" | "ghost" | "danger";
const VARIANTS: Record<Variant, string> = {
  primary: "bg-primary text-canvas font-semibold border-0 hover:brightness-110",
  outline: "bg-transparent border border-line text-fg hover:bg-track",
  ghost: "bg-transparent border-0 text-muted hover:text-fg hover:bg-track",
  danger: "bg-transparent border border-bad/60 text-bad hover:bg-bad/10",
};

export function Button({
  variant = "outline",
  size = "md",
  className,
  ...p
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: "sm" | "md" }) {
  return (
    <button
      type="button"
      {...p}
      className={cx(
        "inline-flex items-center justify-center gap-2 rounded-md cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed",
        size === "sm" ? "min-h-8 px-2.5 text-xs" : "min-h-10 px-4 text-[13px]",
        VARIANTS[variant],
        className,
      )}
    />
  );
}

export function LinkButton({ to, children, variant = "outline" }: { to: string; children: ReactNode; variant?: Variant }) {
  return (
    <Link
      to={to}
      className={cx(
        "inline-flex items-center gap-2 min-h-10 px-4 rounded-md text-[13px] no-underline",
        VARIANTS[variant],
        variant !== "primary" && "!text-fg",
      )}
    >
      {children}
    </Link>
  );
}

/** A button-styled link that leaves the app (another app, a download). */
export function AnchorButton({ href, children, variant = "outline", download, newTab }: { href: string; children: ReactNode; variant?: Variant; download?: boolean; newTab?: boolean }) {
  return (
    <a
      href={href}
      download={download || undefined}
      target={newTab ? "_blank" : undefined}
      rel={newTab ? "noopener noreferrer" : undefined}
      className={cx("inline-flex items-center gap-2 min-h-10 px-4 rounded-md text-[13px] no-underline", VARIANTS[variant], variant !== "primary" && "!text-fg")}
    >
      {children}
    </a>
  );
}

export function Panel({
  title,
  actions,
  children,
  className,
  id,
  subtitle,
}: {
  title?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  id?: string;
  subtitle?: ReactNode;
}) {
  return (
    <section aria-labelledby={id} className={cx("bg-panel border border-track rounded-lg p-4 flex flex-col gap-3 min-w-0", className)}>
      {(title || actions) && (
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex flex-col gap-0.5 min-w-0">
            {title && (
              <h2 id={id} className="m-0 text-sm font-semibold">
                {title}
              </h2>
            )}
            {subtitle && <span className="text-xs text-muted">{subtitle}</span>}
          </div>
          {actions && <div className="ml-auto flex flex-wrap items-center gap-2">{actions}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

export function PageHeader({
  crumbs,
  title,
  subtitle,
  actions,
}: {
  crumbs?: ReactNode;
  title: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <header className="flex flex-wrap items-end justify-between gap-4">
      <div className="flex flex-col gap-1.5 min-w-0">
        {crumbs && <span className="text-xs text-muted">{crumbs}</span>}
        <h1 className="m-0 text-[22px] font-semibold tracking-tight">{title}</h1>
        {subtitle && <span className="text-[13px] text-muted">{subtitle}</span>}
      </div>
      {actions && <div className="flex flex-wrap gap-2.5">{actions}</div>}
    </header>
  );
}

export function Select({ label, className, children, ...p }: SelectHTMLAttributes<HTMLSelectElement> & { label?: ReactNode }) {
  const sel = (
    <select
      {...p}
      className={cx("bg-panel text-fg border border-line rounded-md min-h-[34px] px-2 text-xs", className)}
    >
      {children}
    </select>
  );
  if (!label) return sel;
  return (
    <label className="flex items-center gap-2 text-xs text-muted">
      {label}
      {sel}
    </label>
  );
}

export function Chip({ children, tone = "default", title }: { children: ReactNode; tone?: "default" | "good" | "warn" | "bad" | "info"; title?: string }) {
  const tones = {
    default: "border-line text-muted",
    good: "border-good/50 text-good",
    warn: "border-warn/50 text-warn",
    bad: "border-bad/50 text-bad",
    info: "border-focus/50 text-focus",
  };
  return (
    <span title={title} className={cx("inline-flex items-center gap-1 border rounded px-1.5 py-px text-[11px] whitespace-nowrap", tones[tone])}>
      {children}
    </span>
  );
}

export function StatusDot({ state }: { state: string }) {
  const color =
    {
      up: "bg-good", running: "bg-good", active: "bg-good", verified: "bg-good", done: "bg-good",
      starting: "bg-primary", stopping: "bg-warn", queued: "bg-warn", pending: "bg-subtle", stale: "bg-warn", hosted: "bg-focus",
      stopped: "bg-subtle", released: "bg-subtle", expired: "bg-subtle", down: "bg-subtle",
      failed: "bg-bad", unknown: "bg-bad",
    }[state] ?? "bg-subtle";
  return <span aria-hidden className={cx("inline-block size-2 rounded-full shrink-0", color, ["starting", "stopping"].includes(state) && "animate-pulse")} />;
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="border border-dashed border-line rounded-md p-6 text-center text-[13px] text-muted">{children}</div>;
}

export function ErrorNote({ error }: { error: unknown }) {
  if (!error) return null;
  const msg = error instanceof Error ? error.message : String(error);
  return <div role="alert" className="border border-bad/50 bg-bad/10 text-[13px] text-fg rounded-md px-3 py-2">{msg}</div>;
}

export function Spinner({ label = "Loading" }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-xs text-muted" role="status">
      <span aria-hidden className="size-3 rounded-full border-2 border-line border-t-primary animate-spin" />
      {label}
    </span>
  );
}

export function Bar({ value, color = "var(--color-primary)", max = 1, className }: { value: number | null | undefined; color?: string; max?: number; className?: string }) {
  const w = value === null || value === undefined ? 0 : Math.max(0, Math.min(1, value / max));
  return (
    <span className={cx("block h-1.5 rounded bg-track overflow-hidden", className)}>
      <span className="block h-full rounded" style={{ width: `${w * 100}%`, background: color }} />
    </span>
  );
}

export function Fact({ k, v }: { k: ReactNode; v: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 text-[13px] py-1 border-b border-track last:border-0">
      <span className="text-muted">{k}</span>
      <span className="num">{v}</span>
    </div>
  );
}

export function Toggle({
  options,
  value,
  onChange,
  label,
  className,
}: {
  options: { value: string; label: string }[];
  value: string;
  onChange: (v: string) => void;
  label: string;
  className?: string;
}) {
  return (
    <span role="group" aria-label={label} className={cx("inline-flex border border-line rounded-md overflow-hidden", className)}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          aria-pressed={o.value === value}
          onClick={() => onChange(o.value)}
          className={cx("min-h-8 px-3 text-xs cursor-pointer border-0", o.value === value ? "bg-active text-fg" : "bg-transparent text-muted hover:text-fg")}
        >
          {o.label}
        </button>
      ))}
    </span>
  );
}
