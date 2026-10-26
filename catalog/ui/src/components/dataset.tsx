// Pieces the Dataset Catalog's pages share: the delete confirmation that lists what apps made from a version, a
// reference segment on one line, and a word diff.

import { useId, type ReactNode } from "react";
import { appLabel, labelValue, type DiffOp, type Segment, type Use } from "../lib/datasets";
import { ago, clock } from "../lib/format";
import { Button, ErrorNote } from "./ui";

export type VersionUse = Use & { n?: number };

/** What apps made from a version: their own datasets, which keep working after the version is deleted. */
export function UseList({ uses }: { uses: VersionUse[] }) {
  return (
    <ul className="m-0 pl-5 flex flex-col gap-0.5 text-[13px]">
      {uses.map((u) => (
        <li key={`${u.n}-${u.app}-${u.ref}`}>
          {u.n !== undefined && <span className="num">v{u.n}: </span>}
          {appLabel(u.app)} dataset <span className="num">{u.ref}</span>
          <span className="text-xs text-subtle"> (made {ago(u.created_at)})</span>
        </li>
      ))}
    </ul>
  );
}

/** An inline confirmation for deleting a version or a whole dataset. */
export function ConfirmDelete({
  title,
  children,
  uses,
  action,
  pending,
  error,
  onConfirm,
  onCancel,
}: {
  title: string;
  children: ReactNode;
  uses: VersionUse[];
  action: string;
  pending?: boolean;
  error?: unknown;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const id = useId();
  return (
    <div role="alertdialog" aria-labelledby={id} className="border border-bad/50 bg-bad/10 rounded-md p-3 flex flex-col gap-2 text-[13px]">
      <strong id={id} className="text-sm">{title}</strong>
      <span className="text-muted">{children}</span>
      {uses.length ? (
        <>
          <span>
            Apps made {uses.length === 1 ? "this dataset" : `these ${uses.length} datasets`} from it. They keep working, because each app keeps its own copy of the
            references and holds the audio, and their link to the version shows that it was deleted.
          </span>
          <UseList uses={uses} />
        </>
      ) : (
        <span>No app has made a dataset from it.</span>
      )}
      <ErrorNote error={error} />
      <span className="flex flex-wrap gap-2 pt-1">
        <Button variant="danger" disabled={pending} onClick={onConfirm}>{action}</Button>
        <Button variant="ghost" onClick={onCancel}>Cancel</Button>
      </span>
    </div>
  );
}

export function Times({ s }: { s: Pick<Segment, "start" | "end"> }) {
  return (
    <span className="num text-[11px] text-subtle whitespace-nowrap">
      {clock(s.start)} to {clock(s.end)}
    </span>
  );
}

export function LabelChips({ labels }: { labels: Record<string, unknown> }) {
  const e = Object.entries(labels ?? {});
  if (!e.length) return null;
  return (
    <span className="flex flex-wrap gap-1">
      {e.map(([k, v]) => (
        <span key={k} className="inline-flex items-center rounded bg-track px-1.5 py-px text-[11px] text-muted num">
          {k}={labelValue(v)}
        </span>
      ))}
    </span>
  );
}

/** One reference segment: its times, speaker, text and labels. */
export function SegmentLine({ s, tone }: { s: Segment; tone?: "good" | "bad" }) {
  return (
    <span className={`flex flex-col gap-0.5 ${tone === "good" ? "border-l-2 border-good pl-2" : tone === "bad" ? "border-l-2 border-bad pl-2" : ""}`}>
      <span className="flex flex-wrap items-center gap-2">
        <Times s={s} />
        {s.speaker && <span className="text-[11px] text-focus num">{s.speaker}</span>}
      </span>
      {s.text && <span className="text-[13px]">{s.text}</span>}
      <LabelChips labels={s.labels} />
    </span>
  );
}

/** A word diff from the catalog's comparison: removed words struck through, added words in green. */
export function WordDiff({ ops }: { ops: DiffOp[] }) {
  if (!ops.length) return <span className="text-subtle text-[13px]">no text</span>;
  return (
    <span className="text-[13px] leading-relaxed">
      {ops.map((o, i) => (
        <span key={i}>
          {i > 0 && " "}
          {o.op === "equal" && o.a}
          {(o.op === "delete" || o.op === "replace") && <del className="text-bad bg-bad/10 rounded px-0.5">{o.a}</del>}
          {o.op === "replace" && " "}
          {(o.op === "insert" || o.op === "replace") && <ins className="text-good bg-good/10 rounded px-0.5 no-underline">{o.b}</ins>}
        </span>
      ))}
    </span>
  );
}
