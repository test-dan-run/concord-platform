// A model's card: where it came from, its license, upstream metadata, checkpoint facts, and the parts of the
// upstream model card that describe the model (results collapsed). Assembled and trimmed by catalog/cards.py.

import { useQuery } from "@tanstack/react-query";
import type { ReactNode } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api, type ModelCard as Card } from "../lib/api";
import { ago } from "../lib/format";
import { Chip, ErrorNote, Fact, Panel, Spinner } from "./ui";

const HOST = { huggingface: "Hugging Face", github: "GitHub", provider: "Hosted provider" } as const;

/** A long list (languages, datasets) as its first few items; the rest are in the tooltip. */
function few(items: string[], n = 10): ReactNode {
  if (items.length <= n) return items.join(", ");
  return <span title={items.join(", ")}>{items.slice(0, n).join(", ")} and {items.length - n} more</span>;
}

function Doc({ markdown }: { markdown: string }) {
  // Raw HTML in upstream cards is dropped (skipHtml): only Markdown is rendered, so a card cannot inject markup.
  return (
    <div className="markdown">
      <Markdown remarkPlugins={[remarkGfm]} skipHtml components={{ a: ({ node: _n, ...p }) => <a {...p} target="_blank" rel="noopener noreferrer" /> }}>
        {markdown}
      </Markdown>
    </div>
  );
}

export function ModelCard({ id }: { id: string }) {
  const card = useQuery({ queryKey: ["card", id], queryFn: () => api.get<Card>(`/models/${id}/card`), staleTime: 60_000 });
  if (card.isLoading) return <Spinner label="Reading the model card" />;
  if (!card.data) return <ErrorNote error={card.error ?? "No card"} />;
  const c = card.data;
  const u = c.upstream;
  const src = c.source;
  const about: [string, string[]][] = [
    ["Base model", u.base_model],
    ["Languages", u.languages],
    ["Training data", u.datasets],
  ];
  return (
    <div className="flex flex-col gap-5">
      {c.gaps.length > 0 && (
        <div role="status" className="border border-warn/50 bg-warn/10 text-[13px] rounded-md px-3 py-2 flex flex-col gap-1">
          {c.gaps.map((g) => <span key={g}>{g}</span>)}
        </div>
      )}
      <div className="grid gap-5 lg:grid-cols-3 items-start">
        <Panel title="About">
          {c.tasks.length > 0 && <span className="flex flex-wrap gap-1">{c.tasks.map((t) => <Chip key={t}>{t}</Chip>)}</span>}
          <div>
            {about.filter(([, v]) => v.length).map(([k, v]) => (
              <Fact key={k} k={k} v={<span className="text-right break-words">{few(v)}</span>} />
            ))}
          </div>
        </Panel>
        <Panel title="Source and license">
          <div>
            {src ? (
              <>
                <Fact k={HOST[src.host]} v={<a href={src.url} target="_blank" rel="noopener noreferrer" className="break-all">{src.repo}{src.subpath ? `/${src.subpath}` : ""}</a>} />
                {src.revision && <Fact k="Revision" v={<span title={src.revision}>{src.revision.length > 12 ? src.revision.slice(0, 12) : src.revision}{!src.pinned && " (not pinned)"}</span>} />}
                {src.version && <Fact k="Version" v={src.version} />}
                {src.downloaded && <Fact k="Downloaded" v={src.downloaded} />}
              </>
            ) : (
              <Fact k="Source" v={c.model_weights ? "not recorded" : "this platform (no model weights)"} />
            )}
            <Fact k="License" v={c.license.link ? <a href={c.license.link} target="_blank" rel="noopener noreferrer">{c.license.name ?? "license"}</a> : (c.license.name ?? (c.model_weights ? "not recorded" : "platform code"))} />
          </div>
          {c.license.text && (
            <details className="text-xs">
              <summary className="cursor-pointer text-muted">{c.license.file} ({Math.round(c.license.text.length / 1000)}k characters)</summary>
              <pre className="mt-2 max-h-72 overflow-auto scroll-thin text-[11px] num bg-canvas border border-track rounded-md p-2.5 whitespace-pre-wrap">{c.license.text}</pre>
            </details>
          )}
        </Panel>
        <Panel title="Checkpoint">
          {c.facts.length ? (
            <div>{c.facts.map((f) => <Fact key={f.label} k={f.label} v={f.value} />)}</div>
          ) : (
            <span className="text-[13px] text-muted">{c.model_weights ? "No checkpoint on this host." : "No model: a rule-based service. Its README below describes what it does."}</span>
          )}
        </Panel>
      </div>
      {c.sections.map((s) => (
        <Panel
          key={s.key}
          title={s.title}
          subtitle={s.key === "upstream" ? "What the model is, its training data, intended use and limitations, from the upstream card at the revision above. How to install and run it is left out." : s.origin}
          actions={s.full_url && <a href={s.full_url} target="_blank" rel="noopener noreferrer" className="text-xs">Full upstream card{s.fetched_at ? ` (fetched ${ago(s.fetched_at)})` : ""}</a>}
        >
          {s.markdown ? <Doc markdown={s.markdown} /> : <span className="text-[13px] text-muted">The upstream card has no description beyond its usage instructions.</span>}
          {s.evaluation && (
            <details className="border-t border-track pt-3">
              <summary className="cursor-pointer text-[13px] font-semibold">Evaluation results</summary>
              <div className="mt-3"><Doc markdown={s.evaluation} /></div>
            </details>
          )}
        </Panel>
      ))}
    </div>
  );
}
