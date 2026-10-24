// Free-form tags with suggestions from tags already used on catalog models.

import { useQuery } from "@tanstack/react-query";
import { useId, useState } from "react";
import { api, type Model } from "../lib/api";
import * as I from "./icons";

export function TagInput({ value, onChange, label = "Tags", placeholder = "e.g. call-centre, en-SG, telephony-8k" }: { value: string[]; onChange: (v: string[]) => void; label?: string; placeholder?: string }) {
  const id = useId();
  const [text, setText] = useState("");
  const { data: models } = useQuery({ queryKey: ["models"], queryFn: () => api.get<Model[]>("/models") });
  const counts = new Map<string, number>();
  for (const m of models ?? []) for (const t of m.tags) counts.set(t, (counts.get(t) ?? 0) + 1);
  const data = [...counts].map(([tag, count]) => ({ tag, count })).sort((a, b) => b.count - a.count);
  const add = (raw: string) => {
    const parts = raw.split(/[,\s]+/).map((t) => t.trim().replace(/^#/, "")).filter(Boolean);
    const next = [...value];
    for (const p of parts) if (!next.includes(p)) next.push(p);
    onChange(next);
    setText("");
  };
  const q = text.trim().toLowerCase();
  const suggestions = (data ?? []).filter((t) => !value.includes(t.tag) && (!q || t.tag.toLowerCase().includes(q))).slice(0, 8);
  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor={id} className="text-xs text-muted">
        {label}
      </label>
      <div className="flex flex-wrap items-center gap-1.5 bg-canvas border border-line rounded-md px-2 py-1.5 min-h-10">
        {value.map((t) => (
          <span key={t} className="inline-flex items-center gap-1 bg-track rounded px-1.5 py-0.5 text-xs num">
            #{t}
            <button type="button" aria-label={`Remove tag ${t}`} className="bg-transparent border-0 text-muted hover:text-fg cursor-pointer p-0 inline-flex" onClick={() => onChange(value.filter((x) => x !== t))}>
              <I.X size={12} />
            </button>
          </span>
        ))}
        <input
          id={id}
          value={text}
          placeholder={value.length ? "" : placeholder}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if ((e.key === "Enter" || e.key === ",") && text.trim()) {
              e.preventDefault();
              add(text);
            } else if (e.key === "Backspace" && !text && value.length) onChange(value.slice(0, -1));
          }}
          onBlur={() => text.trim() && add(text)}
          className="flex-1 min-w-[140px] bg-transparent border-0 outline-none text-[13px] text-fg"
        />
      </div>
      {suggestions.length > 0 && (
        <div className="flex flex-wrap gap-1.5" aria-label="Suggested tags">
          {suggestions.map((s) => (
            <button key={s.tag} type="button" onClick={() => add(s.tag)} className="bg-transparent border border-line rounded px-1.5 py-0.5 text-[11px] text-muted hover:text-fg cursor-pointer num">
              +{s.tag} <span className="text-subtle">{s.count}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
