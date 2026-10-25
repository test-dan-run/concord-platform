"""Model cards: what a model is and where it came from, assembled from what the platform already holds.

Sources, all read from disk at request time (nothing is fetched here; `just fetch-cards` does that):

- provenance from `tools/model_cards.source_of` (SOURCE.txt, the manifest comment, or weights.yaml);
- the upstream model card: README.md in the weights folder, else a copy fetched into
  ${AI_ASSETS}/model-cards; its YAML header gives license, languages, base model and datasets;
- facts from the checkpoint's config.json and safetensors headers (architecture, context
  length, precision, quantization, parameters) and the weights folder's size on disk;
- for a rule-based custom service (no model weights), its README.md. A model's own service README and
  WHY.md describe how this platform serves it, not the model, so they stay out of its card.

The upstream card is cut down to what describes the model (`digest`): overview, architecture, inputs and
outputs, languages, training data, intended use and limitations, with evaluation results kept apart.
How to install, run, serve or fine-tune it, citations, news and contacts are dropped, and so is all code:
this platform serves the model its own way, and the card links to the full upstream copy. Relative links
and images are rewritten to the upstream repo at the same revision.
"""

from __future__ import annotations

import json
import posixpath
import re
import struct
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import yaml

from tools import identity, model_cards
from tools.model_cards import Source
from tools.schema import Manifest, Platform

MAX_MARKDOWN = 400_000  # characters per section
MAX_LICENSE = 120_000
_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
_LINK = re.compile(r"(!?)\[([^\]]*)\]\(\s*([^)\s]+)([^)]*)\)")
_FENCE = re.compile(r"^\s*(```|~~~)")
_ATX = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_HTML_H = re.compile(r"^\s*<h([1-6])\b[^>]*>(.*?)</h\1>\s*$", re.I)
# "**Training Data:** ..." or a line of just "**Resources**", used as a heading
_LABEL = re.compile(r"^\*\*([^*\n]{1,60}?)(?::\*\*|\*\*:)\s*(.*)$|^\*\*([^*\n]{1,60})\*\*\s*$")
_IMAGE = re.compile(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)|!\[[^\]]*\]\([^)]*\)")
_TAG = re.compile(r"<[^>]+>")
_HTML_TABLE = re.compile(r"<table\b.*?</table>", re.S | re.I)
_HTML_BLOCK = re.compile(r"<(video|picture|svg|script|style)\b.*?</\1>|<!--.*?-->", re.S | re.I)
# a short line around a link to a demo or chat: "✨ Try the [demo](...)"
_PROMO = re.compile(r"^[^\n]{0,80}\[[^\]\n]*\b(demo|colab|discord|wechat)\b[^\]\n]*\]\([^)\n]*\)[^\n]{0,80}$", re.I | re.M)
_DETAILS = re.compile(r"<details\b.*?</details>", re.S | re.I)
# Sections of an upstream card, by heading: _DROP, then _CORE (training data, intended use, limitations; these
# are kept even under a dropped section or with the results), _EVAL (results, kept apart), _KEEP. Any other
# heading, or one under a dropped section or the results, takes its parent's choice. The preamble and title
# are kept.
_DROP = re.compile(
    r"usage|quick ?start|how to|get(ting)? started|install|setup|set up|environment|requirement|inference|"
    r"deploy|serving|\bserve|vllm|transformers|llama\.cpp|mlx|sglang|ollama|\bsdk\b|\bapi\b|demo|example|"
    r"download|fine-?tun|agentic|tool call|thinking|best practice|long texts|citation|bibtex|reference|"
    r"acknowledg|contact|\bteam\b|news|release|roadmap|zoo|resources|more information|licen[cs]e|terms|"
    r"geography|software|hardware|technical spec|speed|memory|compile|attention|ecosystem|fun fact|embeddings|"
    r"qualitative|model sources|lineage|version|changelog|community|faq|training(?= *$)|switching|fast start|docker|"
    r"links|get in touch|related|join|support|\bto use\b|methodology|note on", re.I)
_CORE = re.compile(r"limitation|bias|risk|ethic|out-of-scope|intended|evaluated use|use case|training|dataset", re.I)
_EVAL = re.compile(r"evaluation|performance|results|benchmark|metric|\b(wer|cer|der)\b|accuracy|leaderboard|testing", re.I)
_KEEP = re.compile(r"\buses\b|domain|overview|abstract|introduction|description|details|facts|highlights|architecture|"
                   r"input|output|language", re.I)
_PACKED = {"awq", "gptq", "compressed-tensors", "modelopt", "bitsandbytes", "nvfp4", "mxfp4", "gguf"}


def _listify(v: Any) -> list[str]:
    if v is None:
        return []
    return [str(x) for x in v] if isinstance(v, list) else [str(v)]


def split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    m = _FRONT.match(text)
    if not m:
        return {}, text
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return (meta if isinstance(meta, dict) else {}), text[m.end():]


def rewrite_links(markdown: str, src: Source, revision: str, folder: str = "") -> str:
    """Point relative links and images at the upstream repo, at the same revision. A link that climbs out of
    the repo is left as it is."""
    def absolute(m: re.Match) -> str:
        bang, label, target, rest = m.groups()
        if re.match(r"^([a-z][a-z0-9+.-]*:|#|/)", target, re.I):
            return m.group(0)
        path = posixpath.normpath(posixpath.join(folder, target))  # as a browser resolves it: ".github" stays
        if path == ".." or path.startswith("../"):
            return m.group(0)
        path = "" if path == "." else path
        if src.host == "huggingface":
            url = f"https://huggingface.co/{src.repo}/{'resolve' if bang else 'blob'}/{revision}/{path}"
        elif bang:
            url = f"https://raw.githubusercontent.com/{src.repo}/{revision}/{path}"
        else:
            url = f"https://github.com/{src.repo}/blob/{revision}/{path}"
        return f"{bang}[{label}]({url}{rest})"

    return _LINK.sub(absolute, markdown)


def _heading(line: str, after_blank: bool) -> tuple[int, str, str] | None:
    """(level, title, text on the same line) of a heading line. A bold label opening a paragraph is a heading
    one level below every real one (7), so it lasts until the next label or heading."""
    if m := _ATX.match(line):
        return len(m.group(1)), m.group(2), ""
    if m := _HTML_H.match(line):
        return int(m.group(1)), _TAG.sub("", m.group(2)).strip(), ""
    if after_blank and (m := _LABEL.match(line)):
        return 7, (m.group(1) or m.group(3)).strip(), m.group(2) or ""
    return None


class _Table(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.cell: list[str] | None = None
        self.span = 1

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th") and self.rows:
            self.cell = []
            span = dict(attrs).get("colspan") or "1"
            self.span = int(span) if span.isdigit() else 1
        elif tag == "br" and self.cell is not None:
            self.cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self.cell is not None and self.rows:
            self.rows[-1].append(" ".join("".join(self.cell).split()).replace("|", "\\|"))
            self.rows[-1].extend([""] * (self.span - 1))
            self.cell = None

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)


def _html_table(m: re.Match) -> str:
    """An HTML table as a Markdown one (cell text only), so it shows on a page that renders no raw HTML."""
    t = _Table()
    t.feed(m.group(0))
    rows = [r for r in t.rows if any(r)]
    if not rows:
        return ""
    width = max(map(len, rows))
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n\n" + "\n".join([lines[0], "|" + "---|" * width, *lines[1:]]) + "\n\n"


def _choice(title: str, parent: str) -> str:
    plain = _TAG.sub("", title).replace("*", "").replace("`", "")
    if _DROP.search(plain):
        return "drop"
    if _CORE.search(plain):
        return "summary"
    if parent in ("drop", "evaluation"):
        return parent
    if _EVAL.search(plain):
        return "evaluation"
    if _KEEP.search(plain):
        return "summary"
    return parent


def digest(markdown: str) -> tuple[str, str]:
    """The parts of an upstream card that describe the model, and its evaluation results, as two Markdown texts.

    Sections are chosen by heading (see _DROP); fenced code and raw HTML (but for tables) go everywhere, and Markdown images
    and collapsed <details> go from the summary (badges, logos, banners, demos) but images stay with the
    results (charts). Headings are kept only when
    something under them is, and the card's own title is left out (the page already names the model).
    """
    nodes: list[tuple[int, str, list[str]]] = [(0, "", [])]  # (level, title, body lines); level 0 = preamble
    fence, blank = None, True
    for line in markdown.splitlines():
        if m := _FENCE.match(line):
            fence = None if fence == m.group(1) else (fence or m.group(1))
            continue
        if fence:
            continue
        if h := _heading(line, blank):
            nodes.append((h[0], h[1], [h[2]] if h[2] else []))
        else:
            nodes[-1][2].append(line)
        blank = not line.strip()

    out: dict[str, list[str]] = {"summary": [], "evaluation": []}
    stack: list[tuple[int, str, str, list]] = []  # (level, title, choice, [emitted?]) of the open ancestors
    title_seen = False
    for level, title, body in nodes:
        while stack and stack[-1][0] >= level > 0:
            stack.pop()
        if level == 0:
            choice = "summary"
        elif level == 1 and not title_seen and len(stack) == 1:  # only the preamble is open
            title_seen, choice = True, "summary"
            title = ""  # the card's title: keep what it says, not the heading
        else:
            choice = _choice(title, stack[-1][2] if stack else "summary")
        stack.append((level, title, choice, [False]))
        if choice == "drop":
            continue
        text = "\n".join(body)
        if choice == "summary":
            text = _PROMO.sub("", _DETAILS.sub("", _IMAGE.sub("", text)))
        # the page renders no raw HTML: tables become Markdown, media and other tags would only be empty space
        text = _HTML_TABLE.sub(_html_table, _HTML_BLOCK.sub("", text))
        text = "\n".join(x.rstrip() for x in _TAG.sub("", text).splitlines()).strip()
        text = re.sub(r"(?<=\S)\n(?=\*\*[^*\n]{1,60}:\*\*)", "\n\n", text)  # "**Input Type(s):** ..." lines stay apart
        if not text:
            continue
        lines = out[choice]
        for lvl, t, c, emitted in stack:  # headings of this section and its ancestors in the same part, once
            if c == choice and t and not emitted[0]:
                emitted[0] = True
                lines.append(f"**{t}:**" if lvl == 7 else f"{'#' * min(lvl + 1, 6)} {t}\n")
        lines.append(text + "\n")
    tidy = lambda xs: re.sub(r"\n{3,}", "\n\n", "\n".join(xs)).strip()  # noqa: E731
    return tidy(out["summary"]), tidy(out["evaluation"])


def _read(path: Path, limit: int) -> str | None:
    if not path.is_file():
        return None
    text = path.read_text(errors="replace")
    return text if len(text) <= limit else text[:limit] + "\n\n*(truncated)*\n"


def _config_facts(folder: Path) -> list[dict[str, str]]:
    cfg_path = folder / "config.json"
    if not cfg_path.is_file():
        return []
    try:
        cfg = json.loads(cfg_path.read_text())
    except ValueError:
        return []
    text = cfg.get("text_config") or cfg.get("thinker_config", {}).get("text_config") or {}

    def pick(*keys: str) -> Any:
        for k in keys:
            for d in (cfg, text):
                if isinstance(d, dict) and d.get(k) not in (None, ""):
                    return d[k]
        return None

    facts = []
    arch = _listify(cfg.get("architectures"))
    if arch:
        facts.append({"label": "Architecture", "value": ", ".join(arch)})
    if (v := pick("model_type")) and not arch:
        facts.append({"label": "Model type", "value": str(v)})
    if (v := pick("max_position_embeddings", "max_source_positions", "n_positions")) is not None:
        facts.append({"label": "Context length", "value": f"{v:,}" if isinstance(v, int) else str(v)})
    dtype = pick("torch_dtype", "dtype")
    quant = cfg.get("quantization_config") or (text.get("quantization_config") if isinstance(text, dict) else None)
    packed = False
    if isinstance(quant, dict) and (method := quant.get("quant_method") or quant.get("quant_algo") or quant.get("format")):
        bits = quant.get("bits") or quant.get("w_bit")
        facts.append({"label": "Quantization", "value": f"{method}" + (f", {bits}-bit" if bits else "")})
        # packed formats store several weights per element, so a header element count is not a parameter count
        packed = str(method).lower() in _PACKED or (str(bits).isdigit() and int(bits) < 8)
    if dtype is not None:
        facts.append({"label": "Precision of unquantized layers" if packed or quant else "Precision", "value": str(dtype)})
    if not packed and (params := _safetensors_params(folder)):
        facts.append({"label": "Parameters", "value": _human(params)})
    return facts


def _safetensors_params(folder: Path) -> int | None:
    """Element count over every tensor in the checkpoint's safetensors headers (header only, no weights read)."""
    total, files = 0, sorted(folder.glob("*.safetensors"))
    if not files:
        return None
    for f in files:
        try:
            with open(f, "rb") as fh:
                (n,) = struct.unpack("<Q", fh.read(8))
                if n > 100 * 1024 * 1024:
                    return None
                header = json.loads(fh.read(n))
        except (OSError, ValueError, struct.error):
            return None
        for name, t in header.items():
            if name != "__metadata__":
                count = 1
                for d in t.get("shape", []):
                    count *= int(d)
                total += count
    return total or None


def _human(n: float) -> str:
    for unit, size in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= size:
            return f"{n / size:.1f}{unit}".replace(".0", "")
    return str(int(n))


def _size(folder: Path) -> int:
    return sum(f.stat().st_size for _, f in identity.weight_files(folder))


def _license(meta: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    name = spec.get("license") or meta.get("license_name") or meta.get("license")
    if isinstance(name, list):
        name = ", ".join(map(str, name))
    link = meta.get("license_link")
    return {"name": str(name) if name else None, "link": str(link) if link else None, "text": None, "file": None}


def build(p: Platform, s: Manifest) -> dict[str, Any]:
    """The card for a platform service."""
    src = model_cards.source_of(p, s)
    folder = model_cards.weights_folder(p, s)
    ws = identity.weights_source(p, s)
    spec = (yaml.safe_load(ws.path.read_text()) or {}) if ws.kind == "yaml" else {}
    gaps: list[str] = []
    sections: list[dict[str, Any]] = []

    # The upstream card: the pinned copy in the weights folder, else one fetched by `just fetch-cards`.
    readme_dir, revision, fetched = None, (src.ref if src else None), None
    if folder is not None and (folder / "README.md").is_file() and src and src.origin == "SOURCE.txt":
        readme_dir = folder
    elif src is not None and (found := model_cards.cache_dirs(p, src)):
        readme_dir = found[0]
        fetched = json.loads((readme_dir / "meta.json").read_text())
        revision = fetched.get("revision") or revision
    meta: dict[str, Any] = {}
    if readme_dir is not None:
        meta, body = split_front_matter(_read(readme_dir / "README.md", MAX_MARKDOWN) or "")
        sub = src.subpath if src and src.host == "github" and fetched and fetched.get("readme_from") else ""
        summary, results = digest(rewrite_links(body, src, revision or "main", sub) if src else body)
        full = None
        if src is not None:
            full = (f"https://huggingface.co/{src.repo}/blob/{revision or 'main'}/README.md" if src.host == "huggingface"
                    else src.url(revision))
        sections.append({"key": "upstream", "title": "Model card", "markdown": summary, "evaluation": results or None,
                         "full_url": full, "origin": "weights folder" if readme_dir == folder else "fetched copy",
                         "fetched_at": (fetched or {}).get("fetched_at")})
    elif src is not None:
        gaps.append(f"No upstream model card on this host: run `just fetch-cards {s.name}`.")
    elif ws.kind == "none" and s.is_custom:
        pass  # a rule-based service with no model: its own README is the card
    else:
        gaps.append("No source is recorded for this model: add SOURCE.txt (repo, revision) to its weights folder, "
                    "then run `just fetch-cards`.")
    if src is not None and not src.pinned:
        gaps.append(f"The source {src.repo} has no pinned revision on record"
                    + (f"; the card was fetched at {str(revision)[:12]}." if fetched else "."))

    lic = _license(meta, spec)
    for d in filter(None, (readme_dir, folder)):
        for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"):
            if (d / name).is_file():
                lic["text"], lic["file"] = _read(d / name, MAX_LICENSE), name
                break
        if lic["text"]:
            break
    if not lic["name"] and not lic["text"] and not (ws.kind == "none" and s.is_custom):
        gaps.append("No license is recorded for this model.")

    facts: list[dict[str, str]] = []
    if folder is not None and folder.is_dir():
        facts += _config_facts(folder)
        size = _size(folder)
        facts.append({"label": "Size on disk", "value": f"{size / 1e9:.2f} GB" if size >= 1e8 else f"{size / 1e6:.1f} MB"})
    elif folder is not None:
        how = f"run `just fetch-weights {s.name}`" if ws.kind == "yaml" else "copy the model there"
        gaps.append(f"The weights folder {folder} is not on this host, so its size and config are unknown; {how}.")

    # A rule-based service has no upstream card: what its README says it does is the card (without its code).
    if ws.kind == "none" and s.is_custom and s.build:
        if (text := _read(p.root / s.build.context / "README.md", MAX_MARKDOWN)) is not None:
            sections.append({"key": "service", "title": "What it does", "markdown": "\n\n".join(filter(None, digest(text))),
                             "evaluation": None, "full_url": None, "origin": f"{s.build.context}/README.md"})

    return {
        "id": s.name,
        "description": s.description,
        "tasks": list(s.tasks),
        "model_weights": ws.kind != "none",  # False: a rule-based service, its README is its card
        "source": None if src is None else {
            "host": src.host, "repo": src.repo, "revision": revision, "pinned": src.pinned, "origin": src.origin,
            "url": src.url(revision), "downloaded": src.downloaded, "subpath": src.subpath or None,
            "version": spec.get("version")},
        "license": lic,
        "upstream": {"base_model": _listify(meta.get("base_model")), "languages": _listify(meta.get("language")),
                     "datasets": _listify(meta.get("datasets"))},
        "facts": facts,
        "sections": sections,
        "gaps": gaps,
    }


def build_hosted(h: Any) -> dict[str, Any]:
    """Hosted endpoints have no local checkpoint: the card says where calls go."""
    return {
        "id": h.id, "description": None, "tasks": [], "model_weights": True,
        "source": {"host": "provider", "repo": h.model, "revision": None, "pinned": False, "origin": "hosted endpoint",
                   "url": h.base_url, "downloaded": None, "subpath": None, "version": None},
        "license": {"name": None, "link": None, "text": None, "file": None},
        "upstream": {"base_model": [], "languages": [], "datasets": []},
        "facts": [{"label": "Provider", "value": h.host}, {"label": "Model at the provider", "value": h.model}],
        "sections": [],
        "gaps": ["Hosted endpoints have no local model card; see the provider's documentation for this model."],
    }
