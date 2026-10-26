"""Model cards: sources, fetching (against a fake Hugging Face and GitHub), and the catalog's card assembly."""

import json
import struct

import httpx
import pytest

from catalog import cards
from tools import model_cards
from tools.model_cards import Source
from tools.schema import load_platform


def test_source_urls_parse_for_both_hosts():
    hf = model_cards._parse_url("https://huggingface.co/nvidia/RE-USE/tree/85575b7", "weights.yaml")
    assert (hf.host, hf.repo, hf.ref, hf.pinned) == ("huggingface", "nvidia/RE-USE", "85575b7", True)
    unpinned = model_cards._parse_url("https://huggingface.co/pyannote/speaker-diarization-3.1", "weights.yaml")
    assert unpinned.ref is None and not unpinned.pinned
    gh = model_cards._parse_url("https://github.com/microsoft/DNS-Challenge/tree/82f1b17/DNSMOS/DNSMOS", "weights.yaml")
    assert (gh.host, gh.repo, gh.ref, gh.subpath) == ("github", "microsoft/DNS-Challenge", "82f1b17", "DNSMOS/DNSMOS")


def test_every_manifest_source_is_found_from_the_platform_records(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_ASSETS", str(tmp_path))  # no SOURCE.txt anywhere: comments and weights.yaml still name sources
    p = load_platform()
    found = {s.name: model_cards.source_of(p, s) for s in p.services}
    assert found["cohere-transcribe-03-2026"] == Source("huggingface", "CohereLabs/cohere-transcribe-03-2026", "d552dad", "", "manifest comment")
    assert found["silero-vad"].host == "github" and found["silero-vad"].ref == "v6.2.3"
    assert found["text-normalizer"] is None and found["wpe-dereverb"] is None  # no model weights at all


def test_front_matter_and_link_rewriting():
    meta, body = cards.split_front_matter("---\nlicense: apache-2.0\nlanguage: [en, zh]\n---\n# Qwen\n![fig](figures/a.png) [doc](docs/x.md) [ext](https://x.y) [top](#usage)")
    assert meta == {"license": "apache-2.0", "language": ["en", "zh"]} and body.startswith("# Qwen")
    hf = cards.rewrite_links(body, Source("huggingface", "Qwen/Qwen3-8B", "abc", "", "SOURCE.txt"), "abc")
    assert "](https://huggingface.co/Qwen/Qwen3-8B/resolve/abc/figures/a.png)" in hf
    assert "](https://huggingface.co/Qwen/Qwen3-8B/blob/abc/docs/x.md)" in hf and "](https://x.y)" in hf and "](#usage)" in hf
    gh = cards.rewrite_links("![a](./b.png)", Source("github", "o/r", "v1", "sub", "weights.yaml"), "v1", "sub")
    assert gh == "![a](https://raw.githubusercontent.com/o/r/v1/sub/b.png)"


def safetensors(path, tensors):
    header = json.dumps({name: {"dtype": "BF16", "shape": shape, "data_offsets": [0, 0]} for name, shape in tensors}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header)


def test_config_facts_count_parameters_only_when_not_packed(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"], "num_hidden_layers": 2,
                                                      "max_position_embeddings": 4096, "torch_dtype": "bfloat16"}))
    safetensors(tmp_path / "model.safetensors", [("a", [1000, 1000]), ("b", [500_000])])
    facts = {f["label"]: f["value"] for f in cards._config_facts(tmp_path)}
    assert facts == {"Architecture": "LlamaForCausalLM", "Context length": "4,096", "Precision": "bfloat16", "Parameters": "1.5M"}
    cfg = json.loads((tmp_path / "config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps({**cfg, "quantization_config": {"quant_method": "awq", "bits": 4}}))
    facts = {f["label"]: f["value"] for f in cards._config_facts(tmp_path)}
    assert "Parameters" not in facts and facts["Quantization"] == "awq, 4-bit" and "Precision of unquantized layers" in facts


class FakeHubs:
    def __call__(self, r: httpx.Request) -> httpx.Response:
        url = str(r.url)
        if url == "https://huggingface.co/api/models/acme/asr/revision/d552dad":
            return httpx.Response(200, json={"sha": "d552dad" + "0" * 33,
                                             "siblings": [{"rfilename": n} for n in ("README.md", "LICENSE", "model.safetensors")]})
        if url.startswith("https://huggingface.co/acme/asr/resolve/d552dad"):
            return httpx.Response(200, text="---\nlicense: mit\n---\n# ACME ASR\nSpeech in, text out.\n## Usage\n```py\nrun()\n```"
                                  if url.endswith("README.md") else "MIT License")
        if url == "https://raw.githubusercontent.com/o/r/v1/README.md":  # no README in o/r/v1/a/b, nor in a/
            return httpx.Response(200, text="# repo root readme")
        if url == "https://raw.githubusercontent.com/o/r/v1/LICENSE":
            return httpx.Response(200, text="Apache")
        return httpx.Response(404)


@pytest.fixture
def platform(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_ASSETS", str(tmp_path / "assets"))
    return load_platform()


def test_fetch_writes_outside_weights_folders_and_records_the_full_revision(platform, tmp_path):
    with httpx.Client(transport=httpx.MockTransport(FakeHubs())) as http:
        src = Source("huggingface", "acme/asr", "d552dad", "", "manifest comment")
        dest = model_cards.fetch(platform, src, http)
        assert dest == tmp_path / "assets/model-cards/huggingface/acme/asr" / ("d552dad" + "0" * 33)
        assert sorted(p.name for p in dest.iterdir()) == ["LICENSE", "README.md", "meta.json"]
        assert model_cards.cache_dirs(platform, src) == [dest]  # the short ref finds the fetched full revision
        gh = model_cards.fetch(platform, Source("github", "o/r", "v1", "a/b", "weights.yaml"), http)
        assert (gh / "README.md").read_text() == "# repo root readme" and (gh / "LICENSE").read_text() == "Apache"


def test_card_for_a_model_with_a_fetched_upstream_card(platform, monkeypatch):
    s = platform.get("cohere-transcribe-03-2026")
    with httpx.Client(transport=httpx.MockTransport(FakeHubs())) as http:
        monkeypatch.setattr(model_cards, "source_of", lambda p, s: Source("huggingface", "acme/asr", "d552dad", "", "manifest comment"))
        model_cards.fetch(platform, model_cards.source_of(platform, s), http)
    card = cards.build(platform, s)
    assert card["source"]["revision"] == "d552dad" + "0" * 33 and card["source"]["pinned"]
    assert card["license"] == {"name": "mit", "link": None, "text": "MIT License", "file": "LICENSE"}
    up = card["sections"][0]
    assert up["key"] == "upstream" and up["markdown"] == "Speech in, text out." and up["evaluation"] is None
    assert up["full_url"] == "https://huggingface.co/acme/asr/blob/" + "d552dad" + "0" * 33 + "/README.md"
    assert any("not on this host" in g for g in card["gaps"])  # the tmp assets have no weights folder


def test_rule_based_service_card_has_no_false_gaps(platform):
    card = cards.build(platform, platform.get("text-normalizer"))
    assert card["source"] is None and card["gaps"] == [] and [x["key"] for x in card["sections"]] == ["service"] and not card["model_weights"]
    assert "```" not in card["sections"][0]["markdown"] and "capitalise" in card["sections"][0]["markdown"]
    # A custom model with weights gets no service section (and no fetched upstream card in the tmp assets).
    assert cards.build(platform, platform.get("silero-vad"))["sections"] == []


CARD = """\
[![badge](https://img.shields.io/x.svg)](https://x)

# Acme ASR

Acme ASR transcribes speech, and a demo of it would show that.

✨ Try the [demo](https://hf.co/spaces/acme)

<table><tr><th>Training objective</th><td>CTC<br>from scratch</td></tr><tr><th colspan="2">Languages</th><td>en</td></tr></table>

## Quick Start
Install it:
```bash
# not a heading
pip install acme
```
### Input format
16 kHz mono.

## Fine-tuning
Do this.
### Evaluated Use
Research on robustness.

**Training Data:** 10k hours of LibriSpeech.

**Release Date**: today

## Results
![chart](https://x/chart.png)
### English
| set | WER |
|---|---|
| clean | 2.0 |

## Limitations
Hallucinates on silence.

## Citation
@article{acme}
"""


def test_digest_keeps_what_describes_the_model_and_sets_results_apart():
    summary, results = cards.digest(CARD)
    assert summary.startswith("Acme ASR transcribes speech, and a demo")  # title heading, badge and plug gone
    assert "| Training objective | CTC from scratch |" in summary and "| Languages |  | en |" in summary  # HTML table kept as Markdown
    for gone in ("Quick Start", "pip install", "not a heading", "Input format", "Fine-tuning", "Do this", "Release Date",
                 "Citation", "@article", "Try the", "shields.io", "<"):
        assert gone not in summary, gone
    assert "#### Evaluated Use\n\nResearch on robustness." in summary  # kept although under a dropped section
    assert "**Training Data:**\n10k hours of LibriSpeech." in summary
    assert summary.index("Training Data") < summary.index("### Limitations")
    assert results.startswith("### Results\n\n![chart](https://x/chart.png)") and "#### English" in results and "| clean | 2.0 |" in results
