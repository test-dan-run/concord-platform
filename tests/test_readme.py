"""The service tables in README.md match services/*.yaml: one row per manifest, every column from the manifest."""

from tools.schema import PLATFORM_DIR, load_platform

HEADER = ["Service", "Alias", "Kind", "Engine", "Contract / API", "Port", "GPU"]


def readme_rows() -> list[list[str]]:
    """The rows of every table in README.md whose header is HEADER."""
    rows, in_table = [], False
    for line in (PLATFORM_DIR / "README.md").read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            in_table = False
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells == HEADER:
            in_table = True
        elif in_table and not set("".join(cells)) <= set("-:"):
            rows.append(cells)
    return rows


def expected_row(s) -> list[str]:
    gpu = "CPU" if s.gpu is None else f"util {s.gpu.util:.2f}" if s.gpu.util is not None else f"{s.gpu.vram_gb:g} GB"
    return [s.name, s.alias or "", s.kind, s.engine, f"{s.contract} / {s.api}", str(s.port), gpu]


def test_readme_service_tables_match_the_manifests():
    services = load_platform().services
    rows = readme_rows()
    names = [r[0] for r in rows]
    assert sorted(names) == sorted(s.name for s in services), "every manifest has exactly one README row"
    by_name = {r[0]: r for r in rows}
    problems = []
    for s in services:
        have = list(by_name[s.name])
        have[4] = have[4].split(" (")[0]  # "openai / chat.completions (images, video)": the note is free text
        if have != expected_row(s):
            problems.append(f"{s.name}: README {have} manifest {expected_row(s)}")
    assert problems == []
