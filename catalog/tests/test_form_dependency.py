"""FastAPI needs python-multipart for the routes that take form data (the Dataset Catalog's package uploads). It
imports it only when such a route is defined, so nothing in the code imports it: the root project must declare it
instead of getting it through aisvc."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_python_multipart_is_a_direct_dependency():
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    assert any(re.match(r"python-multipart\s*>=", d) for d in deps), deps
    lock = (ROOT / "uv.lock").read_text()
    root = lock[lock.index('name = "aiplatform"'):]
    root = root[:root.index("[[package]]")]
    assert '{ name = "python-multipart", specifier = ">=' in root
