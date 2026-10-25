"""Custom build contexts keep host Python caches (__pycache__, *.pyc) out of their images."""

from tools.schema import load_platform

PATTERNS = {"**/__pycache__", "**/*.pyc"}
FROZEN = {"kokoro-tts", "voxcpm2"}  # their manifests say the image must not change, so their contexts stay as they are


def test_every_build_context_ignores_python_caches():
    p = load_platform()
    contexts = [p.root / "custom" / "_bases"]
    contexts += [p.root / s.build.context for s in p.services if s.is_custom and s.build and s.name not in FROZEN]
    missing = {}
    for ctx in contexts:
        f = ctx / ".dockerignore"
        lines = {ln.strip() for ln in f.read_text().splitlines()} if f.is_file() else set()
        if not PATTERNS <= lines:
            missing[str(ctx.relative_to(p.root))] = sorted(PATTERNS - lines)
    assert not missing, f"add these lines to .dockerignore: {missing}"
    assert len(contexts) > 1
