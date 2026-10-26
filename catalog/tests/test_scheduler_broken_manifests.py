"""While the manifests are invalid, platform() loads them once per change: the same broken files give the last good
platform and the stored error without loading them again."""

import catalog.scheduler as scheduler_module
from tools.schema import PlatformError


def test_the_same_broken_manifests_are_loaded_once(cat, monkeypatch):
    s = cat.scheduler
    good = s.platform()
    real = scheduler_module.load_platform
    loads = []
    sig = ["broken"]

    def load_platform(root):
        loads.append(sig[0])
        if sig[0].startswith("broken"):
            raise PlatformError([f"services/x.yaml: bad ({sig[0]})"])
        return real(root)

    monkeypatch.setattr(scheduler_module, "load_platform", load_platform)
    monkeypatch.setattr(s, "_manifest_sig", lambda: (sig[0],))
    assert s.platform() is good and s.platform() is good and s.manifest("silero-vad").name == "silero-vad"
    assert loads == ["broken"] and s.platform_error == "services/x.yaml: bad (broken)"
    sig[0] = "broken again"  # another edit, still broken: loaded once more
    assert s.platform() is good and s.platform() is good
    assert loads == ["broken", "broken again"] and s.platform_error == "services/x.yaml: bad (broken again)"
    sig[0] = "fixed"
    fixed = s.platform()
    assert fixed is not good and s.platform() is fixed and s.platform_error is None
    assert loads == ["broken", "broken again", "fixed"]
