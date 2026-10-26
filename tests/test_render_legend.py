"""The static catalog states the GPU policy vram_check enforces: one GPU service at a time, and each GPU figure is
the minimum the service needs to load, not its footprint."""

from tools import render
from tools.schema import load_platform


def test_catalog_text_states_the_gpu_policy():
    html = render.render_catalog(load_platform())
    assert "Only one GPU service runs at a time" in html
    assert "can run together if their shares add up" not in html
    assert 'aria-label="Minimum GPU memory to load, by service"' in html
    assert 'el("dt", { text: "GPU (minimum to load)" })' in html
    assert "its fill is the minimum GPU memory the service" in html
