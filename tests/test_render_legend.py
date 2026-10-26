"""The static catalog's legend states the GPU policy vram_check enforces: one GPU service at a time."""

from tools import render
from tools.schema import load_platform


def test_legend_says_one_gpu_service_runs_at_a_time():
    html = render.render_catalog(load_platform())
    assert "Only one GPU service runs at a time" in html
    assert "can run together if their shares add up" not in html
