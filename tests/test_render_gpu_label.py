"""The static catalog labels each GPU figure as the minimum the service needs to load, not its footprint."""

from tools import render
from tools.schema import load_platform


def test_gpu_figures_are_labelled_the_minimum_to_load():
    html = render.render_catalog(load_platform())
    assert 'aria-label="Minimum GPU memory to load, by service"' in html
    assert 'el("dt", { text: "GPU (minimum to load)" })' in html
    assert "its fill is the minimum GPU memory the service" in html
