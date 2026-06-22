import numpy as np


def test_viewer_figure_builds():
    from plots.explorer_sa3.viewer_tab import latent_figure
    z = np.random.randn(256, 64).astype(np.float32)
    fig = latent_figure(z, content_frames=40)
    assert fig is not None
    assert len(fig.data) >= 1   # heatmap trace present


def test_analysis_figure_builds():
    from plots.explorer_sa3.analysis_tab import xcorr_figure
    c = np.eye(256, dtype=np.float32)
    fig = xcorr_figure(c)
    assert fig is not None


def test_app_layout_imports():
    from plots.explorer_sa3.app import build_layout
    layout = build_layout(index=[])
    assert layout is not None
