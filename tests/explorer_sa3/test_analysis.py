import numpy as np
from plots.explorer_sa3.analysis import pca_frames, dim_xcorr, dim_feature_corr


def _rng():
    return np.random.default_rng(0)


def test_pca_shapes_and_variance_order():
    r = _rng()
    lats = [r.standard_normal((256, 64)).astype(np.float32) for _ in range(5)]
    comps, evr = pca_frames(lats, k=3)
    assert comps.shape == (3, 256) and evr.shape == (3,)
    assert evr[0] >= evr[1] >= evr[2]


def test_dim_xcorr_shape_and_diag():
    r = _rng()
    lats = [r.standard_normal((256, 64)).astype(np.float32) for _ in range(3)]
    c = dim_xcorr(lats)
    assert c.shape == (256, 256)
    assert np.allclose(np.diag(c), 1.0, atol=1e-5)


def test_dim_feature_corr_detects_link():
    r = _rng()
    feat = r.standard_normal(128).astype(np.float32)
    z = r.standard_normal((256, 128)).astype(np.float32)
    z[7] = feat * 3.0  # dim 7 perfectly correlated with the feature
    corr = dim_feature_corr([z], [feat])
    assert corr.shape == (256,)
    assert abs(corr[7]) > 0.99
