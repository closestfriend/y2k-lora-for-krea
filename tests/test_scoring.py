import numpy as np
from y2k_pipeline.scoring import vibe_scores


def test_vibe_scores_prefers_pos_aligned():
    pos = np.array([[1.0, 0.0]]); neg = np.array([[0.0, 1.0]])
    txt = np.vstack([pos, neg])
    imgs = np.array([[10.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
    s = vibe_scores(imgs, txt, n_pos=1)
    assert s[0] > s[2] > s[1]
    assert abs(s[0] - 1.0) < 1e-6 and abs(s[1] + 1.0) < 1e-6
    assert abs(s[2]) < 1e-6  # equidistant


def test_vibe_scores_norm_invariance():
    txt = np.array([[1.0, 0.0], [0.0, 1.0]])
    a = vibe_scores(np.array([[2.0, 1.0]]), txt, 1)
    b = vibe_scores(np.array([[200.0, 100.0]]), txt, 1)
    assert abs(a[0] - b[0]) < 1e-6
