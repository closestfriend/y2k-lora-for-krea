import numpy as np


def vibe_scores(img_embs, txt_embs, n_pos):
    img = img_embs / np.linalg.norm(img_embs, axis=1, keepdims=True)
    txt = txt_embs / np.linalg.norm(txt_embs, axis=1, keepdims=True)
    sims = img @ txt.T
    return sims[:, :n_pos].mean(axis=1) - sims[:, n_pos:].mean(axis=1)
