"""Event-based encoding (multi-level crossing) and subject-wise dataset builder."""
import numpy as np
from gait_data import make_subject


def fit_normalizer(xs):
    cat = np.concatenate(xs)
    return cat.mean(0), cat.std(0) + 1e-6


def normalize(x, norm):
    return ((x - norm[0]) / norm[1]).astype(np.float32)


def make_levels(n_levels=6, lim=1.5):
    return np.linspace(-lim, lim, n_levels)


def level_crossing_encode(xn, levels):
    """xn [N,C] normalized -> spikes [N, C*L*2] in {0,1}.
    UP spike when a level is crossed upward, DOWN when crossed downward.
    Constant signal -> no spikes."""
    above = xn[:, :, None] > levels[None, None, :]          # N,C,L
    prev = np.concatenate([above[:1], above[:-1]], axis=0)
    up = above & ~prev
    down = ~above & prev
    spk = np.concatenate([up, down], axis=2)                # N,C,2L
    return spk.reshape(len(xn), -1).astype(np.float32)


def build_data(cfg):
    """Subject-wise split. Normalizer is fitted on TRAIN subjects only."""
    sub = {"train": [make_subject(s) for s in range(0, 8)],
           "val":   [make_subject(s) for s in range(100, 102)],
           "test":  [make_subject(s) for s in range(200, 204)]}
    norm = fit_normalizer([x for x, _ in sub["train"]])
    levels = make_levels(cfg.n_levels)
    D = {}
    for k, v in sub.items():
        raw = [normalize(x, norm) for x, _ in v]
        spk = [level_crossing_encode(r, levels) for r in raw]
        D[k] = {"raw": raw, "spk": spk, "y": [y for _, y in v]}
    return D, D["train"]["spk"][0].shape[1]
