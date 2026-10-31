"""Training-set fraction for learning curves.

TDA_FRAC=0.25 keeps a quarter of the training *patients* in every fold, chosen with a
stratified draw that is a deterministic function of (seed, fold), so every method sees the
same reduced training set. Test folds are untouched, so the out-of-fold vector keeps its
full length and the usual collect step applies.
"""
import os
import numpy as np


def cut(tr, patients, y, seed=0, fold=0):
    frac = float(os.environ.get('TDA_FRAC', '1'))
    if frac >= 1.0:
        return tr
    tr = np.asarray(tr)
    pats = patients[tr]
    uniq = np.unique(pats)
    lab = {p: int(y[tr][pats == p].max()) for p in uniq}
    rng = np.random.default_rng(1000 * seed + fold + 17)
    keep = []
    for c in (0, 1):
        ps = np.array([p for p in uniq if lab[p] == c])
        k = max(1, int(round(frac * len(ps)))) if len(ps) else 0
        if k:
            keep.extend(rng.choice(ps, size=k, replace=False))
    keep = set(keep)
    out = np.array([i for i in tr if patients[i] in keep])
    return out if len(np.unique(y[out])) == 2 else tr
