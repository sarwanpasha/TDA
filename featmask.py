"""Zero out blocks of the per-cell feature vector z = [one-hot | distance profile | ranks].

TDA_MASK in {pheno, dist, marker, nomarker}: keep only that block (or drop the marker
block). Applied once after loading, before any fold is formed, so every model in the run
reads the reduced feature.
"""
import os
import numpy as np


def apply(recs, P):
    mask = os.environ.get('TDA_MASK', '')
    if not mask:
        return recs
    for r in recs:
        Z = np.array(r['Z'], dtype=np.float32, copy=True)
        d = Z.shape[1]
        keep = np.zeros(d, bool)
        if mask == 'pheno':
            keep[:P] = True
        elif mask == 'dist':
            keep[P:2 * P] = True
        elif mask == 'marker':
            keep[2 * P:] = True
        elif mask == 'nomarker':
            keep[:2 * P] = True
        else:
            raise SystemExit('unknown TDA_MASK ' + mask)
        Z[:, ~keep] = 0.0
        r['Z'] = Z
    return recs
