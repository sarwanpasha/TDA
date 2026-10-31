"""Per-cohort quantile alignment of the distance-profile block, the domain-adaptation
control for the cross-cohort transfer.

Each distance column of the target cohort is mapped, cell by cell, to the value with the
same quantile in the source cohort. Nothing label-related crosses: only the marginal
distribution of each geometric feature is matched, which is the simplest treatment of a
shift in cell density and core size.
"""
import numpy as np


def align(src, dst):
    P = len(src['meta']['vocab'])
    cols = range(P, 2 * P)
    S = np.concatenate([r['Z'][:, P:2 * P] for r in src['records']], 0)
    D = np.concatenate([r['Z'][:, P:2 * P] for r in dst['records']], 0)
    qs = np.linspace(0, 1, 1001)
    maps = []
    for j, c in enumerate(cols):
        sq = np.quantile(S[:, j], qs)
        dq = np.quantile(D[:, j], qs)
        maps.append((dq, sq))
    for r in dst['records']:
        Z = np.array(r['Z'], dtype=np.float32, copy=True)
        for j, c in enumerate(cols):
            dq, sq = maps[j]
            Z[:, c] = np.interp(Z[:, c], dq, sq)
        r['Z'] = Z
    print('qnorm: aligned %d distance columns of %d target images to the source' %
          (len(maps), len(dst['records'])), flush=True)
