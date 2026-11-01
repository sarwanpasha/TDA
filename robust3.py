"""robust2.py with one change: the training features are recomputed from the subsampled
cells with the same code that the perturbation pipeline uses, so clean and perturbed images
go through an identical feature computation and the clean score is directly comparable.

python robust3.py <cohort> <target> <seed>

Write the results to a separate TDA_RES so they do not overwrite those of robust2.py.
"""
import numpy as np
import robust2
from robust2 import features

_load = robust2.load_cohort


def load_consistent(cohort):
    meta, recs = _load(cohort)
    P = len(meta['vocab'])
    for r in recs:
        r['Z'] = features(np.asarray(r['X'], float), r['pheno'].astype(int),
                          r['Z'][:, 2 * P:].astype(np.float32), P)
    return meta, recs


robust2.load_cohort = load_consistent

if __name__ == '__main__':
    robust2.main()
