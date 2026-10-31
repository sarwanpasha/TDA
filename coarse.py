"""A lineage vocabulary the two cohorts share, so a model fit on one can be applied to the
other. Every phenotype of each cohort is assigned to one of eight lineages. The coarse
one-hot is the indicator of the lineage; the coarse distance to a lineage is the minimum of
the distances to its member phenotypes, which is exactly the distance to the nearest cell of
that lineage. Marker ranks are dropped: the two antibody panels differ, and the point of the
transfer is that nothing cohort-specific crosses.
"""
import numpy as np

LINEAGES = ['Tumour', 'Fibroblast', 'Endothelial', 'T cell', 'B cell',
            'Myeloid', 'Granulocyte', 'Other']

GROUPS = {
    'danenberg': {
        'Tumour': [1, 3, 4, 5, 6, 11, 14, 15, 16, 18, 20, 21, 31],
        'Fibroblast': [0, 2, 13, 27],
        'Endothelial': [7],
        'T cell': [10, 17, 19, 25],
        'B cell': [12],
        'Myeloid': [9, 23, 28, 29],
        'Granulocyte': [8, 26],
        'Other': [22, 24, 30],
    },
    'basel': {
        'Tumour': [2, 4, 5, 7, 8, 9, 10, 12, 14, 15, 16, 17, 18, 21],
        'Fibroblast': [0, 3, 19, 23, 25],
        'Endothelial': [13],
        'T cell': [1, 20],
        'B cell': [22],
        'Myeloid': [11, 24],
        'Granulocyte': [],
        'Other': [6],
    },
}


def check(cohort, P):
    idx = sorted(i for g in GROUPS[cohort].values() for i in g)
    assert idx == list(range(P)), (cohort, len(idx), P)


def coarsen(blob, cohort):
    """Rewrite a graph blob onto the shared vocabulary. Z becomes [one-hot | distance]."""
    P = len(blob['meta']['vocab'])
    check(cohort, P)
    cols = [GROUPS[cohort][L] for L in LINEAGES]
    lut = np.zeros(P, dtype=int)
    for k, g in enumerate(cols):
        for i in g:
            lut[i] = k
    L = len(LINEAGES)
    for r in blob['records']:
        Z = r['Z']
        oh = np.stack([Z[:, g].sum(1) if g else np.zeros(len(Z)) for g in cols], 1)
        # distances live in columns P .. 2P-1, already divided by T and capped at 1
        dd = np.stack([Z[:, [P + i for i in g]].min(1) if g else np.ones(len(Z))
                       for g in cols], 1)
        r['Z'] = np.hstack([oh, dd]).astype(np.float32)
        r['pheno'] = lut[np.asarray(r['pheno'], dtype=int)]
    blob['meta'] = dict(vocab=list(LINEAGES), markers=0, T=blob['meta']['T'],
                        n_cells=blob['meta']['n_cells'], d_in=2 * L)
    return blob
