"""Build the CODEX colorectal cohort in the same record format as the two IMC cohorts.

Schurch et al. (2020): 140 tissue regions from the invasive front of 35 advanced
colorectal tumours, imaged by CODEX rather than imaging mass cytometry, with each patient
assigned by histopathology to one of two groups, Crohn's-like reaction or diffuse
inflammatory infiltration.  That grouping is read off H&E by pathologists and owes nothing
to the cell graph, so it plays the role the METABRIC endpoints play on Danenberg.
"""
import os, pickle, numpy as np, pandas as pd
from scipy.spatial import Delaunay

PX_UM = 0.37744          # 20x CODEX pixel size
T = 100.0
CAP = 800
D = os.environ.get('TDA_DATA', 'data') + ''
SRC = f'{D}/codex/CRC_clusters_neighborhoods_markers.csv'
OUT = f'{D}/graphs/codex_graphs.pkl'
DROP = {'dirt', 'undefined'}
MARKERS = ['CD45 - hematopoietic cells:Cyc_4_ch_2', 'Cytokeratin - epithelia:Cyc_10_ch_2',
           'CD3 - T cells:Cyc_16_ch_4', 'CD20 - B cells:Cyc_8_ch_3',
           'CD68 - macrophages:Cyc_18_ch_4', 'aSMA - smooth muscle:Cyc_11_ch_2',
           'CD31 - vasculature:Cyc_19_ch_3', 'Ki67 - proliferation:Cyc_5_ch_4']


def complex_of(X):
    tri = Delaunay(X).simplices
    e = np.unique(np.sort(np.concatenate(
        [tri[:, [0, 1]], tri[:, [0, 2]], tri[:, [1, 2]]]), axis=1), axis=0)
    el = np.linalg.norm(X[e[:, 0]] - X[e[:, 1]], axis=1)
    keep = el <= T
    e, el = e[keep], el[keep]
    ok = {tuple(r) for r in e}
    good = np.array([all(tuple(sorted(p)) in ok for p in
                         ((t[0], t[1]), (t[0], t[2]), (t[1], t[2]))) for t in tri])
    tri = tri[good] if len(tri) else tri
    td = (np.max([np.linalg.norm(X[tri[:, a]] - X[tri[:, b]], axis=1)
                  for a, b in ((0, 1), (0, 2), (1, 2))], axis=0)
          if len(tri) else np.zeros(0))
    return e, el, np.sort(tri, axis=1), td


def main():
    cols = ['X:X', 'Y:Y', 'ClusterName', 'groups', 'patients', 'File Name'] + MARKERS
    d = pd.read_csv(SRC, usecols=lambda c: c in cols)
    d = d[~d.ClusterName.isin(DROP)].copy()
    vocab = sorted(d.ClusterName.unique())
    code = {v: i for i, v in enumerate(vocab)}
    P = len(vocab)
    print('phenotypes', P, 'cells', len(d), 'images', d['File Name'].nunique())

    rng = np.random.default_rng(0)
    recs = []
    for img, g in d.groupby('File Name'):
        if len(g) < 50:
            continue
        if len(g) > CAP:
            g = g.iloc[rng.choice(len(g), CAP, replace=False)]
        X = np.column_stack([g['X:X'].to_numpy(float), g['Y:Y'].to_numpy(float)]) * PX_UM
        ph = g.ClusterName.map(code).to_numpy(np.int64)
        n = len(X)
        Dm = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        np.fill_diagonal(Dm, np.inf)
        prof = np.full((n, P), T)
        for p in range(P):
            m = ph == p
            if m.any():
                prof[:, p] = np.minimum(Dm[:, m].min(1), T)
        oh = np.zeros((n, P)); oh[np.arange(n), ph] = 1.0
        mk = g[MARKERS].to_numpy(float)
        rk = np.argsort(np.argsort(mk, axis=0), axis=0) / max(n - 1, 1)
        Z = np.concatenate([oh, prof / T, rk], axis=1).astype(np.float32)
        e, el, tri, td = complex_of(X)
        recs.append(dict(image=img, patient='P%02d' % int(g.patients.iloc[0]),
                         y=int(g.groups.iloc[0] == 1), X=X.astype(np.float32), Z=Z,
                         pheno=ph, edges=e, e_len=el.astype(np.float32),
                         tris=tri, t_diam=td.astype(np.float32)))
    meta = dict(vocab=vocab, markers=MARKERS, T=T, n_cells=CAP, d_in=2 * P + len(MARKERS))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pickle.dump(dict(meta=meta, records=recs), open(OUT, 'wb'))
    ys = np.array([r['y'] for r in recs])
    print('images %d  positives %d  patients %d  d_in %d'
          % (len(recs), int(ys.sum()), len({r['patient'] for r in recs}), meta['d_in']))
    print('median edges %d  median triangles %d'
          % (np.median([len(r['edges']) for r in recs]),
             np.median([len(r['tris']) for r in recs])))


if __name__ == '__main__':
    main()
