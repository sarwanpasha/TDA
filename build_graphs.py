"""Precompute, per image, the sparse complex and the per-cell feature matrix used
by the learned filtration.

Cell features z_i = [ one-hot phenotype | distance profile to every phenotype / T
                      | rank-normalised markers ]

The distance profile is what makes the learned family a strict generalisation of the
cross-phenotype filtration: selecting one coordinate of it reproduces w_A exactly.

Usage: python build_graphs.py <cohort> [n_cells]
"""
import os, sys, json, pickle, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import difftda as dt

PROC = os.environ.get('TDA_DATA', 'data') + '/proc'
OUT = os.environ.get('TDA_GRAPHS_OUT', os.environ.get('TDA_DATA', 'data') + '/graphs')
T = float(os.environ.get('TDA_T', 100.0))
MARKERS_D = ['CD31-vWF', 'SMA', 'panCK', 'CD68', 'CD8', 'CD20', 'FSP1', 'Ki-67']
MARKERS_J = ['vWF', 'SMA', 'panCyto', 'CD68', 'CD3epsi', 'CD20', 'Fibrone', 'Ki67']


def pick_markers(names, want):
    lut = {n.lower(): i for i, n in enumerate(names)}
    idx, used = [], []
    for w in want:
        j = lut.get(w.lower())
        if j is None:
            key = w.lower().split()[0]
            j = next((v for k, v in lut.items() if key in k), None)
        if j is not None:
            idx.append(j)
            used.append(names[j])
    return idx, used


def load(cohort):
    if cohort == 'danenberg':
        d = os.path.join(PROC, 'danenberg')
        cells = pd.read_parquet(os.path.join(d, 'cells.parquet'))
        cells = cells.rename(columns={'ImageNumber': 'image', 'metabric_id': 'patient'})
        want = MARKERS_D
    else:
        d = os.path.join(PROC, 'jackson', cohort)
        cells = pd.read_parquet(os.path.join(d, 'cells.parquet'))
        cells = cells.rename(columns={'core': 'image', 'cell_type': 'phenotype'})
        meta = pd.read_parquet(os.path.join(PROC, 'jackson', cohort, 'patient_meta.parquet'))
        pid_col = 'PID' if 'PID' in meta.columns else meta.columns[0]
        m = dict(zip(meta['core'].astype(str), meta[pid_col].astype(str)))
        cells['patient'] = cells['image'].astype(str).map(m)
        want = MARKERS_J
    names = json.load(open(os.path.join(d, 'marker_names.json')))
    M = np.load(os.path.join(d, 'markers.npy'), mmap_mode='r')
    idx, used = pick_markers(names, want)
    print('markers:', used, flush=True)
    return cells, np.asarray(M[:, idx], np.float32), used


def main():
    cohort = sys.argv[1]
    n_cells = int(sys.argv[2]) if len(sys.argv) > 2 else 800
    os.makedirs(OUT, exist_ok=True)
    cells, mk, mnames = load(cohort)
    vocab = list(cells['phenotype'].value_counts().index)
    p2i = {p: i for i, p in enumerate(vocab)}
    P = len(vocab)
    print('phenotypes', P, 'images', cells['image'].nunique(), flush=True)

    xs, ys = cells['x'].to_numpy(float), cells['y'].to_numpy(float)
    phi = cells['phenotype'].map(p2i).to_numpy()
    pat = cells['patient'].to_numpy()

    recs, t0 = [], time.time()
    for k, (img, ix) in enumerate(cells.groupby('image').indices.items()):
        Xall = np.stack([xs[ix], ys[ix]], 1)
        pall = phi[ix]
        if len(Xall) < 8:
            continue
        rng = np.random.default_rng(abs(hash(str(img))) % (2 ** 31))
        sel = rng.choice(len(Xall), min(n_cells, len(Xall)), replace=False)
        X = Xall[sel]
        ph = pall[sel]
        D = np.full((len(X), P), T, np.float32)
        for p in range(P):
            Ap = Xall[pall == p]
            if len(Ap):
                D[:, p] = np.minimum(cKDTree(Ap).query(X, k=1)[0], T)
        onehot = np.zeros((len(X), P), np.float32)
        onehot[np.arange(len(X)), ph] = 1.0
        mm = mk[ix][sel]
        rk = np.argsort(np.argsort(mm, axis=0), axis=0) / max(len(X) - 1, 1)
        Z = np.hstack([onehot, D / T, rk.astype(np.float32)]).astype(np.float32)
        cx = dt.build_complex(X, T)
        recs.append(dict(image=img, patient=pat[ix][0], X=X.astype(np.float32), Z=Z,
                         pheno=ph.astype(np.int16), edges=cx['edges'].astype(np.int32),
                         e_len=cx['e_len'].astype(np.float32),
                         tris=cx['tris'].astype(np.int32),
                         t_diam=cx['t_diam'].astype(np.float32)))
        if (k + 1) % 100 == 0:
            print(k + 1, '%.0fs' % (time.time() - t0), flush=True)

    meta = dict(vocab=vocab, markers=mnames, T=T, n_cells=n_cells, d_in=2 * P + len(mnames))
    with open(os.path.join(OUT, cohort + '_graphs.pkl'), 'wb') as f:
        pickle.dump(dict(meta=meta, records=recs), f, protocol=4)
    print('images kept', len(recs), 'd_in', meta['d_in'],
          'mean edges', np.mean([len(r['edges']) for r in recs]), flush=True)
    print('BUILD_GRAPHS_DONE', cohort, '%.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
