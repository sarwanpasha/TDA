"""Build the Keren et al. (2018) MIBI-TOF triple-negative breast cancer cohort in the record
format of the other cohorts.

41 patients, one 500 x 500 um field each, imaged by multiplexed ion beam imaging. Cell
positions are the centroids of the published segmentation masks; phenotypes are the
published immune subtypes together with the tumor, endothelial, mesenchymal and
keratin-positive tumor groups. The endpoint is the published tumor-immune architecture
class, compartmentalized against mixed, with cold tumors carried as a second endpoint
against the rest.

python keren_prep.py            (expects TNBC_shareCellData.zip unpacked under $D/keren)
"""
import os, glob, pickle, re
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
import difftda as dt

D = os.environ.get('TDA_DATA', 'data') + '/keren'
OUT = os.environ.get('TDA_DATA', 'data') + '/graphs/keren_graphs.pkl'
FOV_UM, FOV_PX = 500.0, 2048.0
T, CAP = 100.0, 800
GROUP = {1: 'Unidentified', 2: 'Immune', 3: 'Endothelial', 4: 'Mesenchymal-like',
         5: 'Tumor', 6: 'Keratin-positive tumor'}
IMMUNE = {1: 'Treg', 2: 'CD4 T', 3: 'CD8 T', 4: 'CD3 T', 5: 'NK', 6: 'B', 7: 'Neutrophil',
          8: 'Macrophage', 9: 'DC', 10: 'DC/Mono', 11: 'Mono/Neu', 12: 'Other immune'}
WANT = ['Pan-Keratin', 'CD45', 'CD3', 'CD20', 'CD68', 'CD31', 'Vimentin', 'Ki67']


def find_file(pattern):
    hits = glob.glob(os.path.join(D, '**', pattern), recursive=True)
    if not hits:
        raise SystemExit('missing ' + pattern)
    return hits[0]


def read_tiff(path):
    try:
        import tifffile
        return tifffile.imread(path)
    except ImportError:
        from PIL import Image
        return np.array(Image.open(path))


def main():
    cells = pd.read_csv(find_file('cellData.csv'))
    cls = pd.read_csv(find_file('patient_class.csv'), header=None)
    if isinstance(cls.iloc[0, 0], str):
        cls = pd.read_csv(find_file('patient_class.csv'))
    cls.columns = ['SampleID', 'cls'][:cls.shape[1]] if cls.shape[1] == 2 else list(cls.columns)
    cls = cls.iloc[:, :2]; cls.columns = ['SampleID', 'cls']
    klass = dict(zip(cls.SampleID.astype(int), cls.cls.astype(int)))
    print('cells', cells.shape, 'patients with class', len(klass), flush=True)
    cols = list(cells.columns)
    marker_cols = []
    for w in WANT:
        c = next((c for c in cols if c.lower() == w.lower()), None) or \
            next((c for c in cols if w.lower().replace('-', '') in c.lower().replace('-', '')), None)
        if c is not None:
            marker_cols.append(c)
    print('markers', marker_cols, flush=True)
    grp = cells['Group'].astype(int)
    img = cells['immuneGroup'].astype(int)
    pheno = np.where(grp == 2, img.map(IMMUNE), grp.map(GROUP))
    cells['phenotype'] = pheno
    cells = cells[cells.phenotype != 'Unidentified'].copy()
    vocab = list(cells.phenotype.value_counts().index)
    code = {v: i for i, v in enumerate(vocab)}
    P = len(vocab)
    rng = np.random.default_rng(0)
    recs = []
    for sid, g in cells.groupby('SampleID'):
        sid = int(sid)
        if sid not in klass:
            continue
        mpath = glob.glob(os.path.join(D, '**', 'p%d_labeledcellData.tiff' % sid), recursive=True)
        if not mpath:
            print('no mask for', sid); continue
        M = read_tiff(mpath[0]).astype(np.int64)
        ids = g['cellLabelInImage'].to_numpy(int)
        present = np.isin(ids, np.unique(M))
        g, ids = g[present], ids[present]
        cy, cx = zip(*ndimage.center_of_mass(np.ones_like(M), M, ids)) if len(ids) else ([], [])
        X = np.column_stack([np.asarray(cx, float), np.asarray(cy, float)]) * (FOV_UM / FOV_PX)
        ph = g.phenotype.map(code).to_numpy(np.int64)
        mk = g[marker_cols].to_numpy(float)
        if len(X) > CAP:
            sel = rng.choice(len(X), CAP, replace=False)
            Xs, phs, mks = X[sel], ph[sel], mk[sel]
        else:
            Xs, phs, mks = X, ph, mk
        n = len(Xs)
        prof = np.full((n, P), T, np.float32)
        for p in range(P):
            A = X[ph == p]
            if len(A):
                prof[:, p] = np.minimum(cKDTree(A).query(Xs, k=1)[0], T)
        oh = np.zeros((n, P), np.float32); oh[np.arange(n), phs] = 1.0
        rk = np.argsort(np.argsort(mks, axis=0), axis=0) / max(n - 1, 1)
        Z = np.hstack([oh, prof / T, rk.astype(np.float32)]).astype(np.float32)
        cx_ = dt.build_complex(Xs, T)
        c = klass[sid]
        ys = {'Compartmentalized vs mixed': (1.0 if c == 1 else (0.0 if c == 0 else np.nan)),
              'Cold vs rest': 1.0 if c == 2 else 0.0}
        recs.append(dict(image=sid, patient='P%02d' % sid, ys=ys, X=Xs.astype(np.float32), Z=Z,
                         pheno=phs.astype(np.int16), edges=cx_['edges'].astype(np.int32),
                         e_len=cx_['e_len'].astype(np.float32), tris=cx_['tris'].astype(np.int32),
                         t_diam=cx_['t_diam'].astype(np.float32)))
        print(sid, 'cells', len(X), 'kept', n, 'class', c, flush=True)
    meta = dict(vocab=vocab, markers=marker_cols, T=T, n_cells=CAP, d_in=2 * P + len(marker_cols))
    pickle.dump(dict(meta=meta, records=recs), open(OUT, 'wb'), protocol=4)
    y1 = np.array([r['ys']['Compartmentalized vs mixed'] for r in recs])
    print('images', len(recs), 'phenotypes', P, 'd_in', meta['d_in'],
          'compartmentalized', int(np.nansum(y1)), 'mixed', int(np.sum(y1 == 0)),
          'cold', int(sum(r['ys']['Cold vs rest'] for r in recs)), flush=True)
    print('KEREN_PREP_DONE', flush=True)


if __name__ == '__main__':
    main()
