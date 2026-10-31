"""Export what the learned filtration does on several held-out images per archetype.

python export_gallery.py <cohort> <target> <seed> [n_per_class]

Trains the learned filtration on folds 1-4 of the first split and exports, for n_per_class
held-out positive and negative images, the cell coordinates and phenotypes, the learned
vertex weight, the tuned hand-designed weight on the same fold, and both H_1 diagrams.
"""
import os, sys
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
import difftda as dt
from train_learned import (Model, load_cohort, image_targets, comp_features, run_fold,
                           choose_reference, RES, T, REF_PHENOTYPE)

cohort, target, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
K = int(sys.argv[4]) if len(sys.argv) > 4 else 3
torch.manual_seed(seed); np.random.seed(seed)
meta, recs = load_cohort(cohort)
P = len(meta['vocab'])
y = np.asarray(image_targets(cohort, recs, meta['vocab'])[target], dtype=float)
keep = ~np.isnan(y)
if not keep.all():
    recs = [r for r, k in zip(recs, keep) if k]; y = y[keep]
y = y.astype(int)
comp_np = comp_features(recs, P)
patients = np.array([str(r['patient']) for r in recs])
for r in recs:
    r['Zt'] = torch.tensor(r['Z'], dtype=torch.float32)
    r['cx'] = dict(n=len(r['X']), edges=r['edges'].astype(int), tris=r['tris'].astype(int),
                   e_len=r['e_len'].astype(float), t_diam=r['t_diam'].astype(float))
comp = [torch.tensor(c, dtype=torch.float32) for c in comp_np]
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
tr, te = next(iter(cv.split(comp_np, y, patients)))
j = choose_reference(recs, tr, y, P)
ref = dict(onehot=j, dist=P + j)
model = Model(meta['d_in'], P, 'learned_nc')
run_fold(model, tr, te, recs, comp, y, ref, 'cpu', log=None)
model.eval()
rng = np.random.default_rng(seed)
pos = rng.permutation([i for i in te if y[i] == 1])[:K]
neg = rng.permutation([i for i in te if y[i] == 0])[:K]
out = {}
for i in list(pos) + list(neg):
    r = recs[i]
    with torch.no_grad():
        w_learned = model.filt(r['Zt']).numpy()
    d = r['Z'][:, ref['dist']] * T
    is_ref = r['Z'][:, ref['onehot']] > 0.5
    w_fixed = np.where(is_ref, 2 * T, d)
    dg = {}
    for name, w in (('learned', w_learned), ('fixed', w_fixed)):
        with torch.no_grad():
            p0, p1 = dt.persistence(r['cx'], torch.tensor(w, dtype=torch.float64),
                                    cap=T, max_pairs=256)
        dg[name] = np.array([q for q in p1.numpy() if q[1] > q[0]]).reshape(-1, 2)
    k = 'img%d' % int(r['image'])
    out[k + '_label'] = int(y[i]); out[k + '_X'] = r['X'].astype(np.float32)
    out[k + '_pheno'] = r['pheno'].astype(np.int16)
    out[k + '_w_learned'] = w_learned.astype(np.float32)
    out[k + '_w_fixed'] = w_fixed.astype(np.float32)
    out[k + '_dg_learned'] = dg['learned'].astype(np.float32)
    out[k + '_dg_fixed'] = dg['fixed'].astype(np.float32)
tag = '%s_%s_s%d' % (cohort, target.replace(' ', '_'), seed)
np.savez_compressed(os.path.join(RES, 'gallery_%s.npz' % tag),
                    vocab=np.array([str(v) for v in meta['vocab']]),
                    ref_name=str(meta['vocab'][j]), T=T, **out)
print('exported', len(pos), 'positive and', len(neg), 'negative images', flush=True)
print('EXPORT_DONE', tag, flush=True)
