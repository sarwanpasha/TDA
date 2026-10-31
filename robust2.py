"""Downstream robustness to segmentation noise, measured where it matters: in predictions.

python robust2.py <cohort> <target> <seed>

Every model is trained once per fold on clean data. Its held-out images are then perturbed
and re-scored: positions jittered by Gaussian noise of 1 to 20 um, a fraction of cells
dropped, or a fraction of cells merged with their nearest neighbour. The per-cell features
that depend on geometry (distance profile) and the complex are rebuilt from the perturbed
cloud, as is the composition vector. We report, per model and level, the AUC of the
seed-averaged out-of-fold rank vector, the fraction of test images whose predicted label
flips relative to the clean prediction, the mean absolute change of the predicted
probability, and the Spearman correlation of the perturbed and clean scores.

Models: the learned filtration, the tuned hand design, the learned sublevel-set filter and
plain Rips, all topology-only; DeepSets and GraphSAGE reading the cell features and the
composition vector.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import difftda as dt
from train_learned import (Model, load_cohort, image_targets, comp_features, run_fold,
                           choose_reference, rank_norm, RES, T, REF_PHENOTYPE)
from train_extra import Extra
from baselines import DeepSets, SAGE, fit_nn

JITTER = (1.0, 2.0, 5.0, 10.0, 20.0)
DROP = (0.01, 0.02, 0.05, 0.10)
MERGE = (0.01, 0.02, 0.05, 0.10)
MERGE_RADIUS = 15.0
FILTS = ('learned_nc', 'fixed_tuned_nc', 'sublevel_nc', 'zero_nc')
NETS = ('deepsets', 'gnn')


def features(X, ph, rk, P):
    D = np.full((len(X), P), T, np.float32)
    for q in range(P):
        Aq = X[ph == q]
        if len(Aq):
            D[:, q] = np.minimum(cKDTree(Aq).query(X, k=1)[0], T)
    oh = np.zeros((len(X), P), np.float32)
    oh[np.arange(len(X)), ph] = 1.0
    return np.hstack([oh, D / T, rk]).astype(np.float32)


def perturb(rec, kind, level, rng, P):
    X = rec['X'].astype(float).copy()
    ph = rec['pheno'].astype(int).copy()
    rk = rec['Z'][:, 2 * P:].astype(np.float32).copy()
    n = len(X)
    if kind == 'jitter':
        X = X + rng.normal(0, level, X.shape)
    elif kind == 'drop':
        keep = rng.random(n) >= level
        if keep.sum() < 8:
            keep[:] = True
        X, ph, rk = X[keep], ph[keep], rk[keep]
    elif kind == 'merge':
        k = int(round(level * n))
        if k > 0 and n > 8:
            d, j = cKDTree(X).query(X, k=2)
            cand = np.where(d[:, 1] <= MERGE_RADIUS)[0]
            rng.shuffle(cand)
            dead = np.zeros(n, bool)
            done = 0
            for i in cand:
                if done >= k:
                    break
                jj = j[i, 1]
                if dead[i] or dead[jj]:
                    continue
                X[i] = 0.5 * (X[i] + X[jj])
                if rng.random() < 0.5:
                    ph[i], rk[i] = ph[jj], rk[jj]
                dead[jj] = True
                done += 1
            keep = ~dead
            X, ph, rk = X[keep], ph[keep], rk[keep]
    Z = features(X, ph, rk, P)
    cxn = dt.build_complex(X, T)
    e, t3 = cxn['edges'].astype(int), cxn['tris'].astype(int)
    cx = dict(n=len(X), edges=e, tris=t3, e_len=cxn['e_len'].astype(float),
              t_diam=cxn['t_diam'].astype(float))
    comp = np.bincount(ph, minlength=P).astype(np.float32)
    comp = comp / max(comp.sum(), 1)
    return dict(X=X.astype(np.float32), Z=Z, pheno=ph,
                Zt=torch.tensor(Z, dtype=torch.float32),
                Et=torch.tensor(e.astype(np.int64)), cx=cx,
                cx_sub=dict(n=len(X), edges=e, tris=t3, e_len=np.zeros(len(e), float),
                            t_diam=np.zeros(len(t3), float)),
                comp=torch.tensor(comp))


def score(name, m, r, comp, ref):
    with torch.no_grad():
        if name in NETS:
            return float(m(r['Zt'], r['Et'], comp))
        return float(m(r, comp, ref))


def main():
    cohort, target, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
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
    ref_idx = next((i for i, v in enumerate(meta['vocab'])
                    if REF_PHENOTYPE in str(v).lower()), 0)
    ref = dict(onehot=ref_idx, dist=P + ref_idx)
    for r in recs:
        r['Zt'] = torch.tensor(r['Z'], dtype=torch.float32)
        r['Et'] = torch.tensor(r['edges'].astype(np.int64))
        e, t3 = r['edges'].astype(int), r['tris'].astype(int)
        r['cx'] = dict(n=len(r['X']), edges=e, tris=t3, e_len=r['e_len'].astype(float),
                       t_diam=r['t_diam'].astype(float))
        r['cx_sub'] = dict(n=len(r['X']), edges=e, tris=t3, e_len=np.zeros(len(e), float),
                           t_diam=np.zeros(len(t3), float))
    comp = [torch.tensor(c, dtype=torch.float32) for c in comp_np]
    levels = [('clean', 0.0)] + [('jitter', j) for j in JITTER] + \
             [('drop', d) for d in DROP] + [('merge', m) for m in MERGE]
    names = FILTS + NETS
    logits = {(nm, k, v): np.full(len(recs), np.nan) for nm in names for k, v in levels}
    ranks = {(nm, k, v): np.zeros(len(recs)) for nm in names for k, v in levels}
    rng = np.random.default_rng(seed)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    t0 = time.time()
    for f, (tr, te) in enumerate(cv.split(comp_np, y, patients)):
        models, refs = {}, {}
        for v in FILTS:
            use_ref = ref
            if v.startswith('fixed_tuned'):
                j = choose_reference(recs, tr, y, P)
                use_ref = dict(onehot=j, dist=P + j)
            if v == 'sublevel_nc':
                m = Extra('sublevel', meta['d_in'], P, False, False)
            else:
                m = Model(meta['d_in'], P, v)
            run_fold(m, tr, te, recs, comp, y, use_ref, 'cpu')
            m.eval(); models[v] = m; refs[v] = use_ref
        for nm, cls in (('deepsets', DeepSets), ('gnn', SAGE)):
            models[nm] = fit_nn(cls(meta['d_in'], P), recs, comp, y, tr, 'cpu').eval()
        print('fold %d trained (%.0fs)' % (f, time.time() - t0), flush=True)
        for k, v in levels:
            out = {nm: np.zeros(len(te)) for nm in names}
            for q, i in enumerate(te):
                r = recs[i] if k == 'clean' else perturb(recs[i], k, v, rng, P)
                c = comp[i] if k == 'clean' else r['comp']
                for nm in names:
                    out[nm][q] = score(nm, models[nm], r, c, refs.get(nm, ref))
            for nm in names:
                logits[(nm, k, v)][te] = out[nm]
                ranks[(nm, k, v)][te] = rank_norm(out[nm])
        print('fold %d perturbed (%.0fs)' % (f, time.time() - t0), flush=True)

    rows = []
    for nm in names:
        base = logits[(nm, 'clean', 0.0)]
        for k, v in levels:
            lg = logits[(nm, k, v)]
            rows.append(dict(model=nm, kind=k, level=v,
                             auc=roc_auc_score(y, ranks[(nm, k, v)]),
                             flip=float(np.mean(np.sign(lg) != np.sign(base))),
                             dprob=float(np.mean(np.abs(1 / (1 + np.exp(-lg)) -
                                                        1 / (1 + np.exp(-base))))),
                             rho=float(spearmanr(lg, base).correlation)))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False), flush=True)
    tag = '%s_%s_s%d' % (cohort, target.replace(' ', '_'), seed)
    df.to_csv(os.path.join(RES, 'robust_%s.csv' % tag), index=False)
    np.savez_compressed(os.path.join(RES, 'robust_%s.npz' % tag), y=y,
                        **{'%s__%s__%g' % key: val for key, val in logits.items()})
    print('ROBUST_DONE', tag, flush=True)


if __name__ == '__main__':
    main()
