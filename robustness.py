"""Two experiments that cash in the stability theorem.

  bound : empirical check that d_B(PD(w), PD(w')) <= ||w - w'||_inf, and that a
          Hausdorff perturbation of the cells moves the diagram by no more than
          the Lipschitz constant of the weight function times that perturbation.
  auc   : downstream AUC under segmentation noise (position jitter and cell dropout)
          for the learned filtration, the fixed cross-phenotype filtration, a GNN,
          DeepSets and composition -- models are trained once on clean data.

python robustness.py <cohort> <target> <seed> {bound|auc}
"""
import os, sys, json, pickle, time
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from gudhi import bottleneck_distance
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import difftda as dt
from train_learned import (Model, image_targets, comp_features, run_fold,
                           GRAPHS, RES, T, REF_PHENOTYPE)
from baselines import DeepSets, SAGE, fit_nn

JITTER = (0.0, 5.0, 15.0, 30.0)
DROPOUT = (0.0, 0.10, 0.30, 0.50)


def diagram(cx, w):
    with torch.no_grad():
        p0, p1 = dt.persistence(cx, torch.tensor(w, dtype=torch.float64),
                                cap=T, max_pairs=256)
    keep = lambda P: np.array([r for r in P.numpy() if r[1] > r[0]]).reshape(-1, 2)
    return keep(p0), keep(p1)


def run_bound(recs, ref, n_images=40, n_trials=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in recs[:n_images]:
        cx = dict(n=len(r['X']), edges=r['edges'].astype(int), tris=r['tris'].astype(int),
                  e_len=r['e_len'].astype(float), t_diam=r['t_diam'].astype(float))
        w0 = np.minimum(r['Z'][:, ref['dist']].astype(float) * T, T)
        d0 = diagram(cx, w0)
        for _ in range(n_trials):
            eps = float(rng.uniform(0.5, 20.0))
            w1 = np.clip(w0 + rng.uniform(-eps, eps, len(w0)), 0, T)
            d1 = diagram(cx, w1)
            linf = float(np.abs(w0 - w1).max())
            for dim in (0, 1):
                db = float(bottleneck_distance(d0[dim], d1[dim]))
                rows.append(dict(image=str(r['image']), dim=dim, linf=linf, bottleneck=db,
                                 ratio=db / max(linf, 1e-9)))
    df = pd.DataFrame(rows)
    print(df.groupby('dim')[['linf', 'bottleneck', 'ratio']].describe().to_string(),
          flush=True)
    viol = int((df['bottleneck'] > df['linf'] + 1e-6).sum())
    print('violations of d_B <= linf norm : %d / %d' % (viol, len(df)), flush=True)
    df.to_csv(os.path.join(RES, 'stability_bound.csv'), index=False)
    return df


def perturb(rec, jitter, drop, rng, P):
    """Simulate segmentation error: jitter positions, drop cells, and recompute the
    per-cell features that depend on geometry.

    The distance-profile block of Z must be rebuilt from the perturbed cloud -- if it
    is carried over unchanged, models that read only Z are perturbation-invariant by
    construction and the comparison is meaningless.
    """
    X = rec['X'].astype(float)
    Z = rec['Z'].copy()
    keep = np.ones(len(X), bool)
    if drop > 0:
        keep = rng.random(len(X)) >= drop
        if keep.sum() < 8:
            keep = np.ones(len(X), bool)
    X = X[keep] + (rng.normal(0, jitter, (int(keep.sum()), 2)) if jitter > 0 else 0)
    Z = Z[keep]
    ph = rec['pheno'][keep].astype(int)
    D = np.full((len(X), P), T, np.float32)
    for q in range(P):
        Aq = X[ph == q]
        if len(Aq):
            D[:, q] = np.minimum(cKDTree(Aq).query(X, k=1)[0], T)
    Z[:, P:2 * P] = D / T
    cxn = dt.build_complex(X, T)
    return dict(X=X.astype(np.float32), Z=Z, pheno=ph,
                Zt=torch.tensor(Z, dtype=torch.float32),
                Et=torch.tensor(cxn['edges'].astype(np.int64)),
                cx=dict(n=len(X), edges=cxn['edges'].astype(int),
                        tris=cxn['tris'].astype(int), e_len=cxn['e_len'].astype(float),
                        t_diam=cxn['t_diam'].astype(float)))


def main():
    cohort, target, seed, mode = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    torch.manual_seed(seed)
    np.random.seed(seed)
    blob = pickle.load(open(os.path.join(GRAPHS, cohort + '_graphs.pkl'), 'rb'))
    meta, recs = blob['meta'], blob['records']
    P = len(meta['vocab'])
    ref_idx = next((i for i, v in enumerate(meta['vocab'])
                    if REF_PHENOTYPE in str(v).lower()), 0)
    ref = dict(onehot=ref_idx, dist=P + ref_idx)
    os.makedirs(RES, exist_ok=True)

    if mode == 'bound':
        run_bound(recs, ref, seed=seed)
        print('ROBUSTNESS_DONE bound', flush=True)
        return

    y = image_targets(cohort, recs, meta['vocab'])[target]
    comp_np = comp_features(recs, P)
    patients = np.array([str(r['patient']) for r in recs])
    for r in recs:
        r['Zt'] = torch.tensor(r['Z'], dtype=torch.float32)
        r['Et'] = torch.tensor(r['edges'].astype(np.int64))
        r['cx'] = dict(n=len(r['X']), edges=r['edges'].astype(int),
                       tris=r['tris'].astype(int), e_len=r['e_len'].astype(float),
                       t_diam=r['t_diam'].astype(float))
    comp = [torch.tensor(c, dtype=torch.float32) for c in comp_np]

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    levels = [('jitter', j) for j in JITTER] + [('drop', d) for d in DROPOUT[1:]]
    preds = {(m, k, v): np.zeros(len(recs)) for m in
             ('learned', 'fixed_cross', 'gnn', 'deepsets') for k, v in levels}
    rng = np.random.default_rng(seed)
    t0 = time.time()

    for f, (tr, te) in enumerate(cv.split(comp_np, y, patients)):
        models = {}
        for variant in ('learned', 'fixed_cross'):
            m = Model(meta['d_in'], P, variant)
            run_fold(m, tr, te, recs, comp, y, ref, 'cpu')
            models[variant] = m
        for name, cls in (('gnn', SAGE), ('deepsets', DeepSets)):
            models[name] = fit_nn(cls(meta['d_in'], P), recs, comp, y, tr, 'cpu')
        print('fold %d trained (%.0fs)' % (f, time.time() - t0), flush=True)
        for k, v in levels:
            pert = {i: perturb(recs[i], v if k == 'jitter' else 0.0,
                               v if k == 'drop' else 0.0, rng, P) for i in te}
            for name, m in models.items():
                m.eval()
                with torch.no_grad():
                    for i in te:
                        r = pert[i]
                        if name in ('learned', 'fixed_cross'):
                            out = m(r, comp[i], ref)
                        else:
                            out = m(r['Zt'], r['Et'], comp[i])
                        preds[(name, k, v)][i] = float(torch.sigmoid(out))
        print('fold %d perturbations done (%.0fs)' % (f, time.time() - t0), flush=True)

    rows = []
    for (name, k, v), p in preds.items():
        rows.append(dict(model=name, kind=k, level=v, auc=roc_auc_score(y, p)))
    df = pd.DataFrame(rows).sort_values(['model', 'kind', 'level'])
    print(df.to_string(index=False), flush=True)
    df.to_csv(os.path.join(RES, 'robustness_%s_s%d.csv' % (cohort, seed)), index=False)
    print('ROBUSTNESS_DONE auc', flush=True)


if __name__ == '__main__':
    main()
