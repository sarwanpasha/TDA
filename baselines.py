"""Non-topological baselines on the same images, splits and targets.

python baselines.py <cohort> <target> <seed> <model>
model in {deepsets, gnn, ripley}
"""
import os, sys, json, pickle, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
from train_learned import (image_targets, comp_features, rank_norm, load_cohort,
                           GRAPHS, RES, T)

EPOCHS = 30
LR = 1e-3
RADII = (10.0, 20.0, 40.0, 80.0)


class DeepSets(nn.Module):
    def __init__(self, d_in, d_comp, hidden=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(2 * hidden + d_comp, 64), nn.ReLU(),
                                 nn.Linear(64, 1))

    def forward(self, Z, edges, comp):
        h = self.phi(Z)
        return self.rho(torch.cat([h.mean(0), h.max(0).values, comp]))


class SAGE(nn.Module):
    """Two-layer GraphSAGE-style network on the Delaunay cell graph."""

    def __init__(self, d_in, d_comp, hidden=64):
        super().__init__()
        self.l1 = nn.Linear(2 * d_in, hidden)
        self.l2 = nn.Linear(2 * hidden, hidden)
        self.rho = nn.Sequential(nn.Linear(2 * hidden + d_comp, 64), nn.ReLU(),
                                 nn.Linear(64, 1))

    @staticmethod
    def agg(h, edges):
        if len(edges) == 0:
            return torch.zeros_like(h)
        src = torch.cat([edges[:, 0], edges[:, 1]])
        dst = torch.cat([edges[:, 1], edges[:, 0]])
        out = torch.zeros_like(h).index_add_(0, dst, h[src])
        cnt = torch.zeros(len(h), device=h.device).index_add_(
            0, dst, torch.ones(len(dst), device=h.device)).clamp(min=1)
        return out / cnt.unsqueeze(1)

    def forward(self, Z, edges, comp):
        h = torch.relu(self.l1(torch.cat([Z, self.agg(Z, edges)], 1)))
        h = torch.relu(self.l2(torch.cat([h, self.agg(h, edges)], 1)))
        return self.rho(torch.cat([h.mean(0), h.max(0).values, comp]))


import frac, featmask  # [tp]
class GIN(nn.Module):  # [tp]
    """Two GIN layers (sum aggregation, learned epsilon) on the Delaunay cell graph."""

    def __init__(self, d_in, d_comp, hidden=64):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(2))
        self.m1 = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.m2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.rho = nn.Sequential(nn.Linear(2 * hidden + d_comp, 64), nn.ReLU(),
                                 nn.Linear(64, 1))

    @staticmethod
    def sagg(h, edges):
        if len(edges) == 0:
            return torch.zeros_like(h)
        src = torch.cat([edges[:, 0], edges[:, 1]])
        dst = torch.cat([edges[:, 1], edges[:, 0]])
        return torch.zeros_like(h).index_add_(0, dst, h[src])

    def forward(self, Z, edges, comp):
        h = torch.relu(self.m1((1 + self.eps[0]) * Z + self.sagg(Z, edges)))
        h = torch.relu(self.m2((1 + self.eps[1]) * h + self.sagg(h, edges)))
        return self.rho(torch.cat([h.mean(0), h.max(0).values, comp]))


class GAT(nn.Module):
    """Two single-head graph attention layers on the Delaunay cell graph."""

    def __init__(self, d_in, d_comp, hidden=64):
        super().__init__()
        self.w1, self.w2 = nn.Linear(d_in, hidden), nn.Linear(hidden, hidden)
        self.a1, self.a2 = nn.Linear(2 * hidden, 1), nn.Linear(2 * hidden, 1)
        self.rho = nn.Sequential(nn.Linear(2 * hidden + d_comp, 64), nn.ReLU(),
                                 nn.Linear(64, 1))

    @staticmethod
    def attend(h, edges, a):
        n = len(h)
        loop = torch.arange(n, device=h.device)
        if len(edges):
            src = torch.cat([edges[:, 0], edges[:, 1], loop])
            dst = torch.cat([edges[:, 1], edges[:, 0], loop])
        else:
            src = dst = loop
        e = torch.nn.functional.leaky_relu(a(torch.cat([h[dst], h[src]], 1)).squeeze(1), 0.2)
        e = torch.exp((e - e.max()).clamp(min=-30))
        den = torch.zeros(n, device=h.device).index_add_(0, dst, e).clamp(min=1e-12)
        out = torch.zeros_like(h).index_add_(0, dst, (e / den[dst]).unsqueeze(1) * h[src])
        return out

    def forward(self, Z, edges, comp):
        h = torch.nn.functional.elu(self.attend(self.w1(Z), edges, self.a1))
        h = torch.nn.functional.elu(self.attend(self.w2(h), edges, self.a2))
        return self.rho(torch.cat([h.mean(0), h.max(0).values, comp]))
def ripley_features(recs, P):
    """Cross-type Ripley K at several radii, for the eight most common phenotypes."""
    top = min(P, 8)
    F = np.zeros((len(recs), top * top * len(RADII)), np.float32)
    for i, r in enumerate(recs):
        X, ph = r['X'].astype(float), r['pheno'].astype(int)
        area = max(float(np.ptp(X[:, 0]) * np.ptp(X[:, 1])), 1.0)
        trees = {p: cKDTree(X[ph == p]) for p in range(top) if (ph == p).sum() > 0}
        v, k = np.zeros(top * top * len(RADII)), 0
        for a in range(top):
            Xa = X[ph == a]
            for b in range(top):
                tb = trees.get(b)
                for rad in RADII:
                    if tb is None or len(Xa) == 0:
                        v[k] = 0.0
                    else:
                        cnt = np.array(tb.query_ball_point(Xa, rad, return_length=True))
                        lam = len(tb.data) / area
                        v[k] = cnt.mean() / max(lam * np.pi * rad ** 2, 1e-9)
                    k += 1
        F[i] = v
    return F


def fit_nn(model, recs, comp, y, tr, device):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        for j, i in enumerate(np.random.permutation(tr)):
            logit = model(recs[i]['Zt'], recs[i]['Et'], comp[i])
            (lossf(logit.squeeze(), torch.tensor(float(y[i]), device=device)) / 8).backward()
            if (j + 1) % 8 == 0:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
    return model


def train_nn(model_cls, recs, comp, y, tr, te, d_in, d_comp, device):
    model = fit_nn(model_cls(d_in, d_comp).to(device), recs, comp, y, tr, device)
    model.eval()
    out = np.zeros(len(te))
    with torch.no_grad():
        for k, i in enumerate(te):
            out[k] = float(torch.sigmoid(model(recs[i]['Zt'], recs[i]['Et'], comp[i])))
    return rank_norm(out)


def main():
    cohort, target, seed, which = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    torch.manual_seed(seed); np.random.seed(seed)
    device = 'cpu'
    meta, recs = load_cohort(cohort)
    P = len(meta['vocab'])
    recs = featmask.apply(recs, P)  # [tp]
    y = image_targets(cohort, recs, meta['vocab'])[target]
    y = np.asarray(y, dtype=float); keep_ = ~np.isnan(y)  # [tp]
    recs = [r for r, k in zip(recs, keep_) if k]; y = y[keep_].astype(int)
    comp_np = comp_features(recs, P)
    patients = np.array([str(r['patient']) for r in recs])
    print(f'{cohort} {target} model={which} seed={seed} images={len(recs)} '
          f'positives={int(y.sum())}', flush=True)

    for r in recs:
        r['Zt'] = torch.tensor(r['Z'], dtype=torch.float32, device=device)
        r['Et'] = torch.tensor(r['edges'].astype(np.int64), device=device)
    comp = [torch.tensor(c, dtype=torch.float32, device=device) for c in comp_np]

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(recs))
    fold_aucs = []
    t0 = time.time()
    if which == 'ripley':
        F = np.hstack([ripley_features(recs, P), comp_np])
        print('ripley features', F.shape, '%.0fs' % (time.time() - t0), flush=True)
    for f, (tr, te) in enumerate(cv.split(comp_np, y, patients)):
        tr = frac.cut(tr, patients, y, seed, f)  # [tp]
        if which == 'ripley':
            sc = StandardScaler().fit(F[tr])
            clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                         max_features='sqrt', random_state=seed, n_jobs=-1)
            clf.fit(sc.transform(F[tr]), y[tr])
            oof[te] = rank_norm(clf.predict_proba(sc.transform(F[te]))[:, 1])
        else:
            cls = {'deepsets': DeepSets, 'gnn': SAGE, 'gin': GIN, 'gat': GAT}[which]  # [tp]
            oof[te] = train_nn(cls, recs, comp, y, tr, te, meta['d_in'], P, device)
        fold_aucs.append(roc_auc_score(y[te], oof[te]))
        print(f'fold {f} auc={fold_aucs[-1]:.4f} ({time.time()-t0:.0f}s)', flush=True)

    auc_fold = float(np.mean(fold_aucs))
    auc_fold_sd = float(np.std(fold_aucs))
    auc_img = roc_auc_score(y, oof)
    g = pd.DataFrame(dict(patient=patients, y=y, p=oof)).groupby('patient').agg(
        y=('y', 'max'), p=('p', 'mean'))
    auc_pat = roc_auc_score(g['y'], g['p'])
    print(f'FOLD-MEAN AUC {auc_fold:.4f} +- {auc_fold_sd:.4f} | '
          f'IMAGE AUC {auc_img:.4f} | PATIENT AUC {auc_pat:.4f}', flush=True)
    os.makedirs(RES, exist_ok=True)
    tag = f'{cohort}_{target.replace(" ", "_")}_{which}{os.environ.get("TDA_TAG", "")}_s{seed}'  # [tp]
    json.dump(dict(cohort=cohort, target=target, variant=which + os.environ.get('TDA_TAG', ''), seed=seed,  # [tp]
                   auc_fold_mean=auc_fold, auc_fold_sd=auc_fold_sd,
                   fold_aucs=[float(a) for a in fold_aucs],
                   auc_image=auc_img, auc_patient=auc_pat),
              open(os.path.join(RES, 'baseline_' + tag + '.json'), 'w'), indent=2)
    np.save(os.path.join(RES, 'oof_' + tag + '.npy'), oof)
    print('BASELINE_DONE', tag, flush=True)


if __name__ == '__main__':
    main()
