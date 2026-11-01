"""Controls on the same pipeline: a coordinate-only learned filtration, a gate without
persistence, a cross-validated reference and designed filtrations on the alpha complex.

python train_review.py <cohort> <target> <seed> <variant>

nishi_nc            coordinate-only learned vertex weight in the style of Nishikawa et al.
                    (2023): an unconstrained network reads an isometry-invariant description
                    of the cell neighbourhood (the sorted distances to the 16 nearest cells)
                    and no phenotype information; filtration (1) on the same complex.
gate_nc             the learned weight network without persistence: cells are soft-gated by
                    their weight and the representation is the gated mean of the one-hot and
                    distance blocks plus 3 weight statistics, read by the same head.
fixed_cv_nc         the cross-phenotype filtration with the reference phenotype chosen per
                    fold by an inner 2-fold cross-validation of the full model on the
                    training split, instead of the separation heuristic of fixed_tuned.
alpha_fixed_tuned_nc the tuned cross-phenotype weight on the weighted alpha complex.
alpha_dtm_nc        the distance-to-measure weight on the weighted alpha complex.

Outputs follow train_learned.py so collect.py aggregates them unchanged.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import difftda as dt
import dtalpha as dta
from train_learned import (Model, load_cohort, image_targets, comp_features, run_fold,
                           choose_reference, rank_norm, PersLayLite, PersLayLip, VEC, T,
                           MAX_PAIRS, RES, EPOCHS, LR, REF_PHENOTYPE)
from train_extra import dtm_weight

KNN = 16
CV_CANDIDATES = 8
CV_EPOCHS = 6


def knn_features(X, k=KNN):
    """Sorted distances to the k nearest cells, divided by T: invariant to rigid motions."""
    kk = min(k, len(X) - 1)
    d, _ = cKDTree(X).query(X, k=kk + 1)
    out = np.ones((len(X), k), np.float32)
    out[:, :kk] = np.minimum(d[:, 1:], T) / T
    return out


class Gate(nn.Module):
    """The weight network of the learned filtration, without persistence."""

    def __init__(self, d_in, P, hidden=64):
        super().__init__()
        self.P = P
        self.filt = dt.LipschitzMLP(d_in, hidden=hidden, depth=2, scale=2 * T, constrained=True)
        self.head = nn.Sequential(nn.Linear(2 * P + 3, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, rec, comp, ref=None):
        Z = rec['Zt']
        w = self.filt(Z)
        g = torch.sigmoid((T - w) / (0.05 * T))
        blocks = Z[:, :2 * self.P]
        gated = (g.unsqueeze(1) * blocks).sum(0) / (g.sum() + 1e-6)
        stats = torch.stack([g.mean(), (w / T).mean(), (w / T).std()])
        return self.head(torch.cat([gated, stats]))


class AlphaFixed(nn.Module):
    """A fixed vertex weight on the weighted alpha complex, with the usual head."""

    def __init__(self):
        super().__init__()
        V = PersLayLip if VEC == 'lip' else PersLayLite
        self.v0, self.v1 = V(), V()
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, rec, comp, ref=None):
        p0, p1 = dta.persistence(rec['Xt'], rec['wfix'], cap=T, max_pairs=MAX_PAIRS)
        return self.head(torch.cat([self.v0(p0), self.v1(p1)]))


def cross_weight(r, P, a):
    d = r['Z'][:, P + a].astype(np.float32) * T
    is_ref = r['Z'][:, a] > 0.5
    return torch.tensor(np.where(is_ref, 2 * T, d), dtype=torch.float32)


def choose_reference_cv(recs, tr, y, P, patients, comp, seed, d_in):
    """Reference phenotype chosen by inner cross-validation of the full fixed_cross model."""
    counts = np.zeros(P)
    for i in tr:
        counts += np.bincount(recs[i]['pheno'].astype(int), minlength=P)
    cands = list(np.argsort(-counts)[:CV_CANDIDATES])
    h = choose_reference(recs, tr, y, P)
    if h not in cands:
        cands.append(h)
    inner = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    best, best_auc = cands[0], -1.0
    for a in cands:
        ref = dict(onehot=int(a), dist=P + int(a))
        aucs = []
        for itr, ite in inner.split(np.zeros(len(tr)), y[tr], patients[tr]):
            itr, ite = tr[itr], tr[ite]
            if len(np.unique(y[ite])) < 2 or len(np.unique(y[itr])) < 2:
                continue
            m = Model(d_in, P, 'fixed_cross_nc')
            out = run_fold(m, itr, ite, recs, comp, y, ref, 'cpu', epochs=CV_EPOCHS)
            aucs.append(roc_auc_score(y[ite], out))
        score = float(np.mean(aucs)) if aucs else -1.0
        if score > best_auc:
            best, best_auc = int(a), score
    return best, best_auc


def main():
    cohort, target, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    variant = sys.argv[4]
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(1)
    meta, recs = load_cohort(cohort)
    P = len(meta['vocab'])
    y = np.asarray(image_targets(cohort, recs, meta['vocab'])[target], dtype=float)
    keep = ~np.isnan(y)
    if not keep.all():
        recs = [r for r, k in zip(recs, keep) if k]; y = y[keep]
    y = y.astype(int)
    comp_np = comp_features(recs, P)
    patients = np.array([str(r['patient']) for r in recs])
    ref_idx = next((i for i, v in enumerate(meta['vocab']) if REF_PHENOTYPE in str(v).lower()), 0)
    print(f'{cohort} {target} variant={variant} seed={seed} images={len(recs)} '
          f'positives={int(y.sum())} d_in={meta["d_in"]}', flush=True)
    for r in recs:
        r['Zt'] = torch.tensor(r['Z'], dtype=torch.float32)
        r['cx'] = dict(n=len(r['X']), edges=r['edges'].astype(int), tris=r['tris'].astype(int),
                       e_len=r['e_len'].astype(float), t_diam=r['t_diam'].astype(float))
        if variant.startswith('nishi'):
            r['Zt'] = torch.tensor(knn_features(np.asarray(r['X'], float)), dtype=torch.float32)
        if variant.startswith('alpha'):
            r['Xt'] = torch.tensor(np.asarray(r['X']), dtype=torch.float32)
            if variant.startswith('alpha_dtm'):
                r['wfix'] = torch.tensor(np.minimum(dtm_weight(np.asarray(r['X'], float)), T),
                                         dtype=torch.float32)
    comp = [torch.tensor(c, dtype=torch.float32) for c in comp_np]

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(recs)); fold_aucs, refs_used, cv_aucs = [], [], []
    t0 = time.time()
    for f, (tr, te) in enumerate(cv.split(comp_np, y, patients)):
        ref = dict(onehot=ref_idx, dist=P + ref_idx)
        if variant.startswith('nishi'):
            model = Model(KNN, P, 'learned_unc_nc')
        elif variant.startswith('gate'):
            model = Gate(meta['d_in'], P)
        elif variant.startswith('fixed_cv'):
            a, s = choose_reference_cv(recs, tr, y, P, patients, comp, seed, meta['d_in'])
            ref = dict(onehot=a, dist=P + a); refs_used.append(str(meta['vocab'][a])); cv_aucs.append(s)
            model = Model(meta['d_in'], P, 'fixed_cross_nc')
        elif variant.startswith('alpha_fixed_tuned'):
            a = choose_reference(recs, tr, y, P); refs_used.append(str(meta['vocab'][a]))
            for r in recs:
                r['wfix'] = cross_weight(r, P, a)
            model = AlphaFixed()
        elif variant.startswith('alpha_dtm'):
            model = AlphaFixed()
        else:
            raise SystemExit('unknown variant ' + variant)
        oof[te] = run_fold(model, tr, te, recs, comp, y, ref, 'cpu', log=(f == 0))
        fold_aucs.append(roc_auc_score(y[te], oof[te]))
        print(f'fold {f} auc={fold_aucs[-1]:.4f} ({time.time()-t0:.0f}s)'
              + (f' ref={refs_used[-1]}' if refs_used else ''), flush=True)

    auc_fold, auc_sd = float(np.mean(fold_aucs)), float(np.std(fold_aucs))
    auc_img = roc_auc_score(y, oof)
    print(f'FOLD-MEAN AUC {auc_fold:.4f} +- {auc_sd:.4f} | IMAGE AUC {auc_img:.4f}', flush=True)
    os.makedirs(RES, exist_ok=True)
    tag = f'{cohort}_{target.replace(" ", "_")}_{variant}{os.environ.get("TDA_TAG", "")}_s{seed}'
    json.dump(dict(cohort=cohort, target=target, variant=variant + os.environ.get('TDA_TAG', ''),
                   seed=seed, auc_fold_mean=auc_fold, auc_fold_sd=auc_sd,
                   fold_aucs=[float(a) for a in fold_aucs], refs_used=refs_used,
                   inner_cv_aucs=[float(a) for a in cv_aucs],
                   auc_image=auc_img, n_images=len(recs), n_pos=int(y.sum())),
              open(os.path.join(RES, 'learned_' + tag + '.json'), 'w'), indent=2)
    np.save(os.path.join(RES, 'oof_' + tag + '.npy'), oof)
    print('TRAIN_REVIEW_DONE', tag, flush=True)


if __name__ == '__main__':
    main()
