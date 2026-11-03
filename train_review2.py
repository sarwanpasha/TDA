"""Further controls and variants on the same pipeline.

python train_review2.py <cohort> <target> <seed> <variant>

gatetopo_nc          the learned filtration with a joint readout: the head reads the
                     vectorized diagrams of Section III together with the gated means and
                     weight statistics of the gate-only control (train_review.py).
nishi_ph_nc          an isometry-invariant learned filtration with phenotype input: an
                     unconstrained network reads the cell's own feature vector and the mean
                     feature vector of its 16 nearest cells, so it sees what the learned
                     weight sees plus a one-hop neighborhood; filtration (1) as usual.
fixed_tuned_keep_nc  the tuned cross-phenotype filtration with the reference cells admitted
                     at weight 0 instead of being pushed above the cap.
learned_meanvec_nc   the learned filtration with the vectorizer normalized by the number of
                     pairs instead of the fixed divisor K0.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import difftda as dt
from train_learned import (Model, load_cohort, image_targets, comp_features, run_fold,
                           choose_reference, rank_norm, PersLayLip, VEC, T, MAX_PAIRS, RES,
                           REF_PHENOTYPE)

KNN = 16


class GateTopo(Model):
    """Learned filtration whose head also reads the gated per-cell features."""

    def __init__(self, d_in, P, hidden=64):
        super().__init__(d_in, P, 'learned_nc', hidden=hidden)
        self.P = P
        self.head = nn.Sequential(nn.Linear(64 + 2 * P + 3, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, rec, comp, ref):
        Z = rec['Zt']
        w = self.filt(Z)
        p0, p1 = dt.persistence(rec['cx'], w, cap=T, max_pairs=MAX_PAIRS)
        g = torch.sigmoid((T - w) / (0.05 * T))
        gated = (g.unsqueeze(1) * Z[:, :2 * self.P]).sum(0) / (g.sum() + 1e-6)
        stats = torch.stack([g.mean(), (w / T).mean(), (w / T).std()])
        return self.head(torch.cat([self.v0(p0), self.v1(p1), gated, stats]))


class PersLayMean(nn.Module):
    """The vectorizer (3) with the sum divided by the number of pairs."""

    def __init__(self, out=32):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, out))

    def forward(self, Pd):
        life = (Pd[:, 1] - Pd[:, 0]).unsqueeze(1) / T
        nz = (Pd.abs().sum(1) > 0).float().sum().clamp(min=1.0)
        feats = self.f(torch.cat([Pd / T, life], 1))
        return (feats * torch.tanh(life)).sum(0) / nz


class KeepRef(Model):
    """Cross-phenotype weight with the reference cells admitted at weight 0."""

    def weights(self, Z, ref):
        d = Z[:, ref['dist']] * T
        is_ref = Z[:, ref['onehot']] > 0.5
        return torch.where(is_ref, torch.zeros_like(d), d)


def neighborhood_features(X, Z, k=KNN):
    kk = min(k, len(X) - 1)
    _, idx = cKDTree(X).query(X, k=kk + 1)
    return np.hstack([Z, Z[idx[:, 1:]].mean(1)]).astype(np.float32)


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
        Z = np.asarray(r['Z'], np.float32)
        if variant.startswith('nishi_ph'):
            Z = neighborhood_features(np.asarray(r['X'], float), Z)
        r['Zt'] = torch.tensor(Z, dtype=torch.float32)
        r['cx'] = dict(n=len(r['X']), edges=r['edges'].astype(int), tris=r['tris'].astype(int),
                       e_len=r['e_len'].astype(float), t_diam=r['t_diam'].astype(float))
    comp = [torch.tensor(c, dtype=torch.float32) for c in comp_np]
    d_in = recs[0]['Zt'].shape[1]

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(recs)); fold_aucs, refs_used = [], []
    t0 = time.time()
    for f, (tr, te) in enumerate(cv.split(comp_np, y, patients)):
        ref = dict(onehot=ref_idx, dist=P + ref_idx)
        if variant.startswith('gatetopo'):
            model = GateTopo(d_in, P)
        elif variant.startswith('nishi_ph'):
            model = Model(d_in, P, 'learned_unc_nc')
        elif variant.startswith('fixed_tuned_keep'):
            a = choose_reference(recs, tr, y, P); refs_used.append(str(meta['vocab'][a]))
            ref = dict(onehot=a, dist=P + a)
            model = KeepRef(d_in, P, 'fixed_cross_nc')
        elif variant.startswith('learned_meanvec'):
            model = Model(d_in, P, 'learned_nc')
            model.v0, model.v1 = PersLayMean(), PersLayMean()
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
                   auc_image=auc_img, n_images=len(recs), n_pos=int(y.sum())),
              open(os.path.join(RES, 'learned_' + tag + '.json'), 'w'), indent=2)
    np.save(os.path.join(RES, 'oof_' + tag + '.npy'), oof)
    print('TRAIN_REVIEW2_DONE', tag, flush=True)


if __name__ == '__main__':
    main()
