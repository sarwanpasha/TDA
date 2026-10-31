"""Filtration baselines from the literature, run through our own pipeline.

python train_extra.py <cohort> <target> <seed> <variant>

sublevel      learned vertex filter with a pure sublevel-set filtration and no
              geometric term: the graph-filtration-learning construction of Hofer
              et al. (2020) and Horn et al. (2022) transplanted onto our complex.
              Unconstrained, as in that work.
sublevel_sn   the same with spectral normalisation, so that architecture and
              filtration form can be separated.
dtm           DTM-weighted Rips (Anai et al. 2020): the vertex weight is the
              empirical distance-to-measure at mass parameter m = k/n.
dowker        relational / Dowker filtration (Stolz et al. 2024) with the fixed
              reference phenotype; simplices enter at min over witnesses of the
              max distance to the witness, restricted to our sparse 1-skeleton.
dowker_tuned  the same with the reference chosen per fold on the training split.

A _nc suffix drops the composition vector, as elsewhere.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from scipy.spatial import cKDTree
import difftda as dt
import frac, featmask  # [tp]
from train_learned import (load_cohort, image_targets, comp_features, choose_reference,
                           PersLayLip, VEC,
                           run_fold, PersLayLite, T, MAX_PAIRS, RES, REF_PHENOTYPE)

DTM_K = 10


def dtm_weight(X):
    """Empirical distance to measure: root mean squared distance to the k nearest
    neighbours, the weight Anai et al. put on a weighted Rips filtration."""
    k = min(DTM_K, len(X) - 1)
    if k < 1:
        return np.zeros(len(X), np.float64)
    d, _ = cKDTree(X).query(X, k=k + 1)
    return np.sqrt((d[:, 1:] ** 2).mean(1))


def dowker_values(X, pheno, edges, tris, a_idx):
    """Dowker sink filtration values against the witness set of phenotype a_idx.

    A simplex enters at min over witnesses a of max over its vertices v of d(v, a).
    We evaluate that on the same sparse 1-skeleton every other filtration uses, so
    the comparison isolates the filtration and not the complex.
    """
    A = X[pheno == a_idx]
    n = len(X)
    if len(A) == 0:
        big = np.full(n, 2 * T)
        return big, np.full(len(edges), 2 * T), np.full(len(tris), 2 * T)
    D = np.sqrt(((X[:, None, :] - A[None, :, :]) ** 2).sum(-1))
    dv = D.min(1)
    if len(edges):
        de = np.maximum(D[edges[:, 0]], D[edges[:, 1]]).min(1)
    else:
        de = np.zeros(0)
    if len(tris):
        dt3 = np.maximum(np.maximum(D[tris[:, 0]], D[tris[:, 1]]), D[tris[:, 2]]).min(1)
    else:
        dt3 = np.zeros(0)
    return dv, de, dt3


def split_width(variant, default=64):
    import re as _re
    m = _re.search(r'_w(\d+)', variant)
    return (variant.replace(m.group(0), ''), int(m.group(1))) if m else (variant, default)


class Extra(nn.Module):
    """One head, three ways of producing the filtration."""

    def __init__(self, kind, d_in, d_comp, use_comp, constrained=False, hidden=64):
        super().__init__()
        self.kind, self.use_comp = kind, use_comp
        self.filt = None
        if kind == 'sublevel':
            self.filt = dt.LipschitzMLP(d_in, hidden=hidden, depth=2, scale=2 * T,
                                        constrained=constrained)
        V = PersLayLip if VEC == 'lip' else PersLayLite
        self.v0, self.v1 = V(), V()
        d = 64 + (d_comp if use_comp else 0)
        self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, rec, comp, ref=None):
        if self.kind == 'sublevel':
            w, cx = self.filt(rec['Zt']), rec['cx_sub']
        else:
            w, cx = rec['wfix'], rec['cx_fix']
        p0, p1 = dt.persistence(cx, w, cap=T, max_pairs=MAX_PAIRS)
        parts = [self.v0(p0), self.v1(p1)] + ([comp] if self.use_comp else [])
        return self.head(torch.cat(parts))


def set_fixed(recs, kind, a_idx):
    """Attach the fixed filtration data for one choice of reference phenotype."""
    for r in recs:
        e, t3 = r['edges'].astype(int), r['tris'].astype(int)
        if kind == 'dtm':
            w = np.minimum(r['dtm'], T)
            el, td = r['e_len'].astype(float), r['t_diam'].astype(float)
        else:
            dv, de, dt3 = dowker_values(r['X'], r['pheno'].astype(int), e, t3, a_idx)
            w, el, td = np.minimum(dv, 2 * T), de, dt3
        r['wfix'] = torch.tensor(w, dtype=torch.float32)
        r['cx_fix'] = dict(n=len(r['X']), edges=e, tris=t3, e_len=el, t_diam=td)


def main():
    cohort, target, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    variant = sys.argv[4]
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(1)

    use_comp = not variant.endswith('_nc')
    base = variant[:-3] if not use_comp else variant
    base, HIDDEN = split_width(base)
    kind = 'sublevel' if base.startswith('sublevel') else ('dtm' if base == 'dtm' else 'dowker')
    constrained = (base == 'sublevel_sn')

    meta, recs = load_cohort(cohort)
    P = len(meta['vocab'])
    recs = featmask.apply(recs, P)  # [tp]
    y = np.asarray(image_targets(cohort, recs, meta['vocab'])[target], dtype=float)
    keep = ~np.isnan(y)
    if not keep.all():
        recs = [r for r, k in zip(recs, keep) if k]
        y = y[keep]
    y = y.astype(int)
    comp_np = comp_features(recs, P)
    patients = np.array([str(r['patient']) for r in recs])
    ref_idx = next((i for i, v in enumerate(meta['vocab'])
                    if REF_PHENOTYPE in str(v).lower()), 0)
    print(f'{cohort} {target} variant={variant} seed={seed} images={len(recs)} '
          f'positives={int(y.sum())} d_in={meta["d_in"]} '
          f'reference={meta["vocab"][ref_idx]}', flush=True)

    for r in recs:
        r['Zt'] = torch.tensor(r['Z'], dtype=torch.float32)
        z = np.zeros(len(r['edges']), float)
        r['cx_sub'] = dict(n=len(r['X']), edges=r['edges'].astype(int),
                           tris=r['tris'].astype(int), e_len=z,
                           t_diam=np.zeros(len(r['tris']), float))
        if kind == 'dtm':
            r['dtm'] = dtm_weight(r['X'])
    comp = [torch.tensor(c, dtype=torch.float32) for c in comp_np]

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(recs))
    fold_aucs, refs_used = [], []
    t0 = time.time()
    for f, (tr, te) in enumerate(cv.split(comp_np, y, patients)):
        tr = frac.cut(tr, patients, y, seed, f)  # [tp]
        if kind != 'sublevel':
            a = ref_idx
            if base == 'dowker_tuned':
                a = choose_reference(recs, tr, y, P)
                refs_used.append(str(meta['vocab'][a]))
            set_fixed(recs, kind, a)
        model = Extra(kind, meta['d_in'], P, use_comp, constrained, hidden=HIDDEN)
        oof[te] = run_fold(model, tr, te, recs, comp, y, None, 'cpu', log=(f == 0))
        fold_aucs.append(roc_auc_score(y[te], oof[te]))
        print(f'fold {f} auc={fold_aucs[-1]:.4f} ({time.time()-t0:.0f}s)'
              + (f' ref={refs_used[-1]}' if refs_used else ''), flush=True)

    auc_fold, auc_sd = float(np.mean(fold_aucs)), float(np.std(fold_aucs))
    auc_img = roc_auc_score(y, oof)
    print(f'FOLD-MEAN AUC {auc_fold:.4f} +- {auc_sd:.4f} | IMAGE AUC {auc_img:.4f}',
          flush=True)
    os.makedirs(RES, exist_ok=True)
    tag = f'{cohort}_{target.replace(" ", "_")}_{variant}{os.environ.get("TDA_TAG", "")}_s{seed}'  # [tp]
    json.dump(dict(cohort=cohort, target=target, variant=variant + os.environ.get('TDA_TAG', ''), seed=seed,  # [tp]
                   auc_fold_mean=auc_fold, auc_fold_sd=auc_sd,
                   fold_aucs=[float(a) for a in fold_aucs], refs_used=refs_used,
                   auc_image=auc_img, n_images=len(recs), n_pos=int(y.sum())),
              open(os.path.join(RES, 'learned_' + tag + '.json'), 'w'), indent=2)
    np.save(os.path.join(RES, 'oof_' + tag + '.npy'), oof)
    print('TRAIN_EXTRA_DONE', tag, flush=True)


if __name__ == '__main__':
    main()
