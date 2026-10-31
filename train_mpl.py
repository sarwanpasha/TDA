"""Multiparameter persistence landscapes (Vipond, 2020; Vipond et al., 2021) as a fixed,
non-learned topological baseline on the same complex.

python train_mpl.py <cohort> <target> <seed> <variant>

mpl_nc         bifiltration (edge length, cross-phenotype weight with the endothelial
               reference), landscape features, the usual classifier head
mpl_tuned_nc   the same with the reference phenotype chosen per fold on the training split
mpl_dtm_nc     (edge length, distance-to-measure weight)

The k-th multiparameter landscape at a point x of the parameter space is
lambda_k(x) = sup{h >= 0 : rank(M(x - h 1) -> M(x + h 1)) >= k}, where 1 = (1, 1). Along
the diagonal line through x the bifiltration restricts to the one-parameter filtration
f(sigma) = max(f1(sigma) - a, f2(sigma) - b), so lambda_k on that line is the k-th ordinary
persistence landscape of that one-parameter filtration. We therefore sample lines by their
offset a - b and evaluate each landscape on a grid of positions along the line; this is
exact for the definition above, with no approximation of the module.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from gudhi import SimplexTree
from train_learned import (load_cohort, image_targets, comp_features, choose_reference,
                           rank_norm, RES, T, EPOCHS, LR, REF_PHENOTYPE)
from train_extra import dtm_weight

N_LINES, N_POS, K_LEVELS = 11, 20, 3
OFFSETS = np.linspace(-T, T, N_LINES)          # a - b, the line's offset from the diagonal
POS = np.linspace(0.0, T, N_POS)               # position t along the line
EPS = 1e-9


def diagram_along_line(cx, f1v, f1e, f1t, f2v, f2e, f2t, a, b):
    """H0/H1 diagrams of the one-parameter filtration max(f1 - a, f2 - b)."""
    n, ne = cx['n'], len(cx['edges'])
    fv = np.maximum(f1v - a, f2v - b)
    fe = np.maximum(f1e - a, f2e - b)
    ft = np.maximum(f1t - a, f2t - b)
    st = SimplexTree()
    st.insert_batch(np.arange(n, dtype=np.int32).reshape(1, -1), fv)
    if ne:
        st.insert_batch(cx['edges'].T.astype(np.int32), fe)
    if len(cx['tris']):
        st.insert_batch(cx['tris'].T.astype(np.int32), ft)
    st.compute_persistence(persistence_dim_max=True)
    out = []
    for d in (0, 1):
        pairs = np.array(st.persistence_intervals_in_dimension(d), dtype=float).reshape(-1, 2)
        if len(pairs):
            pairs[:, 1] = np.minimum(pairs[:, 1], T)      # cap at the cutoff, as elsewhere
            pairs = pairs[pairs[:, 1] - pairs[:, 0] > EPS]
        out.append(pairs)
    return out


def landscape(pairs, pos, k):
    """k-level persistence landscape evaluated at the positions `pos`."""
    L = np.zeros((k, len(pos)))
    if len(pairs) == 0:
        return L
    b, d = pairs[:, 0][:, None], pairs[:, 1][:, None]
    tent = np.maximum(np.minimum(pos[None, :] - b, d - pos[None, :]), 0.0)
    tent = -np.sort(-tent, axis=0)
    m = min(k, len(pairs))
    L[:m] = tent[:m]
    return L


def features(rec, w):
    """Landscape feature vector of one image for a vertex weight w (second parameter)."""
    cx = rec['cx']
    f1v = np.zeros(cx['n']); f1e = cx['e_len']; f1t = cx['t_diam']
    f2v = w
    f2e = w[cx['edges']].max(1) if len(cx['edges']) else np.zeros(0)
    f2t = w[cx['tris']].max(1) if len(cx['tris']) else np.zeros(0)
    F = np.zeros((2, N_LINES, K_LEVELS, N_POS))
    for i, c in enumerate(OFFSETS):
        a, b = (c, 0.0) if c >= 0 else (0.0, -c)
        d0, d1 = diagram_along_line(cx, f1v, f1e, f1t, f2v, f2e, f2t, a, b)
        F[0, i] = landscape(d0, POS, K_LEVELS)
        F[1, i] = landscape(d1, POS, K_LEVELS)
    return (F / T).ravel().astype(np.float32)


class Head(nn.Module):
    def __init__(self, d, d_comp, use_comp):
        super().__init__()
        self.use_comp = use_comp
        self.net = nn.Sequential(nn.Linear(d + (d_comp if use_comp else 0), 64), nn.ReLU(),
                                 nn.Linear(64, 1))

    def forward(self, f, comp):
        return self.net(torch.cat([f, comp]) if self.use_comp else f)


def fit_predict(F, comp, y, tr, te, use_comp, P, seed):
    torch.manual_seed(seed)
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
    Fz = torch.tensor((F - mu) / sd, dtype=torch.float32)
    model = Head(F.shape[1], P, use_comp)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(EPOCHS):
        perm = np.random.permutation(tr); opt.zero_grad()
        for j, i in enumerate(perm):
            (lossf(model(Fz[i], comp[i]).squeeze(), torch.tensor(float(y[i]))) / 8).backward()
            if (j + 1) % 8 == 0:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
    model.eval()
    with torch.no_grad():
        return rank_norm(np.array([float(torch.sigmoid(model(Fz[i], comp[i]))) for i in te]))


def main():
    cohort, target, seed, variant = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(1)
    use_comp = not variant.endswith('_nc')
    base = variant[:-3] if not use_comp else variant
    meta, recs = load_cohort(cohort)
    P = len(meta['vocab'])
    y = np.asarray(image_targets(cohort, recs, meta['vocab'])[target], dtype=float)
    keep = ~np.isnan(y)
    if not keep.all():
        recs = [r for r, k in zip(recs, keep) if k]; y = y[keep]
    y = y.astype(int)
    comp_np = comp_features(recs, P)
    comp = [torch.tensor(c, dtype=torch.float32) for c in comp_np]
    patients = np.array([str(r['patient']) for r in recs])
    ref_idx = next((i for i, v in enumerate(meta['vocab']) if REF_PHENOTYPE in str(v).lower()), 0)
    print(f'{cohort} {target} variant={variant} seed={seed} images={len(recs)} '
          f'positives={int(y.sum())}', flush=True)
    for r in recs:
        r['cx'] = dict(n=len(r['X']), edges=r['edges'].astype(int), tris=r['tris'].astype(int),
                       e_len=r['e_len'].astype(float), t_diam=r['t_diam'].astype(float))
    cache = {}

    def feats_for(ref):
        if ref not in cache:
            t0 = time.time()
            out = []
            for r in recs:
                if ref == 'dtm':
                    w = np.minimum(dtm_weight(r['X'].astype(float)), T)
                else:
                    d = r['Z'][:, P + ref].astype(float) * T
                    w = np.where(r['Z'][:, ref] > 0.5, 2 * T, d)
                out.append(features(r, w))
            cache[ref] = np.stack(out)
            print('landscapes for reference %s: %s (%.0fs)' % (ref, cache[ref].shape,
                                                               time.time() - t0), flush=True)
        return cache[ref]

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(recs)); fold_aucs, refs_used = [], []
    for f, (tr, te) in enumerate(cv.split(comp_np, y, patients)):
        if base == 'mpl_dtm':
            ref = 'dtm'
        elif base == 'mpl_tuned':
            ref = choose_reference(recs, tr, y, P); refs_used.append(str(meta['vocab'][ref]))
        else:
            ref = ref_idx
        F = feats_for(ref)
        oof[te] = fit_predict(F, comp, y, tr, te, use_comp, P, seed * 10 + f)
        fold_aucs.append(roc_auc_score(y[te], oof[te]))
        print(f'fold {f} auc={fold_aucs[-1]:.4f}', flush=True)
    auc_img = roc_auc_score(y, oof)
    print(f'FOLD-MEAN AUC {np.mean(fold_aucs):.4f} +- {np.std(fold_aucs):.4f} | IMAGE AUC {auc_img:.4f}',
          flush=True)
    tag = f'{cohort}_{target.replace(" ", "_")}_{variant}{os.environ.get("TDA_TAG", "")}_s{seed}'
    json.dump(dict(cohort=cohort, target=target, variant=variant + os.environ.get('TDA_TAG', ''),
                   seed=seed, auc_fold_mean=float(np.mean(fold_aucs)),
                   auc_fold_sd=float(np.std(fold_aucs)), fold_aucs=[float(a) for a in fold_aucs],
                   refs_used=refs_used, auc_image=auc_img, n_images=len(recs), n_pos=int(y.sum())),
              open(os.path.join(RES, 'learned_' + tag + '.json'), 'w'), indent=2)
    np.save(os.path.join(RES, 'oof_' + tag + '.npy'), oof)
    print('TRAIN_MPL_DONE', tag, flush=True)


if __name__ == '__main__':
    main()
