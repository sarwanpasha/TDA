"""The learned filtration on the weighted alpha complex.

python train_alpha.py <cohort> <target> <seed> [variant]

alpha_nc       learned Lipschitz vertex weight, weighted alpha complex
alpha_zero_nc  w = 0, the ordinary alpha complex, as the control

Features, network, vectoriser, head, folds and seeds are exactly those of
train_learned.py, so the only thing that differs from learned_nc is the complex
the diagram is computed on: the Delaunay flag complex, for which no modulus of
continuity exists, against the weighted alpha complex, for which the joint
bound holds unconditionally.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import difftda as dt
import dtalpha as dta
from train_learned import (load_cohort, image_targets, comp_features, rank_norm,
                           PersLayLite, PersLayLip, VEC, T, MAX_PAIRS, RES, EPOCHS, LR)

class AlphaModel(nn.Module):
    def __init__(self, d_in, variant, hidden=64):
        super().__init__()
        self.zero = 'zero' in variant
        self.filt = dt.LipschitzMLP(d_in, hidden=hidden, depth=2, scale=2 * T,
                                    constrained=True)
        V = PersLayLip if VEC == 'lip' else PersLayLite
        self.v0, self.v1 = V(), V()
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, rec):
        Z = rec['Zt']
        w = (torch.zeros(len(Z), dtype=Z.dtype, device=Z.device) if self.zero
             else self.filt(Z))
        p0, p1 = dta.persistence(rec['Xt'], w, cap=T, max_pairs=MAX_PAIRS)
        return self.head(torch.cat([self.v0(p0), self.v1(p1)]))

def main():
    cohort, target, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    variant = sys.argv[4] if len(sys.argv) > 4 else 'alpha_nc'
    torch.manual_seed(seed); np.random.seed(seed)
    meta, recs = load_cohort(cohort)
    P = len(meta['vocab'])
    Y = image_targets(cohort, recs, meta['vocab'])
    y = np.asarray(Y[target], dtype=float)
    keep = ~np.isnan(y)
    if not keep.all():
        recs = [r for r, k in zip(recs, keep) if k]; y = y[keep]
    y = y.astype(int)
    comp_np = comp_features(recs, P)
    patients = np.array([str(r['patient']) for r in recs])
    print(f'{cohort} {target} variant={variant} seed={seed} images={len(recs)} '
          f'positives={int(y.sum())} d_in={meta["d_in"]}', flush=True)
    for r in recs:
        r['Zt'] = torch.tensor(r['Z'], dtype=torch.float32)
        r['Xt'] = torch.tensor(np.asarray(r['X']), dtype=torch.float32)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(recs)); fold_aucs = []
    lossf = nn.BCEWithLogitsLoss(); t0 = time.time()
    for f, (tr, te) in enumerate(cv.split(comp_np, y, patients)):
        model = AlphaModel(meta['d_in'], variant)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        for ep in range(EPOCHS):
            model.train(); perm = np.random.permutation(tr); tot = 0.0
            opt.zero_grad()
            for j, i in enumerate(perm):
                loss = lossf(model(recs[i]).squeeze(), torch.tensor(float(y[i])))
                (loss / 8).backward(); tot += float(loss)
                if (j + 1) % 8 == 0:
                    opt.step(); opt.zero_grad()
            opt.step(); opt.zero_grad()
            if f == 0 and (ep + 1) % 5 == 0:
                print(f'  epoch {ep+1:2d} loss={tot/len(perm):.4f}', flush=True)
        model.eval(); out = np.zeros(len(te))
        with torch.no_grad():
            for k, i in enumerate(te):
                out[k] = float(torch.sigmoid(model(recs[i])))
        oof[te] = rank_norm(out)
        fold_aucs.append(roc_auc_score(y[te], oof[te]))
        print(f'fold {f} auc={fold_aucs[-1]:.4f} ({time.time()-t0:.0f}s)', flush=True)
    auc_img = roc_auc_score(y, oof)
    print(f'FOLD-MEAN AUC {np.mean(fold_aucs):.4f} +- {np.std(fold_aucs):.4f} | '
          f'IMAGE AUC {auc_img:.4f}', flush=True)
    tag = f'{cohort}_{target.replace(" ", "_")}_{variant}{os.environ.get("TDA_TAG", "")}_s{seed}'  # [tp]
    json.dump(dict(cohort=cohort, target=target, variant=variant + os.environ.get('TDA_TAG', ''), seed=seed,  # [tp]
                   auc_fold_mean=float(np.mean(fold_aucs)),
                   auc_fold_sd=float(np.std(fold_aucs)),
                   fold_aucs=[float(a) for a in fold_aucs], refs_used=[],
                   auc_image=auc_img, n_images=len(recs), n_pos=int(y.sum())),
              open(os.path.join(RES, 'learned_' + tag + '.json'), 'w'), indent=2)
    np.save(os.path.join(RES, 'oof_' + tag + '.npy'), oof)
    print('TRAIN_LEARNED_DONE', tag, flush=True)

if __name__ == '__main__':
    main()
