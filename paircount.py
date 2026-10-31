"""How many persistence pairs an image actually produces, against the slot count K.

Equation (4) keeps the K most persistent pairs. If the realised count never reaches K
the truncation is inactive and the vectoriser is a plain scaled sum, which is the case
Corollary 1 needs. This measures the realised counts on the whole cohort.

python paircount.py <cohort>
"""
import os, sys, json
import numpy as np
import torch
import difftda as dt
from train_learned import load_cohort, RES, T

BIG = 4096


def counts(cx, w):
    p0, p1 = dt.persistence(cx, w, cap=T, max_pairs=BIG)
    return [int(((p[:, 1] - p[:, 0]) > 0).sum()) for p in (p0, p1)]


def main():
    cohort = sys.argv[1] if len(sys.argv) > 1 else 'danenberg'
    torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(1)
    meta, recs = load_cohort(cohort)
    net = dt.LipschitzMLP(meta['d_in'], hidden=64, depth=2, scale=2 * T, constrained=True)
    rows = []
    for r in recs:
        cx = dict(n=len(r['X']), edges=r['edges'].astype(int), tris=r['tris'].astype(int),
                  e_len=r['e_len'].astype(float), t_diam=r['t_diam'].astype(float))
        Z = torch.tensor(r['Z'], dtype=torch.float32)
        with torch.no_grad():
            wl = net(Z)
        z0, z1 = counts(cx, torch.zeros(len(r['X'])))
        l0, l1 = counts(cx, wl)
        rows.append((z0, z1, l0, l1))
    a = np.array(rows)
    out = dict(cohort=cohort, n_images=len(recs), K=48,
               zero_h0_max=int(a[:, 0].max()), zero_h1_max=int(a[:, 1].max()),
               learned_h0_max=int(a[:, 2].max()), learned_h1_max=int(a[:, 3].max()),
               zero_h1_mean=float(a[:, 1].mean()), learned_h1_mean=float(a[:, 3].mean()),
               frac_images_over_K=float(((a[:, [0, 1, 2, 3]] > 48).any(1)).mean()))
    json.dump(out, open(os.path.join(RES, f'paircount_{cohort}.json'), 'w'), indent=2)
    print(json.dumps(out, indent=2), flush=True)
    print('PAIRCOUNT_DONE', flush=True)


if __name__ == '__main__':
    main()
