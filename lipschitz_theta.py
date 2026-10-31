"""Estimate L_g(R) and check Theorem 2's stated bound directly.

python lipschitz_theta.py <cohort> <seed>

Section 5.1 previously verified the sharper intermediate quantity
    d_B <= eps + |w_theta(Z(X')) - w_theta'(Z(X'))|_inf
because the theorem's constant L_g(R) was left abstract. This script makes it concrete.

g_theta is smooth in theta, so the Lipschitz constant of theta -> g_theta(z) on a parameter
ball is the supremum of the parameter-gradient norm over that ball and over the feature
domain:  L_g(R) = sup ||grad_theta g_theta(z)||_2.  We evaluate that gradient exactly by
autograd at parameter draws from the same ball the stability trials used, and at real cell
features drawn from the cohort, and take the maximum. This is an empirical supremum, not a
proof: it is a lower bound on L_g(R) that is attained, so checking the theorem against it is
a strictly harder test than checking it against any valid upper bound would be.

Note the gradient is taken with respect to the raw parameters (the spectral-norm
parametrisation's weight_orig and the biases), which is exactly the vector whose norm the
stability trials report as ||theta - theta'||, so the two are commensurable.

Writes lipschitz_theta.csv (per-draw gradient norms) and
theorem_check.csv (per-trial ratio of d_B to the theorem's right-hand side).
"""
import os, sys, copy
import numpy as np
import pandas as pd
import torch
import difftda as dt
from train_learned import load_cohort, T, RES

N_THETA = 40
N_CELLS = 4000
SIGMAS = (1e-3, 3e-3, 1e-2, 3e-2)


def flat(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def perturbed(model, sigma, rng):
    m2 = copy.deepcopy(model)
    with torch.no_grad():
        for p in m2.parameters():
            p.add_(torch.tensor(rng.normal(0, sigma, tuple(p.shape)), dtype=p.dtype))
    return m2, float(torch.linalg.vector_norm(flat(model) - flat(m2)))


def grad_norms(model, Z):
    """||grad_theta g_theta(z)||_2 for every row of Z, one backward per row.

    g_theta is the pre-sigmoid network output: w_theta = 2T * sigmoid(g_theta), and the
    T/2 in the theorem is 2T times the 1/4 Lipschitz constant of the sigmoid, so the
    constant the theorem needs is the one for g, not for w.
    """
    out = np.empty(len(Z))
    for i in range(len(Z)):
        model.zero_grad(set_to_none=True)
        g = model.net(Z[i:i + 1]).squeeze()
        g.backward()
        sq = sum(float((p.grad ** 2).sum()) for p in model.parameters() if p.grad is not None)
        out[i] = np.sqrt(sq)
    return out


def main():
    cohort = sys.argv[1] if len(sys.argv) > 1 else 'danenberg'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    meta, recs = load_cohort(cohort)
    P = len(meta['vocab'])

    Zall = np.concatenate([r['Z'] for r in recs[:200]], 0)
    idx = rng.choice(len(Zall), size=min(N_CELLS, len(Zall)), replace=False)
    Z = torch.tensor(Zall[idx], dtype=torch.float32)

    base = dt.LipschitzMLP(meta['d_in'], hidden=64, depth=2, scale=2 * T).eval()
    rows = []
    for k in range(N_THETA):
        sigma = float(rng.choice(SIGMAS)) if k else 0.0
        m = base if k == 0 else perturbed(base, sigma, rng)[0]
        gn = grad_norms(m, Z)
        rows.append(dict(draw=k, sigma=sigma, gn_max=float(gn.max()),
                         gn_mean=float(gn.mean()), gn_p99=float(np.percentile(gn, 99))))
        print(f'draw {k:2d} sigma={sigma:.4g} max||grad||={gn.max():.4f}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, 'lipschitz_theta.csv'), index=False)
    Lg = float(df.gn_max.max())
    print(f'L_g(R) empirical supremum: {Lg:.4f}', flush=True)

    tr = pd.read_csv(os.path.join(RES, 'stability_theta.csv'))
    c_eps = 2.0 + np.sqrt(P)
    tr['thm_bound'] = c_eps * tr['eps'] + (T / 2.0) * Lg * tr['dtheta']
    tr['thm_ratio'] = tr['db'] / tr['thm_bound'].clip(lower=1e-12)
    tr['thm_violation'] = (tr['db'] > tr['thm_bound'] + 1e-6).astype(int)
    g = tr.groupby(['regime', 'dim'])
    summ = pd.DataFrame(dict(
        n=g.size(), thm_bound_mean=g.thm_bound.mean(), db_max=g.db.max(),
        thm_ratio_mean=g.thm_ratio.mean(), thm_ratio_max=g.thm_ratio.max(),
        thm_violations=g.thm_violation.sum())).reset_index()
    summ['L_g'] = Lg
    summ['c_eps'] = c_eps
    summ['P'] = P
    summ.to_csv(os.path.join(RES, 'theorem_check.csv'), index=False)
    print(summ.to_string(index=False), flush=True)
    print('LIPSCHITZ_THETA_DONE', flush=True)


if __name__ == '__main__':
    main()
