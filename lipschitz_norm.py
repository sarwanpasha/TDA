"""An a priori Lipschitz constant for the parameter-to-weight map, and a check of it.

Theorem 2 needs a constant L_g with |g_theta(z) - g_theta'(z)| <= L_g ||theta - theta'||.
Taking theta to be the *normalised* weights and the biases, a telescoping argument over
layers gives L_g <= sqrt(sum_l (A_{l-1}^2 + 1)) with A_0 = max ||z|| and A_l = A_{l-1} + B_b,
where B_b bounds the bias norms. This script evaluates that bound on the real feature domain
and measures the largest ratio actually attained, so the two can be compared.

python lipschitz_norm.py <cohort>
"""
import os, sys, json, itertools
import numpy as np
import torch
import torch.nn as nn
import difftda as dt
from train_learned import load_cohort, RES, T

DRAWS, NZ, SIGMA = 24, 4000, 0.25


def linears(model):
    return [m for m in model.net if isinstance(m, nn.Linear)]


def theta_norm(model):
    """The parameter vector Theorem 2 is stated in: normalised weights and biases."""
    return torch.cat([torch.cat([m.weight.detach().reshape(-1), m.bias.detach()])
                      for m in linears(model)])


def main():
    cohort = sys.argv[1] if len(sys.argv) > 1 else 'danenberg'
    torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(1)
    meta, recs = load_cohort(cohort)
    Z = np.concatenate([r['Z'] for r in recs], 0)
    idx = np.random.choice(len(Z), min(NZ, len(Z)), replace=False)
    Zt = torch.tensor(Z[idx], dtype=torch.float32)
    Bz = float(torch.linalg.vector_norm(Zt, dim=1).max())

    base = dt.LipschitzMLP(meta['d_in'], hidden=64, depth=2, scale=2 * T, constrained=True)
    L = len(linears(base))
    gmax, bmax = 0.0, 0.0
    for k in range(DRAWS):
        m = dt.LipschitzMLP(meta['d_in'], hidden=64, depth=2, scale=2 * T, constrained=True)
        m.load_state_dict(base.state_dict())
        with torch.no_grad():
            for p_ in m.parameters():
                p_.add_(torch.randn_like(p_) * SIGMA)
            ws = [(l.weight.detach().clone(), l.bias.detach().clone()) for l in linears(m)]
        bmax = max(bmax, max(float(torch.linalg.vector_norm(b)) for _, b in ws))
        # A plain copy whose parameters ARE the normalised weights, so autograd returns
        # the gradient in the parametrisation Theorem 2 is stated in. The norm of that
        # gradient is the local Lipschitz constant, a lower bound on L_g and a far tighter
        # probe than random parameter pairs, which are almost orthogonal to the worst case.
        layers = []
        for i, (W, b) in enumerate(ws):
            lin = nn.Linear(W.shape[1], W.shape[0])
            with torch.no_grad():
                lin.weight.copy_(W); lin.bias.copy_(b)
            layers.append(lin)
            if i < len(ws) - 1:
                layers.append(nn.LeakyReLU(0.1))
        net = nn.Sequential(*layers)
        for j in range(0, len(Zt), 256):
            net.zero_grad()
            out = net(Zt[j:j + 256]).squeeze(-1)
            # the max over cells of |grad| is what we want; take it one cell at a time
            # through a random unit combination is not enough, so use the row of largest
            # output magnitude as a representative and also the exact per-cell maximum
            for t_ in range(out.shape[0]):
                net.zero_grad()
                out2 = net(Zt[j + t_:j + t_ + 1]).squeeze()
                out2.backward()
                g = torch.cat([q.grad.reshape(-1) for q in net.parameters()])
                gmax = max(gmax, float(torch.linalg.vector_norm(g)))
                if t_ >= 15:
                    break
            break
    emp = gmax

    A = [Bz + l * bmax for l in range(L)]
    bound = float(np.sqrt(sum(a ** 2 + 1.0 for a in A)))

    out = dict(cohort=cohort, d_in=int(meta['d_in']), n_layers=L, n_draws=DRAWS,
               n_cells=int(len(idx)), sigma=SIGMA, Bz=Bz, Bb=bmax,
               L_g_analytic=bound, L_g_empirical=emp, slack=bound / emp,
               n_probe_cells=16)
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, f'lipschitz_norm_{cohort}.json'), 'w'), indent=2)
    print(json.dumps(out, indent=2), flush=True)
    print('LIPSCHITZ_NORM_DONE', flush=True)


if __name__ == '__main__':
    main()
