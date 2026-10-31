"""Empirical check of the joint data--parameter bound (Theorem 2), not just its
fixed-weight special case (Theorem 1).

python stability_theta.py <cohort> <seed>

Theorem 2 is proved in two steps: the parameter perturbation is turned into a sup-norm
perturbation of the vertex weights, |w_theta - w_theta'|_inf <= (T/2) L_g(R) ||theta-theta'||,
and that is fed to Theorem 1 alongside the Hausdorff term. We check both steps, because
checking only the second is what the earlier version of this experiment did.

  A. parameter only (X' = X). Perturb theta, measure ||theta-theta'||, |w-w'|_inf and the
     bottleneck distance. Two quantities matter: whether d_B <= |w-w'|_inf holds with a
     *learned* weight function rather than a hand-built one, and the ratio
     |w-w'|_inf / ||theta-theta'||, which is an empirical lower bound on (T/2) L_g(R) --
     it says the constant in the theorem is finite and small rather than vacuous.

  B. joint. Also jitter the cell positions, so eps = d_H(X, X') > 0, rebuild the distance
     features and the complex, and check d_B <= eps + |w-w'|_inf. This is the inequality
     the theorem actually asserts.

Writes stability_theta.csv (per trial) and stability_theta_summary.csv (per regime/dim).
"""
import os, sys, copy
import numpy as np
import pandas as pd
import torch
from gudhi import bottleneck_distance
import difftda as dt
from train_learned import load_cohort, T, RES
from robustness import diagram, perturb

N_IMAGES = 30
N_TRIALS = 6
SIGMAS = (1e-3, 3e-3, 1e-2, 3e-2)
JITTERS = (1.0, 3.0, 8.0)


def flat(model):
    """Theorem 2 is stated in the normalised parametrisation: the spectrally
    normalised weight matrices and the biases, not the raw pre-normalisation
    weights. Reading m.weight applies the parametrisation, so this is the theta
    the analytic constant of Proposition 2 is a Lipschitz constant for."""
    import torch.nn as _nn
    ms = [m for m in model.net if isinstance(m, _nn.Linear)]
    return torch.cat([torch.cat([m.weight.detach().reshape(-1), m.bias.detach()])
                      for m in ms])


def jitter_params(model, sigma, rng):
    """A copy of the model with every parameter perturbed, plus ||theta-theta'||."""
    m2 = copy.deepcopy(model)
    with torch.no_grad():
        for p in m2.parameters():
            p.add_(torch.tensor(rng.normal(0, sigma, tuple(p.shape)), dtype=p.dtype))
    return m2, float(torch.linalg.vector_norm(flat(model) - flat(m2)))


def weights(model, Z):
    with torch.no_grad():
        return model(torch.tensor(Z, dtype=torch.float32)).numpy().astype(float)


def cx_of(rec):
    return dict(n=len(rec['X']), edges=rec['edges'].astype(int), tris=rec['tris'].astype(int),
                e_len=rec['e_len'].astype(float), t_diam=rec['t_diam'].astype(float))


def main():
    cohort = sys.argv[1] if len(sys.argv) > 1 else 'danenberg'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    meta, recs = load_cohort(cohort)
    P = len(meta['vocab'])

    model = dt.LipschitzMLP(meta['d_in'], hidden=64, depth=2, scale=2 * T).eval()
    lip_z = model.lipschitz_bound()
    print(f'weight-function Lipschitz bound in z: {lip_z:.4f}', flush=True)

    rows = []
    for r in recs[:N_IMAGES]:
        cx0 = cx_of(r)
        w0 = weights(model, r['Z'])
        d0 = diagram(cx0, w0)
        for _ in range(N_TRIALS):
            sigma = float(rng.choice(SIGMAS))
            m2, dtheta = jitter_params(model, sigma, rng)

            w1 = weights(m2, r['Z'])
            d1 = diagram(cx0, w1)
            linf = float(np.abs(w0 - w1).max())
            for dim in (0, 1):
                db = float(bottleneck_distance(d0[dim], d1[dim]))
                rows.append(dict(regime='param', image=str(r['image']), dim=dim,
                                 sigma=sigma, dtheta=dtheta, eps=0.0, linf=linf,
                                 bound=linf, db=db))

            jit = float(rng.choice(JITTERS))
            pr = perturb(r, jit, 0.0, rng, P)
            eps = float(np.linalg.norm(pr['X'].astype(float) - r['X'].astype(float),
                                       axis=1).max())
            # The parameter term must be measured on a COMMON complex, which is the
            # perturbed one: the proof bounds d_B(PD(F_w(X)), PD(F_w(X'))) by eps and then
            # applies Theorem 1 to the two weight functions on X'. Measuring
            # |w_theta(Z) - w_theta'(Z')| instead folds the data-induced weight change into
            # the parameter term, so eps is counted twice and the test is weaker than the
            # theorem.
            w0j = weights(model, pr['Z'])
            w1j = weights(m2, pr['Z'])
            d1j = diagram(pr['cx'], w1j)
            linf_j = float(np.abs(w0j - w1j).max())
            for dim in (0, 1):
                db = float(bottleneck_distance(d0[dim], d1j[dim]))
                rows.append(dict(regime='joint', image=str(r['image']), dim=dim,
                                 sigma=sigma, dtheta=dtheta, eps=eps, linf=linf_j,
                                 bound=eps + linf_j, db=db))

    df = pd.DataFrame(rows)
    df['slack'] = df['bound'] - df['db']
    df['ratio'] = df['db'] / df['bound'].clip(lower=1e-9)
    df['lip_emp'] = df['linf'] / df['dtheta'].clip(lower=1e-12)
    os.makedirs(RES, exist_ok=True)
    df.to_csv(os.path.join(RES, 'stability_theta.csv'), index=False)

    g = df.groupby(['regime', 'dim'])
    summ = pd.DataFrame(dict(
        n=g.size(),
        dtheta_mean=g.dtheta.mean(), dtheta_max=g.dtheta.max(),
        eps_mean=g.eps.mean(),
        linf_mean=g.linf.mean(), linf_max=g.linf.max(),
        db_mean=g.db.mean(), db_max=g.db.max(),
        bound_mean=g.bound.mean(),
        ratio_mean=g.ratio.mean(), ratio_max=g.ratio.max(),
        violations=g.apply(lambda d: int((d.db > d.bound + 1e-6).sum())),
    )).reset_index()
    summ['lip_emp_max'] = float(df[df.regime == 'param'].lip_emp.max())
    summ['lip_z_bound'] = lip_z
    summ.to_csv(os.path.join(RES, 'stability_theta_summary.csv'), index=False)
    print(summ.to_string(index=False), flush=True)
    print('STABILITY_THETA_DONE', flush=True)


if __name__ == '__main__':
    main()
