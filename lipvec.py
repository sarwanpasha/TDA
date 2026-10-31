"""The explicit constant for the whole parameter-to-representation map.

Reports the bound and spectral norms of a trained phi so the paper can quote
C_phi = sqrt(2) B_phi + 1.62 L_phi and the resulting constant
    N C_phi L_g / (2 K0)
for the map theta -> v(PD(F_w_theta(X))).
"""
import os, sys, json, glob, math
import numpy as np
import torch

RES = os.environ.get('TDA_RES', 'results')


def phi_constants(state, prefix):
    """L_phi = product of spectral norms (ReLU is 1-Lipschitz); B_phi bounds ||phi|| on
    the unit cube [0,1]^3, whose points have norm at most sqrt(3)."""
    W = [v for k, v in state.items() if k.startswith(prefix) and k.endswith('weight')]
    B = [v for k, v in state.items() if k.startswith(prefix) and k.endswith('bias')]
    if not W:
        return None
    L = 1.0
    for w in W:
        L *= float(torch.linalg.matrix_norm(w, 2))
    z = torch.zeros(1, W[0].shape[1])
    for i, (w, b) in enumerate(zip(W, B)):
        z = torch.nn.functional.linear(z, w, b)
        if i < len(W) - 1:
            z = torch.relu(z)
    return dict(L_phi=L, phi0=float(z.norm()), B_phi=float(z.norm()) + L * math.sqrt(3))


def main():
    ck = sorted(glob.glob(os.path.join(RES, 'model_probe.pt')))
    BASE = 'results'
    lz = json.load(open(os.path.join(BASE, 'lipschitz_norm_danenberg.json')))
    pc = json.load(open(os.path.join(BASE, 'paircount_danenberg.json')))
    K0 = int(os.environ.get('TDA_KLIP', 1024))
    N = int(os.environ.get('TDA_NSIMP', 800 + 2359 + 1560))
    out = dict(K0=K0, n_simplices=N, L_g=lz['L_g_analytic'], source=None)
    if ck:
        st = torch.load(ck[0], map_location='cpu')
        st = st.get('model', st)
        for pref in ('v0.f.', 'v1.f.'):
            c = phi_constants(st, pref)
            if c:
                out.update({k + '_' + pref[:2]: v for k, v in c.items()})
                out['source'] = os.path.basename(ck[0])
                break
    Lp = out.get('L_phi_v0')
    if Lp is not None:
        Bp = out['B_phi_v0']
        C = math.sqrt(2) * Bp + 1.62 * Lp
        out['C_phi'] = C
        out['const_theta_to_vec'] = N * C * out['L_g'] / (2.0 * K0)
    json.dump(out, open(os.path.join(RES, 'lipvec_danenberg.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
