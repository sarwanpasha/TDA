"""Differentiable weighted-Rips persistence on a sparse geometric complex.

The complex (vertices, Delaunay edges below a cutoff, and their triangles) is fixed
by the cell positions, so only the filtration values change during training:

    f(sigma) = max( diam(sigma), max_{v in sigma} w(v) )

with diam(sigma) constant and w(v) = w_theta(z_v) produced by a Lipschitz network.
Persistence pairs are found with GUDHI on detached values; the birth/death values
are then recomputed in torch, which gives exact (sub)gradients wherever the pairing
is locally constant -- the standard argument of Carriere et al. (2021).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import Delaunay
from gudhi import SimplexTree

# Every tensor here is tiny (a few thousand entries); with the default intra-op
# thread count on a many-core node the launch overhead dominates and each
# persistence call costs ~400 ms instead of ~6 ms.
torch.set_num_threads(1)


def build_complex(X: np.ndarray, cutoff: float):
    """Vertices, Delaunay edges shorter than `cutoff`, and their induced triangles."""
    n = len(X)
    if n < 4:
        return dict(n=n, edges=np.zeros((0, 2), int), tris=np.zeros((0, 3), int),
                    e_len=np.zeros(0), t_diam=np.zeros(0))
    tri = Delaunay(X)
    e = set()
    for s in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                e.add((min(s[i], s[j]), max(s[i], s[j])))
    edges = np.array(sorted(e), int)
    e_len = np.linalg.norm(X[edges[:, 0]] - X[edges[:, 1]], axis=1)
    keep = e_len <= cutoff
    edges, e_len = edges[keep], e_len[keep]
    ok = {(a, b) for a, b in edges}
    tris, t_diam = [], []
    for s in tri.simplices:
        a, b, c = sorted(s)
        if (a, b) in ok and (a, c) in ok and (b, c) in ok:
            tris.append((a, b, c))
            t_diam.append(max(np.linalg.norm(X[a] - X[b]), np.linalg.norm(X[a] - X[c]),
                              np.linalg.norm(X[b] - X[c])))
    tris = np.array(tris, int).reshape(-1, 3)
    return dict(n=n, edges=edges, tris=tris, e_len=e_len,
                t_diam=np.array(t_diam, float))


def _filtration_values(cx, w):
    """Filtration value of every simplex, as torch tensors."""
    fv = w
    fe = torch.maximum(torch.as_tensor(cx['e_len'], dtype=w.dtype, device=w.device),
                       w[cx['edges']].max(1).values) if len(cx['edges']) else w.new_zeros(0)
    ft = torch.maximum(torch.as_tensor(cx['t_diam'], dtype=w.dtype, device=w.device),
                       w[cx['tris']].max(1).values) if len(cx['tris']) else w.new_zeros(0)
    return fv, fe, ft


def _index_maps(cx):
    """Cache the simplex -> position map; the combinatorics never change."""
    if '_idx' in cx:
        return cx['_idx']
    n = cx['n']
    m = {(int(i),): i for i in range(n)}
    for k, (a, b) in enumerate(cx['edges']):
        m[(int(a), int(b))] = n + k
    off = n + len(cx['edges'])
    for k, (a, b, c) in enumerate(cx['tris']):
        m[(int(a), int(b), int(c))] = off + k
    cx['_idx'] = m
    return m


def persistence(cx, w, cap: float, max_pairs: int = 64):
    """Differentiable (birth, death) tensors for H0 and H1.

    Returns two tensors of shape (max_pairs, 2), zero padded, sorted by persistence.
    """
    fv, fe, ft = _filtration_values(cx, w)
    fall = torch.cat([fv, fe, ft])
    nv = fall.detach().cpu().numpy().astype(np.float64)
    n, ne = cx['n'], len(cx['edges'])

    st = SimplexTree()
    st.insert_batch(np.arange(n, dtype=np.int32).reshape(1, -1), nv[:n])
    if ne:
        st.insert_batch(cx['edges'].T.astype(np.int32), nv[n:n + ne])
    if len(cx['tris']):
        st.insert_batch(cx['tris'].T.astype(np.int32), nv[n + ne:])
    st.compute_persistence(persistence_dim_max=True)

    idx = _index_maps(cx)
    bi = {0: [], 1: []}
    di = {0: [], 1: []}
    for birth_s, death_s in st.persistence_pairs():
        d = len(birth_s) - 1
        if d > 1:
            continue
        bi[d].append(idx[tuple(sorted(int(x) for x in birth_s))])
        di[d].append(idx[tuple(sorted(int(x) for x in death_s))] if len(death_s) else -1)

    res = []
    for d in (0, 1):
        if bi[d]:
            b = fall[torch.as_tensor(bi[d], device=w.device)]
            dj = torch.as_tensor(di[d], device=w.device)
            fin = dj >= 0
            dd = torch.full_like(b, cap)
            if fin.any():
                dd = torch.where(fin, fall[dj.clamp(min=0)], dd)
            P = torch.stack([b, torch.clamp(dd, max=cap)], 1)
            life = (P[:, 1] - P[:, 0]).detach()
            P = P[torch.argsort(life, descending=True)[:max_pairs]]
        else:
            P = w.new_zeros((0, 2))
        if len(P) < max_pairs:
            P = torch.cat([P, w.new_zeros((max_pairs - len(P), 2))])
        res.append(P)
    return res[0], res[1]


class LipschitzMLP(nn.Module):
    """MLP whose Lipschitz constant is controlled by spectral normalisation."""

    def __init__(self, d_in, hidden=64, depth=2, scale=1.0, constrained=True):
        super().__init__()
        # constrained=False drops spectral normalisation: same capacity, no Lipschitz
        # control, hence no Theorem 2. It exists so the cost of the guarantee can be
        # measured instead of asserted.
        sn = (lambda m: nn.utils.parametrizations.spectral_norm(m)) if constrained else (lambda m: m)
        self.constrained = constrained
        layers, d = [], d_in
        for _ in range(depth):
            layers += [sn(nn.Linear(d, hidden)), nn.LeakyReLU(0.1)]
            d = hidden
        layers += [sn(nn.Linear(d, 1))]
        self.net = nn.Sequential(*layers)
        self.scale = scale

    def forward(self, z):
        return self.scale * torch.sigmoid(self.net(z)).squeeze(-1)

    def lipschitz_bound(self):
        """Product of spectral norms times the 1/4 factor from the sigmoid."""
        b = 1.0
        for m in self.net:
            if isinstance(m, nn.Linear):
                b *= float(torch.linalg.matrix_norm(m.weight, 2))
        return 0.25 * self.scale * b
