"""Weighted alpha complex for the learned filtration.

The complex the paper's main experiments use is a Delaunay flag complex filtered by
max edge length.  Its persistence has no modulus of continuity: an arbitrarily small
displacement can flip a co-circular quadrilateral and move a death value by a fixed
amount.  The weighted alpha complex does not have this defect, because its persistence
is isomorphic (nerve lemma) to that of the union of balls

    U(t) = union_i B(x_i, sqrt(t - w_i^2)),   t >= w_i^2,

which is defined without reference to any triangulation.  Reparametrised by s = sqrt(t)
the vertex i still enters at exactly s = w_i, so the filtration is directly comparable
with the weighted Vietoris-Rips one.

Everything here is vectorised; the only per-image geometry is one convex hull.
"""
import numpy as np


def regular_triangulation(X, w):
    """Regular (weighted Delaunay) triangulation: lower hull of the lift
    z_i = |x_i|^2 + w_i^2.  Returns an (m, 3) array of vertex indices."""
    from scipy.spatial import ConvexHull
    z = (X ** 2).sum(1) + w ** 2
    P = np.column_stack([X, z]).astype(float)
    hull = ConvexHull(P)
    keep = hull.equations[:, 2] < -1e-12
    return np.sort(hull.simplices[keep], axis=1)


def _pair_value(X, w2, I, J):
    """Squared alpha value of each edge (I[k], J[k]) and the point at which the two
    balls first touch.  Vectorised closed form."""
    small = np.where(w2[I] <= w2[J], I, J)
    big = np.where(w2[I] <= w2[J], J, I)
    d = np.linalg.norm(X[big] - X[small], axis=1)
    a, b = w2[small], w2[big]
    swallow = np.sqrt(np.maximum(b - a, 0)) >= d           # one ball already covers
    dm = np.maximum(d, 1e-12)
    r = (dm * dm + b - a) / (2 * dm)
    t = np.where(swallow, b, a + r * r)
    frac = np.where(swallow, 1.0, r / dm)[:, None]   # swallowed: centre is the big site
    c = X[small] + (X[big] - X[small]) * frac
    return t, c


def _tri_value(X, w2, tri, pair_t, pair_c):
    """Squared alpha value of each triangle: min_c max_i (|c-x_i|^2 + w_i^2).  The
    minimiser is the radical centre, or a two-site solution whose ball already covers
    the third site; the value is the smallest t at which the three balls meet, so it is
    the minimum over whichever of those candidates is feasible."""
    i, j, k = tri[:, 0], tri[:, 1], tri[:, 2]
    A = np.stack([2 * (X[j] - X[i]), 2 * (X[k] - X[i])], axis=1)     # (m, 2, 2)
    rhs = np.stack([(X[j] ** 2).sum(1) - (X[i] ** 2).sum(1) + w2[j] - w2[i],
                    (X[k] ** 2).sum(1) - (X[i] ** 2).sum(1) + w2[k] - w2[i]], axis=1)
    det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
    det = np.where(np.abs(det) < 1e-14, 1e-14, det)
    cx = (rhs[:, 0] * A[:, 1, 1] - rhs[:, 1] * A[:, 0, 1]) / det
    cy = (A[:, 0, 0] * rhs[:, 1] - A[:, 1, 0] * rhs[:, 0]) / det
    c = np.stack([cx, cy], axis=1)
    t_rad = ((c - X[i]) ** 2).sum(1) + w2[i]
    best = t_rad
    opp = (k, j, i)                       # vertex opposite edge 0,1,2 of the triangle
    for e in range(3):
        te, ce, o = pair_t[:, e], pair_c[:, e], opp[e]
        covered = ((ce - X[o]) ** 2).sum(1) + w2[o] <= te + 1e-12
        best = np.where(covered, np.minimum(best, te), best)
    return best


def alpha_complex(X, w, cap=np.inf):
    """Returns (verts, edges, tris, s_vert, s_edge, s_tri) with s = sqrt(alpha value),
    every array restricted to s <= cap, and faces never entering after their cofaces."""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    w2 = w ** 2
    tri = regular_triangulation(X, w)
    e = np.concatenate([tri[:, [0, 1]], tri[:, [0, 2]], tri[:, [1, 2]]], axis=0)
    e = np.unique(np.sort(e, axis=1), axis=0)
    t_e, c_e = _pair_value(X, w2, e[:, 0], e[:, 1])
    key = {(a, b): n for n, (a, b) in enumerate(map(tuple, e))}
    tri_e = np.array([[key[(t[0], t[1])], key[(t[0], t[2])], key[(t[1], t[2])]]
                      for t in tri])
    t_t = _tri_value(X, w2, tri, t_e[tri_e], c_e[tri_e])
    # monotonicity: an edge cannot enter after a triangle that contains it
    for c in range(3):
        np.minimum.at(t_e, tri_e[:, c], t_t)
    # a site whose power cell is empty is redundant: its ball never reaches the
    # boundary of the union, and the regular triangulation drops it
    used = np.unique(tri)
    s_v, s_e, s_t = w[used], np.sqrt(np.maximum(t_e, 0)), np.sqrt(np.maximum(t_t, 0))
    kv, ke, kt = s_v <= cap, s_e <= cap, s_t <= cap
    return (used[kv], e[ke], tri[kt], s_v[kv], s_e[ke], s_t[kt])


# ---------------------------------------------------------------- torch side

def _torch_values(Xt, wt, e, tri, tri_e):
    """The same closed forms as above, differentiable in the vertex weights."""
    import torch
    w2 = wt ** 2
    I, J = e[:, 0], e[:, 1]
    swap = (w2[I] > w2[J])
    S = torch.where(swap, J, I)
    B = torch.where(swap, I, J)
    d = torch.linalg.norm(Xt[B] - Xt[S], dim=1).clamp_min(1e-12)
    a, b = w2[S], w2[B]
    r = (d * d + b - a) / (2 * d)
    swallow = (b - a).clamp_min(0).sqrt() >= d
    t_e = torch.where(swallow, b, a + r * r)

    i, j, k = tri[:, 0], tri[:, 1], tri[:, 2]
    a11 = 2 * (Xt[j, 0] - Xt[i, 0]); a12 = 2 * (Xt[j, 1] - Xt[i, 1])
    a21 = 2 * (Xt[k, 0] - Xt[i, 0]); a22 = 2 * (Xt[k, 1] - Xt[i, 1])
    r1 = (Xt[j] ** 2).sum(1) - (Xt[i] ** 2).sum(1) + w2[j] - w2[i]
    r2 = (Xt[k] ** 2).sum(1) - (Xt[i] ** 2).sum(1) + w2[k] - w2[i]
    det = a11 * a22 - a12 * a21
    det = torch.where(det.abs() < 1e-14, torch.full_like(det, 1e-14), det)
    cx = (r1 * a22 - r2 * a12) / det
    cy = (a11 * r2 - a21 * r1) / det
    c = torch.stack([cx, cy], 1)
    t_t = ((c - Xt[i]) ** 2).sum(1) + w2[i]

    # a two-site solution that already covers the third site gives a smaller value
    small = torch.where(swap, J, I)
    big = torch.where(swap, I, J)
    frac = torch.where(swallow, torch.ones_like(r), r / d).unsqueeze(1)
    c_e = Xt[small] + (Xt[big] - Xt[small]) * frac
    opp = (k, j, i)
    for m in range(3):
        idx = tri_e[:, m]
        ce, te, o = c_e[idx], t_e[idx], opp[m]
        covered = ((ce - Xt[o]) ** 2).sum(1) + w2[o] <= te + 1e-12
        t_t = torch.where(covered, torch.minimum(t_t, te), t_t)

    # a face never enters after a coface
    for m in range(3):
        t_e = t_e.scatter_reduce(0, tri_e[:, m], t_t, reduce='amin')
    return t_e.clamp_min(0).sqrt(), t_t.clamp_min(0).sqrt()


def persistence(Xt, wt, cap, max_pairs=1024):
    """Degree-0 and degree-1 diagrams of the weighted alpha filtration, differentiable
    in wt.  The complex and the birth/death pairing are computed on detached values and
    are locally constant, exactly as for the Vietoris-Rips filtration."""
    import torch, gudhi
    X = Xt.detach().cpu().numpy().astype(float)
    w = wt.detach().cpu().numpy().astype(float)
    used, e, tri, s_v, s_e, s_t = alpha_complex(X, w, cap=cap)
    if len(used) == 0:
        z = torch.zeros((max_pairs, 2), dtype=wt.dtype, device=wt.device)
        return z, z.clone()
    remap = -np.ones(len(X), dtype=np.int64)
    remap[used] = np.arange(len(used))
    e_l, tri_l = remap[e], remap[tri]
    dev = wt.device
    # an image can have no triangles, or no edges, under the cap; numpy gives those a
    # 1-D empty array, which the column indexing below cannot take
    e_t = torch.as_tensor(np.asarray(e, dtype=np.int64).reshape(-1, 2),
                          dtype=torch.long, device=dev)
    tri_t = torch.as_tensor(np.asarray(tri, dtype=np.int64).reshape(-1, 3),
                            dtype=torch.long, device=dev)
    key = {(a, b): n for n, (a, b) in enumerate(map(tuple, e))}
    tri_e = torch.as_tensor(
        np.array([[key[(t[0], t[1])], key[(t[0], t[2])], key[(t[1], t[2])]]
                                      for t in tri], dtype=np.int64).reshape(-1, 3),
        dtype=torch.long, device=dev)
    te, tt = _torch_values(Xt, wt, e_t, tri_t, tri_e)
    vals = torch.cat([wt[torch.as_tensor(used, dtype=torch.long, device=dev)], te, tt])
    dv = vals.detach().cpu().numpy()

    st = gudhi.SimplexTree()
    for n, v in enumerate(used):
        st.insert([int(remap[v])], dv[n])
    off = len(used)
    for n, (a, b) in enumerate(e_l):
        st.insert([int(a), int(b)], dv[off + n])
    off += len(e)
    for n, (a, b, c) in enumerate(tri_l):
        st.insert([int(a), int(b), int(c)], dv[off + n])
    st.compute_persistence(persistence_dim_max=False)

    sid = {}
    for n, v in enumerate(used):
        sid[(int(remap[v]),)] = n
    for n, ab in enumerate(e_l):
        sid[tuple(sorted(int(x) for x in ab))] = len(used) + n
    for n, abc in enumerate(tri_l):
        sid[tuple(sorted(int(x) for x in abc))] = len(used) + len(e) + n

    out = []
    for dim in (0, 1):
        bi, di = [], []
        for bs, ds in st.persistence_pairs():
            if len(bs) - 1 != dim or not ds:
                continue
            b, d = sid[tuple(sorted(bs))], sid[tuple(sorted(ds))]
            if dv[d] - dv[b] > 1e-12:
                bi.append(b); di.append(d)
        P = torch.zeros((max_pairs, 2), dtype=wt.dtype, device=dev)
        if bi:
            bi, di = bi[:max_pairs], di[:max_pairs]
            P[:len(bi), 0] = vals[torch.as_tensor(bi, dtype=torch.long, device=dev)]
            P[:len(bi), 1] = vals[torch.as_tensor(di, dtype=torch.long, device=dev)]
        out.append(P)
    return out[0], out[1]
