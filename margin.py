"""Measure the margin below which the sparse Delaunay complex cannot change.

The triangulation is decided by the signs of the in-circle predicate on the four points
of each interior edge and its two opposite vertices (Delaunay's lemma). Writing
u = a - d, v = b - d, w = c - d, the predicate is

    I(a,b,c,d) = |u|^2 (v x w) - |v|^2 (u x w) + |w|^2 (u x v),

a polynomial whose gradient with respect to the four points is bounded by 24 D^3 when
all of them lie within distance D of one another. Moving every cell by at most eps
therefore leaves every sign intact once eps < |I| / (24 D^3), and the T-restriction
survives once eps < |edge length - T| / 2.
"""
import os, sys, json, pickle
import numpy as np
from scipy.spatial import Delaunay

T = float(os.environ.get('TDA_T', 100.0))
G = os.environ.get('TDA_DATA', 'data') + '/graphs'
RES = os.environ.get('TDA_RES', 'results')


def incircle_margin(X):
    """Smallest cell displacement that could flip a Delaunay predicate, and the
    smallest that could add or drop an edge at the cutoff T."""
    tri = Delaunay(X)
    S, N = tri.simplices, tri.neighbors
    vals = []
    for t in range(len(S)):
        for k in range(3):
            nb = N[t, k]
            if nb < 0 or nb < t:
                continue                      # hull edge, or the pair already seen
            a, b, c = X[S[t]]
            d = X[S[nb][np.where(N[nb] == t)[0][0]]]
            u, v, w = a - d, b - d, c - d
            cross = lambda p, q: p[0] * q[1] - p[1] * q[0]
            I = (u @ u) * cross(v, w) - (v @ v) * cross(u, w) + (w @ w) * cross(u, v)
            nu, nv, nw = np.linalg.norm(u), np.linalg.norm(v), np.linalg.norm(w)
            ga = 2 * nu * abs(cross(v, w)) + nv * nv * nw + nw * nw * nv
            gb = 2 * nv * abs(cross(u, w)) + nu * nu * nw + nw * nw * nu
            gc = 2 * nw * abs(cross(u, v)) + nu * nu * nv + nv * nv * nu
            # grad_d is minus the sum of the other three, so moving all four points by eps
            # changes I by at most 2 (g_a + g_b + g_c) eps: the same bound as 24 D^3 but
            # evaluated on the tuple at hand, about five times tighter in practice
            L = 2.0 * (ga + gb + gc)
            if L > 0:
                vals.append(abs(I) / L)
    e = np.sqrt(((X[tri.simplices[:, [0, 1, 2]]] -
                  X[tri.simplices[:, [1, 2, 0]]]) ** 2).sum(-1)).ravel()
    gap = np.abs(e - T)
    return np.array(vals), float(gap.min() / 2.0)


def main():
    cohort = sys.argv[1] if len(sys.argv) > 1 else 'danenberg'
    blob = pickle.load(open(os.path.join(G, cohort + '_graphs.pkl'), 'rb'))
    recs = blob['records'] if isinstance(blob, dict) and 'records' in blob else blob
    per_image_min, gaps, pooled, n_tests = [], [], [], 0
    for i, r in enumerate(recs):
        X = np.asarray(r['X'], dtype=float)
        if len(X) < 4:
            continue
        v, b = incircle_margin(X)
        per_image_min.append(float(v.min())); gaps.append(b)
        pooled.append(v); n_tests += len(v)
        if i % 200 == 0:
            print(i, len(recs), v.min(), np.median(v), b, flush=True)
    pooled = np.concatenate(pooled)
    per_image_min, gaps = np.array(per_image_min), np.array(gaps)
    out = dict(cohort=cohort, n_images=len(gaps), T=T, n_predicates=int(n_tests),
               tests_per_image=float(n_tests / max(1, len(gaps))),
               # over all predicates in the cohort, not the per-image minimum
               pred_median=float(np.median(pooled)), pred_q01=float(np.quantile(pooled, 0.01)),
               pred_q10=float(np.quantile(pooled, 0.10)), pred_q90=float(np.quantile(pooled, 0.90)),
               image_min_median=float(np.median(per_image_min)),
               image_min_max=float(per_image_min.max()),
               cutoff_median=float(np.median(gaps)), cutoff_min=float(gaps.min()))
    for eps in (0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
        # share of predicates a displacement of eps could flip, and of images with none
        out['frac_flip_at_%g' % eps] = float((pooled < eps).mean())
        out['frac_images_intact_at_%g' % eps] = float((per_image_min > eps).mean())
    json.dump(out, open(os.path.join(RES, 'margin_%s.json' % cohort), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
