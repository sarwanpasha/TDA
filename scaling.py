"""Cost of one differentiable persistence pass as the number of cells grows.

python scaling.py

For n in a geometric grid up to 10,000 cells, build the Delaunay flag complex and the
weighted alpha complex on real cell clouds (the largest Danenberg images, tiled when an
image has fewer cells than n) and time, on one core, the complex construction and the
forward and forward-plus-backward persistence pass with a random Lipschitz weight.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import torch
import difftda as dt
import dtalpha as dta
from train_learned import PROC, RES, T

torch.set_num_threads(1)
NS = (500, 1000, 2000, 4000, 8000, 10000)
REPS = 3


def clouds():
    cells = pd.read_parquet(os.path.join(PROC, 'danenberg', 'cells.parquet'),
                            columns=['ImageNumber', 'x', 'y'])
    sizes = cells.groupby('ImageNumber').size().sort_values(ascending=False)
    big = sizes.index[:4]
    return [cells[cells.ImageNumber == i][['x', 'y']].to_numpy(float) for i in big]


def sample(X, n, rng):
    if len(X) >= n:
        return X[rng.choice(len(X), n, replace=False)]
    # tile copies of the image side by side until there are enough cells
    w = np.ptp(X[:, 0]) + 20.0
    parts, k = [], 0
    while sum(len(p) for p in parts) < n:
        parts.append(X + np.array([k * w, 0.0])); k += 1
    Y = np.concatenate(parts)
    return Y[rng.choice(len(Y), n, replace=False)]


def main():
    rng = np.random.default_rng(0)
    Xs = clouds()
    net = dt.LipschitzMLP(8, hidden=64, depth=2, scale=2 * T, constrained=True)
    rows = []
    for n in NS:
        for rep in range(REPS):
            X = sample(Xs[rep % len(Xs)], n, rng)
            Z = torch.tensor(rng.random((n, 8)), dtype=torch.float32)
            t0 = time.perf_counter(); cx = dt.build_complex(X, T); t_build = time.perf_counter() - t0
            cx = dict(n=n, edges=cx['edges'].astype(int), tris=cx['tris'].astype(int),
                      e_len=cx['e_len'].astype(float), t_diam=cx['t_diam'].astype(float))
            w = net(Z)
            t0 = time.perf_counter()
            with torch.no_grad():
                dt.persistence(cx, w.detach(), cap=T, max_pairs=1024)
            t_fwd = time.perf_counter() - t0
            t0 = time.perf_counter()
            w = net(Z); p0, p1 = dt.persistence(cx, w, cap=T, max_pairs=1024)
            (p0.sum() + p1.sum()).backward()
            t_fb = time.perf_counter() - t0
            Xt = torch.tensor(X, dtype=torch.float32)
            t0 = time.perf_counter()
            with torch.no_grad():
                dta.persistence(Xt, net(Z).detach(), cap=T, max_pairs=1024)
            a_fwd = time.perf_counter() - t0
            t0 = time.perf_counter()
            w = net(Z); q0, q1 = dta.persistence(Xt, w, cap=T, max_pairs=1024)
            (q0.sum() + q1.sum()).backward()
            a_fb = time.perf_counter() - t0
            rows.append(dict(n=n, rep=rep, edges=len(cx['edges']), tris=len(cx['tris']),
                             build_s=t_build, delaunay_fwd_s=t_fwd, delaunay_fb_s=t_fb,
                             alpha_fwd_s=a_fwd, alpha_fb_s=a_fb))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, 'scaling_danenberg.csv'), index=False)
    print(df.groupby('n').median(numeric_only=True).to_string(), flush=True)
    print('SCALING_DONE', flush=True)


if __name__ == '__main__':
    main()
