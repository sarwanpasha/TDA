"""Train on the Basel cores, test on the Zurich cores: same instrument, same antibody
panel, same phenotyping, different hospital and different patients.

python transfer_jk.py <endpoint> [variants]

No Zurich label is seen; the model is fit on every Basel core carrying the endpoint and
applied once to every Zurich core. The two cohorts share the metacluster vocabulary, so the
feature is remapped onto the Basel column order and nothing is coarsened.
"""
import os, sys, json, pickle, time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
import jclinical
from train_learned import Model, T, EPOCHS, LR, RES, rank_norm, choose_reference

G = os.environ.get('TDA_DATA', 'data') + '/graphs'
N_BOOT = 2000


def load(cohort):
    return pickle.load(open(os.path.join(G, cohort + '_graphs.pkl'), 'rb'))


def remap(dst, src_vocab):
    """Rewrite the target cohort's feature onto the source vocabulary order."""
    dv = [str(v) for v in dst['meta']['vocab']]
    sv = [str(v) for v in src_vocab]
    Pd, Ps = len(dv), len(sv)
    M = dst['meta']['markers'] if isinstance(dst['meta']['markers'], int) else len(dst['meta']['markers'])
    col = {name: i for i, name in enumerate(dv)}
    missing = [s for s in sv if s not in col]
    lut = np.full(Pd, -1, int)
    for i, name in enumerate(dv):
        if name in sv:
            lut[i] = sv.index(name)
    for r in dst['records']:
        Z = r['Z']
        oh = np.zeros((len(Z), Ps), np.float32)
        dd = np.ones((len(Z), Ps), np.float32)
        for s_i, name in enumerate(sv):
            if name in col:
                oh[:, s_i] = Z[:, col[name]]
                dd[:, s_i] = Z[:, Pd + col[name]]
        r['Z'] = np.hstack([oh, dd, Z[:, 2 * Pd:]]).astype(np.float32)
        ph = np.asarray(r['pheno'], int)
        r['pheno'] = np.where(lut[ph] >= 0, lut[ph], Ps - 1)
    print('remap: %d source types, %d missing in target: %s' % (Ps, len(missing), missing),
          flush=True)


def fit_predict(tr_recs, y_tr, te_recs, variant, seed, d_in, ref):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Model(d_in, 0, variant=variant)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss()
    yt = torch.tensor(y_tr, dtype=torch.float32)
    order = np.arange(len(tr_recs))
    for ep in range(EPOCHS):
        np.random.shuffle(order)
        opt.zero_grad()
        for j, i in enumerate(order):
            (lossf(m(tr_recs[i], None, ref).squeeze(), yt[i]) / 8).backward()
            if (j + 1) % 8 == 0:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
    m.eval()
    with torch.no_grad():
        return rank_norm(np.array([float(torch.sigmoid(m(r, None, ref))) for r in te_recs]))


def main():
    endpoint = sys.argv[1]
    variants = sys.argv[2].split(',') if len(sys.argv) > 2 else \
        ['learned_nc', 'fixed_tuned_nc', 'fixed_cross_nc', 'zero_nc']
    seeds = [0, 1, 2, 3, 4]
    src, dst = load('basel'), load('zurich')
    remap(dst, src['meta']['vocab'])
    d_in = src['meta']['d_in']
    P = len(src['meta']['vocab'])
    for b in (src, dst):
        for r in b['records']:
            r['Zt'] = torch.tensor(r['Z'], dtype=torch.float32)
            r['cx'] = dict(n=len(r['X']), edges=r['edges'].astype(int),
                           tris=r['tris'].astype(int), e_len=r['e_len'].astype(float),
                           t_diam=r['t_diam'].astype(float))
    vocab = [str(v) for v in src['meta']['vocab']]
    ref_idx = next((i for i, v in enumerate(vocab) if 'endothelial' in v.lower()), 0)
    ref = dict(onehot=ref_idx, dist=P + ref_idx)

    ys = jclinical.targets('basel', src['records'])[endpoint]
    ok = ~np.isnan(ys)
    tr = [r for r, k in zip(src['records'], ok) if k]
    y_tr = ys[ok].astype(int)
    yd = jclinical.targets('zurich', dst['records'])[endpoint]
    ok2 = ~np.isnan(yd)
    te = [r for r, k in zip(dst['records'], ok2) if k]
    y_te = yd[ok2].astype(int)
    pat_te = np.array([str(r['patient']) for r in te])
    print('train', len(tr), 'pos', int(y_tr.sum()), 'test', len(te), 'pos', int(y_te.sum()),
          flush=True)
    if len(te) < 20 or y_te.sum() in (0, len(y_te)) or y_tr.sum() in (0, len(y_tr)):
        print('ENDPOINT UNUSABLE', endpoint); print('TRANSFER_JK_DONE'); return

    res = {}
    for v in variants:
        t0 = time.time()
        use_ref = ref
        if v.startswith('fixed_tuned'):
            j = choose_reference(tr, np.arange(len(tr)), y_tr, P)
            use_ref = dict(onehot=j, dist=P + j)
            print('tuned reference on source:', vocab[j], flush=True)
        R = np.stack([fit_predict(tr, y_tr, te, v, s, d_in, use_ref) for s in seeds])
        r = R.mean(0)
        auc = roc_auc_score(y_te, r)
        rng = np.random.default_rng(0)
        pats = pd.unique(pat_te)
        boot = []
        for _ in range(N_BOOT):
            pick = rng.choice(pats, len(pats), replace=True)
            idx = np.concatenate([np.where(pat_te == p)[0] for p in pick])
            if len(np.unique(y_te[idx])) == 2:
                boot.append(roc_auc_score(y_te[idx], r[idx]))
        res[v] = dict(auc=float(auc), lo=float(np.quantile(boot, .025)),
                      hi=float(np.quantile(boot, .975)),
                      sd_seeds=float(np.std([roc_auc_score(y_te, x) for x in R])),
                      n_test=len(te), n_pos=int(y_te.sum()), n_train=len(tr),
                      secs=round(time.time() - t0, 1))
        print(v, res[v], flush=True)
    name = endpoint.replace(' ', '_')
    json.dump(res, open(os.path.join(RES, 'transfer_jk_%s.json' % name), 'w'), indent=1)
    print('TRANSFER_JK_DONE', flush=True)


if __name__ == '__main__':
    main()
