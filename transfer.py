"""Train on one cohort, test on the other, on an endpoint both cohorts record.

No refitting, no recalibration and no target-cohort label is seen at any point: the model is
fit on every Danenberg image with the endpoint and applied once to every Jackson core. The
two cohorts are put on the shared lineage vocabulary of coarse.py first, so the input has the
same meaning on both sides.
"""
import os, sys, json, pickle, time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import coarse
import clinical
from train_learned import Model, T, EPOCHS, LR, RES, rank_norm

G = os.environ.get('TDA_DATA', 'data') + '/graphs'
BASEL = (os.environ.get('TDA_DATA', 'data') + '/jackson/Data_publication/BaselTMA/'
         'Basel_PatientMetadata.csv')
N_BOOT = 2000


def load(cohort):
    blob = pickle.load(open(os.path.join(G, cohort + '_graphs.pkl'), 'rb'))
    return coarse.coarsen(blob, cohort)


def basel_labels(recs):
    md = pd.read_csv(BASEL)
    key = next(c for c in md.columns if c.lower() in ('core', 'filename_fullstack'))
    md['_k'] = md[key].astype(str)
    er = md.set_index('_k')['ERStatus'].astype(str).str.strip().str.lower()
    gr = pd.to_numeric(md.set_index('_k').get('grade'), errors='coerce')
    pat = md.set_index('_k')['PID'].astype(str) if 'PID' in md.columns else None
    out = {}
    ks = [str(r['image']) for r in recs]
    hit = sum(k in er.index for k in ks)
    out['_matched'] = hit
    e = np.array([1.0 if er.get(k, '') == 'positive' else
                  (0.0 if er.get(k, '') == 'negative' else np.nan) for k in ks])
    g = np.array([gr.get(k, np.nan) for k in ks], dtype=float)
    out['ER status'] = e
    out['Grade 3'] = np.where(np.isnan(g), np.nan, (g >= 3).astype(float))
    out['_patient'] = np.array([pat.get(k, k) if pat is not None else k for k in ks],
                               dtype=object)
    return out


def fit_predict(tr_recs, y_tr, te_recs, variant, seed, d_in, ref_tr, ref_te):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Model(d_in, 0, variant=variant)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lossf = torch.nn.BCEWithLogitsLoss()
    yt = torch.tensor(y_tr, dtype=torch.float32)
    order = np.arange(len(tr_recs))
    for ep in range(EPOCHS):
        np.random.shuffle(order)
        for i in order:
            opt.zero_grad()
            o = m(tr_recs[i], None, ref_tr)
            lossf(o.squeeze(), yt[i]).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        return rank_norm(np.array([float(torch.sigmoid(m(r, None, ref_te)))
                                   for r in te_recs]))


def main():
    endpoint = sys.argv[1]
    variants = sys.argv[2].split(',') if len(sys.argv) > 2 else ['learned_nc']
    seeds = [0, 1, 2, 3, 4]
    src, dst = load('danenberg'), load('basel')
    if os.environ.get('TDA_QNORM'):  # [tp]
        import qnorm
        qnorm.align(src, dst)
    d_in = src['meta']['d_in']
    for b in (src, dst):
        for r in b['records']:
            r['Zt'] = torch.tensor(r['Z'], dtype=torch.float32)
            r['cx'] = dict(n=len(r['X']), edges=r['edges'].astype(int),
                           tris=r['tris'].astype(int), e_len=r['e_len'].astype(float),
                           t_diam=r['t_diam'].astype(float))
    L = coarse.LINEAGES
    ref = dict(onehot=L.index('Endothelial'), dist=len(L) + L.index('Endothelial'))

    ys = clinical.clinical_targets(np.array([str(r['patient']) for r in src['records']],
                                            dtype=object))[endpoint]
    ok = ~np.isnan(ys)
    tr = [r for r, k in zip(src['records'], ok) if k]
    y_tr = ys[ok]

    lab = basel_labels(dst['records'])
    yd = lab[endpoint]
    ok2 = ~np.isnan(yd)
    te = [r for r, k in zip(dst['records'], ok2) if k]
    y_te = yd[ok2].astype(int)
    pat_te = lab['_patient'][ok2]
    print('matched cores', lab['_matched'], 'train', len(tr), 'test', len(te),
          'test pos', int(y_te.sum()), flush=True)
    if len(te) < 20 or y_te.sum() in (0, len(y_te)):
        print('ENDPOINT UNUSABLE', endpoint); return

    res = {}
    for v in variants:
        t0 = time.time()
        ref_v = ref
        if v.startswith('fixed_tuned'):
            # the reference is chosen on the source cohort only, as in train_learned.py
            from train_learned import choose_reference
            j = int(choose_reference(tr, np.arange(len(tr)), y_tr.astype(int), len(L)))
            ref_v = dict(onehot=j, dist=len(L) + j)
            print('tuned reference', L[j], flush=True)
        R = np.stack([fit_predict(tr, y_tr, te, v, s, d_in, ref_v, ref_v) for s in seeds])
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
    name = endpoint.replace(' ', '_') + os.environ.get('TDA_TAG', '')  # [tp]
    json.dump(res, open(os.path.join(RES, 'transfer_%s.json' % name), 'w'), indent=1)


if __name__ == '__main__':
    main()
