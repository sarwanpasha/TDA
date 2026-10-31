"""Learned Lipschitz filtration: end-to-end training against a spatial target.

python train_learned.py <cohort> <target> <seed> [variant]

variant in {learned, fixed_cross, zero, comp_only} selects the vertex-weight function:
  learned      w = w_theta(z)   (spectrally normalised MLP, the proposed method)
  fixed_cross  w = distance to the reference phenotype, with the reference cells
               themselves pushed above the cutoff (the hand-designed cross-phenotype
               filtration of Definition 1, as a control)
  fixed_tuned  as fixed_cross, but the reference phenotype is selected per fold on the
               training split rather than fixed to endothelium -- the hand-designed
               filtration with its one free choice tuned, which is the baseline a learned
               filtration has to beat to be worth anything
  zero         w = 0            (plain geometric Vietoris-Rips)
  comp_only    no topology at all (composition MLP)

A "_nc" suffix (no composition) drops the composition vector from the classifier
head, so the topological block alone has to carry the signal. With composition
present every variant inherits its ~0.82 AUC, which masks what the filtration
itself contributes.
"""
import os, sys, json, pickle, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import difftda as dt
import frac, featmask  # [tp]

GRAPHS = os.environ.get('TDA_DATA', 'data') + '/graphs'
PROC = os.environ.get('TDA_DATA', 'data') + '/proc'
RES = os.environ.get('TDA_RES', 'results')
T = float(os.environ.get('TDA_T', 100.0))
EPOCHS = 20
LR = 1e-3
VEC = os.environ.get('TDA_VEC', 'lite')
# 'lip' keeps every pair, so the top-K selection never intervenes
K0 = int(os.environ.get('TDA_KLIP', 1024))
MAX_PAIRS = K0 if VEC == 'lip' else int(os.environ.get('TDA_K', 48))
REF_PHENOTYPE = 'endothelial'


class PersLayLite(nn.Module):
    """Permutation-invariant, differentiable vectorisation of a persistence diagram.

    Coordinates are divided by the cut-off T and the pooling is a mean, so the
    output does not scale with the number of persistence pairs; a LayerNorm then
    puts the block on the same footing as the composition vector it is
    concatenated with. Without this the topological block is orders of magnitude
    larger than the composition block and destabilises the classifier head.
    """

    def __init__(self, out=32):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, out))
        self.norm = nn.LayerNorm(out)

    def forward(self, P):
        life = (P[:, 1] - P[:, 0]).unsqueeze(1) / T
        feats = self.f(torch.cat([P / T, life], 1))
        return self.norm((feats * torch.tanh(life)).mean(0))


class PersLayLip(nn.Module):
    """The vectorisation Corollary 1 actually covers.

    Two changes from PersLayLite: the sum runs over every persistence pair rather than
    the K most persistent ones, so no discontinuous selection intervenes, and the
    normalisation is a fixed divisor followed by no LayerNorm, so the map is a scaled
    sum of terms that vanish on the diagonal. Both are what the L_1 stability of
    persistence (Cohen-Steiner et al., 2010) needs.
    """

    def __init__(self, out=32, k0=None):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, out))
        self.k0 = float(K0 if k0 is None else k0)

    def forward(self, P):
        life = (P[:, 1] - P[:, 0]).unsqueeze(1) / T
        feats = self.f(torch.cat([P / T, life], 1))
        return (feats * torch.tanh(life)).sum(0) / self.k0


class PersLayGauss(nn.Module):  # [tp]
    """PersLay with a Gaussian point transformation (Carriere et al., 2020).

    K learned centres in the (birth, death, life)/T cube with learned widths, a learned
    non-negative weight per point, and a sum pooled with the fixed divisor K0. A richer
    vectoriser than the two above, used as a baseline head on the fixed filtrations; it is
    not the map the Lipschitz statement covers.
    """

    def __init__(self, out=32, k0=None, n_gauss=32):
        super().__init__()
        self.c = nn.Parameter(torch.rand(n_gauss, 3))
        self.ls = nn.Parameter(torch.full((n_gauss,), -1.5))
        self.q = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1), nn.Softplus())
        self.lin = nn.Linear(n_gauss, out)
        self.k0 = float(K0 if k0 is None else k0)

    def forward(self, P):
        life = (P[:, 1] - P[:, 0]).unsqueeze(1) / T
        u = torch.cat([P / T, life], 1)
        d2 = ((u.unsqueeze(1) - self.c.unsqueeze(0)) ** 2).sum(-1)
        g = torch.exp(-d2 / (2 * torch.exp(self.ls) ** 2 + 1e-6))
        w = self.q(u) * torch.tanh(life)
        return self.lin((g * w).sum(0) / self.k0)
class Model(nn.Module):
    def __init__(self, d_in, d_comp, variant='learned', hidden=64):
        super().__init__()
        self.use_comp = not variant.endswith('_nc')
        self.variant = variant[:-3] if variant.endswith('_nc') else variant
        if self.variant == 'fixed_tuned':
            self.variant = 'fixed_cross'   # same weight form; only `ref` differs
        # learned_unc: identical network without spectral normalisation, so the accuracy
        #   cost of the Lipschitz constraint can be read off directly.
        # learned_s4: output scale 4T, the scale at which Proposition 1's containment
        #   actually holds; 2T (the default) is half of what the cross-phenotype target needs.
        scale, constrained = 2 * T, True
        if self.variant == 'learned_unc':
            constrained = False
            self.variant = 'learned'
        elif self.variant == 'learned_s4':
            scale = 4 * T
            self.variant = 'learned'
        self.filt = dt.LipschitzMLP(d_in, hidden=hidden, depth=2, scale=scale,
                                    constrained=constrained)
        V = PersLayGauss if os.environ.get('TDA_PL') else (PersLayLip if VEC == 'lip' else PersLayLite)  # [tp]
        self.v0, self.v1 = V(), V()
        d = (0 if self.variant == 'comp_only' else 64) + (d_comp if self.use_comp else 0)
        self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))

    def weights(self, Z, ref):
        if self.variant == 'learned':
            return self.filt(Z)
        if self.variant == 'zero':
            return torch.zeros(len(Z), dtype=Z.dtype, device=Z.device)
        # fixed_cross and fixed_tuned differ only in how `ref` was chosen
        # fixed_cross: weight every non-reference cell by its distance to the reference
        # phenotype and push the reference cells themselves above the cutoff, which is
        # Definition 1 transported to the full cell cloud.
        d = Z[:, ref['dist']] * T
        is_ref = Z[:, ref['onehot']] > 0.5
        return torch.where(is_ref, torch.full_like(d, 2 * T), d)

    def forward(self, rec, comp, ref):
        if self.variant == 'comp_only':
            return self.head(comp)
        w = self.weights(rec['Zt'], ref)
        p0, p1 = dt.persistence(rec['cx'], w, cap=T, max_pairs=MAX_PAIRS)
        parts = [self.v0(p0), self.v1(p1)] + ([comp] if self.use_comp else [])
        return self.head(torch.cat(parts))


def _maybe_coarsen(blob, cohort):
    if os.environ.get('TDA_COARSE'):
        import coarse
        return coarse.coarsen(blob, cohort)
    return blob


def load_cohort(cohort):
    """Load the graph records for a cohort, applying the analysis-set filter.

    For Danenberg we keep only images in which community detection found at least one
    TME structure. Those images are not scored as negatives anywhere else in the paper
    (cf. datasets.danenberg(tme_annotated_only=True)), so including them here would put
    the learned filtration and the fixed-feature baselines on different cohorts.
    """
    blob = pickle.load(open(os.path.join(GRAPHS, cohort + '_graphs' + os.environ.get('TDA_SUF', '') + '.pkl'), 'rb'))
    blob = _maybe_coarsen(blob, cohort)
    meta, recs = blob['meta'], blob['records']
    if cohort == 'danenberg':
        t = pd.read_parquet(os.path.join(PROC, 'danenberg',
                                         'tme_structures.parquet')).set_index('ImageNumber')
        f = [c for c in t.columns if c.startswith('frac::')]
        ann = set(t.index[t[f].sum(1) > 0])
        recs = [r for r in recs if int(r['image']) in ann]
    return meta, recs


def image_targets(cohort, records, vocab):
    # the CODEX colorectal cohort carries one patient-level histopathological
    # grouping, assigned from H&E and not from the cell graph
    if cohort == 'keren':  # [tp]
        return {k: np.array([r['ys'][k] for r in records], dtype=float) for k in records[0]['ys']}
    if cohort == 'codex':
        import numpy as _np
        return {'CLR vs DII': _np.array([r['y'] for r in records], dtype=float)}
    if os.environ.get('TDA_TARGETS') == 'clinical' and cohort in ('basel', 'zurich'):  # [tp]
        import jclinical
        return jclinical.targets(cohort, records)
    if os.environ.get('TDA_TARGETS') == 'clinical':
        import clinical
        return clinical.clinical_targets(
            np.array([str(r['patient']) for r in records], dtype=object))
    imgs = [r['image'] for r in records]
    if cohort == 'danenberg':
        tme = pd.read_parquet(os.path.join(PROC, 'danenberg',
                                           'tme_structures.parquet')).set_index('ImageNumber')
        return {c.replace('frac::', ''): (pd.to_numeric(tme[c].reindex(imgs), errors='coerce')
                                          .fillna(0).to_numpy() >= 0.05).astype(int)
                for c in tme.columns if c.startswith('frac::')}
    # Jackson has no community-detected structures: use the proxy vascular target,
    # images in the top quartile of endothelial-cell fraction.
    vi = [i for i, v in enumerate(vocab) if 'endothelial' in str(v).lower()]
    frac = np.array([np.isin(r['pheno'], vi).mean() for r in records])
    return {'Vascular proxy': (frac >= np.quantile(frac, 0.75)).astype(int)}


def choose_reference(recs, tr, y, P, device='cpu'):
    """Pick the reference phenotype for the hand-designed filtration, using training data.

    For each candidate phenotype p the per-image mean of the normalised distance-to-p column
    is a cheap scalar summary of "how close this image sits to p"; we take the phenotype
    whose summary separates the training labels best. This never touches the test fold, so it
    is a legitimately tuned baseline rather than an oracle.
    """
    best, best_score = 0, -1.0
    ytr = y[tr]
    if len(np.unique(ytr)) < 2:
        return 0
    for p in range(P):
        v = np.array([float(recs[i]['Z'][:, P + p].mean()) for i in tr])
        try:
            a = roc_auc_score(ytr, v)
        except ValueError:
            continue
        score = abs(a - 0.5)
        if score > best_score:
            best, best_score = p, score
    return best


def rank_norm(p):
    """Map a fold's scores to [0, 1] ranks.

    Each fold trains its own network, so raw probabilities are not comparable
    across folds; pooling them directly corrupts a single out-of-fold AUC. Ranks
    are monotone, so per-fold AUC is unchanged, and pooling becomes meaningful.
    """
    return (rankdata(p) - 0.5) / len(p) if len(p) > 1 else np.zeros(len(p))


def comp_features(records, P):
    C = np.zeros((len(records), P), np.float32)
    for i, r in enumerate(records):
        v = np.bincount(r['pheno'].astype(int), minlength=P).astype(np.float32)
        C[i] = v / max(v.sum(), 1)
    return C


def run_fold(model, tr, te, recs, comp, y, ref, device, epochs=EPOCHS, log=None):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(tr)
        tot = 0.0
        opt.zero_grad()
        for j, i in enumerate(perm):
            logit = model(recs[i], comp[i], ref)
            loss = lossf(logit.squeeze(), torch.tensor(float(y[i]), device=device))
            (loss / 8).backward()
            tot += float(loss)
            if (j + 1) % 8 == 0:
                opt.step()
                opt.zero_grad()
        opt.step()
        opt.zero_grad()
        if log and (ep + 1) % 5 == 0:
            print(f'  epoch {ep+1:2d} loss={tot/len(perm):.4f}', flush=True)
    model.eval()
    out = np.zeros(len(te))
    with torch.no_grad():
        for k, i in enumerate(te):
            out[k] = float(torch.sigmoid(model(recs[i], comp[i], ref)))
    return rank_norm(out)


def main():
    cohort, target, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    variant = sys.argv[4] if len(sys.argv) > 4 else 'learned'
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = 'cpu'

    meta, recs = load_cohort(cohort)
    P = len(meta['vocab'])
    recs = featmask.apply(recs, P)  # [tp]
    Y = image_targets(cohort, recs, meta['vocab'])
    y = np.asarray(Y[target], dtype=float)
    keep = ~np.isnan(y)
    if not keep.all():
        recs = [r for r, k in zip(recs, keep) if k]
        y = y[keep]
    y = y.astype(int)
    comp_np = comp_features(recs, P)
    patients = np.array([str(r['patient']) for r in recs])
    ref_idx = next((i for i, v in enumerate(meta['vocab'])
                    if REF_PHENOTYPE in str(v).lower()), 0)
    ref = dict(onehot=ref_idx, dist=P + ref_idx)
    print(f'{cohort} {target} variant={variant} seed={seed} images={len(recs)} '
          f'positives={int(y.sum())} d_in={meta["d_in"]} '
          f'reference={meta["vocab"][ref_idx]}', flush=True)

    for r in recs:
        r['Zt'] = torch.tensor(r['Z'], dtype=torch.float32, device=device)
        r['cx'] = dict(n=len(r['X']), edges=r['edges'].astype(int), tris=r['tris'].astype(int),
                       e_len=r['e_len'].astype(float), t_diam=r['t_diam'].astype(float))
    comp = [torch.tensor(c, dtype=torch.float32, device=device) for c in comp_np]

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(recs))
    fold_aucs, refs_used = [], []
    t0 = time.time()
    for f, (tr, te) in enumerate(cv.split(comp_np, y, patients)):
        tr = frac.cut(tr, patients, y, seed, f)  # [tp]
        fold_ref = ref
        if variant.startswith('fixed_tuned'):
            j = choose_reference(recs, tr, y, P)
            fold_ref = dict(onehot=j, dist=P + j)
            refs_used.append(str(meta['vocab'][j]))
        model = Model(meta['d_in'], P, variant).to(device)
        oof[te] = run_fold(model, tr, te, recs, comp, y, fold_ref, device, log=(f == 0))
        # one fitted network is kept so Proposition 4's constants are read off a
        # trained phi rather than an initialisation
        if os.environ.get('TDA_SAVE') and f == 0:
            torch.save(model.state_dict(), os.path.join(RES, 'model_probe.pt'))
        fold_aucs.append(roc_auc_score(y[te], oof[te]))
        print(f'fold {f} auc={fold_aucs[-1]:.4f} ({time.time()-t0:.0f}s)'
              + (f' ref={refs_used[-1]}' if refs_used else ''), flush=True)

    auc_fold = float(np.mean(fold_aucs))
    auc_fold_sd = float(np.std(fold_aucs))
    auc_img = roc_auc_score(y, oof)
    df = pd.DataFrame(dict(patient=patients, y=y, p=oof))
    g = df.groupby('patient').agg(y=('y', 'max'), p=('p', 'mean'))
    auc_pat = roc_auc_score(g['y'], g['p'])
    print(f'FOLD-MEAN AUC {auc_fold:.4f} +- {auc_fold_sd:.4f} | '
          f'IMAGE AUC {auc_img:.4f} | PATIENT AUC {auc_pat:.4f}', flush=True)

    os.makedirs(RES, exist_ok=True)
    tag = f'{cohort}_{target.replace(" ", "_")}_{variant}{os.environ.get("TDA_TAG", "")}_s{seed}'  # [tp]
    json.dump(dict(cohort=cohort, target=target, variant=variant + os.environ.get('TDA_TAG', ''), seed=seed,  # [tp]
                   auc_fold_mean=auc_fold, auc_fold_sd=auc_fold_sd,
                   fold_aucs=[float(a) for a in fold_aucs], refs_used=refs_used,
                   auc_image=auc_img, auc_patient=auc_pat,
                   n_images=len(recs), n_pos=int(y.sum())),
              open(os.path.join(RES, 'learned_' + tag + '.json'), 'w'), indent=2)
    np.save(os.path.join(RES, 'oof_' + tag + '.npy'), oof)
    print('TRAIN_LEARNED_DONE', tag, flush=True)


if __name__ == '__main__':
    main()
