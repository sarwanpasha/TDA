"""Aggregate every finished run into a tidy summary plus paired-bootstrap contrasts.

python collect.py <cohort>

ONE estimator throughout. For each (target, variant) we average the per-seed out-of-fold
rank vectors and take the AUC of that average; the contrasts bootstrap exactly this
quantity, so a delta really is the difference of the two numbers reported for the two
methods. We also carry the per-seed pooled AUC so the table can show seed-to-seed spread,
and the fold-mean AUC for reference, but nothing in the paper mixes them.

Bootstrap resamples patients, not images: images from one patient are not independent.
The reported delta is the point estimate auc_a - auc_b, not the bootstrap mean.
"""
import os, sys, json, glob, re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from train_learned import load_cohort, image_targets, RES

N_BOOT = 2000
CONTRASTS = [
    ('learned_nc', 'fixed_tuned_nc'),
    ('learned_nc', 'fixed_cross_nc'),
    ('learned_nc', 'zero_nc'),
    ('learned_nc', 'comp_only'),
    ('learned', 'fixed_tuned'),
    ('learned', 'comp_only'),
    ('learned', 'deepsets'),
    ('learned', 'gnn'),
    ('learned', 'ripley'),
    ('learned_nc', 'mpl_nc'),  # [tp]
    ('learned_nc', 'mpl_tuned_nc'),
    ('learned_nc', 'mpl_dtm_nc'),
    ('learned', 'gin'),  # [tp]
    ('learned', 'gat'),
    ('learned_nc', 'fixed_tuned_nc_pl'),
    ('learned_nc_pl', 'fixed_tuned_nc_pl'),
    ('fixed_tuned_nc_pl', 'fixed_tuned_nc'),
    ('learned_nc_pl', 'learned_nc'),
    ('learned_nc', 'learned_nc_mpheno'),
    ('learned_nc', 'learned_nc_mdist'),
    ('learned_nc', 'learned_nc_mmarker'),
    ('learned_nc', 'learned_nc_mnomarker'),
    ('learned_nc', 'gin'),
    ('learned_nc', 'gat'),
    ('learned_nc', 'deepsets'),
    ('learned_nc', 'gnn'),
    ('learned_nc_f10', 'fixed_tuned_nc_f10'),
    ('learned_nc_f10', 'sublevel_nc_f10'),
    ('learned_f10', 'deepsets_f10'),
    ('learned_f10', 'gnn_f10'),
    ('learned_nc_f10', 'zero_nc_f10'),
    ('learned_nc_f25', 'fixed_tuned_nc_f25'),
    ('learned_nc_f25', 'sublevel_nc_f25'),
    ('learned_f25', 'deepsets_f25'),
    ('learned_f25', 'gnn_f25'),
    ('learned_nc_f25', 'zero_nc_f25'),
    ('learned_nc_f50', 'fixed_tuned_nc_f50'),
    ('learned_nc_f50', 'sublevel_nc_f50'),
    ('learned_f50', 'deepsets_f50'),
    ('learned_f50', 'gnn_f50'),
    ('learned_nc_f50', 'zero_nc_f50'),
    ('learned_nc_f100', 'fixed_tuned_nc_f100'),
    ('learned_nc_f100', 'sublevel_nc_f100'),
    ('learned_f100', 'deepsets_f100'),
    ('learned_f100', 'gnn_f100'),
    ('learned_nc_f100', 'zero_nc_f100'),
    # what the Lipschitz constraint costs, and what the scale Proposition 1 needs costs
    ('learned_nc', 'learned_unc_nc'),
    ('learned_nc', 'learned_s4_nc'),
    ('learned', 'learned_unc'),
    ('learned', 'learned_s4'),
    # filtrations from the literature, all topology-only so the comparison is of
    # filtrations and not of classifier heads
    ('learned_nc', 'sublevel_nc'),
    ('learned_nc', 'sublevel_sn_nc'),
    ('learned_nc', 'dtm_nc'),
    ('learned_nc', 'dowker_nc'),
    ('learned_nc', 'dowker_tuned_nc'),
    # capacity sweep of the closest competitor: same features, same vectoriser,
    # same optimisation budget, only the width of the filtration network differs
    # the certified complex: same learned weight function, alpha complex instead of
    # the Delaunay flag complex, and its own unweighted control
    ('learned_nc', 'alpha_nc'),
    ('alpha_nc', 'alpha_zero_nc'),
    ('learned_nc', 'sublevel_w32_nc'),
    ('learned_nc', 'sublevel_w128_nc'),
    ('learned_nc', 'sublevel_w256_nc'),
]


def oof_paths(cohort, target, variant):
    pre = f'oof_{cohort}_{target.replace(" ", "_")}_{variant}_s'
    pat = re.compile(re.escape(pre) + r'\d+[.]npy$')
    return sorted(f for f in glob.glob(os.path.join(RES, 'oof_*.npy'))
                  if pat.match(os.path.basename(f)))


def oof_mean(cohort, target, variant):
    """Seed-averaged out-of-fold rank vector, or None if nothing ran."""
    fs = oof_paths(cohort, target, variant)
    if not fs:
        return None, 0
    return np.stack([np.load(f) for f in fs]).mean(0), len(fs)


def load_runs(cohort):
    rows = []
    for pat in ('learned_', 'baseline_'):
        for f in sorted(glob.glob(os.path.join(RES, f'{pat}{cohort}_*.json'))):
            d = json.load(open(f))
            rows.append(dict(target=d['target'], variant=d['variant'], seed=d['seed'],
                             auc_fold_mean=d['auc_fold_mean'], auc_fold_sd=d['auc_fold_sd'],
                             auc_image=d['auc_image'],
                    # train_extra.py does not compute the patient-level AUC; it is
                    # carried for reference only and no reported number uses it
                    auc_patient=d.get('auc_patient', float('nan')),
                             n_images=d.get('n_images'), n_pos=d.get('n_pos')))
    return pd.DataFrame(rows)


def paired_boot(y, groups, a, b, rng):
    pats = np.unique(groups)
    idx_by_pat = {p: np.where(groups == p)[0] for p in pats}
    out = []
    for _ in range(N_BOOT):
        pick = rng.choice(pats, size=len(pats), replace=True)
        idx = np.concatenate([idx_by_pat[p] for p in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        out.append(roc_auc_score(yy, a[idx]) - roc_auc_score(yy, b[idx]))
    return np.array(out)


def main():
    cohort = sys.argv[1] if len(sys.argv) > 1 else 'danenberg'
    runs = load_runs(cohort)
    if runs.empty:
        print('no results yet'); return

    meta, recs = load_cohort(cohort)
    Y = dict(image_targets(cohort, recs, meta['vocab']))
    pats_all = np.array([str(r['patient']) for r in recs])
    if cohort == 'danenberg':
        import clinical
        Y.update(clinical.clinical_targets(pats_all))
    if cohort in ('basel', 'zurich'):  # [tp]
        import jclinical
        Y.update(jclinical.targets(cohort, recs))

    def resolve(t):
        """Labels and patient ids for target t, with unlabelled images dropped exactly as
        the training runs dropped them."""
        yy = np.asarray(Y[t], dtype=float)
        keep = ~np.isnan(yy)
        return yy[keep].astype(int), pats_all[keep], int(keep.sum())

    rows = []
    for (t, v), g in runs.groupby(['target', 'variant']):
        y, _, n_img = resolve(t)
        a, ns = oof_mean(cohort, t, v)
        if a is None:
            continue
        assert len(a) == n_img, f'{t}/{v}: oof length {len(a)} != cohort {n_img}'
        rows.append(dict(target=t, variant=v, n_seeds=ns,
                         auc=roc_auc_score(y, a),
                         sd_seeds=float(g.auc_image.std()),
                         auc_fold_mean=float(g.auc_fold_mean.mean()),
                         sd_folds=float(g.auc_fold_sd.mean()),
                         auc_patient=float(g.auc_patient.mean()),
                         n_images=n_img, n_pos=int(y.sum()),
                         oof_len=len(a)))
    summ = pd.DataFrame(rows).sort_values(['target', 'auc'], ascending=[True, False])
    summ.to_csv(os.path.join(RES, f'summary_{cohort}.csv'), index=False)
    print(summ.to_string(index=False))

    ref_rows = []
    for f in sorted(glob.glob(os.path.join(RES, f'learned_{cohort}_*fixed_tuned*.json'))):
        d = json.load(open(f))
        for k, r in enumerate(d.get('refs_used', [])):
            ref_rows.append(dict(target=d['target'], variant=d['variant'],
                                 seed=d['seed'], fold=k, reference=r))
    if ref_rows:
        pd.DataFrame(ref_rows).to_csv(os.path.join(RES, f'refs_{cohort}.csv'), index=False)
        print()
        print(pd.DataFrame(ref_rows).groupby(['target', 'reference']).size().to_string())

    rng = np.random.default_rng(0)
    rows = []
    for target in sorted(runs['target'].unique()):
        y, groups, _ = resolve(target)
        for va, vb in CONTRASTS:
            a, na = oof_mean(cohort, target, va)
            b, nb = oof_mean(cohort, target, vb)
            if a is None or b is None:
                continue
            aa, bb = roc_auc_score(y, a), roc_auc_score(y, b)
            d = paired_boot(y, groups, a, b, rng)
            rows.append(dict(target=target, a=va, b=vb, seeds_a=na, seeds_b=nb,
                             auc_a=aa, auc_b=bb,
                             delta=aa - bb,
                             boot_mean=float(np.mean(d)),
                             lo=float(np.percentile(d, 2.5)),
                             hi=float(np.percentile(d, 97.5)),
                             p_gt0=float((d > 0).mean())))
    if rows:
        cmp = pd.DataFrame(rows)
        cmp.to_csv(os.path.join(RES, f'compare_{cohort}.csv'), index=False)
        print()
        print(cmp.to_string(index=False))


if __name__ == '__main__':
    main()
