"""Write the TPAMI worklists. Runs on the cluster, where the cohorts and results are.

python mklists.py
"""
import os, glob, json
import numpy as np
from train_learned import load_cohort, image_targets

P = '$HOME/tda_imc'
J = P + '/jobs/tp'
os.makedirs(J, exist_ok=True)
LIP = 'TDA_VEC=lip,TDA_RES=%s/results_lip' % P
SEEDS = range(5)

meta, recs = load_cohort('danenberg')
ARCH = sorted(image_targets('danenberg', recs, meta['vocab']).keys())
HEAD = ['TLS-like', 'PDPN^{+} active stroma', 'Suppressed expansion', 'Vascular stroma']
CLIN = ['ER status', 'HER2 status', 'PR status', 'Grade 3', 'Basal subtype',
        '5-year OS', '5-year RFS']
JCLIN = ['ER status', 'PR status', 'HER2 status', 'Grade 3', '5-year OS']
assert all(h in ARCH for h in HEAD), ARCH
print('archetypes:', ARCH)


def exists(resdir, cohort, target, variant, seed):
    return os.path.exists('%s/%s/learned_%s_%s_%s_s%d.json' % (P, resdir, cohort, target.replace(' ', '_'), variant, seed)) or \
        os.path.exists('%s/%s/baseline_%s_%s_%s_s%d.json' % (P, resdir, cohort, target.replace(' ', '_'), variant, seed))


SCRIPT = {'learned_nc': 'train_learned.py', 'fixed_tuned_nc': 'train_learned.py',
          'fixed_cross_nc': 'train_learned.py', 'zero_nc': 'train_learned.py',
          'learned': 'train_learned.py', 'comp_only': 'train_learned.py',
          'sublevel_nc': 'train_extra.py', 'dtm_nc': 'train_extra.py',
          'dowker_tuned_nc': 'train_extra.py',
          'deepsets': 'baselines.py', 'gnn': 'baselines.py', 'gin': 'baselines.py',
          'gat': 'baselines.py', 'alpha_nc': 'train_alpha.py', 'alpha_zero_nc': 'train_alpha.py'}


def line(env, script, *args):
    return '\t'.join([env or '-', script] + [str(a) for a in args])


def write(name, lines):
    open('%s/%s.tsv' % (J, name), 'w').write('\n'.join(lines) + '\n')
    print('%-14s %4d runs' % (name, len(lines)))


# L1 certified pipeline: alpha complex with the truncation-free vectoriser
L = [line(LIP, 'train_alpha.py', 'danenberg', t, s, v)
     for v in ('alpha_nc', 'alpha_zero_nc') for t in ARCH for s in SEEDS
     if not exists('results_lip', 'danenberg', t, v, s)]
write('alpha_lip', L)

# L2 learning curves
L = []
for frac, tag in ((0.1, '_f10'), (0.25, '_f25'), (0.5, '_f50'), (1.0, '_f100')):
    env = LIP + ',TDA_FRAC=%g,TDA_TAG=%s' % (frac, tag)
    for v in ('learned_nc', 'fixed_tuned_nc', 'learned', 'sublevel_nc', 'deepsets', 'gnn'):
        for t in HEAD:
            for s in SEEDS:
                if not exists('results_lip', 'danenberg', t, v + tag, s):
                    L.append(line(env, SCRIPT[v], 'danenberg', t, s, v))
write('curves', L)

# L3 robustness
L = [line(LIP, 'robust2.py', 'danenberg', t, s) for t in HEAD for s in SEEDS
     if not os.path.exists('%s/results_lip/robust_danenberg_%s_s%d.csv' % (P, t.replace(' ', '_'), s))]
write('robust', L)

# L4 extra graph baselines with composition, beside the existing with-composition runs
L = []
for v in ('gin', 'gat', 'learned', 'deepsets', 'gnn'):
    for t in ARCH + CLIN:
        env = 'TDA_TARGETS=clinical' if t in CLIN else ''
        for s in SEEDS:
            if not exists('results', 'danenberg', t, v, s):
                L.append(line(env, SCRIPT[v], 'danenberg', t, s, v))
write('graphnets', L)

# L5 PersLay head
L = []
for v in ('fixed_tuned_nc', 'learned_nc'):
    for t in ARCH + CLIN:
        env = LIP + ',TDA_PL=1,TDA_TAG=_pl' + (',TDA_TARGETS=clinical' if t in CLIN else '')
        for s in SEEDS:
            if not exists('results_lip', 'danenberg', t, v + '_pl', s):
                L.append(line(env, 'train_learned.py', 'danenberg', t, s, v))
write('perslay', L)

# L6 feature-block ablation
L = []
for m in ('pheno', 'dist', 'marker', 'nomarker'):
    env = LIP + ',TDA_MASK=%s,TDA_TAG=_m%s' % (m, m)
    for t in ARCH:
        for s in SEEDS:
            if not exists('results_lip', 'danenberg', t, 'learned_nc_m' + m, s):
                L.append(line(env, 'train_learned.py', 'danenberg', t, s, 'learned_nc'))
write('masks', L)

# L7 colorectal cohort, 20 seeds, all methods
L = []
for v in ('learned_nc', 'fixed_tuned_nc', 'fixed_cross_nc', 'zero_nc', 'learned', 'comp_only',
          'sublevel_nc', 'dtm_nc', 'dowker_tuned_nc', 'deepsets', 'gnn', 'gin', 'gat'):
    for s in range(20):
        if not exists('results_lip', 'codex', 'CLR vs DII', v, s):
            L.append(line(LIP, SCRIPT[v], 'codex', 'CLR vs DII', s, v))
write('codex20', L)

# L0 prerequisites: zurich graphs and the larger cell caps
write('prep', [line('', 'build_graphs.py', 'zurich'), line('', 'build_cap.py', 2000),
               line('', 'build_cap.py', 5000)])

# L8 Basel and Zurich, clinical endpoints and the vascular proxy
L = []
for c in ('basel', 'zurich'):
    for v in ('learned_nc', 'fixed_tuned_nc', 'fixed_cross_nc', 'zero_nc', 'learned', 'comp_only',
              'sublevel_nc', 'dtm_nc', 'dowker_tuned_nc', 'deepsets', 'gnn', 'gin', 'gat'):
        for t in JCLIN + ['Vascular proxy']:
            env = LIP + (',TDA_TARGETS=clinical' if t != 'Vascular proxy' else '')
            for s in SEEDS:
                if not exists('results_lip', c, t, v, s):
                    L.append(line(env, SCRIPT[v], c, t, s, v))
write('jackson', L)

# L9 transfers
L = [line(LIP + ',TDA_QNORM=1,TDA_TAG=_qnorm', 'transfer.py', e, 'learned_nc,fixed_tuned_nc,zero_nc')
     for e in ('ER status', 'Grade 3')]
L += [line(LIP, 'transfer_jk.py', e) for e in ('ER status', 'Grade 3', 'HER2 status', 'PR status')]
write('transfer', L)

# L10 scaling and the cap sweep
write('scaling', [line(LIP, 'scaling.py')])
for n in (2000, 5000):
    env = 'TDA_VEC=lip,TDA_RES=%s/results_sens_c%d,TDA_SUF=_c%d' % (P, n, n)
    os.makedirs('%s/results_sens_c%d' % (P, n), exist_ok=True)
    L = [line(env, 'train_learned.py', 'danenberg', t, s, v)
         for v in ('learned_nc', 'fixed_tuned_nc', 'zero_nc')
         for t in ('TLS-like', 'Suppressed expansion') for s in SEEDS
         if not exists('results_sens_c%d' % n, 'danenberg', t, v, s)]
    write('cap%d' % n, L)

# L12 gallery
write('gallery', [line(LIP, 'export_gallery.py', 'danenberg', t, 0, 3) for t in HEAD])
print('MKLISTS_DONE')
