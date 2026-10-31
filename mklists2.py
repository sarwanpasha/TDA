"""Worklists for the multiparameter-landscape baseline and the Keren MIBI cohort."""
import os
from train_learned import load_cohort, image_targets
P = '$HOME/tda_imc'; J = P + '/jobs/tp'
LIP = 'TDA_VEC=lip,TDA_RES=%s/results_lip' % P
meta, recs = load_cohort('danenberg')
ARCH = sorted(image_targets('danenberg', recs, meta['vocab']).keys())
CLIN = ['ER status', 'HER2 status', 'PR status', 'Grade 3', 'Basal subtype', '5-year OS', '5-year RFS']
SCRIPT = {'learned_nc': 'train_learned.py', 'fixed_tuned_nc': 'train_learned.py',
          'fixed_cross_nc': 'train_learned.py', 'zero_nc': 'train_learned.py',
          'learned': 'train_learned.py', 'comp_only': 'train_learned.py',
          'sublevel_nc': 'train_extra.py', 'dtm_nc': 'train_extra.py', 'dowker_tuned_nc': 'train_extra.py',
          'deepsets': 'baselines.py', 'gnn': 'baselines.py', 'gin': 'baselines.py', 'gat': 'baselines.py',
          'mpl_nc': 'train_mpl.py', 'mpl_tuned_nc': 'train_mpl.py', 'mpl_dtm_nc': 'train_mpl.py'}
def line(env, script, *a): return '\t'.join([env or '-', script] + [str(x) for x in a])
def write(name, L):
    open('%s/%s.tsv' % (J, name), 'w').write('\n'.join(L) + '\n'); print(name, len(L))
L = []
for v in ('mpl_nc', 'mpl_tuned_nc', 'mpl_dtm_nc'):
    for t in ARCH + CLIN:
        env = LIP + (',TDA_TARGETS=clinical' if t in CLIN else '')
        L += [line(env, 'train_mpl.py', 'danenberg', t, s, v) for s in range(5)]
L += [line(LIP, 'train_mpl.py', 'codex', 'CLR vs DII', s, 'mpl_tuned_nc') for s in range(5)]
for c, ts in (('basel', ['ER status', 'PR status', 'HER2 status', 'Grade 3', '5-year OS', 'Vascular proxy']),
              ('zurich', ['ER status', 'PR status', 'HER2 status', 'Grade 3', 'Vascular proxy'])):
    for t in ts:
        env = LIP + (',TDA_TARGETS=clinical' if t != 'Vascular proxy' else '')
        L += [line(env, 'train_mpl.py', c, t, s, 'mpl_tuned_nc') for s in range(5)]
write('mpl', L)
write('kprep', [line('', 'keren_prep.py')])
L = [line(LIP, SCRIPT[v], 'keren', t, s, v) for v in SCRIPT for t in ('Compartmentalized vs mixed', 'Cold vs rest')
     for s in range(5) if v not in ('mpl_nc', 'mpl_dtm_nc')]
write('keren', L)
print('MKLISTS2_DONE')
