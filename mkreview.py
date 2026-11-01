"""Worklists for the review controls (train_review.py, robust3.py)."""
import os
P = os.environ.get('TDA_HOME', os.path.expanduser('~/tda_imc')); J = P + '/jobs/tp'
LIP = 'TDA_VEC=lip,TDA_RES=%s/results_lip' % P
ARCH = ['APC enriched', 'Active IR ', 'Active stroma', 'CD8^{+} & macrophages', 'FSP1^{+} enriched',
        'Granulocyte enriched', 'PDPN^{+} active stroma', 'Suppressed expansion', 'TLS-like', 'Vascular stroma']
CLIN = ['ER status', 'HER2 status', 'PR status', 'Grade 3', 'Basal subtype', '5-year OS', '5-year RFS']
def line(env, script, *a): return '\t'.join([env or '-', script] + [str(x) for x in a])
def write(name, L):
    open('%s/%s.tsv' % (J, name), 'w').write('\n'.join(L) + '\n'); print(name, len(L))
def env_of(t): return LIP + (',TDA_TARGETS=clinical' if t in CLIN else '')
write('review_a', [line(env_of(t), 'train_review.py', 'danenberg', t, s, v)
                   for v in ('nishi_nc', 'gate_nc') for t in ARCH + CLIN for s in range(5)])
write('review_b', [line(env_of(t), 'train_review.py', 'danenberg', t, s, 'fixed_cv_nc')
                   for t in ARCH + CLIN for s in range(5)])
write('review_c', [line(LIP, 'train_review.py', 'danenberg', t, s, v)
                   for v in ('alpha_fixed_tuned_nc', 'alpha_dtm_nc') for t in ARCH for s in range(5)])
write('review_r', [line('TDA_VEC=lip,TDA_RES=%s/results_cf' % P, 'robust3.py', 'danenberg', t, s)
                   for t in ('Suppressed expansion', 'TLS-like', 'PDPN^{+} active stroma', 'Vascular stroma')
                   for s in range(5)])
print('MKREVIEW_DONE')
