"""Worklists for the second round of controls."""
import os
P = os.environ.get('TDA_HOME', os.path.expanduser('~/tda_imc')); J = P + '/jobs/tp'
LIP = 'TDA_VEC=lip,TDA_RES=%s/results_lip' % P
ARCH = ['APC enriched', 'Active IR ', 'Active stroma', 'CD8^{+} & macrophages', 'FSP1^{+} enriched',
        'Granulocyte enriched', 'PDPN^{+} active stroma', 'Suppressed expansion', 'TLS-like', 'Vascular stroma']
CLIN = ['ER status', 'HER2 status', 'PR status', 'Grade 3', 'Basal subtype', '5-year OS', '5-year RFS']
FOUR = ['Suppressed expansion', 'TLS-like', 'PDPN^{+} active stroma', 'Vascular stroma']
def line(env, script, *a): return '\t'.join([env or '-', script] + [str(x) for x in a])
def write(name, L):
    open('%s/%s.tsv' % (J, name), 'w').write('\n'.join(L) + '\n'); print(name, len(L))
def env_of(t): return LIP + (',TDA_TARGETS=clinical' if t in CLIN else '')
# joint readout and the other variants on all 17 METABRIC targets
L = [line(env_of(t), 'train_review2.py', 'danenberg', t, s, v)
     for v in ('gatetopo_nc', 'nishi_ph_nc', 'fixed_tuned_keep_nc', 'learned_meanvec_nc') for t in ARCH + CLIN for s in range(5)]
write('review2_a', L)
# composition-only and Ripley K on the 13 targets they were missing
L = [line(env_of(t), 'train_learned.py', 'danenberg', t, s, 'comp_only') for t in ARCH + CLIN if t not in FOUR for s in range(5)]
L += [line(env_of(t), 'baselines.py', 'danenberg', t, s, 'ripley') for t in ARCH + CLIN if t not in FOUR for s in range(5)]
# ablations with the final vectorizer on the 4 archetypes
L += [line(LIP, 'train_learned.py', 'danenberg', t, s, v) for v in ('learned_unc_nc', 'learned_s4_nc') for t in FOUR for s in range(5)]
L += [line(LIP, 'train_extra.py', 'danenberg', t, s, 'sublevel_sn_nc') for t in FOUR for s in range(5)]
for tag, TT in (('T75', 75), ('T125', 125), ('c400', 100), ('c1200', 100)):
    L += [line(LIP + ',TDA_T=%d,TDA_SUF=_%s,TDA_TAG=_%s' % (TT, tag, tag), 'train_learned.py', 'danenberg', t, s, 'learned_nc')
          for t in FOUR for s in range(5)]
write('review2_b', L)
# gate and joint readout on the other cohorts
L = []
for c, ts in (('basel', ['ER status', 'PR status', 'HER2 status', 'Grade 3', '5-year OS', 'Vascular proxy']),
              ('zurich', ['ER status', 'PR status', 'HER2 status', 'Grade 3', 'Vascular proxy'])):
    for t in ts:
        env = LIP + (',TDA_TARGETS=clinical' if t != 'Vascular proxy' else '')
        L += [line(env, 'train_review.py', c, t, s, 'gate_nc') for s in range(5)]
        L += [line(env, 'train_review2.py', c, t, s, 'gatetopo_nc') for s in range(5)]
L += [line(LIP, 'train_review.py', 'codex', 'CLR vs DII', s, 'gate_nc') for s in range(20)]
L += [line(LIP, 'train_review2.py', 'codex', 'CLR vs DII', s, 'gatetopo_nc') for s in range(20)]
L += [line(LIP, 'train_review.py', 'keren', 'Compartmentalized vs mixed', s, 'gate_nc') for s in range(5)]
L += [line(LIP, 'train_review2.py', 'keren', 'Compartmentalized vs mixed', s, 'gatetopo_nc') for s in range(5)]
L += [line('TDA_VEC=lip,TDA_RES=%s/results_margin_j05' % P, 'margin_jitter.py', 'danenberg', '0.5')]
L += [line('TDA_VEC=lip,TDA_RES=%s/results_margin_j0' % P, 'margin_jitter.py', 'danenberg', '0.0')]
write('review2_c', L)
write('review2_r', [line('TDA_VEC=lip,TDA_RES=%s/results_cf2' % P, 'robust4.py', 'danenberg', t, s) for t in FOUR for s in range(5)])
print('MKREVIEW2_DONE')
