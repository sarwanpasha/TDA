"""collect_review.py with the contrasts of the second round added.

python collect_review2.py <cohort>
"""
import collect, collect_review

collect.CONTRASTS += [
    ('learned_nc', 'gatetopo_nc'), ('gatetopo_nc', 'gate_nc'), ('gatetopo_nc', 'learned_nc'),
    ('learned_nc', 'nishi_ph_nc'), ('gate_nc', 'nishi_ph_nc'),
    ('fixed_tuned_keep_nc', 'fixed_tuned_nc'), ('learned_nc', 'fixed_tuned_keep_nc'),
    ('learned_nc', 'learned_meanvec_nc'),
    ('learned_nc', 'learned_nc_T75'), ('learned_nc', 'learned_nc_T125'),
    ('learned_nc', 'learned_nc_c400'), ('learned_nc', 'learned_nc_c1200'),
    ('gate_nc', 'comp_only'), ('gate_nc', 'deepsets'), ('gatetopo_nc', 'deepsets'), ('gatetopo_nc', 'gat'),
    ('gatetopo_nc', 'comp_only'), ('learned', 'gatetopo_nc'),
]

if __name__ == '__main__':
    collect.main()
