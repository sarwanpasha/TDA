"""collect.py with the contrasts of the review controls added.

python collect_review.py <cohort>
"""
import collect

collect.CONTRASTS += [
    ('learned_nc', 'nishi_nc'),
    ('learned_nc', 'gate_nc'),
    ('learned_nc', 'fixed_cv_nc'),
    ('fixed_cv_nc', 'fixed_tuned_nc'),
    ('alpha_nc', 'alpha_fixed_tuned_nc'),
    ('alpha_nc', 'alpha_dtm_nc'),
    ('learned_nc', 'alpha_fixed_tuned_nc'),
    ('fixed_tuned_nc', 'alpha_fixed_tuned_nc'),
    ('dtm_nc', 'alpha_dtm_nc'),
    ('nishi_nc', 'zero_nc'),
    ('gate_nc', 'comp_only'),
]

if __name__ == '__main__':
    collect.main()
