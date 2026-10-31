"""Clinical endpoints for the Jackson et al. Basel and Zurich cohorts, per image.

Read from the patient metadata that ships with the cohort; none is derived from the cell
graph. Every core inherits its patient's label; unknown values are NaN and are dropped by
the training scripts exactly as the METABRIC endpoints are.
"""
import os
import numpy as np
import pandas as pd

PROC = os.environ.get('TDA_DATA', 'data') + '/proc/jackson'
FIVE_YEARS = 60.0


def _norm(s):
    return s.astype(str).str.strip().str.lower()


def targets(cohort, records):
    md = pd.read_parquet('%s/%s/patient_meta.parquet' % (PROC, cohort))
    md['_k'] = md['core'].astype(str)
    md = md.drop_duplicates('_k').set_index('_k')
    ks = [str(r['image']) for r in records]
    out = {}

    def put(name, series):
        m = series.to_dict()
        out[name] = np.array([m.get(k, np.nan) for k in ks], dtype=float)

    for name, cols in (('ER status', ('ERStatus', 'ER')), ('PR status', ('PRStatus', 'PR')),
                       ('HER2 status', ('HER2Status', 'HER2'))):
        # Basel codes receptor status as positive/negative in *Status; Zurich carries
        # the usable coding as +/- in the short column. Take whichever parses more.
        best = None
        for col in cols:
            if col in md.columns:
                s = _norm(md[col])
                v = np.where(s.isin(['positive', 'pos', '+']), 1.0,
                             np.where(s.isin(['negative', 'neg', '-']), 0.0, np.nan))
                if best is None or np.isfinite(v).sum() > np.isfinite(best).sum():
                    best = v
        if best is not None and np.isfinite(best).sum() > 0:
            put(name, pd.Series(best, index=md.index))
    if 'grade' in md.columns:
        g = pd.to_numeric(md['grade'], errors='coerce')
        put('Grade 3', pd.Series(np.where(g.isna(), np.nan, (g >= 3).astype(float)),
                                 index=md.index))
    if 'OSmonth' in md.columns and 'Patientstatus' in md.columns:
        t = pd.to_numeric(md['OSmonth'], errors='coerce').to_numpy()
        st = _norm(md['Patientstatus'])
        ev = (st.str.contains('death') | st.str.contains('dead')).to_numpy()
        y = np.full(len(md), np.nan)
        y[ev & (t <= FIVE_YEARS)] = 1.0
        y[t > FIVE_YEARS] = 0.0
        y[(~ev) & (t <= FIVE_YEARS)] = np.nan
        if np.isfinite(y).sum() > 0:
            put('5-year OS', pd.Series(y, index=md.index))
    return out
