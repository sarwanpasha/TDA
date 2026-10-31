"""Image-level targets built from METABRIC clinical and genomic annotation.

None of these labels is derived from the cell graph, so they are independent of the
spatial features the filtration reads. Every image inherits its patient's label; the
evaluation aggregates image ranks to patients before scoring.
"""
import numpy as np
import pandas as pd
import datasets as ds

FIVE_YEARS = 60.0


def _patient_frame(pids):
    pat = pd.read_csv(ds.CBIO + '/data_clinical_patient.txt', sep='\t', comment='#')
    smp = pd.read_csv(ds.CBIO + '/data_clinical_sample.txt', sep='\t',
                      comment='#').drop_duplicates('PATIENT_ID')
    df = pat.merge(smp, on='PATIENT_ID', how='left').set_index('PATIENT_ID')
    return df.reindex(pd.Index(list(pids)))


def _norm(s):
    return s.astype(str).str.strip().str.lower()


def clinical_targets(patients):
    """patients: array of per-image patient ids. Returns {name: y} with NaN for unknown."""
    pids = pd.unique(np.asarray(patients, dtype=object))
    df = _patient_frame(pids)
    out = {}

    def put(name, vals):
        m = dict(zip(df.index, vals))
        out[name] = np.array([m.get(p, np.nan) for p in patients], dtype=float)

    for name, col, pos in (('ER status', 'ER_STATUS', 'positive'),
                           ('HER2 status', 'HER2_STATUS', 'positive'),
                           ('PR status', 'PR_STATUS', 'positive')):
        if col in df.columns:
            s = _norm(df[col])
            put(name, np.where(s.isin(['nan', '', 'na', 'none']), np.nan,
                               (s == pos).astype(float)))
    if 'GRADE' in df.columns:
        g = pd.to_numeric(df['GRADE'], errors='coerce')
        put('Grade 3', np.where(g.isna(), np.nan, (g >= 3).astype(float)))
    if 'CLAUDIN_SUBTYPE' in df.columns:
        s = _norm(df['CLAUDIN_SUBTYPE'])
        put('Basal subtype', np.where(s.isin(['nan', '', 'nc']), np.nan,
                                      (s == 'basal').astype(float)))
    for name, mcol, scol in (('5-year OS', 'OS_MONTHS', 'OS_STATUS'),
                             ('5-year RFS', 'RFS_MONTHS', 'RFS_STATUS')):
        if mcol in df.columns and scol in df.columns:
            t = pd.to_numeric(df[mcol], errors='coerce').to_numpy()
            ev = _norm(df[scol]).str.startswith('1').to_numpy()
            y = np.full(len(df), np.nan)
            y[(ev) & (t <= FIVE_YEARS)] = 1.0          # event inside the window
            y[t > FIVE_YEARS] = 0.0                    # followed past it, no event yet
            y[(~ev) & (t <= FIVE_YEARS)] = np.nan      # censored early: unknown
            put(name, y)
    return out
