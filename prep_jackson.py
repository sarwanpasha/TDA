"""Preprocess the Jackson et al. 2020 Basel/Zurich breast IMC cohorts."""
import os, json, re
import numpy as np
import pandas as pd

ROOT = os.environ.get('TDA_DATA', 'data') + '/jackson'
DP = os.path.join(ROOT, 'Data_publication')
LOC = os.path.join(ROOT, 'singlecell_locations')
CL = os.path.join(ROOT, 'singlecell_cluster_labels', 'Cluster_labels')
OUT = os.environ.get('TDA_DATA', 'data') + '/proc/jackson'

COHORTS = {
    'basel': dict(loc='Basel_SC_locations.csv', mc='Basel_metaclusters.csv',
                  sc=os.path.join(DP, 'BaselTMA', 'SC_dat.csv'),
                  meta=os.path.join(DP, 'BaselTMA', 'Basel_PatientMetadata.csv')),
    'zurich': dict(loc='Zurich_SC_locations.csv', mc='Zurich_matched_metaclusters.csv',
                   sc=os.path.join(DP, 'ZurichTMA', 'SC_dat.csv'),
                   meta=os.path.join(DP, 'ZurichTMA', 'Zuri_PatientMetadata.csv')),
}


def clean_channel(c):
    m = re.match(r'^\s*\S*?Di\s+(.*)$', str(c))
    name = m.group(1) if m else str(c)
    return name.strip()


def build(name, cfg, ann):
    loc = pd.read_csv(os.path.join(LOC, cfg['loc']))
    mc = pd.read_csv(os.path.join(CL, cfg['mc']))
    print(name, 'loc', loc.shape, 'mc', mc.shape, flush=True)
    mc.columns = ['id', 'cluster'] if len(mc.columns) == 2 else mc.columns
    cells = loc.merge(mc, on='id', how='inner')
    cells = cells.rename(columns={'Location_Center_X': 'x', 'Location_Center_Y': 'y'})
    cells['cluster'] = cells['cluster'].astype(int)
    cells = cells.merge(ann, on='cluster', how='left')
    cells = cells[['id', 'core', 'x', 'y', 'cluster', 'cell_type', 'cell_class']]
    cells = cells.dropna(subset=['x', 'y']).reset_index(drop=True)
    print(name, 'cells', cells.shape, 'cores', cells.core.nunique(),
          'types', cells.cell_type.nunique(), flush=True)

    idx = pd.Series(np.arange(len(cells)), index=cells['id'].values)
    n = len(cells)
    chans, cols, mat = {}, [], None
    for chunk in pd.read_csv(cfg['sc'], usecols=['id', 'channel', 'mc_counts'],
                             chunksize=5000000):
        for c in chunk['channel'].unique():
            if c not in chans:
                chans[c] = len(chans); cols.append(clean_channel(c))
        if mat is None or mat.shape[1] < len(chans):
            new = np.zeros((n, len(chans)), np.float32)
            if mat is not None:
                new[:, :mat.shape[1]] = mat
            mat = new
        r = idx.reindex(chunk['id'].values).to_numpy()
        ok = ~np.isnan(r)
        cc = chunk['channel'].map(chans).to_numpy()
        mat[r[ok].astype(int), cc[ok].astype(int)] = chunk['mc_counts'].to_numpy(np.float32)[ok]
    print(name, 'markers', mat.shape, cols[:4], flush=True)

    os.makedirs(os.path.join(OUT, name), exist_ok=True)
    cells.to_parquet(os.path.join(OUT, name, 'cells.parquet'), index=False)
    np.save(os.path.join(OUT, name, 'markers.npy'), mat)
    json.dump(cols, open(os.path.join(OUT, name, 'marker_names.json'), 'w'))
    meta = pd.read_csv(cfg['meta'])
    meta.to_parquet(os.path.join(OUT, name, 'patient_meta.parquet'), index=False)
    print(name, 'meta', meta.shape, list(meta.columns)[:8], flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    ann = pd.read_csv(os.path.join(CL, 'Metacluster_annotations.csv'), sep=';')
    ann.columns = [c.strip() for c in ann.columns]
    ann = ann.rename(columns={ann.columns[0]: 'cluster', ann.columns[1]: 'cell_type',
                              ann.columns[2]: 'cell_class'})
    ann['cluster'] = ann['cluster'].astype(int)
    print(ann.to_string(), flush=True)
    ann.to_parquet(os.path.join(OUT, 'metacluster_annotations.parquet'), index=False)
    for name, cfg in COHORTS.items():
        build(name, cfg, ann)
    print('PREP_JACKSON_DONE', flush=True)


if __name__ == '__main__':
    main()
