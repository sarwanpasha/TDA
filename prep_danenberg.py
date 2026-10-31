"""Preprocess the Danenberg et al. 2022 METABRIC IMC cohort into compact arrays."""
import os, json
import numpy as np
import pandas as pd

RAW = os.environ.get('TDA_DATA', 'data') + '/danenberg/MBTMEIMCPublic'
OUT = os.environ.get('TDA_DATA', 'data') + '/proc/danenberg'
META = ['ImageNumber', 'ObjectNumber', 'metabric_id', 'cellPhenotype', 'is_epithelial',
        'is_tumour', 'is_normal', 'is_dcis', 'is_interface', 'is_perivascular',
        'is_hotAggregate', 'Location_Center_X', 'Location_Center_Y', 'AreaShape_Area']


def main():
    os.makedirs(OUT, exist_ok=True)
    sc = pd.read_csv(os.path.join(RAW, 'SingleCells.csv'))
    print('raw', sc.shape, flush=True)
    n0 = len(sc)
    sc = sc[sc['is_tumour'] == 1]
    print('is_tumour==1:', n0, '->', len(sc), flush=True)
    n0 = len(sc)
    sc = sc[sc['is_hotAggregate'] == 0]
    print('is_hotAggregate==0:', n0, '->', len(sc), flush=True)
    sc = sc.reset_index(drop=True)

    markers = [c for c in sc.columns if c not in META]
    print('n markers', len(markers), markers[:5], markers[-3:], flush=True)

    cells = sc[['ImageNumber', 'ObjectNumber', 'metabric_id', 'cellPhenotype',
                'Location_Center_X', 'Location_Center_Y', 'is_epithelial',
                'is_perivascular', 'AreaShape_Area']].copy()
    cells = cells.rename(columns={'Location_Center_X': 'x', 'Location_Center_Y': 'y',
                                  'cellPhenotype': 'phenotype'})
    cells.to_parquet(os.path.join(OUT, 'cells.parquet'), index=False)
    np.save(os.path.join(OUT, 'markers.npy'), sc[markers].to_numpy(np.float32))
    json.dump(markers, open(os.path.join(OUT, 'marker_names.json'), 'w'))

    print('images', cells.ImageNumber.nunique(), 'patients', cells.metabric_id.nunique(),
          'phenotypes', cells.phenotype.nunique(), flush=True)

    ann = pd.read_csv(os.path.join(RAW, 'TMEStructuresAnnotation.csv'))
    sid2name = dict(zip(ann['Structure'], ann['Label']))
    print('TME structures', sid2name, flush=True)

    counts = {}
    for chunk in pd.read_csv(os.path.join(RAW, 'CellNeighbours.csv'),
                             usecols=['ImageNumber', 'from', 'TMEStructure'],
                             chunksize=2000000):
        chunk = chunk.dropna(subset=['TMEStructure'])
        g = chunk.drop_duplicates(['ImageNumber', 'from'])
        for (img, s), c in g.groupby(['ImageNumber', 'TMEStructure']).size().items():
            counts[(img, int(s))] = counts.get((img, int(s)), 0) + c
    tot = cells.groupby('ImageNumber').size()
    rows = []
    for img in tot.index:
        r = {'ImageNumber': img, 'n_cells': int(tot[img])}
        for s, name in sid2name.items():
            r['frac::' + name] = counts.get((img, int(s)), 0) / tot[img]
        rows.append(r)
    tme = pd.DataFrame(rows)
    tme.to_parquet(os.path.join(OUT, 'tme_structures.parquet'), index=False)
    fc = [c for c in tme.columns if c.startswith('frac::')]
    print('TME coverage (frac>0.05):', flush=True)
    print((tme[fc] > 0.05).sum().to_string(), flush=True)
    print('images with any structure:', int((tme[fc].sum(1) > 0).sum()), flush=True)
    print('PREP_DANENBERG_DONE', flush=True)


if __name__ == '__main__':
    main()
