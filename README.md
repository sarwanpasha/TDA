# Learned Lipschitz filtrations for multiplexed tissue images

Code for training a vertex-weighted Vietoris-Rips filtration end to end on multiplexed
tissue images (imaging mass cytometry, CODEX, MIBI-TOF), with the stability checks,
baselines and evaluation scripts used in the accompanying paper.

## Layout

| File | Purpose |
| --- | --- |
| `difftda.py` | Delaunay flag complex, differentiable weighted-Rips persistence, spectrally normalized weight network |
| `dtalpha.py` | Weighted alpha complex with differentiable alpha values (the certified complex) |
| `train_learned.py` | Learned filtration and the fixed, tuned and plain-Rips controls, with the truncation-free vectorizer |
| `train_extra.py` | Sublevel-set filtration learning, DTM-Rips and Dowker filtrations on the same complex |
| `train_alpha.py` | Learned filtration on the weighted alpha complex |
| `train_mpl.py` | Multiparameter persistence landscapes computed exactly on diagonal slices |
| `baselines.py` | DeepSets, GraphSAGE, GIN, GAT and Ripley-K baselines |
| `collect.py`, `collect_review.py` | Seed-averaged out-of-fold AUCs and paired patient bootstrap contrasts |
| `robust2.py`, `robustness.py`, `robust3.py` | Segmentation-noise robustness of trained models (`robust3.py`: features recomputed from the subsampled cells) |
| `train_review.py` | Controls: coordinate-only learned filtration, gate without persistence, cross-validated reference, designed filtrations on the alpha complex |
| `stability_theta.py`, `lipschitz_norm.py`, `lipschitz_theta.py`, `lipvec.py`, `margin.py`, `paircount.py` | Empirical checks of the stability results and their constants |
| `transfer.py`, `transfer_jk.py`, `qnorm.py`, `coarse.py` | Cross-cohort transfer without refitting |
| `scaling.py` | Cost of one persistence pass against the number of cells |
| `export_gallery.py` | Export of learned weights and diagrams for held-out images |
| `build_graphs.py`, `prep_danenberg.py`, `prep_jackson.py`, `codexprep.py`, `keren_prep.py`, `build_cap.py` | Cohort preprocessing into the shared record format |
| `clinical.py`, `jclinical.py` | Clinical endpoints from the published patient tables |
| `frac.py`, `featmask.py` | Learning-curve subsampling and feature-block ablations |
| `runlist.py`, `mklists.py`, `mklists2.py`, `mkreview.py`, `job_tp.sbatch`, `submit_tp.sh` | Worklists and batch submission |

## Data and paths

The scripts read and write through 2 environment variables so that no site-specific path
is hard-coded:

- `TDA_DATA`: root of the preprocessed data (default `data`), with the subfolders
  `graphs`, `proc`, `danenberg`, `jackson`, `codex`, `keren` created by the
  preprocessing scripts from the public releases of the respective cohorts.
- `TDA_RES`: output folder for result files (default `results`).

Other switches read by the training scripts: `TDA_VEC` (`lip` for the truncation-free
vectorizer, `lite` for the top-K one), `TDA_PL` (PersLay Gaussian head), `TDA_FRAC`
(training fraction for learning curves), `TDA_MASK` (feature-block ablation), `TDA_TAG`
(suffix appended to the variant name), `TDA_T` (cutoff in micrometers, default 100).

## Requirements

Python 3.10 or later with `numpy`, `scipy`, `pandas`, `scikit-learn`, `torch` and
`gudhi`.

## Example

```
export TDA_DATA=/path/to/data TDA_RES=results
python build_graphs.py danenberg
python train_learned.py danenberg "Vascular stroma" 0 learned_nc
python collect.py danenberg
```
