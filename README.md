# dCLIMBA: Differentiable Climate Model Bias Adjustment

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

A differentiable framework for climate model bias adjustment. dCLIMBA combines
neural architectures with domain-specific climate science knowledge to produce
bias-corrected precipitation from coarse-resolution climate model outputs.

> **Manuscript.** Sawadekar, K., McGinnis, S., Li, P., Lawson, K., & Shen, C.
> *A Differentiable Framework for General Circulation Model Precipitation Bias Correction.*
> [arXiv:2604.23045v3]
>
> The frozen code, processed input data and model outputs used for the manuscript are
> archived on Zenodo: **DOI: _to be added_**. This GitHub repository contains the code only.

> **NOTE:** This release performs bias *correction* only. The downscaling component
> is not included in this version.

## Overview

Climate models produce valuable projections but at spatial resolutions (~25–100 km)
too coarse for many impact studies. **dCLIMBA** addresses this by:

- **Bias adjustment**: corrects systematic biases in climate-model output using a
  physically-informed, monotonic parametric transformation.
- **Spatial–temporal modeling**: leverages spatial correlations and temporal patterns.
- **Multi-model support**: works with CMIP6 climate models against a gridded
  observational reference (unsplit-Livneh).

### Core model

**SpatioTemporalQM** ([model/model.py](model/model.py)):
- Temporal encoder (Conv1D in the paper; LSTM, MLP and Transformer also available)
- Spatial attention over LOCA-style seasonal neighbors, with geographic relative-position bias
- Monotone-basis transformation of the raw GCM precipitation (order-preserving, non-negative)

## Requirements

```bash
git clone https://github.com/kasProg/dCLIMBA-release.git
cd dCLIMBA-release
conda env create -f env.yaml
conda activate dCLIMAD
```

The pinned environment (`env.yaml`) installs PyTorch 2.4.1 built against CUDA 11.8. Make sure
your driver supports CUDA 11.8, or swap the pin for a build matching your hardware.

Key dependencies: PyTorch, xarray, netCDF4, geopandas, rioxarray, numpy, scipy,
scikit-learn, Hydra, ibicus.

## Data

Download the processed data from the Zenodo record and place it in the repository root,
so the paths in [configs/config.yaml](configs/config.yaml) resolve:

```
processed_data/
├── cmip6/<gcm>/historical/precipitation/clipped_US.nc     # CMIP6 historical precipitation (CONUS)
├── cmip6/<gcm>/ssp5_8_5/precipitation/clipped_US.nc       # CMIP6 SSP5-8.5 precipitation
├── cmip6/<gcm>/{elev,slope,aspect,landcover}.nc           # static attributes on the GCM grid
├── Livneh/unsplit/prec/<gcm>/prec.YYYY.nc                 # unsplit-Livneh, regridded to each GCM grid
└── shapefiles/{conus,us_huc}/                             # CONUS boundary and HUC2 basins
```

GCMs: `access_cm2`, `gfdl_esm4`, `ipsl_cm6a_lr`, `miroc6`, `mpi_esm1_2_lr`, `mri_esm2_0`.
To regenerate the processed data from the raw sources, see [download_scripts/](download_scripts/)
and the preprocessing scripts in [data/](data/).

To reproduce the paper figures without retraining, also download `outputs/` and `benchmark/`
from the Zenodo record into the repository root.

## Usage

### 1. Configuration

[Hydra](https://hydra.cc) configuration. [configs/config.yaml](configs/config.yaml) holds the
defaults used in the paper; the sweep files define the hyperparameter grid:

```
configs/
├── config.yaml          # Main config (paper defaults)
└── sweep/
    ├── conv1d.yaml      # Conv1D temporal encoder (paper)
    ├── lstm.yaml        # LSTM temporal encoder
    └── mlp.yaml         # MLP temporal encoder
```

`configs/sweep/conv1d.yaml`:
```yaml
# @package _global_
clim: ['access_cm2', 'gfdl_esm4', 'ipsl_cm6a_lr', 'miroc6', 'mpi_esm1_2_lr','mri_esm2_0']
degree: [8]
emph_quantile: [0.5, 0.9]
temp_enc: 'Conv1d'
layers: 2
```

### 2. Training

```bash
# Sweep over all combinations in the sweep config, one job per free GPU
python launcher.py sweep=conv1d

# Single run
python run_exp.py sweep=conv1d clim=access_cm2 degree=8 emph_quantile=0.9
```

Each run is saved to `<save_path>/jobs_LOCAspatioTemp<temp_enc>/<gcm>-livneh/<config>/<run_id>_<train_start>_<train_end>/`
with a checkpoint every 10 epochs, `train_config.yaml`, `statDict.json` (normalization
statistics), and validation metrics (`<val_start>_<val_end>/val_metrics.jsonl`).

### 3. Model selection and testing

```bash
# Temporal test: select the best run/epoch per GCM on 1965-1978, test on 2001-2014
./auto_eval.sh

# Spatial test: select on the Upper Mississippi (HUC 07), test on the Ohio (HUC 05), 1990-2014
SPATIAL=true ./auto_eval.sh

# Overrides
BASE_DIR=outputs/<exp>/jobs_LOCAspatioTempConv1d TEST_PERIOD=1995,2010 ./auto_eval.sh
```

`auto_eval.sh` runs `run_model_selector.sh` (ranks every run and epoch from its validation
log) and then `eval_exp.py` on the best checkpoint of each GCM. Corrected precipitation is
written to `<run>/<start>_<end>/ep<epoch>/xt.pt` and `xt.nc` (`<run>/['05']/ep<epoch>/` for the spatial test), and the SSP5-8.5 projection to
`<run>/ssp5_8_5_2015_2099/xt.pt`.

Individual steps:
```bash
# Re-validate every checkpoint of a run (e.g. after a metrics change)
python run_val.py --run_path <run_dir> [--val_period 1965,1978]
./run_val_batch.sh outputs/<exp>          # all runs of an experiment, from a GPU node

# Rank runs of one GCM
python run_model_selector.py --exp_root outputs/<exp>/jobs_LOCAspatioTempConv1d/<gcm>-livneh --val_period 1965,1978

# Test one run
python eval_exp.py --run_id <run_id> --testepoch <epoch> --base_dir outputs/<exp> --test_period 2001,2014
```

### 4. Benchmarks

Statistical baselines (Quantile Mapping, ISIMIP, ECDFM, Quantile Delta Mapping, via
[ibicus](https://github.com/ecmwf-projects/ibicus)), written to `benchmark/<method>/conus/<gcm>-livneh/`:

```bash
./benchmark.sh
```

## Reproducing the paper

| Manuscript item | How to reproduce |
|-----------------|------------------|
| Training (all 6 GCMs), temporal test | `python launcher.py sweep=conv1d` |
| Training (all 6 GCMs), spatial test | set `save_path`/`logging_path` in `configs/config.yaml` to `outputs/spatial_Adam_harmonic0`, then `python launcher.py sweep=conv1d spatial_test=true train_start=1990 train_end=2014 val_start=1990 val_end=2014 batch_size=2 learning_rate=1e-3` |
| Testing (all 6 GCMs), temporal test | `./auto_eval.sh` |
| Testing (all 6 GCMs), spatial test | `SPATIAL=true ./auto_eval.sh` |
| Benchmarks | `./benchmark.sh` |
| Fig. 3 (quantile comparison), Figs. 4–5 (ETCCDI bias), Fig. 6 (fractal dimension) | [analysis_notebooks/analysis_ensemble.ipynb](analysis_notebooks/analysis_ensemble.ipynb) |
| Fig. 7 (trend preservation, GFDL-ESM4 SSP5-8.5) | [analysis_notebooks/analysis_future.ipynb](analysis_notebooks/analysis_future.ipynb) |
| Fig. 8 (data-scarce / Ohio holdout) | [analysis_notebooks/analysis_spatial.ipynb](analysis_notebooks/analysis_spatial.ipynb) |

## Project structure

```
dCLIMBA-release/
├── model/
│   ├── model.py              # SpatioTemporalQM and building blocks
│   ├── loss.py               # Distributional, rainy-day and spatial-correlation losses
│   └── benchmark.py          # ibicus baseline wrapper
├── data/
│   ├── loader.py             # Data loading, seasonal neighbors, spatial patches
│   ├── process.py            # Normalization and reference loading
│   ├── helper.py             # Units, time labels, run lookup
│   ├── valid_crd.py          # Valid grid cells, HUC filtering, NetCDF reconstruction
│   └── *.py                  # Preprocessing (clipping, coarsening, attributes)
├── download_scripts/         # Raw data download (CMIP6, Livneh, LOCA2)
├── eval/
│   ├── metrics.py            # ETCCDI indices and evaluation metrics
│   └── plot.py               # Plotting helpers
├── configs/                  # Hydra configuration
├── analysis_notebooks/       # Paper figures
├── launcher.py               # Hyperparameter sweep across GPUs
├── run_exp.py                # Training (with inline validation)
├── run_val.py, run_val_batch.sh            # Re-validation of saved checkpoints
├── run_model_selector.py, run_model_selector.sh  # Run/epoch selection
├── eval_exp.py, auto_eval.sh               # Testing
└── benchmarking.py, benchmark.sh           # Statistical baselines
```

## Contact

- Kamlesh Sawadekar — kas7897@psu.edu
- Corresponding author — Chaopeng Shen (cshen@engr.psu.edu)
- Issues: [GitHub Issues](https://github.com/kasProg/dCLIMBA-release/issues)

## Acknowledgments

- CMIP6 climate modeling community
- PyTorch and the scientific Python ecosystem
- Computing resources: NERSC (Perlmutter)

## Related work

- [ibicus](https://github.com/ecmwf-projects/ibicus): statistical bias-adjustment toolkit
- [LOCA](https://loca.ucsd.edu/): Localized Constructed Analogs downscaling
