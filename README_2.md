# dCLIMBA: Differentiable Climate Model Bias Adjustment

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

A differentiable framework for climate model bias adjustment. This project combines
neural architectures with domain-specific climate science knowledge to produce
bias-corrected precipitation from coarse-resolution climate model outputs.

> **This archive accompanies a manuscript.** This Zenodo record is the frozen
> snapshot of the code, processed input data, and representative outputs
> corresponding to the manuscript:
>
> Sawadekar, K., McGinnis, S., Li, P., Lawson, K., & Shen, C.
> *A Differentiable Framework for General Circulation Model Precipitation Bias Correction.*
> Manuscript: [arXiv:2604.23045v3]

> Ongoing development continues on GitHub; that repository is **not** the citable
> artifact for the paper — this archived record is.

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

## Architecture

### Core model

**SpatioTemporalQM**:
- Temporal encoders (Conv1D, LSTM, Transformer)
- Spatial attention with geographic awareness
- Monotone-basis transformation for bias adjustment

### Key features

- **Monotone mapping**: preserves precipitation order relationships
- **Seasonal neighbors**: time-varying spatial correlation modeling
- **Multi-scale processing**: daily to seasonal temporal patterns
- **Physical constraints**: non-negative precipitation with trace thresholds

## Data

This archive includes the **processed** inputs the model actually ingests (regridded
to each GCM's native grid) and representative bias-corrected outputs. Raw third-party
datasets are **not** redistributed in full; see Data Provenance below for how to
obtain and regenerate them.

### Included data inventory

The archive includes processed inputs organized by GCM, plus representative
training, evaluation, and benchmark artifacts under `outputs/` and `benchmark/`.

| Path | Description | Source dataset | Variable | Spatial coverage / resolution | Temporal range | Format |
|------|-------------|----------------|----------|-------------------------------|----------------|--------|
| `processed_data/cmip6/<gcm>/historical/precipitation/clipped_US.nc` | Historical CMIP6 precipitation on the model grid | CMIP6 (see provenance) | precipitation | CONUS / native GCM grid | 1979–2014 | NetCDF |
| `processed_data/cmip6/<gcm>/historical/precipitation/loca/coarse_USclip.nc` | Historical precipitation subset used by the LOCA-style pipeline | CMIP6 (see provenance) | precipitation | CONUS / native GCM grid | 1979–2014 | NetCDF |
| `processed_data/cmip6/<gcm>/ssp5_8_5/precipitation/clipped_US.nc` | SSP5-8.5 precipitation on the model grid | CMIP6 (see provenance) | precipitation | CONUS / native GCM grid | 2015–2099 | NetCDF |
| `processed_data/cmip6/<gcm>/ssp5_8_5/precipitation/loca/coarse_USclip*.nc` | SSP5-8.5 precipitation subsets used in the LOCA-style pipeline | CMIP6 (see provenance) | precipitation | CONUS / native GCM grid | 2015–2099 | NetCDF |
| `processed_data/cmip6/<gcm>/{elev.nc, slope.nc, aspect.nc, landcover.nc}` | Static predictors regridded to each GCM grid | SRTMGL1 / NALCMS | elevation, slope, aspect, landcover | CONUS / native GCM grid | static | NetCDF |
| `processed_data/Livneh/unsplit/prec/<gcm>/prec.YYYY.nc` | Annual unsplit-Livneh precipitation files aligned to each GCM | unsplit-Livneh | precipitation | CONUS / native GCM grid | 1960–2014 | NetCDF |
| `processed_data/shapefiles/conus/*`, `processed_data/shapefiles/us_huc/contents/*` | Boundary and basin shapefiles used for spatial masking and evaluation | USGS / CONUS boundary sources | vector geometry | CONUS / US HUC | static | Shapefile |
| `outputs/Final_repeat_nowghtdecay_5b/jobs_LOCAspatioTempConv1d/<gcm>-livneh/QM_monotone_layers2_degree8_quantile0.9_scalejulian-day/<run_id>/` | Representative model run artifacts, checkpoints, and tensors | model output | precipitation model state | CONUS | historical + SSP5-8.5 | PyTorch / JSON / CSV |
| `outputs/Final_repeat_nowghtdecay_5b` | Representative model run artifacts, checkpoints, and tensors for temporal test | model output | precipitation model state | CONUS | historical + SSP5-8.5 | PyTorch / JSON / CSV |
| `outputs/spatial_Adam_harmonic0` | Representative model run artifacts, checkpoints, and tensors for spatial test | model output | precipitation model state | CONUS | historical + SSP5-8.5 | PyTorch / JSON / CSV |
| `benchmark/{ECDFM,ISIMIP,QuantileMapping,QuantileDeltaMapping}/conus/<gcm>-livneh/[1979, 2000]_historical_[2001, 2014].pt` | Benchmark comparison outputs for the archived baseline methods | benchmark model output | precipitation | CONUS / native GCM grid | historical + evaluation window | PyTorch |

### Data provenance

Each source is cited in the manuscript by its persistent identifier where one exists.

| Source dataset | DOI/Source |
|----------------|----------------|
| CMIP6 precipitation | https://doi.org/10.24381/cds.c866074c
| unsplit-Livneh | https://doi.org/10.1175/JHM-D-20-0212.1 
| SRTMGL1 elevation | https://doi.org/10.5067/MEASURES/SRTM/SRTMGL1.003
| NALCMS landcover | https://www.cec.org/north-american-environmental-atlas/land-cover-30m-2020/|




## Requirements

### Environment setup

```bash
conda env create -f env.yaml
conda activate dCLIMBA
```

> The pinned environment is in `env.yaml` (included in this archive), which conda-installs
> PyTorch 2.4.1 built against CUDA 11.8 (`pytorch=2.4.1=py3.10_cuda11.8_cudnn9.1.0_0` with
> `pytorch-cuda=11.8`) rather than the CUDA 12.x wheels PyTorch ships by default on PyPI —
> make sure your driver supports CUDA 11.8, or swap the pin for a build matching your
> hardware before creating the environment.

### Key dependencies

- **Deep learning**: PyTorch 2.4.1
- **Climate data**: xarray, netCDF4, rasterio
- **Geospatial**: geopandas, rioxarray, pyproj
- **Scientific**: numpy, scipy, scikit-learn
- **Config**: Hydra
- **Evaluation**: ibicus

## Quick start

### 1. Configuration

Hydra configuration with sweep configs:

```
configs/
├── config.yaml          # Main config with defaults
└── sweep/
    ├── conv1d.yaml      # Conv1D temporal encoder experiments
    ├── lstm.yaml        # LSTM-based experiments
    └── mlp.yaml         # MLP-based experiments
```

Example (`configs/sweep/conv1d.yaml`):
```yaml
# @package _global_
clim: ['access_cm2', 'gfdl_esm4', 'ipsl_cm6a_lr', 'miroc6', 'mpi_esm1_2_lr', 'mri_esm2_0']
degree: [8, 10]
emph_quantile: [0.5, 0.9]
temp_enc: 'Conv1d'
epochs: 500
layers: 2
```

### 2. Training

```bash
# Default sweep config (conv1d)
python run_exp.py

# Specific sweep config
python run_exp.py sweep=conv1d

# Override parameters
python run_exp.py sweep=conv1d clim=access_cm2 epochs=100 degree=8 emph_quantile=0.5
```

Hyperparameter sweeps:
```bash
python launcher.py
python launcher.py sweep=conv1d
python launcher.py sweep=conv1d clim=access_cm2 epochs=200
```

SLURM batch jobs:
```bash
sbatch slurm1.sbatch
squeue -u $USER
```

### 3. Evaluation

```bash
# Default test period
./auto_eval.sh

# Custom experiment tree
BASE_DIR=/outputs/Final_repeat_nowghtdecay_5b/jobs_LOCAspatioTempConv1d ./auto_eval.sh

# Override test period
TEST_PERIOD=1995,2010 ./auto_eval.sh

# Spatial test instead of full-domain test
SPATIAL_EXTENT="05" ./auto_eval.sh
```

Single-model validation:
```bash
python run_val.py --run_id <run_id> --base_dir outputs/
python run_val.py --run_id <run_id> --base_dir outputs/ --val_period 1965,1978
```

Model selection and ranking (`run_model_selector.sh` reads the same `BASE_DIR` as `auto_eval.sh`):
```bash
python run_model_selector.py --exp_root outputs/experiment_name/
python run_model_selector.py --exp_root outputs/experiment_name/ --val_period 1965,1978
python run_model_selector.py --exp_root outputs/experiment_name/ --spatial_extent "['05']"
```

## Reproducing the paper

| Manuscript item | How to reproduce |
|-----------------|------------------|
| Training (all 6 GCMs) for temporal test | `python launcher.py sweep=conv1d` |
| Training (all 6 GCMs) for spatial test | `python launcher.py sweep=conv1d spatial_test=true train_start=1990 train_end=2014 val_start=1990 val_end=2014 batch_size=2 learning_rate=1e-3` |
| Testing (all 6 GCMs) for temporal test | `./auto_eval.sh` (default values already set) |
| Testing (all 6 GCMs) for spatial test | `SPATIAL_EXTENT="05" VAL_SPATIAL_EXTENT="07" TEST_PERIOD="1990,2014" ./auto_eval.sh` |
| Fig. 3 (quantile comparison), Figs. 4–5 (ETCCDI bias), Fig. 6 (fractal dimension) | run analysis_notebooks/analysis_ensemble.ipynb |
| Fig. 7 (trend preservation, GFDL-ESM4 SSP5-8.5) | run analysis_notebooks/analysis_future.ipynb|
| Fig. 8 (data-scarce / Ohio holdout) | run analysis_notebooks/analysis_spatial.ipynb |

## Project structure

```
dCLIMBA-release/
├── model/
│   ├── model.py           # Neural network architectures
│   └── loss.py            # Climate-specific loss functions
├── data/
│   ├── loader.py          # Data loading with spatial patches
│   ├── helper.py          # Utilities and time processing
│   └── process.py         # Preprocessing and normalization
├── eval/
│   └── metrics.py         # Climate evaluation metrics
├── configs/               # Hydra configuration files
├── outputs/               # Model outputs and checkpoints
├── slurm/                 # HPC batch scripts
├── launcher.py            # Hyperparameter sweep orchestration
├── run_exp.py             # Single experiment training
├── run_val.py             # Model validation
└── benchmarking.py        # Baseline comparisons
```

## Scientific features

- **Trace precipitation**: handles values < 0.254 mm appropriately
- **Seasonal correlations**: time-varying spatial neighbor selection
- **Physical constraints**: monotonic transformations preserve order relationships
- **Spatial processing**: Haversine-distance neighbor selection, patch-based training,
  geographic-aware attention
- **Evaluation**: ETCCDI indices (Rx1day, Rx5day, CDD, CWD, SDII, R10mm, R20mm,
  R95pTOT, R99pTOT), fractal-dimension spatial structure, trend-bias metrics


## Development

`tests/` covers the pure-logic modules (loss functions, model shape/monotonicity
contracts, normalization round-trips, climate indices, model-selector scoring) with
`pytest` — no GPU or real climate data required: `pip install pytest && pytest tests/ -v`.

## Contact

- Kamlesh Sawadekar — kas7897@psu.edu
- Corresponding author — Chaopeng Shen (cshen@engr.psu.edu)

## Acknowledgments

- CMIP6 climate modeling community
- PyTorch and the scientific Python ecosystem
- Computing resources: NERSC (Perlmutter)

## Related work

- [ibicus](https://github.com/ecmwf-projects/ibicus): statistical bias-adjustment toolkit
- [LOCA](https://loca.ucsd.edu/): Localized Constructed Analogs downscaling
