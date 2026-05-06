# dCLIMBA: Differentiable Climate Model Bias Adjustment

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

A differentiable framework for climate model bias adjustment. This project combines advanced neural architectures with domain-specific climate science knowledge to produce bias-corrected precipitation data from coarse-resolution climate model outputs.

NOTE 1: This code currently only does bias correction, the downscaling model will be available soon! 

NOTE 2: The manuscript for this work is under review in GMD 📜 

## 🌍 Overview

Climate models produce valuable projections but at spatial resolutions (25-100km) too coarse for many impact studies. **dCLIMBA** addresses this challenge by:

- **Bias Adjustment**: Corrects systematic biases in climate model outputs using physically-informed transformations
- **Spatial-Temporal Modeling**: Leverages spatial correlations and temporal patterns for enhanced accuracy
- **Multi-Model Support**: Works with CMIP6 climate models and observational datasets (Livneh, GridMET)

## 🏗️ Architecture

### Core Models

**SpatioTemporalQM**: Advanced neural architecture with:
   - Temporal encoders (Conv1D, LSTM, Transformer)
   - Spatial attention with geographic awareness
   - Monotone-basis transformations for adjusting biases


### Key Features

- **Monotone Mapping**: Preserves precipitation order relationships
- **Seasonal Neighbors**: LOCA-style spatial correlation modeling
- **Multi-Scale Processing**: Daily to seasonal temporal patterns
- **Physical Constraints**: Non-negative precipitation with trace thresholds

## 📋 Requirements

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/kasProg/dCLIMBA-release.git
cd dCLIMBA-release

# Create conda environment
conda env create -f env.yml
conda activate dCLIMAD
```

### Key Dependencies

- **Deep Learning**: PyTorch 2.4.1, CUDA 11.8
- **Climate Data**: xarray, netCDF4, rasterio
- **Geospatial**: geopandas, rioxarray, pyproj
- **Scientific**: numpy, scipy, scikit-learn
- **Hydra**: Configuration management
- **Ibicus**: Climate evaluation metrics

## 🚀 Quick Start

### 1. Configuration

The project uses **Hydra** configuration management with sweep configs. Main configuration structure:

```
configs/
├── config.yaml          # Main config with defaults
└── sweep/               # Hyperparameter sweep configurations
    ├── conv1d.yaml      # Conv1D temporal encoder experiments
    ├── lstm.yaml        # LSTM-based experiments  
    └── mlp.yaml         # MLP-based experiments
```

Example sweep configuration (`configs/sweep/conv1d.yaml`):
```yaml
# @package _global_
clim: ['access_cm2', 'gfdl_esm4', 'ipsl_cm6a_lr', 'miroc6', 'mpi_esm1_2_lr','mri_esm2_0']
degree: [8, 10]
emph_quantile: [0.5, 0.9]
temp_enc: 'Conv1d'
epochs: 500
layers: 2
```

### 2. Training

#### Single Experiment
```bash
# Using default sweep config (conv1d)
python run_exp.py

# Using specific sweep config
python run_exp.py sweep=conv1d

# Override individual parameters
python run_exp.py sweep=conv1d clim=access_cm2 epochs=100 degree=8 emph_quantile=0.5
```

#### Hyperparameter Sweeps
```bash
# Launch sweep with automatic GPU management
python launcher.py

# Use specific sweep configuration
python launcher.py sweep=conv1d

# Override sweep parameters
python launcher.py sweep=conv1d clim=access_cm2 epochs=200

# Dry run to see what would be executed
python launcher.py sweep=lstm dry_run=true
```

#### SLURM Batch Jobs
```bash
# Submit to SLURM queue
sbatch slurm1.sbatch

# Monitor job status
squeue -u $USER
```

### 3. Evaluation

#### Auto hyperparameter sweep and testing
# Select base experiment in run_model_selector.sh (for validation tuning) and in auto_val.sh (for testing and triggering validation)
```bash
# sweep testing run and giving the best results based on hyperparamters selected from validation
./auto_eval.sh
```

#### Single Model Validation (However validation is already done during training)
```bash
# Validate specific model run
python run_val.py --run_id <run_id> --base_dir outputs/ 

# Validate with specific validation period
python run_val.py --run_id <run_id> --base_dir outputs/ --val_period 1965,1978
```

#### Batch Validation
```bash
# Validate all models in directory
./run_val_batch.sh outputs/experiment_name/ 1965,1978

# Run in background mode
RUN_IN_BACKGROUND=true ./run_val_batch.sh outputs/experiment_name/ 1965,1978
```

#### Model Selection and Ranking
```bash
# Rank all models by performance metrics
python run_model_selector.py --exp_root outputs/experiment_name/

# Use specific validation period for ranking
python run_model_selector.py --exp_root outputs/experiment_name/ --val_period 1965,1978

# Save results to custom files
python run_model_selector.py --exp_root outputs/experiment_name/ \
    --out_csv my_results.csv --out_json my_best_model.json
```

## 📁 Project Structure

```
dCLIMBA-release/
├── model/
│   ├── model.py           # Neural network architectures
│   └── loss.py            # Climate-specific loss functions
├── data/
│   ├── loader.py          # Advanced data loading with spatial patches
│   ├── helper.py          # Utility functions and time processing
│   └── process.py         # Data preprocessing and normalization
├── eval/
│   └── metrics.py         # Climate evaluation metrics
├── config_files/         # Hydra configuration files
├── outputs/              # Model outputs and checkpoints
├── runs/                 # TensorBoard logging
├── slurm/               # HPC batch scripts
├── launcher.py          # Hyperparameter sweep orchestration
├── run_exp.py          # Single experiment training
├── run_val.py          # Model validation
└── benchmarking.py     # Baseline comparisons
```

## 🔬 Scientific Features

### Climate-Aware Design

- **Trace Precipitation**: Handles values < 0.254mm appropriately
- **Seasonal Correlations**: Uses time-varying spatial neighbor selection
- **Physical Constraints**: Monotonic transformations preserve order relationships
- **Multi-Scale Temporal**: Processes daily, monthly, and seasonal patterns

### Spatial Processing

- **Haversine Distance**: Geographic distance calculations for spatial relationships
- **Patch-Based Training**: Processes spatial neighborhoods for context
- **Attention Mechanisms**: Geographic-aware positional encoding

### Evaluation Metrics

- **Climate Indices**: Rx1day, Rx5day, CDD, CWD, SDII
- **Extreme Precipitation**: R10mm, R20mm, R95pTOT, R99pTOT
- **Bias Metrics**: Comprehensive bias assessment with baseline comparisons

## 🔧 Advanced Usage

### Custom Model Training

```python
from model.model import SpatioTemporalQM
from data.loader import DataLoaderWrapper

# Initialize model
model = SpatioTemporalQM(
    f_in=9,                    # Input features
    f_model=64,                # Hidden dimensions
    heads=4,                   # Attention heads
    degree=8,                  # Transform complexity
    transform_type='monotone'  # Physical constraints
)

# Load data with spatial patches
loader = DataLoaderWrapper(
    clim='access_cm2',
    scenario='historical',
    ref='livneh',
    period=[1950, 1980],
    # ... other parameters
)

# Get spatial dataloader
dataloader = loader.get_spatial_dataloader(K=16)  # 16 neighbors
```

### Hyperparameter Sweeps

The launcher supports automatic GPU management and job distribution:

```bash
# Run sweep with dry-run mode
python launcher.py sweep=lstm clim=access_cm2 epochs=100 dry_run=true

# Full execution across available GPUs
python launcher.py sweep=conv1d clim=['access_cm2','gfdl_esm4'] epochs=400
```

### Model Selection

Automated model ranking based on climate metrics:

```python
from demo_model_selector import scan_and_rank

# Evaluate all models in directory
results = scan_and_rank(
    root='outputs/experiment_suite',
    val_period='1965,1978'
)

print(f"Best model: {results['best']['trial_dir']}")
print(f"Metrics: J={results['best']['best_J']:.4f}")
```

## 📊 Performance Monitoring

### TensorBoard Integration

```bash
tensorboard --logdir runs/
```

### GPU Monitoring

```bash
# Monitor GPU usage during training
./auto_eval.sh

# Check job status
squeue -u $USER
```

## 🎯 Applications

- **Climate Impact Studies**: High-resolution precipitation for hydrology
- **Agricultural Planning**: Crop modeling with bias-corrected climate data  
- **Water Resource Management**: Streamflow and drought analysis
- **Urban Planning**: Infrastructure design under climate change

## 🤝 Contributing

Contributions are welcome! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact

- **Primary Contact**: kas7897@psu.edu
- **Issues**: [GitHub Issues](https://github.com/kasProg/dCLIMBA-release/issues)
- **Discussions**: [GitHub Discussions](https://github.com/kasProg/dCLIMBA-release/discussions)


## 🙏 Acknowledgments

- CMIP6 climate modeling community
- Livneh and GridMET observational datasets
- PyTorch and scientific Python ecosystem
- High-performance computing resources

## 🔗 Related Work

- [Ibicus](https://github.com/btschwertfeger/Ibicus): Climate bias adjustment toolkit
- [LOCA](https://loca.ucsd.edu/): Localized Constructed Analogs downscaling
- [DeepSD](https://github.com/jjgomezcadenas/DeepSD): Deep learning statistical downscaling

---

**Note**: This is a research project under active development. Please report issues and contribute to make it better for the climate science community! 🌍
