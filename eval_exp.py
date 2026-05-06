import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import xarray as xr
from torch.utils.tensorboard import SummaryWriter
import os
import pandas as pd
import numpy as np
from model.model import QuantileMappingModel, SpatioTemporalQM
from model.loss import rainy_day_loss, distributional_loss_interpolated, compare_distributions, rmse, kl_divergence_loss, wasserstein_distance_loss, trend_loss
import data.process as process
from sklearn.preprocessing import StandardScaler
from ibicus.evaluate import assumptions, correlation, marginal, multivariate, trend
from ibicus.evaluate.metrics import *
from data.loader import DataLoaderWrapper
from model.benchmark import BiasCorrectionBenchmark
import data.valid_crd as valid_crd
import data.helper as helper
import yaml
import argparse

###-----The code is currently accustomed to CMIP6-Livneh/gridmet Data format ----###

torch.manual_seed(42)
cuda_device = 0  # could be 'cpu' or an integer like '0', '1', etc.

if cuda_device == 'cpu':
    device = torch.device('cpu')
else:
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{cuda_device}')
    else:
        raise RuntimeError(f"CUDA device {cuda_device} requested but CUDA is not available.")

parser = argparse.ArgumentParser(description="Evaluate experiment")
parser.add_argument('--run_id', type=str, required=True, help='Run ID')
parser.add_argument('--testepoch', type=int, required=True, help='Test epoch')
parser.add_argument('--base_dir', type=str, required=True, help='Base directory for outputs')
parser.add_argument('--validation', action='store_true')

## add argument for test period list
parser.add_argument('--test_period', type=str, required=False, help='Test period, format: start_year,end_year')
parser.add_argument('--spatial_extent', type=str, required=False, help='Spatial extent for evaluation, format: min_lat,max_lat,min_lon,max_lon')


args = parser.parse_args()

if args.test_period:
    args.test_period = [int(x) for x in args.test_period.split(',')]
    test_period= args.test_period



run_id = args.run_id
testepoch = args.testepoch
validation = args.validation
base_dir = args.base_dir

run_path = helper.load_run_path(run_id, base_dir=base_dir)
print(run_path)
# Load the config.yaml file
with open(os.path.join(run_path, 'train_config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

logging = True

if validation:
    test_period = [config['val_start'], config['val_end']]


if args.spatial_extent:
    spatial_extent = [str(x) for x in args.spatial_extent.split(',')]
    shapefile_filter_path = config['shapefile_filter_path']
else:
    spatial_extent =  None
    shapefile_filter_path = None



cmip6_dir = config['cmip_dir']
ref_path = config['ref_dir']



clim = config['clim']
ref = config['ref']
train = False

input_x = {'precipitation': ['pr', 'prec', 'prcp' 'PRCP', 'precipitation']}
clim_var = 'pr'
ref_var = config['ref_var']

input_attrs = config['input_attrs'].split(';')
# input_attrs = {}


### FOR TREND ANALYSIS
if 'trend_analysis' not in config:
    trend_analysis = True
    scenario = 'ssp5_8_5'
    trend_future_period = [2015, 2099]

else:
    trend_analysis = config['trend_analysis']
    scenario = config['scenario']
    trend_future_period = [config['trend_start'], config['trend_end']]





train_period = [config['train_start'], config['train_end']]

# model params
transform_type = config['transform_type'] #[SST, Poly2]
temp_enc = config['temp_enc']
degree = config['degree'] # degree of transformation
layers = config['layers'] #number of layers to ANN
time_scale = config['time_scale'] #choose from [daily, month, year-month, julian-day, season]
emph_quantile = config['emph_quantile']
batch_size = config['batch_size']
epochs = config['epochs']
autoregression = config['autoregression']
lag = config['lag']
wet_dry_flag = config['wet_dry_flag']
# pca_mode = config['pca_mode']
logging_path = config['logging_path']
hidden_size = config['hidden_size']
neighbors = config['neighbors'] if 'neighbors' in config else 16
n_harmonics = config['n_harmonics'] if 'n_harmonics' in config else 0


# ny = 4 # number of params


# crd =  [14, 15, 16, 17, 18] 
# shape_file_filter = '/pscratch/sd/k/kas7897/us_huc/contents/WBDHU2.shp'

if logging:
    exp = f'{logging_path}/{clim}-{ref}/{transform_type}_{layers}Layers_{degree}degree_quantile{emph_quantile}_scale{time_scale}/{run_id}_{train_period[0]}_{train_period[1]}_{test_period[0]}_{test_period[1]}'
    writer = SummaryWriter(f"runs_revised/{exp}")


###-------- Developer section here -----------###


save_path = run_path
model_save_path = save_path
print(save_path)
if spatial_extent:
    save_path = save_path + f'/{spatial_extent}/'
else:
    save_path =  save_path + f'/{test_period[0]}_{test_period[1]}/'
test_save_path = save_path + f'ep{testepoch}'
os.makedirs(test_save_path, exist_ok=True)



data_loader = DataLoaderWrapper(
    clim=clim, scenario='historical', ref=ref, period=test_period, ref_path=ref_path, cmip6_dir=cmip6_dir, 
    input_x=input_x, input_attrs=input_attrs, ref_var=ref_var, save_path=save_path, stat_save_path = model_save_path,
    crd=spatial_extent, shapefile_filter_path=shapefile_filter_path, batch_size=batch_size, train=train, autoregression=autoregression, 
    lag=lag, wet_dry_flag=wet_dry_flag, time_scale=time_scale, device=device)

dataloader = data_loader.get_spatial_dataloader(K=neighbors)
valid_coords = data_loader.get_valid_coords()

if trend_analysis:
    future_save_path = model_save_path + f'/{scenario}_{trend_future_period[0]}_{trend_future_period[1]}/'
    os.makedirs(future_save_path, exist_ok=True)
    data_loader_future = DataLoaderWrapper( 
    clim=clim, scenario = scenario, ref=ref, period=trend_future_period, ref_path=ref_path, cmip6_dir=cmip6_dir, 
    input_x=input_x, input_attrs=input_attrs, ref_var='', save_path=future_save_path, stat_save_path = model_save_path, 
    crd=spatial_extent, shapefile_filter_path=shapefile_filter_path, batch_size=batch_size, train=train, autoregression=autoregression,lag=lag,
    wet_dry_flag=wet_dry_flag, time_scale=time_scale, device=device)

    dataloader_future = data_loader_future.get_spatial_dataloader(K=neighbors)

_, time_x = data_loader.load_dynamic_inputs()
nx = len(input_x)+ len(input_attrs)

if autoregression:
    nx += lag

if wet_dry_flag:
    nx += 1  


model = SpatioTemporalQM(f_in=nx, f_model=hidden_size, heads=2, t_blocks=layers, st_layers=1, degree=degree, dropout=0.1, transform_type=transform_type, temp_enc=temp_enc, n_harmonics=n_harmonics).to(device)


model_path = f'{model_save_path}/model_{testepoch}.pth'    
ckpt = torch.load(model_path, map_location=device)

# Extract state dict, handling both formats
try:
    state_dict = ckpt['model_state']
except KeyError:
    state_dict = ckpt

# Try loading first
try:
    model.load_state_dict(state_dict, strict=False)
except RuntimeError as e:
    # Handle renamed parameters (to_params -> to_coeffs)
    if 'to_params.weight' in state_dict and 'to_coeffs.weight' not in state_dict:
        state_dict['to_coeffs.weight'] = state_dict.pop('to_params.weight')
        state_dict['to_coeffs.bias'] = state_dict.pop('to_params.bias')
        print("Remapped 'to_params' -> 'to_coeffs' for compatibility")
        model.load_state_dict(state_dict, strict=False)
    else:
        raise
# optimizer.load_state_dict(ckpt["optimizer_state"])
# start_epoch = ckpt["epoch"]

model.eval()
transformed_x = []
transformed_x_future = []
patch_future = []
x_future = []
params_all = []
patch_all = []
x = []
y = []
with torch.no_grad():
    for batch in dataloader:
        patches, batch_input_norm, batch_x, batch_y, time_labels = [b.to(device) for b in batch]
        patches_latlon = torch.tensor(valid_coords[patches.cpu().numpy()], dtype=batch_x.dtype).to(device)  # (B,P,2), numpy

        # Forward pass
        # predictions, params = model(batch_x, batch_input_norm, time_scale = time_labels)
        predictions, params = model(batch_input_norm, patches_latlon, batch_x, t_idx = time_labels)
      

        transformed_x.append(predictions.cpu())

        y.append(batch_y.cpu())
        x.append(batch_x.cpu())
        patch_all.append(patches.cpu())
        params_all.append(params.cpu())

    if trend_analysis:
        for batch in dataloader_future:
            patches, batch_input_norm, batch_x, time_labels_future = [b.to(device) for b in batch]
            patches_latlon = torch.tensor(valid_coords[patches.cpu().numpy()], dtype=batch_x.dtype).to(device)  # (B,P,2), numpy

            # Forward pass
            # predictions, _ = model(batch_x, batch_input_norm, time_scale = time_labels_future)
            predictions, _ = model(batch_input_norm, patches_latlon, batch_x, t_idx = time_labels_future)



            # Store predictions
            transformed_x_future.append(predictions.cpu())

            x_future.append(batch_x.cpu())
            patch_future.append(patches.cpu())
        


x = data_loader.reconstruct_from_patches(patch_all, x, mode='mean').numpy().T ##time, coords
transformed_x = data_loader.reconstruct_from_patches(patch_all, transformed_x, mode='mean').numpy().T
y = data_loader.reconstruct_from_patches(patch_all, y, mode='mean').numpy().T
params = data_loader.reconstruct_from_patches(patch_all, params_all, mode='mean').numpy().T

transformed_x_nc = valid_crd.reconstruct_nc(transformed_x, valid_coords, time_x, input_x['precipitation'][0])
transformed_x_nc.to_netcdf(f'{test_save_path}/xt.nc')



torch.save(params, f'{test_save_path}/params.pt')

torch.save(transformed_x, f'{test_save_path}/xt.pt')
avg_improvement, individual_improvements = compare_distributions(transformed_x, x, y)

quantile_rmse_model = torch.sqrt(distributional_loss_interpolated(torch.tensor(x), torch.tensor(y), device='cpu', num_quantiles=1000, emph_quantile=None))
quantile_rmse_bs = torch.sqrt(distributional_loss_interpolated(torch.tensor(transformed_x), torch.tensor(y), device='cpu', num_quantiles=1000, emph_quantile=None))
print(f"Average distribution improvement: {avg_improvement:.4f}")

print(f"Quantile RMSE between Model and Target: {quantile_rmse_model}")
print(f"Quantile RMSE between Corrected and Target: {quantile_rmse_bs}")
print(f"Quantile RMSE Improvement: {quantile_rmse_model - quantile_rmse_bs}")


if trend_analysis:
    # transformed_x_future = torch.cat(transformed_x_future, dim=0).numpy().T
    transformed_x_future = data_loader_future.reconstruct_from_patches(patch_future, transformed_x_future, mode='mean').numpy().T
    transformed_x_nc = valid_crd.reconstruct_nc(transformed_x, valid_coords, time_x, input_x['precipitation'][0])

    torch.save(transformed_x_future, f'{future_save_path}/xt.pt')
