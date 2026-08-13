import torch
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import datetime

import data.helper as helper
from data.loader import DataLoaderWrapper
from model.build import build_model, load_checkpoint
from eval.validate import run_validation_epoch

###-----The code is currently accustomed to CMIP6-Livneh Data format ----###
# Standalone re-validation of a trained run's checkpoints against a chosen
# validation period. Training already validates inline every 10 epochs
# (see run_exp.py); this script exists for re-running that same validation
# later -- e.g. against a different val_period, or after the fact -- without
# retraining. It shares its core validation logic with run_exp.py via
# eval.validate.run_validation_epoch, so the two stay in sync.

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
parser.add_argument('--base_dir', type=str, required=True, help='Base directory for outputs')

## add argument for validation period list
parser.add_argument('--val_period', type=str, required=False, help='Validation period, format: start_year,end_year')

args = parser.parse_args()

run_id = args.run_id
base_dir = args.base_dir
val_period = [int(x) for x in args.val_period.split(',')]

run_path, config = helper.load_trial_config(run_id, base_dir=base_dir)

logging = True
cmip6_dir = config['cmip_dir']
ref_path = config['ref_dir']

clim = config['clim']
ref = config['ref']
train = False

input_x = {'precipitation': ['pr', 'prec', 'prcp', 'PRCP', 'precipitation']}
clim_var = 'pr'
ref_var = config['ref_var']

input_attrs = config['input_attrs'].split(';')

train_period = [config['train_start'], config['train_end']]

# model params
## if transform_type in config:
if 'transform_type' in config:
    transform_type = config['transform_type']  # [SST, Poly2]
    temp_enc = config['temp_enc']
else:  # temporary fix for trials saved before transform_type/temp_enc existed
    transform_type = 'monotone'
    temp_enc = 'Conv1d'
degree = config['degree']  # degree of transformation
layers = config['layers']  # number of layers to ANN
time_scale = config.get('time_scale', 'daily')  # choose from [daily, month, year-month, julian-day, season]
emph_quantile = config['emph_quantile']
num_epochs = config['epochs']
stride = config['stride']
chunk = config['chunk']
chunk_size = config['chunk_size']
autoregression = config['autoregression']
lag = config['lag']
wet_dry_flag = config['wet_dry_flag']
logging_path = config['logging_path']
loss_func = config['loss']
neighbors = config['neighbors'] if 'neighbors' in config else 16
batch_size = config['batch_size']

#####----- For spatial Tests--------#####
## For Spatial Test
spatial_test = config['spatial_test']
try:
    spatial_extent_val = None if not spatial_test else config['spatial_extent_test']
except KeyError:
    spatial_extent_val = None if not spatial_test else config['spatial_extent_val']
shapefile_filter_path = None if not spatial_test else config['shapefile_filter_path']

if logging:
    exp = f'{logging_path}/{clim}-{ref}/{transform_type}_{layers}Layers_{degree}degree_quantile{emph_quantile}_scale{time_scale}/{run_id}_{train_period[0]}_{train_period[1]}_{val_period[0]}_{val_period[1]}'
    writer = SummaryWriter(f"runs_revised/{exp}")

save_path_address = config['save_path']
# Must match how run_exp.py builds job_path, so the two scripts agree on
# where the one-time baseline_<period>.jsonl for this clim/ref lives.
job_path = f'{save_path_address}/jobs_LOCAspatioTemp{temp_enc}'

save_path = run_path
model_save_path = save_path
val_save_path = save_path + f'/{val_period[0]}_{val_period[1]}/'
os.makedirs(val_save_path, exist_ok=True)

data_loader_val = DataLoaderWrapper(
    clim=clim, scenario='historical', ref=ref, period=val_period, ref_path=ref_path, cmip6_dir=cmip6_dir,
    input_x=input_x, input_attrs=input_attrs, ref_var=ref_var, save_path=val_save_path, stat_save_path=model_save_path,
    crd=spatial_extent_val, shapefile_filter_path=shapefile_filter_path, batch_size=batch_size, train=train,
    autoregression=autoregression, lag=lag, chunk=False, chunk_size=chunk_size, stride=stride,
    wet_dry_flag=wet_dry_flag, time_scale=time_scale, device=device)

dataloader_val = data_loader_val.get_spatial_dataloader(K=neighbors)

nx = len(input_x) + len(input_attrs)

if autoregression:
    nx += lag
if wet_dry_flag:
    nx += 1  # Adding wet/dry flag as an additional feature

start_epoch = 0

for epoch in range(start_epoch + 1, num_epochs + 1):
    if epoch % 10 == 0:
        ckpt_path = f'{model_save_path}/model_{epoch}.pth'
        if not os.path.exists(ckpt_path):
            continue

        model = build_model(config, nx=nx, device=device)
        load_checkpoint(model, ckpt_path, device=device, strict=True)
        model.eval()

        run_validation_epoch(
            model=model, dataloader_val=dataloader_val, data_loader_val=data_loader_val,
            device=device, val_period=val_period, val_save_path=val_save_path,
            job_path=job_path, clim=clim, ref=ref, loss_func=loss_func,
            emph_quantile=emph_quantile, epoch=epoch,
            writer=writer if logging else None,
        )

# Save finished.txt to mark successful completion
finished_file = os.path.join(model_save_path, "finished.txt")
with open(finished_file, "w") as f:
    f.write(f"Finished successfully at {datetime.datetime.now()}\n")
