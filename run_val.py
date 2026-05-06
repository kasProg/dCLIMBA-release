import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from model.model import QuantileMappingModel, SpatioTemporalQM
from model.loss import CorrelationLoss, rainy_day_loss, distributional_loss_interpolated, autocorrelation_loss, fourier_spectrum_loss, totalPrecipLoss, spatial_correlation_loss
from ibicus.evaluate.metrics import *
from data.loader import DataLoaderWrapper
import argparse
import yaml
import data.helper as helper
import datetime
import os
from eval.metrics import *
import json

###-----The code is currently accustomed to CMIP6-Livneh Data format ----###

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

# Generate unique run ID
args_dict = vars(args)


run_id = args.run_id
base_dir = args.base_dir
args.val_period = [int(x) for x in args.val_period.split(',')]
val_period = args.val_period

run_path = helper.load_run_path(run_id, base_dir=base_dir)
# Load the config.yaml file
with open(os.path.join(run_path, 'train_config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

logging = True
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
trend_analysis = config['trend_analysis']
scenario = config['scenario']
trend_future_period = [config['trend_start'], config['trend_end']]




train_period = [config['train_start'], config['train_end']]
benchmarking = config['benchmarking']


# model params
## if transform_type in config:
if 'transform_type' in config:
    transform_type = config['transform_type'] #[SST, Poly2]
    temp_enc = config['temp_enc']
else: ## temporary fix
    transform_type = 'monotone'
    temp_enc = 'Conv1d'
degree = config['degree'] # degree of transformation
layers = config['layers'] #number of layers to ANN
time_scale = config['time_scale'] #choose from [daily, month, year-month, julian-day, season]
emph_quantile = config['emph_quantile']
batch_size = config['batch_size']
num_epochs = config['epochs']
stride = config['stride']
chunk = config['chunk']
chunk_size = config['chunk_size']
autoregression = config['autoregression']
lag = config['lag']
wet_dry_flag = config['wet_dry_flag']
pca_mode = config['pca_mode']
logging_path = config['logging_path']
hidden_size = config['hidden_size']
loss_func = config['loss']
neighbors = config['neighbors'] if 'neighbors' in config else 16

# ny = 4 # number of params

#####----- For spatial Tests--------#####
## For Spatial Test
spatial_test = config['spatial_test']
try:
    spatial_extent_val =  None if not spatial_test  else config['spatial_extent_test']
except KeyError:
    spatial_extent_val =  None if not spatial_test  else config['spatial_extent_val']
shapefile_filter_path =  None if not spatial_test  else config['shapefile_filter_path']
# crd =  [14, 15, 16, 17, 18] 
# shape_file_filter = '/pscratch/sd/k/kas7897/us_huc/contents/WBDHU2.shp'

if logging:
    exp = f'{logging_path}/{clim}-{ref}/{transform_type}_{layers}Layers_{degree}degree_quantile{emph_quantile}_scale{time_scale}/{run_id}_{train_period[0]}_{train_period[1]}_{val_period[0]}_{val_period[1]}'
    writer = SummaryWriter(f"runs_revised/{exp}")

save_path_address = config['save_path']

####------------FIXED INPUTS------------####


input_x = {'precipitation': ['pr', 'prec', 'prcp' 'PRCP', 'precipitation']}
clim_var = 'pr'

## loss params
w1 = 0.99
w2 = 0.01
# ny = 4 # number of params

save_path = run_path
model_save_path = save_path
val_save_path =  save_path + f'/{val_period[0]}_{val_period[1]}/'
os.makedirs(val_save_path, exist_ok=True)

data_loader_val = DataLoaderWrapper(
clim=clim, scenario='historical', ref=ref, period=val_period, ref_path=ref_path, cmip6_dir=cmip6_dir, 
input_x=input_x, input_attrs=input_attrs, ref_var=ref_var, save_path=val_save_path, stat_save_path = model_save_path,
crd=spatial_extent_val, shapefile_filter_path=shapefile_filter_path, batch_size=batch_size, train=train, autoregression = autoregression, lag = lag, 
chunk=False, chunk_size=chunk_size, stride=stride, wet_dry_flag=wet_dry_flag, device=device)

dataloader_val = data_loader_val.get_spatial_dataloader(K=neighbors)
valid_coords = data_loader_val.get_valid_coords()


if time_scale == 'daily':
    time_labels = time_labels_val = 'daily'
else:
    time_labels = helper.extract_time_labels(data_loader_val.load_dynamic_inputs()[1], label_type=time_scale)
    time_labels_val = helper.extract_time_labels(data_loader_val.load_dynamic_inputs()[1], label_type=time_scale)


nx = len(input_x)+ len(input_attrs)

if autoregression:
    nx += lag
if wet_dry_flag:
    nx += 1  # Adding wet/dry flag as an additional feature

# model = QuantileMappingModel(nx=nx, degree=degree, hidden_dim=64, num_layers=layers, modelType=transform_type, pca_mode=pca_mode,
#                               monotone=monotone).to(device)
model = SpatioTemporalQM(f_in=nx, f_model=hidden_size, heads=2, t_blocks=layers, st_layers=1, degree=degree, dropout=0.1, transform_type=transform_type, temp_enc=temp_enc).to(device)

# --- Resume training if checkpoint exists ---
start_epoch = 0
    
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(start_epoch + 1, num_epochs + 1):
    if epoch % 10 == 0:
        model = SpatioTemporalQM(f_in=nx, f_model=hidden_size, heads=2, t_blocks=layers, st_layers=1, degree=degree, dropout=0.1, transform_type=transform_type, temp_enc=temp_enc).to(device)

        model.load_state_dict(torch.load(f'{model_save_path}/model_{epoch}.pth', weights_only=True, map_location=device))
        
        model.eval()
        val_epoch_loss = 0
        patch_val = []
        xt_val = []
        x_val = []
        y_val = []
        with torch.no_grad():
            for patches, batch_input_norm, batch_x, batch_y in dataloader_val:
                
                patches_latlon = torch.tensor(valid_coords[patches.cpu().numpy()], dtype=batch_x.dtype).to(device) 
                
                batch_input_norm = batch_input_norm.to(device)
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                # transformed_x, _ = model(batch_x, batch_input_norm, time_labels_val)

                transformed_x, _ = model(batch_input_norm, patches_latlon, batch_x)

                # val_loss_l1 = model.get_weighted_l1_penalty(lambda_l1=1e-4)
                if 'quantile' in loss_func:
                    val_dist_loss = w1 * distributional_loss_interpolated(transformed_x.movedim(-1, 0), batch_y.movedim(-1, 0), device=device, num_quantiles=1000, emph_quantile=emph_quantile)
                    val_loss = val_dist_loss

                if 'autocorrelation' in loss_func:
                    val_autocorr_loss = w2 * autocorrelation_loss(transformed_x, batch_y)
                    val_loss += val_autocorr_loss

                if 'fourier' in loss_func:
                    val_fourier_loss = w2 * fourier_spectrum_loss(transformed_x, batch_y)
                    val_loss += val_fourier_loss

                if 'rainy_day' in loss_func:
                    val_rainy_day_loss = w2 * rainy_day_loss(transformed_x.movedim(-1, 0), batch_y.movedim(-1, 0))
                    val_loss += val_rainy_day_loss

                if 'correlation' in loss_func:
                    val_corr_loss = w2 * CorrelationLoss(transformed_x, batch_y)
                    val_loss += val_corr_loss

                if 'totalP' in loss_func:
                    val_total_precip_loss = 0.0001*totalPrecipLoss(transformed_x, batch_y)
                    val_loss += val_total_precip_loss

                if 'spatial_correlation' in loss_func:
                    val_spatial_corr_loss = spatial_correlation_loss(transformed_x, batch_y)
                    val_loss += val_spatial_corr_loss

                val_epoch_loss += val_loss.item()
                # Store predictions
                xt_val.append(transformed_x.detach().cpu())
                patch_val.append(patches.detach().cpu())
                y_val.append(batch_y.detach().cpu())
                x_val.append(batch_x.detach().cpu())

        


            avg_val_loss = val_epoch_loss / len(dataloader_val)

            x_val = data_loader_val.reconstruct_from_patches(patch_val, x_val, mode='mean').numpy().T ##time, coords
            xt_val = data_loader_val.reconstruct_from_patches(patch_val, xt_val, mode='mean').numpy().T
            y_val = data_loader_val.reconstruct_from_patches(patch_val, y_val, mode='mean').numpy().T
            # xt_val = torch.cat(xt_val, dim=0).numpy().T ##time, coords
            # x_val = torch.cat(x_val, dim=0).numpy().T
            # y_val = torch.cat(y_val, dim=0).numpy().T

            x_val_time = torch.load(f'{val_save_path}/time.pt', weights_only = False)

            ## to manage time
            x_val_time_np = np.array([pd.Timestamp(str(t)) for t in x_val_time])
            x_val_time_np = np.array([pd.Timestamp(t).replace(hour=0, minute=0, second=0) for t in x_val_time_np], dtype='datetime64[D]')
            # Generate a daily time array following the standard Gregorian calendar
            y_val_time = pd.date_range(start=f"{val_period[0]}-01-01", end=f"{val_period[1]}-12-31", freq="D")
            # Convert to NumPy array for indexing and comparison
            y_val_time_np = y_val_time.to_numpy()
            # Find indices where observed time matches model time
            matched_indices = np.where(np.isin(y_val_time_np, x_val_time_np))[0]
            y_val = y_val[matched_indices,:]

            # Initialize climate indices manager
            climate_indices = ClimateIndices()          

            # day_bias_percentages = get_day_bias_percentages(x_val, y_val, xt_val, climate_indices)
            mean_bias_percentages = get_mean_bias_percentages(x_val, y_val, xt_val, x_val_time_np, climate_indices)
            day_bias_percentages = get_day_bias_percentages(x_val, y_val, xt_val, climate_indices)

            keys_mean = ['SDII (Monthly)','CDD (Yearly)', 'CWD (Yearly)', "Rx1day", "Rx5day", "R10mm",  "R20mm", "R95pTOT", "R99pTOT"]
            mean_bias_percentages = dict(filter(lambda item: item[0] in keys_mean , mean_bias_percentages.items()))

            row = {"epoch": int(epoch), "loss": float(avg_val_loss), "metrics": {k: float(np.nanmedian(v[1])) for k, v in mean_bias_percentages.items()}}
            with open(f"{val_save_path}/val_metrics.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            
            if not os.path.exists(f"{save_path_address}/{clim}-{ref}/baseline.jsonl"):
                row_baseline = {k: float(np.nanmedian(v[0])) for k, v in mean_bias_percentages.items()}
                with open(f"{save_path_address}/{clim}-{ref}/baseline_{val_period[0]}_{val_period[1]}.jsonl", "a") as f:
                    f.write(json.dumps(row_baseline) + "\n")

            if logging:
                writer.add_scalar("Loss/validation", avg_val_loss, epoch)
            
                print(f"Epoch {epoch}: Validation Loss = {avg_val_loss:.4f}")

                # Extract and log median(corrected) per metric
                for name, values in mean_bias_percentages.items():
                    corrected = values[1]  # extract corrected values
                    median_corrected = float(np.nanmedian(corrected))
                    writer.add_scalar(f'median_adjusted/{name}', median_corrected, epoch)

                # Extract and log median(corrected) per metric
                for name, values in day_bias_percentages.items():
                    corrected = values[1]  # extract corrected values
                    median_corrected = float(np.nanmedian(corrected))
                    writer.add_scalar(f'median_adjusted/{name}', median_corrected, epoch)
                
                

# Save finished.txt to mark successful completion
finished_file = os.path.join(model_save_path, "finished.txt")
with open(finished_file, "w") as f:
    f.write(f"Finished successfully at {datetime.datetime.now()}\n")