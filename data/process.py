import xarray as xr
import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import glob
from data.helper import UnitManager

def load_and_process_year(path, year, valid_coords, var):

    search_pattern = os.path.join(path, f'*{year}*.nc')
    matching_files = glob.glob(search_pattern)
    if not matching_files:
        raise FileNotFoundError(f"No NetCDF file found in '{path}' containing year '{year}'.")

    # Select the first available file (or implement logic to choose the best one)
    path = matching_files[0]

    with xr.open_dataset(path) as x_year:
        unit_identifier = UnitManager(x_year)
        units = unit_identifier.get_units()

        lat_coords = valid_coords[:, 0]
        lon_coords = valid_coords[:, 1]

        x_data = x_year[var].sel(
            lat=xr.DataArray(lat_coords, dims='points'),
            lon=xr.DataArray(lon_coords, dims='points'),
            method='nearest'
        ).values

        # managing units
        x_data = unit_identifier.convert(x_data, var, units[var])
    
    # else:
    #     print(f"Variable '{var}' not found in the Reference NetCDF file. Available variables: {list(x_year.variables.keys())}")
    
    return x_data


def process_multi_year_data(path, train_period, valid_coords, device, var):
    # Use sequential I/O for better stability with netCDF/HDF5 backends on HPC.
    x_list = []
    for year in range(train_period[0], train_period[1] + 1):
        x_list.append(load_and_process_year(path, year, valid_coords, var))

    x = np.concatenate(x_list, axis=0)

    return torch.tensor(x).to(device)


def calStatgamma(x):
    x = x.cpu().detach().numpy()
    a = x.flatten()
    b = a[~np.isnan(a)]  # kick out NaN
    b = np.log10(np.sqrt(b) + 0.1)  # transformation to change gamma characteristics
    p10 = np.percentile(b, 0.1)
    p90 = np.percentile(b, 0.9)
    mean = np.mean(b)
    std = np.std(b)
    if std < 0.001:
        std = np.tensor(1.0)
    return [p10, p90, mean, std]

def calStat(data):
    data = data.cpu().detach().numpy()
    data = data.flatten()
    data = data[~np.isnan(data)]
    mean = np.mean(data)
    std = np.std(data)
    pct10 = np.percentile(data, 0.1)
    pct90 = np.percentile(data, 0.9)
    return [pct10, pct90, mean, std]

def getStatDic(flow_regime, attrLst=None, attrdata=None, seriesLst=None, seriesdata=None):
    statDict = {}
    # series data
    if seriesLst is not None:
        for k in range(len(seriesLst)):
            var = seriesLst[k]
            if flow_regime == 0:
                if var in ["prcp", "Precip", "runoff", "Runoff", "Runofferror", "noisy_prcp", 'pr']:
                    statDict[var] = calStatgamma(seriesdata[:, :, k])
                else:
                    statDict[var] = calStat(seriesdata[:, :, k])
            elif flow_regime == 1:
                statDict[var] = calStat(seriesdata[:, :, k])
    # const attribute
    if attrLst is not None:
        for k in range(len(attrLst)):
            var = attrLst[k]
            statDict[var] = calStat(attrdata[:, k])
    return statDict


def transNormbyDic(x, varLst, staDic, toNorm, flow_regime):
    if isinstance(varLst, str):
        varLst = [varLst]
    out = torch.zeros_like(x)

    special_vars = [
        "prcp", "usgsFlow", "Precip", "runoff", "Runoff", "Runofferror", "noisy_prcp", 'pr'
    ]

    for k, var in enumerate(varLst):
        stat = staDic[var]
        if toNorm:
            if x.dim() == 3:
                if flow_regime == 0 and var in special_vars:
                    temp = torch.log10(torch.sqrt(x[:, :, k]) + 0.1)
                    out[:, :, k] = (temp - stat[2]) / stat[3]
                else:
                    out[:, :, k] = (x[:, :, k] - stat[2]) / stat[3]
            elif x.dim() == 2:
                if flow_regime == 0 and var in special_vars:
                    temp = torch.log10(torch.sqrt(x[:, k]) + 0.1)
                    out[:, k] = (temp - stat[2]) / stat[3]
                else:
                    out[:, k] = (x[:, k] - stat[2]) / stat[3]
        else:
            if x.dim() == 3:
                out[:, :, k] = x[:, :, k] * stat[3] + stat[2]
                if flow_regime == 0 and var in special_vars:
                    temptrans = torch.pow(10, out[:, :, k]) - 0.1
                    temptrans = torch.clamp(temptrans, min=0)  # set negative as zero
                    out[:, :, k] = temptrans ** 2
            elif x.dim() == 2:
                out[:, k] = x[:, k] * stat[3] + stat[2]
                if flow_regime == 0 and var in special_vars:
                    temptrans = torch.pow(10, out[:, k]) - 0.1
                    temptrans = torch.clamp(temptrans, min=0)
                    out[:, k] = temptrans ** 2
    return out


# Saving the dictionary
def save_dict(dictionary, filename):
    def convert_values(obj):
        if isinstance(obj, dict):
            return {key: convert_values(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_values(item) for item in obj]
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy arrays to lists
        else:
            return obj

    # Convert the dictionary
    converted_dict = convert_values(dictionary)
    with open(filename, 'w') as f:
        json.dump(converted_dict, f)

# Loading the dictionary
def load_dict(filename):
    with open(filename, 'r') as f:
        return json.load(f)
    

def log_plots_to_tensorboard(writer, y, x, xt, elev_data, n, ln, exp, epoch):
    """
    Logs combined scatter plots to TensorBoard as one figure using a SummaryWriter.
    
    Parameters:
    - writer: TensorBoard SummaryWriter object.
    - y, x, xt: Data for Original vs. Recovered plot.
    - elev_data, n, ln: Data for Elevation vs. Noise plot.
    - epoch: The epoch number to use as the global_step for logging.
    """
    
    # Create a figure with two subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    fig.suptitle(f"Analysis of Learned Noise \n {exp}")

    # First subplot: Original vs. Recovered
    ax1.scatter(x=y, y=x, c='red', s=10, label='Original')
    ax1.scatter(x=y, y=xt, c='green', s=10, label='Transformed')
    ax1.plot(ax1.get_xlim(), ax1.get_xlim(), 'blue', linestyle='--', label='1 - 1')
    ax1.set_xlabel('Original')
    ax1.set_ylabel('Perturbed')
    ax1.legend()
    ax1.set_title("Original vs. Perturbed")

    # Second subplot: Elevation vs. Noise
    ax2.scatter(elev_data, n, c='red', s=10, label='Original')
    ax2.scatter(elev_data, ln, c='green', s=10, label='Learned')
    ax2.set_xlabel('Elevation')
    ax2.set_ylabel('Noise')
    ax2.legend()
    ax2.set_title("Elevation vs. Noise")

    mean_n = np.nanmean(n, axis=0)
    mean_ln = np.nanmean(ln, axis=0)
    ax3.scatter(elev_data[0,:], mean_n, c='red', s=10, label='Original')
    ax3.scatter(elev_data[0,:], mean_ln, c='green', s=10, label='Learned')
    ax3.set_xlabel('Elevation')
    ax3.set_ylabel('Mean Noise')
    ax3.legend()
    ax3.set_title("Elevation vs. Mean Noise")
    # Log the combined figure to TensorBoard
    writer.add_figure("Analysis of Learned Noise", fig, global_step=epoch)
    plt.close(fig)  # Close the figure to save memory

