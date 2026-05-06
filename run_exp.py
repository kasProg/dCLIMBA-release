import os
import glob
import json
import datetime
import yaml

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import hydra
from omegaconf import DictConfig, OmegaConf

from model.model import QuantileMappingModel, SpatioTemporalQM
from model.loss import (
    CorrelationLoss, rainy_day_loss, distributional_loss_interpolated,
    autocorrelation_loss, fourier_spectrum_loss, totalPrecipLoss,
    spatial_correlation_loss
)
from ibicus.evaluate.metrics import *
from data.loader import DataLoaderWrapper
from data.helper import generate_run_id
import data.helper as helper
from eval.metrics import *
import time

###-----The code is currently accustomed to CMIP6-Livneh Data format ----###

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function with Hydra config"""
    script_start_time = time.time()
    torch.manual_seed(42)

    # Print config for verification
    print("="*80)
    print("Training Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("="*80)
    
    # Convert config to dict for compatibility
    args_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Extract values from config (replace all args.xxx with cfg.xxx)
    cuda_device = cfg.get('cuda_device', '0')    
    
    # Generate unique run ID
    run_id_dict = {k: v for k, v in args_dict.items() if k not in ['cuda_device', 'available_gpus', 'epochs']}
    run_id = generate_run_id(run_id_dict)
    
    
    logging = cfg.logging
    
    cmip6_dir = cfg.cmip_dir
    ref_path = cfg.ref_dir
    clim = cfg.clim
    ref = cfg.ref
    ref_var = cfg.ref_var
    
    train = cfg.train
    validation = cfg.validation

    train_period = [cfg.train_start, cfg.train_end]
    val_period = [cfg.val_start, cfg.val_end]
    epochs = cfg.epochs


    # model params
    transform_type = cfg.transform_type
    temp_enc = cfg.temp_enc
    batch_size = cfg.batch_size
    degree = cfg.degree
    layers = cfg.layers
    hidden_size = cfg.hidden_size
    time_scale = cfg.time_scale
    emph_quantile = cfg.emph_quantile
    chunk = cfg.chunk
    chunk_size = cfg.chunk_size
    stride = cfg.stride
    loss_func = cfg.loss
    wet_dry_flag = cfg.wet_dry_flag
    learning_rate = cfg.learning_rate
    monotone = cfg.monotone

    neighbors = cfg.neighbors
    n_harmonics = cfg.n_harmonics


    ## For Spatial Test
    spatial_test = cfg.spatial_test
    spatial_extent =  None if not spatial_test  else cfg.spatial_extent
    spatial_extent_val =  None if not spatial_test  else cfg.spatial_extent_val
    shapefile_filter_path =  None if not spatial_test  else cfg.shapefile_filter_path


    autoregression = cfg.autoregression
    lag = cfg.lag

    save_path_address = cfg.save_path
    logging_path_address = cfg.logging_path

    ## INPUTS
    input_attrs = cfg.input_attrs.split(';')



    ####------------FIXED INPUTS------------####


    input_x = {'precipitation': ['pr', 'prec', 'prcp', 'PRCP', 'precipitation']}
    clim_var = 'pr'

    ## fixed loss params
    w1 = 0.99
    w2 = 0.01



    ###------------ Developer section here --------------###
    if cuda_device == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{cuda_device}')
        else:
            raise RuntimeError(f"CUDA device {cuda_device} requested but CUDA is not available.")

    if logging:
        exp = f'conus_{transform_type}{degree}_LOCAspatioTemp{temp_enc}/{clim}-{ref}/{transform_type}_{layers}Layers_{degree}degree_quantile{emph_quantile}_scale{time_scale}/{run_id}_{train_period[0]}_{train_period[1]}_{val_period[0]}_{val_period[1]}'
        writer = SummaryWriter(f"{logging_path_address}/{exp}")

    job_path = f'{save_path_address}/jobs_LOCAspatioTemp{temp_enc}'
    save_path = f'{job_path}/{clim}-{ref}/QM_{transform_type}_layers{layers}_degree{degree}_quantile{emph_quantile}_scale{time_scale}/{run_id}_{train_period[0]}_{train_period[1]}/'
    model_save_path = save_path
    if validation:
        if spatial_test:
            val_save_path =  save_path + f'{spatial_extent_val}/'
        else:
            val_save_path =  save_path + f'{val_period[0]}_{val_period[1]}/'
        # test_save_path = val_save_path + f'ep{testepoch}'
        os.makedirs(val_save_path, exist_ok=True)

    os.makedirs(save_path, exist_ok=True)

     # Save current arguments into config.yaml inside save_path
    with open(os.path.join(save_path, "train_config.yaml"), "w") as f:
        # Save Hydra config instead of args_dict
        OmegaConf.save(cfg, f)

    data_loader = DataLoaderWrapper(
        clim=clim, scenario='historical', ref=ref, period=train_period, ref_path=ref_path, cmip6_dir=cmip6_dir, 
        input_x=input_x, input_attrs=input_attrs, ref_var=ref_var, save_path=save_path, stat_save_path = model_save_path,
        crd=spatial_extent, shapefile_filter_path=shapefile_filter_path, batch_size=batch_size, train=train, autoregression = autoregression, lag = lag, 
        chunk=chunk, chunk_size=chunk_size, stride=stride, wet_dry_flag=wet_dry_flag, time_scale=time_scale, device=device)

    dataloader = data_loader.get_spatial_dataloader(K=neighbors)

    valid_coords = data_loader.get_valid_coords()

    if validation:
        data_loader_val = DataLoaderWrapper(
        clim=clim, scenario='historical', ref=ref, period=val_period, ref_path=ref_path, cmip6_dir=cmip6_dir, 
        input_x=input_x, input_attrs=input_attrs, ref_var=ref_var, save_path=val_save_path, stat_save_path = model_save_path,
        crd=spatial_extent_val, shapefile_filter_path=shapefile_filter_path, batch_size=batch_size, train=train, autoregression = autoregression, lag = lag, 
        chunk=False, chunk_size=chunk_size, stride=stride, wet_dry_flag=wet_dry_flag, time_scale=time_scale, device=device)

        dataloader_val = data_loader_val.get_spatial_dataloader(K=neighbors)


   

    nx = len(input_x)+ len(input_attrs)

    if autoregression:
        nx += lag
    if wet_dry_flag:
        nx += 1  # Adding wet/dry flag as an additional feature

    model = SpatioTemporalQM(f_in=nx, f_model=hidden_size, heads=2, t_blocks=layers, st_layers=1, degree=degree, dropout=0.1, transform_type=transform_type, temp_enc=temp_enc, n_harmonics=n_harmonics).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Resume training if checkpoint exists ---
    start_epoch = 0
    latest_ckpt = None

    ckpt_files = sorted(glob.glob(f"{save_path}/model_*.pth"), key=os.path.getmtime)
    if ckpt_files:
        latest_ckpt = ckpt_files[-1]
        # Extract epoch number from filename
        start_epoch = int(os.path.basename(latest_ckpt).split('_')[1].split('.')[0])
        print(f"Resuming from checkpoint: {latest_ckpt}, epoch {start_epoch}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"]
        

    balance_loss = 0  # Adjust this weight to balance between distributional and rainy day losses

    # Training loop
    num_epochs = epochs
    loss_list = []
    for epoch in range(start_epoch + 1, num_epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()

        
        loss1 = 0
        loss2 = 0
        loss3 = 0

        for batch_idx, (patches, batch_input_norm, batch_x, batch_y, time_labels) in enumerate(dataloader):
            # Move batch to device
            batch_start = time.time()

            patches_latlon = torch.tensor(valid_coords[patches.cpu().numpy()], dtype=batch_x.dtype).to(device)  # (B,P,2), numpy

            batch_input_norm = batch_input_norm.to(device)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            time_labels = time_labels.to(device)

            fwd_start = time.time()
            # Forward pass
            transformed_x, _ = model(batch_input_norm, patches_latlon, batch_x, t_idx=time_labels)

            fwd_time = time.time() - fwd_start

            #trasform log back
            # transformed_x= torch.expm1(transformed_x)

            if 'quantile' in loss_func:
                dist_loss = w1 * distributional_loss_interpolated(transformed_x.movedim(-1, 0), batch_y.movedim(-1, 0), device=device, num_quantiles=1000, emph_quantile=emph_quantile)
                loss = dist_loss
                loss1 += dist_loss.item()

            if 'autocorrelation' in loss_func:
                autocorr_loss = w2 * autocorrelation_loss(transformed_x, batch_y)
                loss+= autocorr_loss 
                loss2 += autocorr_loss.item()
            
            if 'fourier' in loss_func:
                fourier_loss = w2 * fourier_spectrum_loss(transformed_x, batch_y)
                loss+= fourier_loss
                
            if 'rainy_day' in loss_func:
                rainy_loss = w2 * rainy_day_loss(transformed_x.movedim(-1, 0), batch_y.movedim(-1, 0))
                loss+= rainy_loss
                loss2 += rainy_loss.item()

            if 'correlation' in loss_func:
                corr_loss = CorrelationLoss(transformed_x, batch_y)
                loss+= corr_loss
                loss2 += corr_loss.item()
                
            if 'totalP' in loss_func:
                total_precip_loss = 0.0001*totalPrecipLoss(transformed_x, batch_y)
                loss+= total_precip_loss
                loss3 += total_precip_loss.item()

            if 'spatial_correlation' in loss_func:
                spatial_corr_loss = spatial_correlation_loss(transformed_x, batch_y)
                loss+= spatial_corr_loss
                loss3 += spatial_corr_loss.item()
            


            # Backward pass and optimization
            bwd_start = time.time()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bwd_time = time.time() - bwd_start
            batch_time = time.time() - batch_start


            epoch_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}/{len(dataloader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Quantile: {dist_loss.item():.4f} | "
                    f"Rainy Day: {rainy_loss.item():.4f} | "
                    f"Spatial Correlation: {spatial_corr_loss.item():.4f} | "
                    f"Fwd: {fwd_time:.3f}s | "
                    f"Bwd: {bwd_time:.3f}s | "
                    f"Total: {batch_time:.3f}s")
        
        epoch_time = time.time() - epoch_start


        # Average loss for the epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        avg_epoch_loss1 = loss1 / len(dataloader)
        avg_epoch_loss2 = loss2 / len(dataloader)
        avg_epoch_loss3 = loss3 / len(dataloader)
        print(f"Epoch {epoch} done in {epoch_time:.2f}s | Avg Loss: {avg_epoch_loss:.4f}")



        if logging:
            writer.add_scalar("Loss/train", avg_epoch_loss, epoch)
            writer.add_scalar("Loss1/train", avg_epoch_loss1, epoch)
            writer.add_scalar("Loss2/train", avg_epoch_loss2, epoch)
            writer.add_scalar("Loss3/train", avg_epoch_loss3, epoch)

        loss_list.append(avg_epoch_loss)

        if epoch % 10 == 0:

            print(f'Epoch {epoch}, Average Loss: {avg_epoch_loss:.4f}, Average Loss1: {avg_epoch_loss1:.4f}, Average Loss2: {avg_epoch_loss2:.4f}')
            checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            }
            torch.save(checkpoint, f"{save_path}/model_{epoch}.pth")

            
            # ====== VALIDATION SECTION ====== #
            if validation:
                model.eval()
                val_epoch_loss = 0
                patch_val = []
                xt_val = []
                x_val = []
                y_val = []
                with torch.no_grad():
                    for patches, batch_input_norm, batch_x, batch_y, time_labels_val in dataloader_val:
                        
                        patches_latlon = torch.tensor(valid_coords[patches.cpu().numpy()], dtype=batch_x.dtype).to(device) 
                        
                        batch_input_norm = batch_input_norm.to(device)
                        batch_x = batch_x.to(device)
                        batch_y = batch_y.to(device)
                        time_labels_val = time_labels_val.to(device)
                        # transformed_x, _ = model(batch_x, batch_input_norm, time_labels_val)

                        transformed_x, _ = model(batch_input_norm, patches_latlon, batch_x, t_idx=time_labels_val)

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

                if not os.path.exists(f"{job_path}/{clim}-{ref}/baseline.jsonl"):
                    row_baseline = {k: float(np.nanmedian(v[0])) for k, v in mean_bias_percentages.items()}
                    with open(f"{job_path}/{clim}-{ref}/baseline_{val_period[0]}_{val_period[1]}.jsonl", "a") as f:
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
                    
                    

    total_runtime_seconds = time.time() - script_start_time
    total_runtime_hms = str(datetime.timedelta(seconds=int(total_runtime_seconds)))

    # Save finished.txt to mark successful completion
    finished_file = os.path.join(model_save_path, "finished.txt")
    with open(finished_file, "w") as f:
        f.write(f"Finished successfully at {datetime.datetime.now()}\n")
        f.write(f"Total runtime (seconds): {total_runtime_seconds:.2f}\n")
        f.write(f"Total runtime (hh:mm:ss): {total_runtime_hms}\n")

    
    print("\n" + "="*80)
    print("Training completed successfully!")
    print(f"Run ID: {run_id}")
    print(f"Total epochs: {num_epochs}")
    print(f"Final training loss: {loss_list[-1]:.4f}")
    print(f"Model saved to: {save_path}")
    print("="*80 + "\n")
    
    # Save finished.txt to mark successful completion
    finished_file = os.path.join(model_save_path, "finished.txt")
    with open(finished_file, "w") as f:
        f.write(f"Training completed successfully\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Total epochs: {num_epochs}\n")
        f.write(f"Final training loss: {loss_list[-1]:.4f}\n")
        f.write(f"Total runtime (seconds): {total_runtime_seconds:.2f}\n")
        f.write(f"Total runtime (hh:mm:ss): {total_runtime_hms}\n")
        f.write(f"Completed at: {datetime.datetime.now()}\n")
    
    return save_path



if __name__ == "__main__":
    main()