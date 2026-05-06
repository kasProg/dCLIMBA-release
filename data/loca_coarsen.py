
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from helper import interpolate_time_slice, clip_netcdf, search_path

######------THIS CODE INTERPOLATES (COARSENS) LOCA2 TO CLIMATE RESOLUTION AND CLIPS INTO CONUS REGION--------#######
models= {'access_cm2':'ACCESS-CM2',
        'gfdl_esm4': 'GFDL-ESM4',
        'ipsl_cm6a_lr': 'IPSL-CM6A-LR',
        'miroc6': 'MIROC6',
        'mpi_esm1_2_lr': 'MPI-ESM1-2-LR',
        'mri_esm2_0': 'MRI-ESM2_0'}

variables = {'precipitation':'pr'}
exps = {'ssp5_8_5': 'ssp585'}
conus_shape_file = "/conus/conus.shp" #conus shapefile path
cmip6_dir = "/cmip6" #cmip6_dir

for model in models:
    for var in variables:
      for exp in exps:
        save_path = f'{cmip6_dir}/{model}/{exp}/{var}/loca'
        os.makedirs(save_path, exist_ok=True)
        if exp == 'historical':
            start_date, end_date = "19500101", "20141231"
            loca_path_search = f'{save_path}/{variables[var]}.{models[model]}.{exps[exp]}.*.nc'
        else:
            loca_path_search = f'{cmip6_dir}/{model}/{exps[exp]}/{var}/loca/{variables[var]}.{models[model]}.{exps[exp]}.*.2045-2074.*.nc'
            start_date, end_date = "20150101", "20991231"

        clim_path_search= f"{cmip6_dir}/{model}/{exp}/{var}/{variables[var]}_day_{models[model]}_{exps[exp]}_r1i1p1f1_*_{start_date}-{end_date}.nc"
        
        
        clim_path = search_path(clim_path_search)
        loca_path = search_path(loca_path_search)

        loca_ds = xr.open_dataset(loca_path, chunks={'time': 10})
        ds_og = xr.open_dataset(clim_path, chunks={'time': 10})

        if not os.path.exists(f'{save_path}/coarse.nc'):
            # Create a new dataset with noisy precipitation
            # loca_ds_copy = loca_ds.copy()
            time = loca_ds.time.values
            prcp_n = loca_ds.pr.values

            # ## Bicubic interpolation
            lat_og = ds_og.lat.values
            lon_og = ds_og.lon.values
            lat_A = loca_ds.lat.values
            lon_A = loca_ds.lon.values

            final = np.zeros((len(time), len(lat_og), len(lon_og)))

            # Perform interpolation
            for t in range(len(time)):
                final[t] = interpolate_time_slice(prcp_n[t], lat_A, lon_A, lat_og, lon_og)



            ds_C = xr.Dataset(
                data_vars={
                    variables[var]: (['time', 'lat', 'lon'], final)
                },
                coords={
                    'time': time,
                    'lat': lat_og,
                    'lon': lon_og
                }
            )
            
            ds_C['pr'] = ds_C['pr'].where(ds_C['pr'] >= 0, 0)

            ds_C.to_netcdf(f'{save_path}/coarse.nc')
            del loca_ds
            print(f'Yo Yo LOCA Coarsened {model}_{exp}_{var}')


        ds_C_clipped = clip_netcdf(nc_file = f'{save_path}/coarse.nc', shape_file = conus_shape_file)
        ds_C_clipped[variables[var]].to_netcdf(f'{save_path}/coarse_USclip2.nc')
        print(f'Yo Yo LOCA Clipped {model}_{exp}_{var}')




