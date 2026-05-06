import xarray as xr
import numpy as np
from scipy.interpolate import griddata
import cftime
import pandas as pd
from data.helper import interpolate_time_slice, interpolate_missing_day, find_missing_dates, search_path

######-----THIS INTERPOLATES CMIP6 CLIMATE DATA TO LIVNEH'S RESOLUTION-----######

spatial_interp = True ## Set False to only do temporal interpolation
time_interp = False
### Interpolating Clim Models to Livneh crds
models= {'access_cm2':'ACCESS-CM2'}
variables = {'precipitation':'pr'}
exps = {'historical':'historical', 'ssp2_4_5': 'ssp245'}
cmip6_dir = "/cmip6" #cmip6_dir
ref_livneh_path = f"/Livneh/prec.1980.nc" #reference livneh file path

for model in models:
    for var in variables:
        for exp in exps:
            if exp == 'historical':
                start_date, end_date = "19500101", "20141231"
            else:
                start_date, end_date = "20150101", "20991231"

            clim_path_search= f"cmip6/{model}/{exp}/{var}/{variables[var]}_day_{models[model]}_{exps[exp]}_r1i1p1f1_*_{start_date}-{end_date}.nc"
            clim_path = search_path(clim_path_search)
            clim_ds = xr.open_dataset(clim_path)

            # for year in range(1980,1996):
            ref_ds = xr.open_dataset(ref_livneh_path)

            if var =='precipitation':
                var_clim = clim_ds[variables[var]].values*86400 ##converting to mm/day
            else:
                var_clim = clim_ds[variables[var]].values


            if time_interp:
                if isinstance(clim_ds['time'].values[0], cftime.datetime):
                    # Convert cftime to pandas datetime
                    new_time = pd.to_datetime([t.strftime('%Y-%m-%d') for t in clim_ds['time'].values])

                    # Replace the time coordinate
                    clim_ds = clim_ds.assign_coords(time=new_time)
                    
                # Find missing dates in clim_ds
                missing_dates = find_missing_dates(time_clim, time_ref)
                # Handle missing dates by interpolation
                for missing_date in missing_dates:
                    interpolated_day = interpolate_missing_day(var_clim, time_clim, missing_date)
                    insert_idx = np.searchsorted(time_clim, missing_date)  # Find where to insert the missing date
                    time_clim = np.insert(time_clim, insert_idx, missing_date)
                    var_clim = np.insert(var_clim, insert_idx, interpolated_day, axis=0)
            
            ## Bicubic interpolation
            lat_clim = clim_ds.lat.values
            lon_clim = clim_ds.lon.values
            time_clim = clim_ds.time.values

            
            # Extract target lat and lon from file B
            lat_ref = ref_ds.lat.values
            lon_ref = ref_ds.lon.values
            time_ref = ref_ds.time.values
            

            if spatial_interp:
                # Create meshgrid for target coordinates
                lon_mesh, lat_mesh = np.meshgrid(lon_ref, lat_ref)
                var_interp = np.zeros((len(time_clim), len(lat_ref), len(lon_ref)))

                # Perform interpolation
                for t in range(len(time_ref)):
                    var_interp[t] = interpolate_time_slice(var_clim[t], lat_clim, lon_clim, lat_ref, lon_ref)

                ds_interp = xr.Dataset(
                    data_vars={
                        variables[var]: (['time', 'lat', 'lon'], var_interp)
                    },
                    coords={
                        'time': time_clim,
                        'lat': lat_ref,
                        'lon': lon_ref
                    }
                )
                if var == 'precipitation':
                    ds_interp[variables[var]] = ds_interp[variables[var]].where(ds_interp[variables[var]] >= 0, 0)
                    ref_ds_aligned = ref_ds.reindex(time=ds_interp['time'], method='nearest')
                    ds_interp[variables[var]] = ds_interp[variables[var]].where(ref_ds_aligned.prec == ref_ds_aligned.prec, np.nan)

            else:
                ds_interp = xr.Dataset(
                    data_vars={
                        variables[var]: (['time', 'lat', 'lon'], var_clim)
                    },
                    coords={
                        'time': time_clim,
                        'lat': lat_clim,
                        'lon': lon_clim
                    }
                )

            # Add attributes if necessary
            ds_interp.attrs = clim_ds.attrs
            

            # Save the new dataset as a NetCDF file
            ds_interp.to_netcdf(f"{cmip6_dir}/{model}/{exp}/{var}/livneh_grid.nc")
            
            # noisy_clim_ds.to_netcdf(f'/data/kas7897/Livneh/noisy_new/prec_{year}.nc')
            print(f'{model}_{var}_{exps} Yo Killed!')