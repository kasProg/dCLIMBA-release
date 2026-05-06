
import xarray as xr
import geopandas as gpd
import rioxarray
import pandas as pd
import warnings
import numpy as np
import regionmask
import glob
from helper import clip_netcdf, search_path

#######-------THIS CODE CLIPS CLIMATE-MODELS TO CONUS SHAPEFILE-------#######

models= {'ipsl_cm6a_lr': 'IPSL-CM6A-LR',
         'mpi_esm1_2_lr':'MPI-ESM1-2-LR', 
         'mri_esm2_0':'MRI-ESM2-0', 
         'access_cm2':'ACCESS-CM2', 
         'miroc6':'MIROC6',
         'gfdl_esm4':'GFDL-ESM4'}
variables = {'precipitation':'pr', 'near_surface_air_temperature':'tas'}
exps = {'historical': 'historical', 'ssp5_8_5':'ssp585'}
shapefile_path = "/conus/conus.shp" #conus shapefile path
cmip6_location = "cmip6" #cmip6_location

for model in models:
    for var in variables:
      for exp in exps:

        if exp == 'historical':
                start_date, end_date = "19500101", "20141231"
        else:
            start_date, end_date = "20150101", "20991231"

        # Construct the search pattern for files with any grid type (*)
        search_pattern = f"{cmip6_location}/{model}/{exp}/{var}/{variables[var]}_day_{models[model]}_{exps[exp]}_r1i1p1f1_*_{start_date}-{end_date}.nc"
        netcdf_file = search_path(search_pattern)
          
        # Step 1: Open the NetCDF file
        clipped_ds = clip_netcdf(nc_file = netcdf_file, shape_file = shapefile_path)

        # Step 5: Save the clipped dataset
        output_path = f"{cmip6_location}/{model}/{exp}/{var}/clipped_US.nc"
        clipped_ds[variables[var]].to_netcdf(output_path)

        print(f"Clipped NetCDF saved to {output_path}")




