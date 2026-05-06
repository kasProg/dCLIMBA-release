from PIL import Image
import rasterio
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from scipy.interpolate import griddata
from rasterio.windows import Window


#######---------------THIS CODE DID NOT WORK INTIALLY. THIS DOES CHUNK INTERPOLATION OF ATTRIBUTES TO GCM's RESOLUTION-------------#########

def convert_lon_to_neg180(lons):
    return np.where(lons > 180, lons - 360, lons)

clim_models = ['gfdl_esm4']
cmip6_dir = 'cmip6' #cmip6_dir
tiff_file = "/usa_land_cover_2020v2_30m_tif/USA_NALCMS_landcover_2020v2_30m/data/landcover_project.tif" #landcover tiff file path

chunk_size = 10000  # Adjust this based on available memory

with rasterio.open(tiff_file) as attri_ds:
    tiff_transform = attri_ds.transform
    tiff_bounds = attri_ds.bounds
    tiff_res = attri_ds.res
    tiff_width = attri_ds.width
    tiff_height = attri_ds.height

    for clim in clim_models:
        precip_nc_file = f'{cmip6_dir}/{clim}/historical/precipitation/clipped_US.nc'
        attri_nc_file = f'{cmip6_dir}/{clim}/landcover.nc'

        with nc.Dataset(precip_nc_file, 'r') as precip_ds:
            lats = precip_ds.variables['lat'][:]
            lons = precip_ds.variables['lon'][:]
            lons_converted = convert_lon_to_neg180(lons)
            lon_grid, lat_grid = np.meshgrid(lons_converted, lats)

        # Create the NetCDF output file
        with nc.Dataset(attri_nc_file, 'w', format='NETCDF4') as attri_ds_out:
            lat_dim = attri_ds_out.createDimension('lat', len(lats))
            lon_dim = attri_ds_out.createDimension('lon', len(lons))

            latitudes = attri_ds_out.createVariable('lat', np.float32, ('lat',))
            longitudes = attri_ds_out.createVariable('lon', np.float32, ('lon',))

            latitudes[:] = lats
            longitudes[:] = lons

            attri = attri_ds_out.createVariable('landcover', np.float32, ('lat', 'lon'))
            attri.units = 'None'
            attri.long_name = 'Land Cover Type at the given lat/lon'

            # Initialize the output array with NaNs
            attri_interp = np.full((len(lats), len(lons)), np.nan, dtype=np.float32)

            # Interpolate in chunks
            for i in range(0, tiff_height, chunk_size):
                for j in range(0, tiff_width, chunk_size):
                    window = Window(j, i, min(chunk_size, tiff_width - j), min(chunk_size, tiff_height - i))
                    chunk_data = attri_ds.read(1, window=window)

                    # Apply bounds filtering
                    chunk_data = np.where((chunk_data < 1) | (chunk_data > 19), np.nan, chunk_data.astype(np.float32))

                    # Get the corresponding coordinates
                    chunk_transform = attri_ds.window_transform(window)
                    x_min, y_min = chunk_transform * (0, chunk_data.shape[0])
                    x_max, y_max = chunk_transform * (chunk_data.shape[1], 0)

                    # Generate lat/lon for this chunk
                    chunk_lons = np.linspace(x_min, x_max - tiff_res[0], chunk_data.shape[1])
                    chunk_lats = np.linspace(y_min, y_max + tiff_res[1], chunk_data.shape[0])
                    chunk_lon_grid, chunk_lat_grid = np.meshgrid(chunk_lons, chunk_lats)

                     # Find overlapping lat/lon indices
                    lat_min_idx = np.searchsorted(lats, chunk_lats.min(), side='left')
                    lat_max_idx = np.searchsorted(lats, chunk_lats.max(), side='right')
                    lon_min_idx = np.searchsorted(lons_converted, chunk_lons.min(), side='left')
                    lon_max_idx = np.searchsorted(lons_converted, chunk_lons.max(), side='right')

                    # Extract overlapping region from the target grid
                    target_lon_grid, target_lat_grid = np.meshgrid(
                        lons_converted[lon_min_idx:lon_max_idx],
                        lats[lat_min_idx:lat_max_idx]
                    )

                    # Interpolate this chunk only for the overlapping area
                    chunk_interp = griddata(
                        (chunk_lon_grid.flatten(), chunk_lat_grid.flatten()),
                        chunk_data.flatten(),
                        (target_lon_grid, target_lat_grid),
                        method='nearest'
                    )

                    # Update only the overlapping portion
                    attri_interp[lat_min_idx:lat_max_idx, lon_min_idx:lon_max_idx] = chunk_interp

            # Write the full array to the NetCDF
            attri[:] = attri_interp
