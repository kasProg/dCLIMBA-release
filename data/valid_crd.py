import xarray as xr
import numpy as np
from scipy.stats import wasserstein_distance
import geopandas as gpd
from shapely.geometry import Point

def valid_lat_lon(ds, var_name='prec', shapefile_path=None, attr='OBJECTID', attrList = None):
    if 'lat' in ds.variables:
        lat = ds.lat.values
        lon = ds.lon.values
    else:
        lat = ds.latitude.values
        lon = ds.longitude.values
    
    # Create meshgrid for target coordinates
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    # Create a mask for valid (non-NaN) values
    valid_mask = ds[var_name].values[0] == ds[var_name].values[0]

    # Extract valid latitude and longitude pairs
    valid_lats = lat_mesh[valid_mask]
    valid_lons = lon_mesh[valid_mask]
    valid_coords = np.column_stack((valid_lats, valid_lons))
    if shapefile_path is not None:
        # Load shapefile
        gdf = gpd.read_file(shapefile_path)

        if attrList is not None:
            gdf = gdf[gdf[attr].isin(attrList)]


        # Optional: ensure consistent CRS
        if hasattr(ds, 'crs') and gdf.crs != ds.crs:
            gdf = gdf.to_crs(ds.crs)

        # Create GeoSeries of valid Points
        points = gpd.GeoSeries([Point(lon, lat) for lat, lon in valid_coords], crs=gdf.crs)

        # Create a dictionary to store grouped coords
        grouped_coords = {}

        for idx, shape in gdf.iterrows():
            name = shape[attr] if attr else idx
            mask = points.within(shape.geometry)
            coords_in_shape = valid_coords[mask.values]
            if len(coords_in_shape) > 0:
                grouped_coords[name] = coords_in_shape

        valid_coords = np.vstack(list(grouped_coords.values()))

    return valid_coords



def reconstruct_nc(x_values, valid_coords, time_values, var):
    """
    Reform an xarray dataset with (time, lat, lon) dimensions and save as NetCDF.

    Parameters:
    - x_values: 2D NumPy array of shape (time, points), filtered values.
    - valid_coords: Nx2 NumPy array containing valid (lat, lon) pairs.
    - time_values: 1D NumPy array containing time values.
    - output_nc_path: Path to save the NetCDF file.
    """

    # Extract unique lat/lon values to create a grid
    unique_lats = np.sort(np.unique(valid_coords[:, 0]))
    unique_lons = np.sort(np.unique(valid_coords[:, 1]))

    # Initialize an empty 3D array (time, lat, lon) filled with NaNs
    x_reshaped = np.full((len(time_values), len(unique_lats), len(unique_lons)), np.nan)

    # Create lookup dictionaries to map lat/lon values to array indices
    lat_to_idx = {lat: i for i, lat in enumerate(unique_lats)}
    lon_to_idx = {lon: i for i, lon in enumerate(unique_lons)}

    # Fill the 3D array with x_values based on valid_coords
    for point_idx, (lat, lon) in enumerate(valid_coords):
        lat_idx = lat_to_idx[lat]
        lon_idx = lon_to_idx[lon]
        x_reshaped[:, lat_idx, lon_idx] = x_values[:, point_idx]  # Assign values at correct positions

    # Create an xarray Dataset
    ds_filtered = xr.Dataset(
        {
            var : (["time", "lat", "lon"], x_reshaped)  # Define variable with correct dimensions
        },
        coords={
            "time": ("time", time_values),
            "lat": ("lat", unique_lats),
            "lon": ("lon", unique_lons)
        }
    )

    return ds_filtered


## To test files
#
# ds_B = xr.open_dataset(f"/data/kas7897/Livneh/prec.1980.nc")
# ds_A = xr.open_dataset(f"/data/kas7897/Livneh/upscale_1by4_bci/prec_1980.nc")
# ds_A = xr.open_dataset("/data/shared_data/NLDAS/NLDAS_FORA0125_H002/NLDAS_FORA0125_H.A19790101.0000.002.grb.SUB.nc4")
# #
# valid_coords = valid_lat_lon(ds_B)
# #
# # ## calling precpitation values
# #
# prec_B = ds_B['prec'].sel(lat=xr.DataArray(valid_coords[:, 0], dims='points'), lon=xr.DataArray(valid_coords[:, 1], dims='points'), method='nearest').values
# prec_A = ds_A['prec'].sel(lat=xr.DataArray(valid_coords[:, 0], dims='points'), lon=xr.DataArray(valid_coords[:, 1], dims='points'), method='nearest').values
#
# # Finding num of crd with all zeros
# print(np.sum(np.all(prec_B == 0, axis=0)))
# print(np.sum(np.all(prec_A == 0, axis=0)))
#
#
#
# plt.figure(figsize=(12, 6))
# plt.hist(prec_A[:,400], bins=30, alpha=0.6, label="Noisy")
# plt.hist(prec_B[:,400], bins=30, alpha=0.6, label="Original", color='orange')
# plt.title("Original Gaussian and Target Skewed Distributions")
# plt.legend()
