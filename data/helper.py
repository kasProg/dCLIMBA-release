

import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np
import regionmask
import glob
from scipy.interpolate import griddata
import cftime
import hashlib
import yaml
import os



def search_path(search_pattern):
    matching_files = glob.glob(search_pattern)
    if not matching_files:
        print(f"⚠️ No matching file found for: {search_pattern}")
        return
    else:
        path = matching_files[0]
        return path


def interpolate_time_slice(slice_data, lat_A, lon_A, lat_B, lon_B, method='nearest'):
    lon_A_2d, lat_A_2d = np.meshgrid(lon_A, lat_A)
    valid_mask = ~np.isnan(slice_data)
    points = np.column_stack((lat_A_2d[valid_mask], lon_A_2d[valid_mask]))
    values = slice_data[valid_mask]
    lon_B_2d, lat_B_2d = np.meshgrid(lon_B, lat_B)
    return griddata(points, values, (lat_B_2d, lon_B_2d), method='nearest')



def clip_netcdf(nc_file, shape_file):

    ds = xr.open_dataset(nc_file)

    # Step 2: Load the shapefile
    shapefile_path = shape_file
    shape = gpd.read_file(shapefile_path)
    shape = shape.to_crs(epsg=4326)
    shape = shape.dissolve()

    # Step 3: Set spatial dimensions and CRS for the dataset
    # Ensure NetCDF has latitude and longitude dimensions
    # if 'x' in ds.dims and 'y' in ds.dims:
    ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180)
    ds = ds.rio.write_crs("EPSG:4326", inplace=True)
    mask = regionmask.from_geopandas(shape, names="geometry")
    mask_2d = mask.mask(ds)

    # Step 4: Mask the dataset
    clipped_ds = ds.where(mask_2d==0)

    return clipped_ds



def interpolate_missing_day(var_data, time_clim, missing_date):
    """Interpolate missing day between the adjacent dates"""
    # Create cftime objects for the previous and next day
    previous_day = missing_date - np.timedelta64(1, 'D')
    next_day = missing_date + np.timedelta64(1, 'D')

    # Ensure the neighbors are in the available time data
    prev_idx = np.where(time_clim == previous_day)[0][0] if previous_day in time_clim else None
    next_idx = np.where(time_clim == next_day)[0][0] if next_day in time_clim else None

    # If both neighbors exist, interpolate
    if prev_idx is not None and next_idx is not None:
        interpolated_day = (var_data[prev_idx] + var_data[next_idx]) / 2.0
        return interpolated_day
    else:
        raise ValueError(f"Cannot interpolate for {missing_date}, missing neighbors.")
    

def get_calendar_type(time_variable):
    """Retrieve the calendar type from the dataset's time variable"""
    if hasattr(time_variable, 'calendar'):
        return time_variable.calendar
    elif isinstance(time_variable.values[0], cftime.datetime):
        return time_variable.values[0].calendar
    else:
        return 'standard'
    

def find_missing_dates(time_ds_A, time_ref_ds):
    """Find missing dates in clim_ds (time_ds_A) compared to ref_ds (time_ref_ds)"""
    missing_dates = np.setdiff1d(time_ref_ds, time_ds_A)
    return missing_dates

def get_time_units(time_variable):
    """Retrieve the units from the time variable's attributes"""
    if 'units' in time_variable.attrs:
        return time_variable.attrs['units']
    else:
        # Default to a common unit system if not specified
        return 'days since 1900-01-01'
    

class UnitManager:
    """
    A class for handling unit identification and conversion for climate datasets.
    """

    # Standard unit definitions
    STANDARD_UNITS = {
        "precipitation": "mm/day",
        "temperature": "°C",
        "wind_speed": "m/s"
    }

    # Known unit mappings to standardized units
    UNIT_MAPPING = {
        "kg m-2 s-1": "kg/m²/s",
        "kg/m^2/s": "kg/m²/s",
        "mm/day": "mm/day",
        "m/s": "m/s",
        "K": "K",
        "°C": "°C",
        "Celsius": "°C",
        "hPa": "hPa",
        "Pa": "Pa"
    }

    # Conversion factors (to standard units)
    CONVERSION_FACTORS = {
        # Convert kg/m²/s (climate model) to mm/day (standard)
        ("kg/m²/s", "mm/day"): 86400,
        ("mm", "mm/day"): 1,

        # Convert K to °C
        ("K", "°C"): lambda x: x - 273.15,

        # Convert m/s to km/h
        ("m/s", "km/h"): 3.6
    }

    def __init__(self, dataset=None):
        """
        Initializes the UnitManager.
        
        Parameters:
            dataset (xarray dataset): NetCDF/xarray dataset to extract unit information.
        """
        self.dataset = dataset
        self.unit_metadata = {}  # Stores extracted unit data

        if dataset:
            self._extract_units()

    def _extract_units(self):
        """Extracts unit metadata from an xarray dataset (NetCDF)."""
        ds = self.dataset

        for var in ds.data_vars:
            unit = ds[var].attrs.get("units", "").strip()
            standardized_unit = self.UNIT_MAPPING.get(unit, unit)
            self.unit_metadata[var] = standardized_unit

    def identify_unit(self, unit_name):
        """
        Identifies the standardized unit.

        Parameters:
            unit_name (str): Raw unit string.

        Returns:
            str: Standardized unit name.
        """
        return self.UNIT_MAPPING.get(unit_name.strip(), unit_name)

    def get_units(self):
        """Returns the extracted unit metadata from the dataset."""
        return self.unit_metadata

    def convert(self, data, variable, current_unit):
        """
        Converts data to the standard unit for the given variable.
        
        Parameters:
            data (torch.Tensor, np.ndarray, or xarray.DataArray): Input data to convert.
            variable (str): Climate variable (e.g., "precipitation", "temperature").
            current_unit (str): The current unit of the data.
        
        Returns:
            Converted data in standard units.
        """
        if current_unit in ('', None):
            print(f"Warning: No unit metadata found for {variable}. Returning dataset unchanged.")
            return data

        target_unit = self.STANDARD_UNITS.get(variable, current_unit)
        
        # If units already match, return as is
        if current_unit == target_unit:
            return data

        # Find conversion factor
        conversion_key = (current_unit, target_unit)
        if conversion_key in self.CONVERSION_FACTORS:
            factor = self.CONVERSION_FACTORS[conversion_key]

            # Apply conversion
            if callable(factor):  # Some conversions require functions (e.g., Kelvin to Celsius)
                return factor(data)
            else:
                return data * factor
        else:
            raise ValueError(f"Conversion from {current_unit} to {target_unit} not defined!")



# unit_identifier = UnitManager("/pscratch/sd/k/kas7897/Livneh/unsplit/precipitation/gfdl_esm4/prec.1950.nc")

# Extracted metadata
# print(unit_identifier.get_units())

def get_season(month):
    return {
        12: "DJF", 1: "DJF", 2: "DJF",
        3: "MAM", 4: "MAM", 5: "MAM",
        6: "JJA", 7: "JJA", 8: "JJA",
        9: "SON", 10: "SON", 11: "SON"
    }[month]

def extract_time_labels(time_array, label_type='month'):
    """
    Extracts labels from time_array for grouping:
    - 'month': groups all Januaries together (ignores year)
    - 'year-month': unique for each calendar month
    - 'season': DJF, MAM, JJA, SON
    """
    try:
        dt = pd.to_datetime(time_array)
        if label_type == 'month':
            return dt.month.astype(str).str.zfill(2).tolist()  # '01', '02', ...
        elif label_type == 'season':
            return dt.month.map(get_season).tolist()
        elif label_type == 'year-month':
            return dt.to_period("M").astype(str).tolist()
        elif label_type == 'julian-day':
            return dt.dayofyear.tolist()
    except (TypeError, ValueError):
        # For cftime or non-standard calendars
        if label_type == 'month':
            return [f"{t.month:02d}" for t in time_array]  # '01', '02', ...
        elif label_type == 'season':
            return [get_season(t.month) for t in time_array]
        elif label_type == 'year-month':
            return [f"{t.year}-{t.month:02d}" for t in time_array]
        elif label_type == 'julian-day':
            return [t.timetuple().tm_yday for t in time_array]


def generate_run_id(args_dict):
    # Convert to sorted string to ensure consistent hashing
    config_str = yaml.dump(args_dict, default_flow_style=False, sort_keys=True)
    run_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]  # Short, 8-char hash
    return run_hash

def load_run_path(run_id, base_dir='/pscratch/sd/k/kas7897/diffDownscale/jobs/'):
    # Find path for run_id
    # pattern = os.path.join(base_dir, '*','*', f'*{run_id}*')  # Wildcard to match structure
    # matching_dirs = glob.glob(pattern)

    # Recursively search for any directory containing run_id in its name
    pattern = os.path.join(base_dir, '**', f'*{run_id}*')
    matching_dirs = [d for d in glob.glob(pattern, recursive=True) if os.path.isdir(d)]
    
    if not matching_dirs:
        raise FileNotFoundError(f"No directory found for run_id {run_id}")
    if len(matching_dirs) > 1:
        raise ValueError(f"Multiple directories found for run_id {run_id}, please resolve ambiguity:\n{matching_dirs}")
    
    run_path = matching_dirs[0]


    return run_path