import cdsapi
import os
import zipfile
import traceback

year_ssp245 =  [
        "2015", "2016", "2017",
        "2018", "2019", "2020",
        "2021", "2022", "2023",
        "2024", "2025", "2026",
        "2027", "2028", "2029",
        "2030", "2031", "2032",
        "2033", "2034", "2035",
        "2036", "2037", "2038",
        "2039", "2040", "2041",
        "2042", "2043", "2044",
        "2045", "2046", "2047",
        "2048", "2049", "2050",
        "2051", "2052", "2053",
        "2054", "2055", "2056",
        "2057", "2058", "2059",
        "2060", "2061", "2062",
        "2063", "2064", "2065",
        "2066", "2067", "2068",
        "2069", "2070", "2071",
        "2072", "2073", "2074",
        "2075", "2076", "2077",
        "2078", "2079", "2080",
        "2081", "2082", "2083",
        "2084", "2085", "2086",
        "2087", "2088", "2089",
        "2090", "2091", "2092",
        "2093", "2094", "2095",
        "2096", "2097", "2098",
        "2099"
    ]


year_historical = [
        "1950", "1951", "1952",
        "1953", "1954", "1955",
        "1956", "1957", "1958",
        "1959", "1960", "1961",
        "1962", "1963", "1964",
        "1965", "1966", "1967",
        "1968", "1969", "1970",
        "1971", "1972", "1973",
        "1974", "1975", "1976",
        "1977", "1978", "1979",
        "1980", "1981", "1982",
        "1983", "1984", "1985",
        "1986", "1987", "1988",
        "1989", "1990", "1991",
        "1992", "1993", "1994",
        "1995", "1996", "1997",
        "1998", "1999", "2000",
        "2001", "2002", "2003",
        "2004", "2005", "2006",
        "2007", "2008", "2009",
        "2010", "2011", "2012",
        "2013", "2014"
    ]


#models= ['access_cm2', 'mri_esm2_0', 'gfdl_esm4', 'ipsl_cm6a_lr', 'miroc6', 'mpi_esm1_2_lr']
models= ['miroc6']
#variables = ['precipitation', 'near_surface_air_temperature']
variables = ['precipitation']

experiments = ['ssp5_8_5']

cmip6_dir = '/cmip6' #cmip6_dir
# Error log file
error_log_file = f"{cmip6_dir}/error_log.txt"
os.makedirs(os.path.dirname(error_log_file), exist_ok=True)

# Open error log file in append mode
with open(error_log_file, "a") as error_log:
    for model in models:
        for experiment in experiments:
            # Determine year range based on experiment
            if experiment == 'historical':
                year = year_historical
            else:
                year = year_ssp245

            for variable in variables:
                try:
                    # Create folder for storing data
                    folder_path = f'{cmip6_dir}/{model}/{experiment}/{variable}/'
                    os.makedirs(folder_path, exist_ok=True)

                    # Define dataset and request
                    dataset = "projections-cmip6"
                    request = {
                        "temporal_resolution": "daily",
                        "experiment": f"{experiment}",
                        "variable": f"{variable}",
                        "model": f"{model}",
                        "month": [
                            "01", "02", "03",
                            "04", "05", "06",
                            "07", "08", "09",
                            "10", "11", "12"
                        ],
                        "day": [
                            "01", "02", "03",
                            "04", "05", "06",
                            "07", "08", "09",
                            "10", "11", "12",
                            "13", "14", "15",
                            "16", "17", "18",
                            "19", "20", "21",
                            "22", "23", "24",
                            "25", "26", "27",
                            "28", "29", "30",
                            "31"
                        ],
                        "year": year,
                        "area": [60, -140, 20, -60],
                        "format": "netcdf"
                    }

                    # Define target and file path
                    target = "pr.zip"
                    file_path = os.path.join(folder_path, target)

                    # Retrieve and download the dataset
                    client = cdsapi.Client()
                    result = client.retrieve(dataset, request)
                    result.download(file_path)

                    # Extract and clean up
                    with zipfile.ZipFile(file_path, "r") as zip_ref:
                        zip_ref.extractall(folder_path)

                    os.remove(file_path)  # Remove the zip file after extraction

                except Exception as variable_error:
                    # Log variable-specific errors
                    error_message = f"Error with model: {model}, experiment: {experiment}, variable: {variable}\n"
                    error_message += traceback.format_exc()
                    error_log.write(error_message)
                    error_log.write("\n" + "="*80 + "\n")
                    continue  # Skip to the next variable
