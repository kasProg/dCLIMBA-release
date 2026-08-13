import torch
import os
from ibicus.evaluate.metrics import *
from model.benchmark import BiasCorrectionBenchmark
import data.helper as helper
import argparse

###-----The code is currently accustomed to CMIP6-Livneh/gridmet Data format ----###

torch.manual_seed(42)

device = torch.device('cpu')


parser = argparse.ArgumentParser(description="Evaluate experiment")
parser.add_argument('--run_id', type=str, required=True, help='Run ID')
parser.add_argument('--base_dir', type=str, required=True, help='Base directory for outputs')
parser.add_argument('--validation', action='store_true')

## add argument for test period list
parser.add_argument('--test_period', type=str, required=False, help='Test period, format: start_year,end_year')

args = parser.parse_args()

if args.test_period:
    args.test_period = [int(x) for x in args.test_period.split(',')]
    test_period= args.test_period


run_id = args.run_id
validation = args.validation
base_dir = args.base_dir

run_path, config = helper.load_trial_config(run_id, base_dir=base_dir)

logging = True

if validation:
    test_period = [config['val_start'], config['val_end']]

cmip6_dir = config['cmip_dir']
ref_path = config['ref_dir']

clim = config['clim']
ref = config['ref']

input_x = {'precipitation': ['pr', 'prec', 'prcp' 'PRCP', 'precipitation']}
clim_var = 'pr'
ref_var = config['ref_var']

### FOR TREND ANALYSIS
trend_analysis = True
scenario = 'ssp5_8_5'
# trend_future_period = [config['trend_start'], config['trend_end']]
trend_future_period = [2015, 2099]

train_period = [config['train_start'], config['train_end']]


save_path = run_path
model_save_path = save_path
save_path =  save_path + f'/{test_period[0]}_{test_period[1]}/'

methods = ['QuantileMapping','ISIMIP', 'ECDFM', 'QuantileDeltaMapping']

bench = BiasCorrectionBenchmark(clim = clim,
                                ref = ref,
                                hist_period = train_period, 
                                test_period = test_period, 
                                scenario = 'historical', 
                                clim_var = clim_var, 
                                correction_methods = methods,  
                                model_path = model_save_path, 
                                test_path = save_path)  
bench.apply_correction()


if trend_analysis:
    future_save_path = model_save_path + f'/{scenario}_{trend_future_period[0]}_{trend_future_period[1]}/'
    os.makedirs(future_save_path, exist_ok=True)

    bench = BiasCorrectionBenchmark(clim = clim,
                                ref = ref,
                                hist_period = train_period, 
                                test_period = trend_future_period, 
                                scenario = scenario, 
                                clim_var = clim_var, 
                                correction_methods = methods,  
                                model_path = model_save_path, 
                                test_path = future_save_path)
    bench.apply_correction()



