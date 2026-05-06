import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import subprocess
import os
import time
from itertools import product

@hydra.main(version_base=None, config_path="configs", config_name="config")
def launcher(cfg: DictConfig):
    """Launch multiple jobs with parameter sweeps, managing GPU availability"""
    launcher_start_time = time.time()
    
    print("="*80)
    print("Launcher Configuration:")
    print("="*80)
    print(OmegaConf.to_yaml(cfg))
    print("="*80)
    
    # Automatically detect sweep parameters
    sweep_params = {}
    
    skip_keys = {
        'available_gpus', 'save_path', 'logging_path', 'cmip_dir', 
        'ref_dir', 'defaults', 'loss', 'spatial_extent', 'spatial_extent_val', 'spatial_extent_test',
        'check_interval'
    }
    
    for key, value in cfg.items():
        if key in skip_keys:
            continue
        
        if isinstance(value, (list, tuple, ListConfig)) and len(value) > 1:
            sweep_params[key] = list(value)
            print(f"✓ Sweep parameter: {key} = {value}")
        elif isinstance(value, (list, tuple, ListConfig)) and len(value) == 1:
            # Single item lists become single values
            sweep_params[key] = [value[0]]
            print(f"✓ Single parameter: {key} = {value[0]}")
        elif not isinstance(value, (list, tuple, ListConfig)):
            # Single values become single-item lists for consistency
            sweep_params[key] = [value]
            print(f"✓ Single parameter: {key} = {value}")
    
    if not sweep_params:
        print("\n❌ No sweep parameters detected!")
        print("Make sure your sweep config has parameters to process.")
        return
    
    # Generate all combinations
    keys = list(sweep_params.keys())
    values = [sweep_params[k] for k in keys]
    combinations = list(product(*values))
    
    # Configuration
    check_interval = cfg.get('check_interval', 5)  # Seconds between checks
    available_gpus = list(cfg.available_gpus)
    
    print(f"\n{'='*80}")
    print(f"Sweep Summary:")
    print(f"{'='*80}")
    print(f"Total jobs: {len(combinations)}")
    print(f"Available GPUs: {available_gpus}")
    print(f"Check interval: {check_interval}s")
    print(f"Sweep parameters: {keys}")
    print(f"{'='*80}\n")
    
    # Determine which script to run
    if cfg.get('train', True):
        run_file = "run_exp.py"
    else:
        run_file = "run_val.py"
    
    # Create list of parameter combinations (jobs to run)
    param_combinations = [dict(zip(keys, combo)) for combo in combinations]
    
    # Track GPU jobs: {gpu_id: process_object or None}
    gpu_jobs = {gpu: None for gpu in available_gpus}
    
    print(f"Starting job queue with {len(param_combinations)} jobs...\n")
    
    def launch_job(params, gpu_id, run_file):
        """Launch a single job on specified GPU"""
        command = ["python", run_file]
        
        # Add sweep parameters as overrides
        for k, v in params.items():
            if isinstance(v, str) and ';' in v:
                command += [f"{k}='{v}'"]
            else:
                command += [f"{k}={v}"]
        
        print(f"Launching on GPU {gpu_id}: {params}")
        print(f"  Command: {' '.join(command)}")
        
        return subprocess.Popen(
            command,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu_id)}
        )
    
    try:
        # Main job distribution loop
        while param_combinations or any(gpu_jobs.values()):
            # Check for finished jobs and launch new ones
            for gpu_id, job in list(gpu_jobs.items()):
                # Check if job finished
                if job is not None and job.poll() is not None:
                    print(f"\n✓ Job finished on GPU {gpu_id}")
                    gpu_jobs[gpu_id] = None
                
                # Launch new job if GPU is free and jobs remaining
                if gpu_jobs[gpu_id] is None and param_combinations:
                    params = param_combinations.pop(0)
                    gpu_jobs[gpu_id] = launch_job(params, gpu_id, run_file)
            
            # Progress update
            completed = len(combinations) - len(param_combinations) - sum(1 for j in gpu_jobs.values() if j is not None)
            running = sum(1 for j in gpu_jobs.values() if j is not None)
            pending = len(param_combinations)
            
            gpu_status = ', '.join([
                f"GPU{gpu}:{'busy' if gpu_jobs[gpu] is not None else 'free'}" 
                for gpu in available_gpus
            ])
            
            print(f"\r[Progress] Completed: {completed}/{len(combinations)} | "
                  f"Running: {running} ({gpu_status}) | Pending: {pending}  ",
                  end='', flush=True)
            
            # Short sleep to reduce CPU usage
            time.sleep(check_interval)
    
    except KeyboardInterrupt:
        elapsed = time.time() - launcher_start_time
        print("\n\n⚠️  Interrupted! Cleaning up running jobs...")
        for gpu_id, job in gpu_jobs.items():
            if job is not None:
                job.terminate()
        elapsed_h = int(elapsed // 3600)
        elapsed_m = int((elapsed % 3600) // 60)
        elapsed_s = int(elapsed % 60)
        print(f"Elapsed runtime before interrupt: {elapsed:.2f}s ({elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d})")
        print("Cleanup complete.")
        return
    
    elapsed = time.time() - launcher_start_time
    elapsed_h = int(elapsed // 3600)
    elapsed_m = int((elapsed % 3600) // 60)
    elapsed_s = int(elapsed % 60)

    print(f"\n\n{'='*80}")
    print(f"✓ All jobs finished!")
    print(f"Total launcher runtime: {elapsed:.2f}s ({elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d})")
    print(f"{'='*80}")

if __name__ == "__main__":
    launcher()
