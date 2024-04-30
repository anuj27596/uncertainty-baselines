import time
import itertools
import subprocess
import logging
import numpy as np

from absl import app
from absl import flags
import threading
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import time

# os.makedirs('.logs', exist_ok=True)

if len(sys.argv) == 1:
    exp_dir = "debug"
else:
    exp_dir = sys.argv[1]
os.makedirs(f'.logs/{exp_dir}', exist_ok=True)
os.makedirs(f'.logs/{exp_dir}/slurm_jobs', exist_ok=True)


logging.basicConfig(filename=f".logs/{exp_dir}/status.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

DATASET_NAME = "histopathology"

prefix_slurm = f'''#!/bin/sh
#SBATCH --job-name=cx # Job name
#SBATCH --ntasks=32
#SBATCH --time=46:00:00 # Time limit hrs:min:sec
#SBATCH --output=.logs/{exp_dir}/slurm_jobs/cx%j.out # Standard output and error log
#SBATCH --gres=gpu:1 
#SBATCH --partition=low_unl_1gpu
'''

run_settings = [

    # DAN+IW
 	('baselines.chest_xray.jax_finetune_impwt', {
			'output_dir': '/data3/home/karmpatel/karm_8T/outputs/chest_xray/<_organ>/vit_dan_iw/lr_<_lr>/dlc_<_dlc>/nl_<_nl>/hdim_<_hdim>/<_seed>',
			'config': 'configs/chest_xray/vit_finetune_impwt.py',
			'config.seed': '<_seed>',
            'config.lr.base':'<_lr>',
            'config.model.domain_predictor.num_layers':'<_nl>',
			'config.model.domain_predictor.hid_dim':'<_hdim>',
   			'config.model.domain_predictor.grl_coeff':'<_grl>',
			'config.dp_loss_coeff':'<_dlc>',
	},  
  {	'_seed': range(1),
	} ,
  {
	'_lr': [1e-4, 1e-3]  
	} ,
 {	'_dlc': [0.1, 0.3, 1, 3], #[0.05, 0.1, 3], # need to add 10
	},
  {	'_grl': [1] #[0.05, 0.1, 3],
	},
 {	'_nl': [2, 3, 5],
	},
 {	'_hdim': [256, 512],
	},
 {'_organ':['vit-cxfr-new']
 }
 ),

    # IW
#  	('baselines.chest_xray.jax_finetune_impwt', {
# 			'output_dir': '/data3/home/karmpatel/karm_8T/outputs/chest_xray/<_organ>/vit_iw/lr_<_lr>/dlc_<_dlc>/nl_<_nl>/hdim_<_hdim>/<_seed>',
# 			'config': 'configs/chest_xray/vit_finetune_impwt.py',
# 			'config.seed': '<_seed>',
#             'config.lr.base':'<_lr>',
#             'config.model.domain_predictor.num_layers':'<_nl>',
# 			'config.model.domain_predictor.hid_dim':'<_hdim>',
#    			'config.model.domain_predictor.grl_coeff':'<_grl>',
# 			'config.dp_loss_coeff':'<_dlc>',
   			
# 	},  
#   {	'_seed': range(1),
# 	} ,
#   {
# 	'_lr': [1e-4, 1e-3]  
# 	} ,
#  {	'_dlc': [0.1, 0.3, 1, 3], #[0.05, 0.1, 3], # need to add 10
# 	},
#   {	'_grl': [0] #[0.05, 0.1, 3],
# 	},
#  {	'_nl': [2, 3, 5],
# 	},
#  {	'_hdim': [256, 512],
# 	},
#  {'_organ':['vit-cxfr-new']
#  }
#  ),
 ]

run_id = 0
runs = []
for module, static_args, *iter_args in run_settings:
    keys = [k for d in iter_args for k in d.keys()]
    for vals in itertools.product(*[zip(*d.values()) for d in iter_args]):
        vals = [v2 for v1 in vals for v2 in v1]
        run = {'module': module, 'args': {}}
        run['args'].update(static_args)
        for k, v in zip(keys, vals):
            if not k.startswith('_'):
                run['args'][k] = v
            for s_key, s_val in run['args'].items():
                if isinstance(s_val, str):
                    run['args'][s_key] = s_val.replace(f'<{k}>', str(v))
        run['run_id'] = run_id
        runs.append(run)
        run_id+=1

cmds = ""
def cmd_from_run(run, gpu=0):
    cmd = [
                'python', '-m', run['module'],
                *[f'--{key}={val}' for key, val in run['args'].items()]
            ]
    cmd = ' '.join(cmd[:3]) + ' \\\n    '.join([''] + cmd[3:])
    prefix = f"\nrun={run['run_id']}" + f'\nCUDA_VISIBLE_DEVICES="{gpu}" '
    # cmd = "echo trial\nsleep 5" # debug purpose
    cmd = prefix_slurm + prefix + cmd
    return cmd


FLAGS = flags.FLAGS
flags.DEFINE_integer('exp_id', None, 'experiment id')
flags.DEFINE_integer('run_id', None, 'run id')
flags.DEFINE_bool('run', False, 'whether to run')
flags.DEFINE_bool('loop', False, 'whether to run')

Q = [1, 3, 4]
lock = threading.Lock()
def create_cmd_sh(run):
    global Q
    global logger
    id = run['run_id']
    # Acquire the lock before accessing shared resource
    lock.acquire()
    try:
        # Safely perform operations on Q
        gpu = Q[0]
        Q.append(Q.pop(0))
    finally:
        # Always release the lock, even if an error occurs
        lock.release()

    cmd = cmd_from_run(run, gpu)
    with open(f'slurm_{gpu}.sh', "w") as fp:
        fp.write(cmd)

    slurm_cmd = f"sbatch slurm_{gpu}.sh"    
    logger.debug(cmd)
    result = subprocess.run(slurm_cmd, shell=True, capture_output=True, text=True)
    slurm_id = result.stdout.split('job')[-1].strip()
    status = subprocess.run(f"squeue -j {slurm_id}", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        # Command failed, print standard error
        with open(f'.logs/{exp_dir}/failed_{id}.txt', "w") as fp:
            fp.write(str(id) + '\n')
            fp.write("slurm id:" + slurm_id + "\n")
            fp.write(cmd + "\n")
            fp.write(result.stderr)
    else:
        # check if process is running
        while slurm_id in status.stdout:
            time.sleep(30)
            status = subprocess.run(f"squeue -j {slurm_id}", shell=True, capture_output=True, text=True)

        with open(f'.logs/{exp_dir}/succ_{id}.txt', "w") as fp:
            fp.write("id:" + str(id) + '\n')
            fp.write("slurm id:" + slurm_id + "\n")
            fp.write(cmd + "\n")
        
      
cmds = ""
for run in runs:
    cmds += cmd_from_run(run) + "\n======================================\n"

with open(f'.logs/{exp_dir}/cmds.txt', "w") as fp:
    fp.write(cmds)

runs_to_execute = [9, 16] + list(range(18,100))
filtered_runs = [run for run in runs if run['run_id'] in runs_to_execute]  
print(f'running {len(filtered_runs)}......')

# for run in filtered_runs:
#     create_cmd_sh(run)

with ThreadPoolExecutor(max_workers=len(Q)) as executor:
    executor.map(create_cmd_sh, filtered_runs)