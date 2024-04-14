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
from check_gpus import check_gpu_availability

os.makedirs('.logs', exist_ok=True)

exp_dir = sys.argv[1]
os.makedirs(f'.logs/{exp_dir}', exist_ok=True)

logging.basicConfig(filename=f".logs/{exp_dir}/status.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

DATASET_NAME = "histopathology"
run_settings = [

    # DAN+IW
#  	('baselines.histopathology.jax_finetune_impwt', {
# 			'output_dir': '/data/home/karmpatel/karm_8T/outputs/histopathology/<_organ>/vit_dan_iw/lr_<_lr>/dlc_<_dlc>/nl_<_nl>/hdim_<_hdim>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune_impwt.py',
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
#   {	'_grl': [1] #[0.05, 0.1, 3],
# 	},
#  {	'_nl': [2, 3, 5],
# 	},
#  {	'_hdim': [256, 512],
# 	},
#  {'_organ':['processed_onehot_tum_swap2']
#  }
#  ),

    # IW
 	('baselines.histopathology.jax_finetune_impwt', {
			'output_dir': '/data/home/karmpatel/karm_8T/outputs/histopathology/<_organ>/vit_iw/lr_<_lr>/dlc_<_dlc>/nl_<_nl>/hdim_<_hdim>/<_seed>',
			'config': 'configs/histopathology/vit_finetune_impwt.py',
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
  {	'_grl': [0] #[0.05, 0.1, 3],
	},
 {	'_nl': [2, 3, 5],
	},
 {	'_hdim': [256, 512],
	},
 {'_organ':['processed_onehot_tum_swap2']
 }
 ),
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
for run in runs:
    gpu=0
    cmd = [
                'python', '-m', run['module'],
                *[f'--{key}={val}' for key, val in run['args'].items()]
            ]
    cmd = ' '.join(cmd[:3]) + ' \\\n    '.join([''] + cmd[3:])
    prefix = f"run={run['run_id']}" + f'\nCUDA_VISIBLE_DEVICES="{gpu}" '
    cmd = prefix + cmd
    cmds += cmd + "\n =========================== \n"

with open(f'.logs/{exp_dir}/cmds.txt', "w") as fp:
    fp.write(cmds)

def get_run_ids():
    return range(len(runs))

def get_output_dir(run_id, exp_id):
    return runs[run_id]['args']['output_dir'].replace('<exp_id>', str(exp_id))


FLAGS = flags.FLAGS
flags.DEFINE_integer('exp_id', None, 'experiment id')
flags.DEFINE_integer('run_id', None, 'run id')
flags.DEFINE_bool('run', False, 'whether to run')
flags.DEFINE_bool('loop', False, 'whether to run')

Q = [6]
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
    cmd = [
                'python', '-m', run['module'],
                *[f'--{key}={val}' for key, val in run['args'].items()]
            ]
    cmd = ' '.join(cmd[:3]) + ' \\\n    '.join([''] + cmd[3:])

    prefix = f'run={id}\nCUDA_VISIBLE_DEVICES="{gpu}" '
    cmd = prefix + cmd

    logger.debug(cmd)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        # Command failed, print standard error
        with open(f'.logs/{exp_dir}/failed_{id}.txt', "w") as fp:
            fp.write(str(id) + '\n')
            fp.write(cmd + "\n")
            fp.write(result.stderr)
    else:
        with open(f'.logs/{exp_dir}/succ_{id}.txt', "w") as fp:
            fp.write(str(id) + '\n')
            fp.write(cmd + "\n")
        
def create_cmd_sh_gpu_list(run):
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
    cmd = [
                'python', '-m', run['module'],
                *[f'--{key}={val}' for key, val in run['args'].items()]
            ]
    cmd = ' '.join(cmd[:3]) + ' \\\n    '.join([''] + cmd[3:])
    
    gpu = ",".join(map(str, gpu))
    prefix = f'run={id}\nCUDA_VISIBLE_DEVICES="{gpu}" '
    cmd = prefix + cmd

    logger.debug(cmd)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        # Command failed, print standard error
        with open(f'.logs/{exp_dir}/failed_{id}.txt', "w") as fp:
            fp.write(str(id) + '\n')
            fp.write(cmd + "\n")
            fp.write(result.stderr)
    else:
        with open(f'.logs/{exp_dir}/succ_{id}.txt', "w") as fp:
            fp.write(str(id) + '\n')
            fp.write(cmd + "\n")

# runs_to_execute = list(range(5,16)) # gpu 6
# runs_to_execute = list(range(16,32)) # gpu 3
# runs_to_execute = list(range(32,100)) # gpu 3
runs_to_execute =  [32, 40]
print("checking GPU availability....")
# Q = check_gpu_availability(21, 4, exlude_ids=[])
Q = [[2,3], [4,5]]
print(f"GPUS occupied.... {Q}")
filtered_runs = [run for run in runs if run['run_id'] in runs_to_execute]  
print(f'running {len(filtered_runs)}......')    
with ThreadPoolExecutor(max_workers=len(Q)) as executor:
    # executor.map(create_cmd_sh, filtered_runs)
    executor.map(create_cmd_sh_gpu_list, filtered_runs)