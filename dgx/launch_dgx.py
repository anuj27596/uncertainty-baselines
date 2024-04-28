import time
import itertools
import subprocess
import logging
import numpy as np
import subprocess
import xml.etree.ElementTree as ET
import time

from absl import app
from absl import flags
import threading
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import multiprocessing as mp
# from check_gpus import check_gpu_availability

def check_gpu_availability(required_space_gb=25, required_gpus=2, exlude_ids = [], include_ids = list(range(8))):
    while True:
        # Run nvidia-smi command to get GPU info in XML format
        print(f"checking gpus excluding {exlude_ids}...")
        nvidia_smi_output = subprocess.run(['nvidia-smi', '-q', '-x'], stdout=subprocess.PIPE, check=True).stdout
        root = ET.fromstring(nvidia_smi_output)

        # Parse XML to find GPUs with enough free memory
        available_gpus = []
        for gpu in root.findall('gpu'):
            gpu_id = gpu.find('minor_number').text
            free_memory_mb = int(gpu.find('fb_memory_usage/free').text.replace(' MiB', ''))
            
            # Convert MB to GB for comparison
            free_memory_gb = free_memory_mb / 1024
            if free_memory_gb > required_space_gb:
                if int(gpu_id) not in exlude_ids and int(gpu_id) in include_ids:
                    available_gpus.append(gpu_id)

            if len(available_gpus) >= required_gpus:
                return list(map(int, available_gpus))
        
        # Pause for a short time before checking again
        time.sleep(30)

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

#     # DAN+IW
#     # best lr_0.001/dlc_1/nl_3/hdim_256/0
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
#   {	'_seed': range(1,6),
# 	} ,
#   {
# 	'_lr': [1e-3] # [1e-4, 1e-3]  
# 	} ,
#  {	'_dlc': [1] # [0.1, 0.3, 1, 3], # need to add 10
# 	},
#   {	'_grl': [1] 
# 	},
#  {	'_nl': [3] # [2, 3, 5],
# 	},
#  {	'_hdim': [256] # [256, 512],
# 	},
#  {'_organ':['processed_onehot_tum_swap2']
#  }
#  ),

# # histo dan
#     ('baselines.histopathology.jax_finetune_dan', {
# 			'output_dir': '/data/home/karmpatel/karm_8T/outputs/histopathology/<_organ>/vit_dan/lr_<_lr>/dlc_<_dlc>/grl_<_grl>/nl_<_nl>/hdim_<_hdim>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune_dan.py',
# 			'config.seed': '<_seed>',
# 			'config.lr.base':'<_lr>',
#             'config.model.domain_predictor.num_layers':'<_nl>',
# 			'config.model.domain_predictor.hid_dim':'<_hdim>',
#    			'config.model.domain_predictor.grl_coeff':'<_grl>',
# 			'config.dp_loss_coeff':'<_dlc>',
# 			'config.distribution_shift':'<_organ>'
# 				},  
#   {	'_seed': range(6),
# 	} ,
#   {
# 	'_lr':  [1e-3] # [1e-3, 1e-4]  
# 	} ,
#  {	'_dlc': [0.1] # [0.1, 0.3, 1, 3], # need to add 10
# 	},
#   {	'_grl':  [1],
# 	},
#  {	'_nl':  [3] # [2, 3, 5],
# 	},
#  {	'_hdim': [512] # [256, 512],
# 	},
#  {'_organ':['processed_onehot_tum_swap2']
#  }
#  ),


#     # IW
#     # best - lr_0.001/dlc_3/nl_3/hdim_512
#  	('baselines.histopathology.jax_finetune_impwt', {
# 			'output_dir': '/data/home/karmpatel/karm_8T/outputs/histopathology/<_organ>/vit_iw/lr_<_lr>/dlc_<_dlc>/nl_<_nl>/hdim_<_hdim>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune_impwt.py',
# 			'config.seed': '<_seed>',
#             'config.lr.base':'<_lr>',
#             'config.model.domain_predictor.num_layers':'<_nl>',
# 			'config.model.domain_predictor.hid_dim':'<_hdim>',
#    			'config.model.domain_predictor.grl_coeff':'<_grl>',
# 			'config.dp_loss_coeff':'<_dlc>',
   			
# 	},  
#   {	'_seed': range(1,6),
# 	} ,
#   {
# 	'_lr': [1e-3] # [1e-4, 1e-3]  
# 	} ,
#  {	'_dlc': [3] # [0.1, 0.3, 1, 3], # need to add 10
# 	},
#   {	'_grl': [0] # [0.05, 0.1, 3],
# 	},
#  {	'_nl': [3] # [2, 3, 5],
# 	},
#  {	'_hdim': [512] # [256, 512],
# 	},
#  {'_organ':['processed_onehot_tum_swap2']
#  }
#  ),

# isic - Dan+IW
  	('baselines.isic.jax_finetune_impwt', {
			'output_dir': '/data/home/karmpatel/karm_8T/outputs/isic/<_organ>/vit_dan_iw/lr_<_lr>/dlc_<_dlc>/nl_<_nl>/hdim_<_hdim>/<_seed>',
			'config': 'configs/isic/vit_finetune_impwt.py',
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
	'_lr':  [1e-3, 1e-4]  
	} ,
 {	'_dlc': [0.1, 0.3, 1, 3], # need to add 10
	},
  {	'_grl':  [1],
	},
 {	'_nl': [2, 3, 5],
	},
 {	'_hdim': [256, 512],
	},
 {'_organ':['upper_extremity']
 }
 ),
 ]

def cmd_from_run(run, gpu=0):
    id = run['run_id']
    cmd = [
                'python', '-m', run['module'],
                *[f'--{key}={val}' for key, val in run['args'].items()]
            ]
    cmd = ' '.join(cmd[:3]) + ' \\\n    '.join([''] + cmd[3:])

    if isinstance(gpu, list):
        gpu = ",".join(map(str, gpu))
    prefix = f'run={id}\nXLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES="{gpu}" '
    cmd = prefix + cmd + f" 2> .logs/{exp_dir}/running_{id}.txt"
    return cmd

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
    cmd = cmd_from_run(run, gpu)
    cmds += cmd + "\n =========================== \n"

with open(f'.logs/{exp_dir}/cmds.txt', "w") as fp:
    fp.write(cmds)

Q = []
lock = threading.Lock()


def create_cmd_sh(run):
    global Q
    global logger
    id = run['run_id']
    # Acquire the lock before accessing shared resource
    lock.acquire()
    try:
        # Safely perform operations on Q
        gpu = check_gpu_availability(21, 1, exlude_ids=Q)[0]
        Q.append(gpu)
        print(f"GPUS occupied.... {Q}")
        
    finally:
        # Always release the lock, even if an error occurs
        lock.release()

    cmd = cmd_from_run(run, gpu)

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
    
    # free that gpu
    Q.remove(gpu)

# runs_to_execute = list(range(5,16)) # gpu 6
# runs_to_execute = list(range(16,32)) # gpu 3
# runs_to_execute = list(range(32,100)) # gpu 3
# 1 to 5 and 11 done isic_dan_iw
# runs_to_execute = [3, 4, 12, 18, 23, 28, 30, 34, 35, 39, 40, 41, 42, 45, 47] # histo dan left
# runs_to_execute = list(range(3,48))
runs_to_execute = [44] # isic_dan_iw

# print("checking GPU availability....")
# Q = check_gpu_availability(21, 6, exlude_ids=[])
# Q = [2,4,6]

MAX_GPUS = 6
filtered_runs = [run for run in runs if run['run_id'] in runs_to_execute]  
print(f'running {len(filtered_runs)}......')    
with ThreadPoolExecutor(max_workers=MAX_GPUS) as executor:
    executor.map(create_cmd_sh, filtered_runs)
    print("running")