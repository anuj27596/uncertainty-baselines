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
# from check_gpus import check_gpu_availability

DATASET_NAME = "histopathology"
run_settings = [

    ('baselines.histopathology.jax_finetune_dan', {
			'output_dir': '/data/home/karmpatel/karm_8T/outputs/histopathology/<_organ>/vit_dan/lr_<_lr>/dlc_<_dlc>/grc_<_grc>/nl_<_nl>/hdim_<_hdim>/<_seed>',
			'config': 'configs/histopathology/vit_finetune_dan.py',
			'config.seed': '<_seed>',
			'config.lr.base':'<_lr>',
            'config.model.domain_predictor.num_layers':'<_nl>',
			'config.model.domain_predictor.hid_dim':'<_hdim>',
   			'config.model.domain_predictor.grl_coeff':'<_grl>',
			'config.dp_loss_coeff':'<_dlc>',
			'config.distribution_shift':'<_organ>'
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
 {'_organ':['processed_onehot_tum_swap2']
 }
 ),

 ]

run_id = 0
runs = []

with open("txt_files/histo-dan-done-info.txt") as fp:
    done = fp.readlines()

done = list(map(lambda x: x.strip(), done))
left = []
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

        run_args = run['args']
        run_string = f"lr_{run_args['config.lr.base']}/dlc_{run_args['config.dp_loss_coeff']}/layers_{run_args['config.model.domain_predictor.num_layers']}/dim_{run_args['config.model.domain_predictor.hid_dim']} *"
        if run_string not in done:
            left.append(run_id)
        run_id+=1

print(left)