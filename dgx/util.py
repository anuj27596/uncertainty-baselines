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

def run_to_op_dir(run_setting, find_run_id):
# find_run_id = 20
    module, static_args, *iter_args  = run_setting
    keys = [k for d in iter_args for k in d.keys()]

    for run_id, vals in enumerate(itertools.product(*[zip(*d.values()) for d in iter_args])):
        if run_id == find_run_id:
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
            op_dir = run['args']['output_dir']
            break

    return op_dir

# run_setting =  	('baselines.chest_xray.jax_finetune_impwt', {
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
# 	'_lr': [1e-3, 1e-4]  
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
#  )