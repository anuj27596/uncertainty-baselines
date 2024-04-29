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
from util import run_to_op_dir

run_setting =  	('baselines.chest_xray.jax_finetune_impwt', {
			'output_dir': '/data3/home/karmpatel/karm_8T/outputs/chest_xray/<_organ>/vit_iw/lr_<_lr>/dlc_<_dlc>/nl_<_nl>/hdim_<_hdim>/<_seed>',
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
	'_lr': [1e-3, 1e-4]  
	} ,
 {	'_dlc': [0.1, 0.3, 1, 3], #[0.05, 0.1, 3], # need to add 10
	},
  {	'_grl': [0] #[0.05, 0.1, 3],
	},
 {	'_nl': [2, 3, 5],
	},
 {	'_hdim': [256, 512],
	},
 {'_organ':['vit-cxfr-new']
 }
 )

which_runs = list(range(20,26))

for run_id in which_runs:
    op_dir = run_to_op_dir(run_setting, run_id)
    ckpts = os.listdir(os.path.join(op_dir, "checkpoints"))
    steps_done = list(map(lambda x: int(x.split("_")[-1].replace(".npz", "")) if "_" in x else 0 ,ckpts))
    max_step = max(steps_done)
    os.system(f"ln -s {os.path.join(op_dir, "checkpoints", f"checkpoint_{max_step}.npz")} {os.path.join(op_dir, "checkpoints", "checkpoint.npz")}")