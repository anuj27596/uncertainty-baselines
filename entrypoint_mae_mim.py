import time
import itertools
import subprocess

import numpy as np

from absl import app
from absl import flags


run_settings = [
	('baselines.isic.jax_pretrain_mim', {
			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/isic/head_neck/vit_mim/mr_<_mr>/<_seed>',
			'config': 'configs/isic/vit_mim_pretrain.py',
			'config.seed': '<_seed>',
            'config.pp_train' : 'isic_preprocess(512)|patch_mim_mask(<_mr>)',
            'config.pp_eval' : 'isic_preprocess(512)|patch_mim_mask(<_mr>)',
	},  {	'_seed': range(6),
	} ,
 {	'_mr': [0.1, 0.2, 0.4, 0.6, 0.8],
	}
 ),

 ('baselines.isic.jax_finetune_deterministic', {
			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/isic/head_neck/vit_mim_finetune/mr_<_mr>/<_seed>',
			'config': 'configs/isic/vit_finetune.py',
			'config.seed': '<_seed>',
   			'config.model_init': 'gs://ue-usrl-anuj/karm/outputs/isic/head_neck/vit_mim/mr_<_mr>/<_seed>/checkpoints/checkpoint_47300.npz'
	}, {	'_lr': [1e-3],
	}, {	'_seed': range(1)
	},
  {	'_mr': [0.1, 0.2, 0.4, 0.6, 0.8],
	}
 ),
 
	('baselines.isic.jax_pretrain_mae', {
			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/isic/head_neck/vit_mae/mr_<_mr>/<_seed>',
			'config': 'configs/isic/vit_pretrain_mae.py',
			'config.seed': '<_seed>',
            'config.pp_train' : 'isic_preprocess(512)|mae_mask(<_mr>)',
            'config.pp_eval' : 'isic_preprocess(512)|mae_mask(<_mr>)',
	},  {	'_seed': range(1),
	} ,
 {	'_mr': [0.1, 0.2, 0.25, 0.4, 0.6, 0.8],
	}
 ),
 

]

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
		runs.append(run)


def get_run_ids():
	return range(len(runs))

def get_output_dir(run_id, exp_id):
	return runs[run_id]['args']['output_dir'].replace('<exp_id>', str(exp_id))


FLAGS = flags.FLAGS
flags.DEFINE_integer('exp_id', None, 'experiment id')
flags.DEFINE_integer('run_id', None, 'run id')
flags.DEFINE_bool('run', False, 'whether to run')
flags.DEFINE_bool('loop', False, 'whether to run')

def main(_):
	if FLAGS.loop:
		while True:
			pass
	else:
		run = runs[FLAGS.run_id]
		run['args']['output_dir'] = get_output_dir(FLAGS.run_id, FLAGS.exp_id)
		cmd = [
			'python', '-m', run['module'],
			*[f'--{key}={val}' for key, val in run['args'].items()]
		]
		print(' '.join(cmd[:3]) + ' \\\n    '.join([''] + cmd[3:]))
		if FLAGS.run:
			subprocess.run(cmd)

if __name__ == '__main__':
	app.run(main)
