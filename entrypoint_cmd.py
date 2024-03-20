import time
import itertools
import subprocess

import numpy as np

from absl import app
from absl import flags


run_settings = [
	('baselines.isic.jax_finetune_cmd', {
			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/isic/head_neck/vit_cmd/clc_<_clc>/cmo_<_cmo>/<_seed>',
			'config': 'configs/isic/vit_finetune_cmd.py',
			'config.seed': '<_seed>',
			'config.model.cmd_loss_coeff':'<_clc>',
			'config.model.cmd_max_order':'<_cmo>'
	},  
  {	'_seed': range(1),
	} ,
 {	'_clc': [0.1, 1.0, 10.0],
	},
 {	'_cmo': [1, 2, 3, 5, 8],
	}
 ),
	# ('baselines.diabetic_retinopathy_detection.jax_finetune_cmd', {
	# 		'output_dir': 'gs://ue-usrl-anuj/vit/vit-cmd-sweep/eval/lr_<_lr>/cmo_<_cmo>/clc_<_clc>/<_seed>',
	# 		'config': 'configs/drd/vit_eval_cmd.py',
	# 		'config.seed': '<_seed>',
	# 		'config.cmd_max_order': '<_cmo>',
	# 		'config.cmd_loss_coeff': '<_clc>',
	# 		'config.model_init': 'gs://ue-usrl-anuj/vit/vit-cmd-sweep/finetune/lr_<_lr>/cmo_<_cmo>/clc_<_clc>/<_seed>/checkpoints/checkpoint_<_step>.npz',
	# }, {	'_lr': [1e-3],
	# }, {	'_cmo': [1],
	# }, {	'_clc': [0.1],
	# }, {	'_seed': range(6),
	# 		'_step': [6582, 6582, 4388, 6582, 6582, 6582],
	# }),


	# ('baselines.diabetic_retinopathy_detection.jax_finetune_dan', {
	# 		'output_dir': 'gs://ue-usrl-anuj/vit/vit-dan-ovp/eval/lr_0.0001/ovp_<_ovp>/<_seed>',
	# 		'config': 'configs/drd/vit_eval_dan.py',
	# 		'config.model_init': 'gs://ue-usrl-anuj/vit/vit-dan-ovp/finetune/lr_0.0001/ovp_<_ovp>/<_seed>/checkpoints/checkpoint_<_step>.npz',
	# }, {	'_seed': np.tile(range(6), 3),
	# 		'_ovp': np.repeat([25, 50, 75], 6),
	# 		'_step': [28522, 17552, 24134, 28522, 32910, 26328, 24134, 21940, 24134, 37298, 28522, 26328, 21940, 17552, 21940, 43880, 28522, 26328],
	# }),

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
