import time
import itertools
import subprocess

import numpy as np

from absl import app
from absl import flags


run_settings = [
	('baselines.isic.jax_finetune_dan', {
			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/isic/head_neck/vit_dan/gc_<_gc>/nl_<_nl>/hdim_<_hdim>/<_seed>',
			'config': 'configs/isic/vit_finetune_dan.py',
			'config.seed': '<_seed>',
			'config.model.domain_predictor.grl_coeff': '<_gc>',
    		'config.model.domain_predictor.num_layers':'<_nl>',
			'config.model.domain_predictor.hid_dim':'<_hdim>' 
	},  
  {	'_seed': range(1),
	} ,
 {	'_gc': [0.1, 0.3, 1, 3],
	},
 {	'_nl': [2, 3, 5],
	},
 {	'_hdim': [256, 512],
	},
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
