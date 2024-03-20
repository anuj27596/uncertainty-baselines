import os
import sys
import itertools

import numpy as np
import tensorflow as tf
import sklearn.metrics as skm

from tqdm import tqdm


if __name__ == '__main__':
	
	result_datasets = ['train', 'in_domain_test', 'ood_validation', 'ood_test']
	DATASET_NAME = "histopathology"
	organ='upper_extremity'

	for src_dir, dest_dir in [
     
     (f'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit_dan/gc_*/nl_*/hdim_*/', ''),
     
    #  (f'gs://ue-usrl-anuj/karm/outputs/isic/{organ}/vit_cmd/clc_*/cmo_*/', ''),
    #  (f'gs://ue-usrl-anuj/karm/outputs/isic/{organ}/vit_simclr_dan/gc_*/nl_*/hdim_*/', ''),
    #  (f'gs://ue-usrl-anuj/karm/outputs/isic/{organ}/vit_mim_dan/gc_*/nl_*/hdim_*/', ''),
    #  (f'gs://ue-usrl-anuj/karm/outputs/isic/{organ}/vit_mae_dan/gc_*/nl_*/hdim_*/', ''),
     
    #  (f'gs://ue-usrl-anuj/karm/outputs/{DATASET_NAME}/vit/lr_0.001/', ''),
	# 	(f'gs://ue-usrl-anuj/karm/outputs/isic/{organ}/vit/lr_0.001/', ''),
	# 	(f'gs://ue-usrl-anuj/karm/outputs/isic/{organ}/vit_simclr_finetune/crop_*/', ''),
  	# 	(f'gs://ue-usrl-anuj/karm/outputs/isic/{organ}/vit_osp/llr_*/plr_*/', ''),
    # (f'gs://ue-usrl-anuj/karm/outputs/isic/{organ}/vit_mcd/dr_*/', ''),
    # (f'gs://ue-usrl-anuj/karm/outputs/isic/{organ}/vit_iw/dlc_*', ''), # this is dan+iw
    # (f'gs://ue-usrl-anuj/karm/outputs/isic/{organ}/vit_iw2/dlc_*', ''), # this is iw
    # (f'gs://ue-usrl-anuj/karm/outputs/isic/{organ}/vit_cmd/clc_0.1/cmo_2/', ''),
    
    # (f'gs://ue-usrl-anuj/karm/outputs/isic/{organ}/vit_dan_intermed/grl_2/', ''),
    # (f'gs://ue-usrl-anuj/karm/outputs/isic/{organ}/vit_dan_intermed/grl_5/', ''),
    # (f'gs://ue-usrl-anuj/karm/outputs/isic/{organ}/vit_dan_intermed/grl_8/', ''),

    # (f'gs://ue-usrl-anuj/karm/outputs/isic/{organ}/vit_mim_finetune/mr_*', ''),
    # (f'gs://ue-usrl-anuj/karm/outputs/isic/{organ}/vit_dan/gc_*/nl_*/hdim_*/', ''),
	]:

		print('-' * 79)
		print('source:', src_dir)
		print('source:', src_dir, file = sys.stderr)
		# print('destination:', dest_dir)

		print('fetching file list', file = sys.stderr)
		y_path_regex = os.path.join(src_dir, '*', 'in_domain_test', 'eval_results*', 'y_true.npy')
		y_list = list(filter(None, os.popen(f'gsutil -m ls {y_path_regex}').read().split('\n')))


		print('processing results', file = sys.stderr)
		best = dict()
		y_true = None

		for y_file in tqdm(y_list):
			*key, _, step, _ = y_file.split('/')
			key = '/'.join(key)
			pred_file = y_file.replace('y_true', 'y_pred')

			if y_true is None:
				with tf.io.gfile.GFile(y_file, 'rb') as f:
					y_true = np.load(f)

			with tf.io.gfile.GFile(pred_file, 'rb') as f:
				pred = np.load(f)
			
			auroc = skm.roc_auc_score(y_true, pred)

			best[key] = max(
				best.get(key, (-1, '_')),
				(auroc, step.split('_')[-1]),
			)

		print('best validation steps')
		for k, (score, step) in best.items():
			print(f'{k} : {score}\n{step}')

		# print('downloading best validation results')
		# for seed, dataset in tqdm(itertools.product(best.keys(), result_datasets), total = len(best) * len(result_datasets)):
		# 	_, step = best[seed]
		# 	src_regex = os.path.join(src_dir, seed, dataset, step, '*')
		# 	dest_subdir = os.path.join(dest_dir, seed, dataset, 'eval_results_1')
		# 	os.makedirs(dest_subdir, exist_ok = True)
		# 	os.popen(f'gsutil -m cp {src_regex} {dest_subdir} 2>> gsutil-err.log').read()

		# 	if str(seed) == '0':
		# 		chkpt_src = os.path.join(src_dir, seed, 'checkpoints', f'checkpoint_{step}.npz')
		# 		chkpt_dest = dest_dir.replace('./eval_results/in21k', '/troy/anuj/gub-og/checkpoints/vit_drd/vit')
		# 		os.popen(f'gsutil -m cp {chkpt_src} {chkpt_dest} 2>> gsutil-err.log').read()

		# print('done')
