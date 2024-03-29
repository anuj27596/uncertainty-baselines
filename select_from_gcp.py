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


	for idx, src_dir in enumerate([
		path
		for datadomain in ['isic-ue'] #, ['retina']
		for path in (
			[f'gs://ue-usrl-anuj/vit-isic-ue-new/vit-osp/finetune/lr_{_lr}/llr_{_llr}/plr_{_plr}'
				for _lr in [1e-3]
				for _llr in [1e-2]
				for _plr in [0.1]] +
			[f'gs://ue-usrl-anuj/vit-{datadomain}-new/vit-dan/finetune/lr_{_lr}/dlc_{_dlc}'
				for _lr in [1e-3, 1e-4]
				for _dlc in [0.3]] +
			[f'gs://ue-usrl-anuj/vit-{datadomain}-new/vit-iw/finetune/lr_{_lr}/dlc_{_dlc}'
				for _lr in [1e-3]
				for _dlc in [0.1]] +
			[f'gs://ue-usrl-anuj/vit-{datadomain}-new/vit-dan-iw/finetune/lr_{_lr}/dlc_{_dlc}'
				for _lr in [1e-4]
				for _dlc in [0.3, 1]] +
			[f'gs://ue-usrl-anuj/vit-{datadomain}-new/vit-cmd/finetune/lr_{_lr}/cmo_{_cmo}/clc_{_clc}'
				for _lr in [1e-3]
				for _cmo, _clc in [(3, 0.3), (8, 0.1)]]
		)
	]):

		print('-' * 79)
		print(f'[{idx:3}] source:', src_dir)
		print(f'[{idx:3}] source:', src_dir, file = sys.stderr)

		use_r_acc = "vit-dan/" in src_dir or "vit-cmd/" in src_dir or "vit-dan-iw/" in src_dir

		print('fetching file list', file = sys.stderr)
		y_path_regex = os.path.join(src_dir, '[1-5]',
			('in_domain_test' if 'isic-ue' in src_dir else 'in_domain_validation'),
			'eval_results*', 'y_true.npy')
		y_list = list(filter(None, os.popen(f'gsutil -m ls {y_path_regex}').read().split('\n')))


		print('processing results', file = sys.stderr)
		best, best_r_acc, best_r_auroc = dict(), dict(), dict()
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

			if use_r_acc:
				# Accuracy
				acc_file = y_file.replace('y_true', 'reverse_accuracy')
				with tf.io.gfile.GFile(acc_file, 'rb') as f:
					y_r_acc = np.load(f).item()
				
				best_r_acc[key] = max(
					best_r_acc.get(key, (-1, '_')),
					(y_r_acc, step.split('_')[-1]),
				)

				# AUROC
				r_pred_file = y_file.replace('y_true', 'reverse_pred')
				with tf.io.gfile.GFile(r_pred_file, 'rb') as f:
					r_pred = np.load(f)

				r_auroc = skm.roc_auc_score(y_true, r_pred)
				best_r_auroc[key] = max(
					best_r_auroc.get(key, (-1, '_')),
					(r_auroc, step.split('_')[-1]),
				)

		print('best validation steps')
		for k, (score, step) in best.items():
			print(f'{k} : {score} @ {step}')

		if use_r_acc:
			print('\nbest validation Reverse Accuracy steps')
			for k, (score, step) in best_r_acc.items():
				print(f'{k} : {score} @ {step}')

			print('\nbest validation Reverse AUROC steps')
			for k, (score, step) in best_r_auroc.items():
				print(f'{k} : {score} @ {step}')
