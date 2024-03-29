import os
import sys
import itertools

import numpy as np
import tensorflow as tf
import sklearn.metrics as skm

from tqdm import tqdm


if __name__ == '__main__':
	prefix_local_dirs = [
	# "/home/anuj/troy/gub-og/outputs/vit/new_sweep_results",
	"/home/anuj/troy/gub-og/outputs/chest_xray/new_sweep_results",
	# "/home/anuj/troy/gub-og/outputs/isic/new_sweep_results",
 ]
 
	result_datasets = ['in_domain_validation' ,'in_domain_test', 'ood_validation', 'ood_test']

	for dataset_ind, dataset_path in enumerate([
		# 'gs://ue-usrl-anuj/vit-retina-new/eval',
		'gs://ue-usrl-anuj/vit-cxfr-new/eval',
		# 'gs://ue-usrl-anuj/karm/outputs/isic/upper_extremity',
	]):

		for src_dir, dest_dir in [
		# (f'{dataset_path}/vit_simclr_finetune/lr_*/crop_*', ''),
		(f'{dataset_path}/vit_simclr_finetune_debiased/lr_*/crop_*', ''),
		# (f'{dataset_path}/vit_mim_finetune/lr_*/mr_*', ''),
		# (f'{dataset_path}/vit_mae_finetune/lr_*/mr_*', ''),
		]:

			print('-' * 79)
			print('source:', src_dir)
			print('source:', src_dir, file = sys.stderr)	
			use_r_acc = False
			if "vit_dan/" in src_dir or "vit_cmd/" in src_dir or "vit_dan_iw" in src_dir:
				use_r_acc = True
			# print('destination:', dest_dir)

			print('fetching file list', file = sys.stderr)
			if "isic" in dataset_path:
				y_path_regex = os.path.join(src_dir, '*', 'in_domain_test', 'eval_results*', 'y_true.npy')
			else:
				y_path_regex = os.path.join(src_dir, '*', 'in_domain_validation', 'eval_results*', 'y_true.npy')
	
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

			print('best validation AUROC steps\n')
			for k, (score, step) in best.items():
				print(f'{k} : {score}\n{step}')
			
			if use_r_acc:
				print('best validation Reverse Accuracy steps\n')
				for k, (score, step) in best_r_acc.items():
					print(f'{k} : {score}\n{step}')
		
				print('best validation Reverse AUROC steps\n')
				for k, (score, step) in best_r_auroc.items():
					print(f'{k} : {score}\n{step}')

			print('downloading best validation results')
			# if use_r_acc:
			# 	best = best_r_auroc
			for model_path, dataset in tqdm(itertools.product(best.keys(), result_datasets), total = len(best) * len(result_datasets)):
				_, step = best[model_path]
				
				src_regex = os.path.join(model_path, dataset, f"eval_results_{step}", '*')
				dest_subdir = os.path.join(model_path.replace(dataset_path, prefix_local_dirs[dataset_ind]), dataset, 'eval_results_1')
				os.makedirs(dest_subdir, exist_ok = True)
				os.popen(f'gsutil -m cp {src_regex} {dest_subdir} 2>> .logs/gsutil-err.log').read()
				print()
				# if str(seed) == '0':
				# 	chkpt_src = os.path.join(src_dir, seed, 'checkpoints', f'checkpoint_{step}.npz')
				# 	chkpt_dest = dest_dir.replace('./eval_results/in21k', '/troy/anuj/gub-og/checkpoints/vit_drd/vit')
				# 	os.popen(f'gsutil -m cp {chkpt_src} {chkpt_dest} 2>> gsutil-err.log').read()

			print('done')
