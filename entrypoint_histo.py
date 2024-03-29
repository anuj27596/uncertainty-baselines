import time
import itertools
import subprocess

import numpy as np

from absl import app
from absl import flags

DATASET_NAME = "histopathology"
run_settings = [

#   # 1. DR
#   ('baselines.diabetic_retinopathy_detection.jax_finetune_deterministic', {
# 			'output_dir': 'gs://ue-usrl-anuj/vit-retina-new/eval/vit_simclr_finetune/lr_<_lr>/crop_<_crop>/<_seed>',
# 			'config': 'configs/drd/vit_finetune_eval.py',
# 			'config.seed': '<_seed>',
# 			'config.lr.base':'<_lr>',
#    			'config.model_init': 'gs://ue-usrl-anuj/vit-retina-new/vit_simclr_finetune/lr_<_lr>/crop_<_crop>/0/checkpoints/checkpoint_<_step>.npz'
# 	}, 
#    {	'_lr': [1e-3],
# 	}, 
#    {	'_seed': range(6),
#    '_step': [3294, 4392, 3294, 3294, 3294, 4392],
    
# 	},
#   {	'_crop': [36, 64, 81][1:2], #[0.1, 0.2, 0.25, 0.4, 0.6, 0.8] 
#    '_lrp': [1e-4,1e-3,1e-3][1:2]
# 	},
#    {'_organ':['upper_extremity']
#  }
#  ),

#   ('baselines.diabetic_retinopathy_detection.jax_finetune_deterministic', {
# 			'output_dir': 'gs://ue-usrl-anuj/vit-retina-new/eval/vit_simclr_finetune_debiased/lr_<_lr>/crop_<_crop>/<_seed>',
# 			'config': 'configs/drd/vit_finetune_eval.py',
# 			'config.seed': '<_seed>',
# 			'config.lr.base':'<_lr>',
#    			'config.model_init': 'gs://ue-usrl-anuj/vit-retina-new/vit_simclr_finetune_debiased/lr_<_lr>/crop_<_crop>/0/checkpoints/checkpoint_<_step>.npz'
# 	}, 
#    {	'_lr': [1e-3],
# 	}, 
#    {	'_seed': range(6),
#    '_step': [3294]*6,
    
# 	},
#   {	'_crop': [36, 64, 81][0:1], #[0.1, 0.2, 0.25, 0.4, 0.6, 0.8] 
#    '_lrp': [1e-4,1e-3,1e-3][0:1]
# 	},
#    {'_organ':['upper_extremity']
#  }
#  ),
# 	('baselines.diabetic_retinopathy_detection.jax_finetune_deterministic', {
# 			'output_dir': 'gs://ue-usrl-anuj/vit-retina-new/eval/vit_mim_finetune/lr_<_lr>/mr_<_mr>/<_seed>',
# 			'config': 'configs/drd/vit_finetune_eval.py',
# 			'config.seed': '<_seed>',
# 'config.lr.base':'<_lr>',
#    			'config.model_init': 'gs://ue-usrl-anuj/vit-retina-new/vit_mim_finetune/lr_<_lr>/mr_<_mr>/0/checkpoints/checkpoint_<_step>.npz'
# 	},
#    {	'_lr': [1e-3], # 1e-4 done
# 	}, 
#    {	'_seed': range(6),
#    '_step': [4392, 3294, 4392, 4392, 3294, 5490],
# 	},
#   {	'_mr': [0.1, 0.2, 0.4, 0.6, 0.8][2:3],
#    '_lrp': [1e-3, 1e-3, 1e-3, 1e-3, 1e-3][2:3]
# 	},
#    {'_organ':['upper_extremity']
#  }
#  ),

# ('baselines.diabetic_retinopathy_detection.jax_finetune_deterministic', {
# 			'output_dir': 'gs://ue-usrl-anuj/vit-retina-new/eval/vit_mae_finetune/lr_<_lr>/mr_<_mr>/<_seed>',
# 			'config': 'configs/drd/vit_finetune_eval.py',
# 			'config.seed': '<_seed>',
# 'config.lr.base':'<_lr>',
#    			'config.model_init': 'gs://ue-usrl-anuj/vit-retina-new/vit_mae_finetune/lr_<_lr>/mr_<_mr>/0/checkpoints/checkpoint_<_step>.npz'
# 	},
#    {	'_lr': [1e-3], # 1e-4 done
# 	}, 
#    {	'_seed': range(6),
#    '_step':[2196,3294,3294,2196,3294,2196],
    
# 	},
#   {	'_mr': [0.1, 0.2, 0.4, 0.6, 0.8][3:4],
#    '_lrp': [1e-4, 1e-3, 1e-4, 1e-4, 1e-3][3:4]
# 	},
#    {'_organ':['upper_extremity']
#  }
#  ),

# 	# 2. CX
#   ('baselines.chest_xray.jax_finetune_deterministic', {
# 			'output_dir': 'gs://ue-usrl-anuj/vit-cxfr-new/eval/vit_simclr_finetune/lr_<_lr>/crop_<_crop>/<_seed>',
# 			'config': 'configs/chest_xray/vit_finetune_eval.py',
# 			'config.seed': '<_seed>',
# 			'config.lr.base':'<_lr>',
#    			'config.model_init': 'gs://ue-usrl-anuj/vit-cxfr-new/vit_simclr_finetune/lr_<_lr>/crop_<_crop>/0/checkpoints/checkpoint_<_step>.npz'
# 	}, 
#    {	'_lr': [1e-3],
# 	}, 
#    {	'_seed': range(6),
#    '_step': [10444,11936,16412,8206,5968,8206],
    
# 	},
#   {	'_crop': [36, 64, 81][1:2], #[0.1, 0.2, 0.25, 0.4, 0.6, 0.8] 
#    '_lrp': [1e-4,1e-3,1e-3][1:2]
# 	},
#    {'_organ':['upper_extremity']
#  }
#  ),

  ('baselines.chest_xray.jax_finetune_deterministic', {
			'output_dir': 'gs://ue-usrl-anuj/vit-cxfr-new/eval/vit_simclr_finetune_debiased/lr_<_lr>/crop_<_crop>/<_seed>',
			'config': 'configs/chest_xray/vit_finetune_eval.py',
			'config.seed': '<_seed>',
			'config.lr.base':'<_lr>',
   			'config.model_init': 'gs://ue-usrl-anuj/vit-cxfr-new/vit_simclr_finetune_debiased/lr_<_lr>/crop_<_crop>/0/checkpoints/checkpoint_<_step>.npz'
	}, 
   {	'_lr': [1e-3],
	}, 
   {	'_seed': range(6),
   '_step': [7460, 6714, 8206, 5968, 8952, 9698],
    
	},
  {	'_crop': [36, 64, 81][1:2], #[0.1, 0.2, 0.25, 0.4, 0.6, 0.8] 
   '_lrp': [1e-4,1e-3,1e-3][1:2]
	},
   {'_organ':['upper_extremity']
 }
 ),
# 	('baselines.chest_xray.jax_finetune_deterministic', {
# 			'output_dir': 'gs://ue-usrl-anuj/vit-cxfr-new/eval/vit_mim_finetune/lr_<_lr>/mr_<_mr>/<_seed>',
# 			'config': 'configs/chest_xray/vit_finetune_eval.py',
# 			'config.seed': '<_seed>',
# 'config.lr.base':'<_lr>',
#    			'config.model_init': 'gs://ue-usrl-anuj/vit-cxfr-new/vit_mim_finetune/lr_<_lr>/mr_<_mr>/0/checkpoints/checkpoint_<_step>.npz'
# 	},
#    {	'_lr': [1e-3], # 1e-4 done
# 	}, 
#    {	'_seed': range(6),
#    '_step': [11190, 9698, 12682, 9698, 8952, 10444],
# 	},
#   {	'_mr': [0.1, 0.2, 0.4, 0.6, 0.8][2:3],
#    '_lrp': [1e-3, 1e-3, 1e-3, 1e-3, 1e-3][2:3]
# 	},
#    {'_organ':['upper_extremity']
#  }
#  ),

# ('baselines.chest_xray.jax_finetune_deterministic', {
# 			'output_dir': 'gs://ue-usrl-anuj/vit-cxfr-new/eval/vit_mae_finetune/lr_<_lr>/mr_<_mr>/<_seed>',
# 			'config': 'configs/chest_xray/vit_finetune_eval.py',
# 			'config.seed': '<_seed>',
# 'config.lr.base':'<_lr>',
#    			'config.model_init': 'gs://ue-usrl-anuj/vit-cxfr-new/vit_mae_finetune/lr_<_lr>/mr_<_mr>/0/checkpoints/checkpoint_<_step>.npz'
# 	},
#    {	'_lr': [1e-3], # 1e-4 done
# 	}, 
#    {'_seed': range(6),
#    '_step':[14174, 11190, 6714, 6714, 11936, 11936],
    
# 	},
#   {	'_mr': [0.1, 0.2, 0.4, 0.6, 0.8][3:4],
#    '_lrp': [1e-4, 1e-3, 1e-4, 1e-4, 1e-3][3:4]
# 	},
#    {'_organ':['upper_extremity']
#  }
#  ),


 
# # Pretraing + DAN

# # 1. Histopathology
# # SimCLR+DAN
# ('baselines.histopathology.jax_finetune_dan', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit_simclr_dan/lr_<_lr>/dlc_<_dlc>/grc_<_grc>/nl_<_nl>/hdim_<_hdim>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune_dan.py',
# 			'config.seed': '<_seed>',
# 			'config.lr.base':'<_lr>',
# 			'config.model.domain_predictor.grl_coeff': '<_grc>',
#     		'config.model.domain_predictor.num_layers':'<_nl>',
# 			'config.model.domain_predictor.hid_dim':'<_hdim>',
# 			'config.dp_loss_coeff':'<_dlc>',
# 			'config.model_init': 'gs://ue-usrl-anuj/karm/outputs/histopathology/processed_onehot_tum_swap/vit_simclr/lr_<_lrp>/crop_<_crop>/0/checkpoints/checkpoint_<_step>.npz'
   
# 	},  
#   {	'_seed': range(1),
# 	} ,
#   {
# 	'_lr': [1e-4, 1e-3]  
  
# 	} ,
#    {	'_dlc': [0.1, 0.3, 1, 3, 10]
# 	},
#  {	'_grc': [1] #[0.05, 0.1, 0.3, 1, 3],
# 	},
#  {	'_nl': [3] #[2, 3, 5],
# 	},
#  {	'_hdim': [256] #[256, 512, 768],
# 	},
#  {'_organ':['processed_onehot_tum_swap2']
#  },
#    {'_crop': [81], #[36, 64, 81], 
#    '_step': [204800], #[40960, 40960, 204800],
#    '_lrp': [1e-3] #[1e-4,1e-4,1e-3]
# 	},
#  ),


# # SimCLR Deb +DAN
# ('baselines.histopathology.jax_finetune_dan', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit_simclr_debiased_dan/lr_<_lr>/dlc_<_dlc>/grc_<_grc>/nl_<_nl>/hdim_<_hdim>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune_dan.py',
# 			'config.seed': '<_seed>',
# 			'config.lr.base':'<_lr>',
# 			'config.model.domain_predictor.grl_coeff': '<_grc>',
#     		'config.model.domain_predictor.num_layers':'<_nl>',
# 			'config.model.domain_predictor.hid_dim':'<_hdim>',
# 			'config.dp_loss_coeff':'<_dlc>',
# 			'config.model_init': 'gs://ue-usrl-anuj/karm/outputs/histopathology/processed_onehot_tum_swap/vit_simclr_debiased/lr_<_lrp>/crop_<_crop>/0/checkpoints/checkpoint_<_step>.npz'
   
# 	},  
#   {	'_seed': range(1),
# 	} ,
#   {
# 	'_lr': [1e-4, 1e-3]  
  
# 	} ,
#    {	'_dlc': [0.1, 0.3, 1, 3, 10]
# 	},
#  {	'_grc': [1] #[0.05, 0.1, 0.3, 1, 3],
# 	},
#  {	'_nl': [3] #[2, 3, 5],
# 	},
#  {	'_hdim': [256] #[256, 512, 768],
# 	},
#  {'_organ':['processed_onehot_tum_swap2']
#  },
#   {	'_crop': [81], # [36, 64, 81], #[0.1, 0.2, 0.25, 0.4, 0.6, 0.8] 
#    '_step': [122880], # [40960, 40960, 122880],
#    '_lrp': [1e-3] # [1e-4,1e-4,1e-3]
# 	},
#  ),

# # SimCLR MIM+DAN
# ('baselines.histopathology.jax_finetune_dan', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit_mae_dan/lr_<_lr>/dlc_<_dlc>/grc_<_grc>/nl_<_nl>/hdim_<_hdim>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune_dan.py',
# 			'config.seed': '<_seed>',
# 			'config.lr.base':'<_lr>',
# 			'config.model.domain_predictor.grl_coeff': '<_grc>',
#     		'config.model.domain_predictor.num_layers':'<_nl>',
# 			'config.model.domain_predictor.hid_dim':'<_hdim>',
# 			'config.dp_loss_coeff':'<_dlc>',
# 			'config.model_init': 'gs://ue-usrl-anuj/karm/outputs/histopathology/processed_onehot_tum_swap/vit_mae/lr_<_lrp>/mr_<_mr>/0/checkpoints/checkpoint_<_step>.npz'   
# 	},  
#   {	'_seed': range(1),
# 	} ,
#   {
# 	'_lr': [1e-4, 1e-3]    
# 	} ,
#    {	'_dlc': [0.1, 0.3, 1, 3, 10]
# 	},
#  {	'_grc': [1] #[0.05, 0.1, 0.3, 1, 3],
# 	},
#  {	'_nl': [3] #[2, 3, 5],
# 	},
#  {	'_hdim': [256] #[256, 512, 768],
# 	},
#  {'_organ':['processed_onehot_tum_swap2']
#  },
#   {	'_mr': [0.4], #[0.4, 0.6, 0.8],
#    '_step':[307200], #[307200, 40960, 225280],
#    '_lrp': [1e-3] # [1e-3, 1e-3, 1e-3]
# 	},
#  ),

# # SimCLR MAE+DAN
# ('baselines.histopathology.jax_finetune_dan', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit_mae_dan/lr_<_lr>/dlc_<_dlc>/grc_<_grc>/nl_<_nl>/hdim_<_hdim>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune_dan.py',
# 			'config.seed': '<_seed>',
# 			'config.lr.base':'<_lr>',
# 			'config.model.domain_predictor.grl_coeff': '<_grc>',
#     		'config.model.domain_predictor.num_layers':'<_nl>',
# 			'config.model.domain_predictor.hid_dim':'<_hdim>',
# 			'config.dp_loss_coeff':'<_dlc>',
# 			'config.model_init': 'gs://ue-usrl-anuj/karm/outputs/histopathology/processed_onehot_tum_swap/vit_mae/lr_<_lrp>/mr_<_mr>/0/checkpoints/checkpoint_<_step>.npz'
   
# 	},  
#   {	'_seed': range(1),
# 	} ,
#   {
# 	'_lr': [1e-4, 1e-3]  
  
# 	} ,
#    {	'_dlc': [0.1, 0.3, 1, 3, 10]
# 	},
#  {	'_grc': [1] #[0.05, 0.1, 0.3, 1, 3],
# 	},
#  {	'_nl': [3] #[2, 3, 5],
# 	},
#  {	'_hdim': [256] #[256, 512, 768],
# 	},
#  {'_organ':['processed_onehot_tum_swap2']
#  },
#   {	'_mr':[0.8], #[0.1, 0.2, 0.4, 0.6, 0.8],
#    '_step': [20480], #[307200, 266240, 20480, 20480, 20480],
#    '_lrp': [1e-4] #[1e-3, 1e-3, 1e-4, 1e-4, 1e-4] 
# 	},
#  ),

# # 2. ISIC

# # SimCLR+DAN
# ('baselines.isic.jax_finetune_dan', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/isic/<_organ>/vit_simclr_dan/lr_<_lr>/dlc_<_dlc>/grc_<_grc>/nl_<_nl>/hdim_<_hdim>/<_seed>',
# 			'config': 'configs/isic/vit_finetune_dan.py',
# 			'config.seed': '<_seed>',
# 			'config.lr.base':'<_lr>',
# 			'config.model.domain_predictor.grl_coeff': '<_grc>',
#     		'config.model.domain_predictor.num_layers':'<_nl>',
# 			'config.model.domain_predictor.hid_dim':'<_hdim>',
# 			'config.dp_loss_coeff':'<_dlc>',
# 			'config.model_init': 'gs://ue-usrl-anuj/karm/outputs/isic/<_organ>/vit_simclr/lr_<_lrp>/crop_<_crop>/0/checkpoints/checkpoint_<_step>.npz'
   
# 	},  
#   {	'_seed': range(1),
# 	} ,
#   {
# 	'_lr': [1e-4, 1e-3]  
  
# 	} ,
#    {	'_dlc': [0.1, 0.3, 1, 3, 10]
# 	},
#  {	'_grc': [1] #[0.05, 0.1, 0.3, 1, 3],
# 	},
#  {	'_nl': [3] #[2, 3, 5],
# 	},
#  {	'_hdim': [256] #[256, 512, 768],
# 	},
#  {'_organ':['upper_extremity']
#  },
#   {	'_crop': [36, 64, 81][1:2], #[0.1, 0.2, 0.25, 0.4, 0.6, 0.8] 
#    '_step': [141900, 141900, 42570][1:2],
#    '_lrp': [1e-4,1e-3,1e-3][1:2]
# 	},
#  ),


# # SimCLR Deb +DAN
# ('baselines.isic.jax_finetune_dan', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/isic/<_organ>/vit_simclr_debiased_dan/lr_<_lr>/dlc_<_dlc>/grc_<_grc>/nl_<_nl>/hdim_<_hdim>/<_seed>',
# 			'config': 'configs/isic/vit_finetune_dan.py',
# 			'config.seed': '<_seed>',
# 			'config.lr.base':'<_lr>',
# 			'config.model.domain_predictor.grl_coeff': '<_grc>',
#     		'config.model.domain_predictor.num_layers':'<_nl>',
# 			'config.model.domain_predictor.hid_dim':'<_hdim>',
# 			'config.dp_loss_coeff':'<_dlc>',
# 			'config.model_init': 'gs://ue-usrl-anuj/karm/outputs/isic/<_organ>/vit_simclr_debiased/lr_<_lrp>/crop_<_crop>/0/checkpoints/checkpoint_<_step>.npz'
   
# 	},  
#   {	'_seed': range(1),
# 	} ,
#   {
# 	'_lr': [1e-4, 1e-3]  
  
# 	} ,
#    {	'_dlc': [0.1, 0.3, 1, 3, 10]
# 	},
#  {	'_grc': [1] #[0.05, 0.1, 0.3, 1, 3],
# 	},
#  {	'_nl': [3] #[2, 3, 5],
# 	},
#  {	'_hdim': [256] #[256, 512, 768],
# 	},
#  {'_organ':['upper_extremity']
#  },
#   {	'_crop': [36, 64, 81][0:1], #[0.1, 0.2, 0.25, 0.4, 0.6, 0.8] 
#    '_step': [14190, 70950, 70950][0:1],
#    '_lrp': [1e-3,1e-3,1e-3][0:1]
# 	},
#  ),

# # SimCLR MIM+DAN
# ('baselines.isic.jax_finetune_dan', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/isic/<_organ>/vit_mim_dan/lr_<_lr>/dlc_<_dlc>/grc_<_grc>/nl_<_nl>/hdim_<_hdim>/<_seed>',
# 			'config': 'configs/isic/vit_finetune_dan.py',
# 			'config.seed': '<_seed>',
# 			'config.lr.base':'<_lr>',
# 			'config.model.domain_predictor.grl_coeff': '<_grc>',
#     		'config.model.domain_predictor.num_layers':'<_nl>',
# 			'config.model.domain_predictor.hid_dim':'<_hdim>',
# 			'config.dp_loss_coeff':'<_dlc>',
# 			'config.model_init': 'gs://ue-usrl-anuj/karm/outputs/isic/<_organ>/vit_mim/lr_<_lrp>/mr_<_mr>/0/checkpoints/checkpoint_<_step>.npz'   
# 	},  
#   {	'_seed': range(1),
# 	} ,
#   {
# 	'_lr': [1e-4, 1e-3]    
# 	} ,
#    {	'_dlc': [0.1, 0.3, 1, 3, 10]
# 	},
#  {	'_grc': [1] #[0.05, 0.1, 0.3, 1, 3],
# 	},
#  {	'_nl': [3] #[2, 3, 5],
# 	},
#  {	'_hdim': [256] #[256, 512, 768],
# 	},
#  {'_organ':['upper_extremity']
#  },
#   {	'_mr': [0.1, 0.2, 0.4, 0.6, 0.8][-1:],
#    '_step': [37840, 37840, 37840, 9460, 9460][-1:],
#    '_lrp': [1e-3, 1e-3, 1e-3, 1e-3, 1e-3][-1:]
# 	},
#  ),

# # SimCLR MAE+DAN
# ('baselines.isic.jax_finetune_dan', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/isic/<_organ>/vit_mae_dan/lr_<_lr>/dlc_<_dlc>/grc_<_grc>/nl_<_nl>/hdim_<_hdim>/<_seed>',
# 			'config': 'configs/isic/vit_finetune_dan.py',
# 			'config.seed': '<_seed>',
# 			'config.lr.base':'<_lr>',
# 			'config.model.domain_predictor.grl_coeff': '<_grc>',
#     		'config.model.domain_predictor.num_layers':'<_nl>',
# 			'config.model.domain_predictor.hid_dim':'<_hdim>',
# 			'config.dp_loss_coeff':'<_dlc>',
# 			'config.model_init': 'gs://ue-usrl-anuj/karm/outputs/isic/<_organ>/vit_mae/lr_<_lrp>/mr_<_mr>/0/checkpoints/checkpoint_<_step>.npz'
   
# 	},  
#   {	'_seed': range(1),
# 	} ,
#   {
# 	'_lr': [1e-4, 1e-3]  
  
# 	} ,
#    {	'_dlc': [0.1, 0.3, 1, 3, 10]
# 	},
#  {	'_grc': [1] #[0.05, 0.1, 0.3, 1, 3],
# 	},
#  {	'_nl': [3] #[2, 3, 5],
# 	},
#  {	'_hdim': [256] #[256, 512, 768],
# 	},
#  {'_organ':['upper_extremity']
#  },
#   {	'_mr': [0.1, 0.2, 0.4, 0.6, 0.8][-1:],
#    '_step':[28380],
#    '_lrp': [1e-4]
# 	},
#  ),

# # 3. CX
# # SimCLR+DAN
# ('baselines.chest_xray.jax_finetune_dan', {
# 			'output_dir': 'gs://ue-usrl-anuj/vit-cxfr-new/vit_simclr_dan/lr_<_lr>/dlc_<_dlc>/grc_<_grc>/nl_<_nl>/hdim_<_hdim>/<_seed>',
# 			'config': 'configs/chest_xray/vit_finetune_dan.py',
# 			'config.seed': '<_seed>',
# 			'config.lr.base':'<_lr>',
# 			'config.model.domain_predictor.grl_coeff': '<_grc>',
#     		'config.model.domain_predictor.num_layers':'<_nl>',
# 			'config.model.domain_predictor.hid_dim':'<_hdim>',
# 			'config.dp_loss_coeff':'<_dlc>',
# 			'config.model_init': 'gs://ue-usrl-anuj/vit-cxfr-new/vit_simclr/lr_<_lrp>/crop_<_crop>/0/checkpoints/checkpoint_<_step>.npz'
   
# 	},  
#   {	'_seed': range(1),
# 	} ,
#   {
# 	'_lr': [1e-4, 1e-3]    
# 	} ,
#    {	'_dlc': [0.1, 0.3, 1, 3, 10]
# 	},
#  {	'_grc': [1] #[0.05, 0.1, 0.3, 1, 3],
# 	},
#  {	'_nl': [3] #[2, 3, 5],
# 	},
#  {	'_hdim': [256] #[256, 512, 768],
# 	},
#  {'_organ':['upper_extremity']
#  },
#   {	'_crop': [36, 64, 81][1:2], #[0.1, 0.2, 0.25, 0.4, 0.6, 0.8] 
#    '_step': [298400, 29840, 29840][1:2],
#    '_lrp': [1e-4,1e-4,1e-4][1:2]
# 	},
#  ),


# # SimCLR Deb +DAN
# ('baselines.chest_xray.jax_finetune_dan', {
# 			'output_dir': 'gs://ue-usrl-anuj/vit-cxfr-new/vit_simclr_debiased_dan/lr_<_lr>/dlc_<_dlc>/grc_<_grc>/nl_<_nl>/hdim_<_hdim>/<_seed>',
# 			'config': 'configs/chest_xray/vit_finetune_dan.py',
# 			'config.seed': '<_seed>',
# 			'config.lr.base':'<_lr>',
# 			'config.model.domain_predictor.grl_coeff': '<_grc>',
#     		'config.model.domain_predictor.num_layers':'<_nl>',
# 			'config.model.domain_predictor.hid_dim':'<_hdim>',
# 			'config.dp_loss_coeff':'<_dlc>',
# 			'config.model_init': 'gs://ue-usrl-anuj/vit-cxfr-new/vit_simclr_debiased/lr_<_lrp>/crop_<_crop>/0/checkpoints/checkpoint_<_step>.npz'
   
# 	},  
#   {	'_seed': range(1),
# 	} ,
#   {
# 	'_lr': [1e-4, 1e-3]  
  
# 	} ,
#    {	'_dlc': [0.1, 0.3, 1, 3, 10]
# 	},
#  {	'_grc': [1] #[0.05, 0.1, 0.3, 1, 3],
# 	},
#  {	'_nl': [3] #[2, 3, 5],
# 	},
#  {	'_hdim': [256] #[256, 512, 768],
# 	},
#  {'_organ':['upper_extremity']
#  },
#   {	'_crop': [64, 81][0:1], #[0.1, 0.2, 0.25, 0.4, 0.6, 0.8] 
#    '_step': [119360, 238720][0:1],
#    '_lrp': [1e-3,1e-3][0:1]
# 	},
#  ),

# # SimCLR MIM+DAN
# ('baselines.chest_xray.jax_finetune_dan', {
# 			'output_dir': 'gs://ue-usrl-anuj/vit-cxfr-new/vit_mim_dan/lr_<_lr>/dlc_<_dlc>/grc_<_grc>/nl_<_nl>/hdim_<_hdim>/<_seed>',
# 			'config': 'configs/chest_xray/vit_finetune_dan.py',
# 			'config.seed': '<_seed>',
# 			'config.lr.base':'<_lr>',
# 			'config.model.domain_predictor.grl_coeff': '<_grc>',
#     		'config.model.domain_predictor.num_layers':'<_nl>',
# 			'config.model.domain_predictor.hid_dim':'<_hdim>',
# 			'config.dp_loss_coeff':'<_dlc>',
# 			'config.model_init': 'gs://ue-usrl-anuj/vit-cxfr-new/vit_mim/lr_<_lrp>/mr_<_mr>/0/checkpoints/checkpoint_<_step>.npz'   
# 	},  
#   {	'_seed': range(1),
# 	} ,
#   {
# 	'_lr': [1e-4, 1e-3]    
# 	} ,
#    {	'_dlc': [0.1, 0.3, 1, 3, 10]
# 	},
#  {	'_grc': [1] #[0.05, 0.1, 0.3, 1, 3],
# 	},
#  {	'_nl': [3] #[2, 3, 5],
# 	},
#  {	'_hdim': [256] #[256, 512, 768],
# 	},
#  {'_organ':['upper_extremity']
#  },
#   {	'_mr': [0.4, 0.6, 0.8][-1:],
#    '_step': [147708, 98472, 135772][-1:],
#    '_lrp': [1e-3, 1e-3, 1e-3][-1:]
# 	},
#  ),

# # SimCLR MAE+DAN
# ('baselines.chest_xray.jax_finetune_dan', {
# 			'output_dir': 'gs://ue-usrl-anuj/vit-cxfr-new/vit_mae_dan/lr_<_lr>/dlc_<_dlc>/grc_<_grc>/nl_<_nl>/hdim_<_hdim>/<_seed>',
# 			'config': 'configs/chest_xray/vit_finetune_dan.py',
# 			'config.seed': '<_seed>',
# 			'config.lr.base':'<_lr>',
# 			'config.model.domain_predictor.grl_coeff': '<_grc>',
#     		'config.model.domain_predictor.num_layers':'<_nl>',
# 			'config.model.domain_predictor.hid_dim':'<_hdim>',
# 			'config.dp_loss_coeff':'<_dlc>',
# 			'config.model_init': 'gs://ue-usrl-anuj/vit-cxfr-new/vit_mae/lr_<_lrp>/mr_<_mr>/0/checkpoints/checkpoint_<_step>.npz'
   
# 	},  
#   {	'_seed': range(1),
# 	} ,
#   {
# 	'_lr': [1e-4, 1e-3]  
# 	} ,
#    {	'_dlc': [0.1, 0.3, 1, 3, 10]
# 	},
#  {	'_grc': [1] #[0.05, 0.1, 0.3, 1, 3],
# 	},
#  {	'_nl': [3] #[2, 3, 5],
# 	},
#  {	'_hdim': [256] #[256, 512, 768],
# 	},
#  {'_organ':['upper_extremity']
#  },
#   {	'_mr': [0.1, 0.2, 0.4, 0.6, 0.8][0:1],
#    '_step':[149200, 149200, 7460, 149200, 7460][0:1],
#    '_lrp': [1e-3, 1e-3, 1e-3, 1e-3, 1e-3][0:1]
# 	},
#  ),


# #  0 ViT	
#  (f'baselines.{DATASET_NAME}.jax_finetune_deterministic', {
# 			'output_dir': f'gs://ue-usrl-anuj/karm/outputs/{DATASET_NAME}/<_organ>/vit/lr_<_lr>/<_seed>',
# 			'config': f'configs/{DATASET_NAME}/vit_finetune.py',
# 			'config.seed': '<_seed>',
# 			'config.lr.base':'<_lr>',
# 			'config.builder_config':'processed_onehot_tum_swap'
# 	}, {	'_lr': [1e-3, 1e-4],
# 	}, {	'_seed': range(6)
# 	},
#   {'_organ':['processed_onehot_tum_swap2']
#  }
#  ),
 
# #  1 osp --
#  	('baselines.histopathology.jax_finetune_osp', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit_osp/lr_<_lr>/llr_<_llr>/plr_<_plr>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune_osp.py',
# 			'config.seed': '<_seed>',
# 			'config.lr.base':'<_lr>',
# 			'config.model.lagrangian.lambda_grad':'<_llr>',
# 			'config.model.lagrangian.phi_grad':'<_plr>'
# 	},  
#   {	'_seed': range(1,6),
# 	} ,
#   {
# 	'_lr': [1e-3] #[1e-4, 1e-3]  
# 	} ,
#  {	'_llr': [0.1]#[0.01, 0.1],
# 	},
#  {	'_plr': [0.1]#[0.1, 1, 10],
# 	},
#  {'_organ':['processed_onehot_tum_swap2']
#  }
#  ),
  
# #   2 mcd --
# 	('baselines.histopathology.jax_finetune_mcd', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit_mcd/lr_<_lr>/dr_<_dr>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune_mcd.py',
# 			'config.seed': '<_seed>',
# 			'config.lr.base':'<_lr>',
# 			'config.model.transformer.attention_dropout_rate':'<_dr>',
# 			'config.model.transformer.dropout_rate':'<_dr>'
# 	},  
#   {	'_seed': range(1,6),
# 	} ,
#   {
# 	'_lr': [1e-3] #[1e-4, 1e-3]  
# 	} ,
#  {	'_dr': [0.01] #[0.01, 0.03, 0.1, 0.3],
# 	},
#  {'_organ':['processed_onehot_tum_swap2']
#  }
#  ),

 # 3 dan+iw: done
#  	('baselines.histopathology.jax_finetune_impwt', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit_dan_iw/lr_<_lr>/dlc_<_dlc>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune_impwt.py',
# 			'config.seed': '<_seed>',
# 'config.lr.base':'<_lr>',
#    			'config.model.domain_predictor.grl_coeff':'<_grl>',
# 			'config.dp_loss_coeff':'<_dlc>',
   			
# 	},  
#   {	'_seed': range(1,6),
# 	} ,
#   {
# 	'_lr': [1e-3] #[1e-4, 1e-3]  
  
# 	} ,
#  {	'_dlc': [0.3]# [0.1, 0.3, 1, 3, 10], #[0.05, 0.1, 3],
# 	},
#   {	'_grl': [1] #[0.05, 0.1, 3],
# 	},
#  {'_organ':['processed_onehot_tum_swap2']
#  }
#  ),
  

#  #4  iw 
#  	('baselines.histopathology.jax_finetune_impwt', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit_iw/lr_<_lr>/dlc_<_dlc>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune_impwt.py',
# 			'config.seed': '<_seed>',
# 			'config.lr.base':'<_lr>',
#    			'config.model.domain_predictor.grl_coeff':'0',
# 			'config.dp_loss_coeff':'<_dlc>',
   			
# 	},  
#   {	'_seed': range(1,6),
# 	} ,
#   {
# 	'_lr': [1e-3] #[1e-4, 1e-3]  
  
# 	} ,
#  {	'_dlc': [0.1] #[0.1, 0.3, 1, 3, 10],
# 	}
#  ,
#  {'_organ':['processed_onehot_tum_swap2']
#  }
#  ),
  
#   # 5 mim_finetune -- done
#   ('baselines.histopathology.jax_pretrain_mim', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit_mim/lr_<_lr>/mr_<_mr>/<_seed>',
# 			'config': 'configs/histopathology/vit_mim_pretrain.py',
# 			'config.seed': '<_seed>',
# 'config.lr.base':'<_lr>',
#             'config.pp_train' : 'histopathology_preprocess(224)|patch_mim_mask(<_mr>)',
#             'config.pp_eval' : 'histopathology_preprocess(224)|patch_mim_mask(<_mr>)',
            
# 	},  
#    {	'_seed': range(1),
# 	} ,
#  {	'_mr': [0.1, 0.2, 0.4, 0.6, 0.8],
# 	} ,  {
# 	'_lr': [1e-4, 1e-3]  
# 	} ,
#  {'_organ':['processed_onehot_tum_swap2']
#  }
#  ),
  

# #   done
#   ('baselines.histopathology.jax_finetune_deterministic', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit_mim_finetune/lr_<_lr>/mr_<_mr>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune.py',
# 			'config.seed': '<_seed>',
# 'config.lr.base':'<_lr>',
#    			'config.model_init': 'gs://ue-usrl-anuj/karm/outputs/histopathology/processed_onehot_tum_swap/vit_mim/lr_<_lrp>/mr_<_mr>/0/checkpoints/checkpoint_<_step>.npz'
# 	},
#    {	'_lr': [1e-3]#[1e-4, 1e-3], # 1e-4 done
# 	}, 
#    {	'_seed': range(6)
# 	},
#   {	'_mr': [0.4], #[0.4, 0.6, 0.8],
#    '_step':[307200], #[307200, 40960, 225280],
#    '_lrp': [1e-3] # [1e-3, 1e-3, 1e-3]
# 	},
#    {'_organ':['processed_onehot_tum_swap2']
#  }
#  ),
  

  
#   # 5.2 MAE:
#   	('baselines.histopathology.jax_pretrain_mae', {
# 		'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit_mae/lr_<_lr>/mr_<_mr>/<_seed>',
# 		'config': 'configs/histopathology/vit_pretrain_mae.py',
# 		'config.seed': '<_seed>',
# 'config.lr.base':'<_lr>',
#   		'config.mask_rate': '<_mr>',
# 		'config.pp_train' : 'histopathology_preprocess(224)|mae_mask(<_mr>)',
# 		'config.pp_eval' : 'histopathology_preprocess(224)|mae_mask(<_mr>)',
# },  
#     {	'_seed': range(1),
# } ,
# {	'_mr': [0.1, 0.2, 0.25, 0.4, 0.6, 0.8],
# },  {
# 	'_lr': [1e-4, 1e-3]  
# 	} ,
#  {'_organ':['processed_onehot_tum_swap2']
#  }
# ),

#   ('baselines.histopathology.jax_finetune_deterministic', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit_mae_finetune/lr_<_lr>/mr_<_mr>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune.py',
# 			'config.seed': '<_seed>',
# 'config.lr.base':'<_lr>',
#    			'config.model_init': 'gs://ue-usrl-anuj/karm/outputs/histopathology/processed_onehot_tum_swap/vit_mae/lr_<_lrp>/mr_<_mr>/0/checkpoints/checkpoint_<_step>.npz'
# 	}, {	'_lr': [1e-3],
# 	}, {	'_seed': range(6)
# 	},
#   {	'_mr':[0.8], #[0.1, 0.2, 0.4, 0.6, 0.8],
#    '_step': [20480], #[307200, 266240, 20480, 20480, 20480],
#    '_lrp': [1e-4] #[1e-3, 1e-3, 1e-4, 1e-4, 1e-4] 
# 	},
#    {'_organ':['processed_onehot_tum_swap2']
#  }
#  ),
  
#  # 5.3 - debiased
#   ('baselines.histopathology.jax_pretrain_simclr_debiased', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit_simclr_debiased/lr_<_lr>/crop_<_crop>/<_seed>',
# 			'config': 'configs/histopathology/vit_simclr_pretrain.py',
# 			'config.seed': '<_seed>',
# 'config.lr.base':'<_lr>',
# 			'config.pp_train' : 'histopathology_preprocess(224)|simclr_aug(<_crop>, 45)',
#    			'config.pp_eval' : 'histopathology_preprocess(224)|simclr_aug(<_crop>, 45)'
# 	},  
#   {	'_seed': range(1),
# 	} ,
#   {
# 	'_lr': [1e-4, 1e-3] 
# 	} ,
#  {	'_crop': [36,64,81],
# 	},
#  {'_organ':['processed_onehot_tum_swap2']
#  }
#  ),

  
#   ('baselines.histopathology.jax_finetune_deterministic', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit_simclr_finetune_debiased/lr_<_lr>/crop_<_crop>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune.py',
# 			'config.seed': '<_seed>',
# 			'config.lr.base':'<_lr>',
#    			'config.model_init': 'gs://ue-usrl-anuj/karm/outputs/histopathology/processed_onehot_tum_swap/vit_simclr_debiased/lr_<_lrp>/crop_<_crop>/0/checkpoints/checkpoint_<_step>.npz'
# 	}, 
#    {	'_lr': [1e-3] #[1e-3, 1e-4],
# 	}, 
#    {	'_seed': range(6)
# 	},
#   {	'_crop': [81], # [36, 64, 81], #[0.1, 0.2, 0.25, 0.4, 0.6, 0.8] 
#    '_step': [122880], # [40960, 40960, 122880],
#    '_lrp': [1e-3] # [1e-4,1e-4,1e-3]
# 	},
#    {'_organ':['processed_onehot_tum_swap2']
#  }
#  ),
  
#   # 6. SimcLR + finetune: left [0.64]
  
#   ('baselines.histopathology.jax_pretrain_simclr', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit_simclr/lr_<_lr>/crop_<_crop>/<_seed>',
# 			'config': 'configs/histopathology/vit_simclr_pretrain.py',
# 			'config.seed': '<_seed>',
# 'config.lr.base':'<_lr>',
# 			'config.pp_train' : 'histopathology_preprocess(224)|simclr_aug(<_crop>, 45)',
#    			'config.pp_eval' : 'histopathology_preprocess(224)|simclr_aug(<_crop>, 45)',
      		
# 	},  
#   {	'_seed': range(1),
# 	} ,
#   {
# 	'_lr': [1e-4, 1e-3]  
# 	} ,
#  {	'_crop': [36, 64, 81],
# 	},
#  {'_organ':['processed_onehot_tum_swap2']
#  }
#  ),

#   ('baselines.histopathology.jax_finetune_deterministic', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit_simclr_finetune/lr_<_lr>/crop_<_crop>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune.py',
# 			'config.seed': '<_seed>',
# 'config.lr.base':'<_lr>',
#    			'config.model_init': 'gs://ue-usrl-anuj/karm/outputs/histopathology/processed_onehot_tum_swap/vit_simclr/lr_<_lrp>/crop_<_crop>/0/checkpoints/checkpoint_<_step>.npz'
# 	}, 
#    {	'_lr': [1e-3] #[1e-4, 1e-3],
# 	}, 
#    {	'_seed': range(6)
# 	},
#   {	'_crop': [81], #[36, 64, 81], #[0.1, 0.2, 0.25, 0.4, 0.6, 0.8] 
#    '_step': [204800], #[40960, 40960, 204800],
#    '_lrp': [1e-3] #[1e-4,1e-4,1e-3]
# 	},
#    {'_organ':['processed_onehot_tum_swap2']
#  }
#  ),
  
  
# 7 DAN
# 	('baselines.histopathology.jax_finetune_dan', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit_dan/lr_<_lr>/dlc_<_dlc>/grc_<_grc>/nl_<_nl>/hdim_<_hdim>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune_dan.py',
# 			'config.seed': '<_seed>',
# 			'config.lr.base':'<_lr>',
# 			'config.model.domain_predictor.grl_coeff': '<_grc>',
#     		'config.model.domain_predictor.num_layers':'<_nl>',
# 			'config.model.domain_predictor.hid_dim':'<_hdim>',
# 			'config.dp_loss_coeff':'<_dlc>',
# 			'config.distribution_shift':'<_organ>'
   
# 	},  
#   {	'_seed': range(1),
# 	} ,
#   {
# 	'_lr': [1e-4, 1e-3]  
  
# 	} ,
#    {	'_dlc': [1] #[0.01, 0.03] #[0.1, 0.3, 1, 3, 10]
# 	},
#  {	'_grc': [0.05, 0.1, 0.3, 1, 3],
# 	},
#  {	'_nl': [3] #[2, 3, 5],
# 	},
#  {	'_hdim': [256] #[256, 512, 768],
# 	},
#  {'_organ':['processed_onehot_tum_swap2']
#  }
#  ),
  
#   ('baselines.histopathology.jax_finetune_deterministic', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit_simclr_finetune/crop_<_crop>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune.py',
# 			'config.seed': '<_seed>',
#'config.lr.base':'<_lr>',
#    			'config.model_init': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit_simclr/crop_<_crop>/0/checkpoints/checkpoint_141900.npz'
# 	}, {	'_lr': [1e-3],
# 	}, {	'_seed': [1,5]
# 	},
#   {	'_crop': [0.81],
# 	},
#    {'_organ':['processed_onehot_tum_swap2']
#  }
#  ),

# #   8. CMD finetune
  
#   	('baselines.histopathology.jax_finetune_cmd', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit_cmd/lr_<_lr>/clc_<_clc>/cmo_<_cmo>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune_cmd.py',
# 			'config.seed': '<_seed>',
# 'config.lr.base':'<_lr>',
# 			'config.cmd_loss_coeff':'<_clc>',
# 			'config.cmd_max_order':'<_cmo>'
# 	},  
#   {	'_seed': range(1,6)
# 	} ,
#  {	'_clc': [0.3] #[0.1, 0.3, 1.0, 3, 10.0]
# 	},
#  {	'_cmo': [8] #[3, 5, 8]
# 	},
#   {
# 	'_lr': [1e-3] #[1e-4, 1e-3]  
  
# 	},
 
#  {'_organ':['processed_onehot_tum_swap2']
#  }
#  ),

# # 9 Simclr + DAN
# 	('baselines.histopathology.jax_finetune_dan', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit_simclr_dan/gc_<_gc>/nl_<_nl>/hdim_<_hdim>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune_dan.py',
# 			'config.seed': '<_seed>',
#'config.lr.base':'<_lr>',
# 			'config.model.domain_predictor.grl_coeff': '<_gc>',
#     		'config.model.domain_predictor.num_layers':'<_nl>',
# 			'config.model.domain_predictor.hid_dim':'<_hdim>', 
#    'config.model_init': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit_simclr2/crop_0.64/0/checkpoints/checkpoint_819200.npz'
# 	},  
#   {	'_seed': range(6),
# 	} ,
#  {	'_gc': [1] #[0.1, 0.3, 1, 3, 10],
# 	},
#  {	'_nl': [5] #[2, 3, 4, 5],
# 	},
#  {	'_hdim': [768] # [256, 512, 768],
# 	},
#  {'_organ':['processed_onehot_tum_swap2']
#  }
#  ),


# # 10 SimMIM+DAN
# 	('baselines.histopathology.jax_finetune_dan', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit_mim_dan/gc_<_gc>/nl_<_nl>/hdim_<_hdim>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune_dan.py',
# 			'config.seed': '<_seed>',
#'config.lr.base':'<_lr>',
# 			'config.model.domain_predictor.grl_coeff': '<_gc>',
#     		'config.model.domain_predictor.num_layers':'<_nl>',
# 			'config.model.domain_predictor.hid_dim':'<_hdim>', 
#    'config.model_init': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit_mim/mr_0.6/0/checkpoints/checkpoint_1064960.npz'
# 	},  
#   {	'_seed': range(6),
# 	} ,
#  {	'_gc': [1] #[0.1, 0.3, 1, 3, 10],
# 	},
#  {	'_nl': [5] #[2, 3, 4, 5],
# 	},
#  {	'_hdim': [768] # [256, 512, 768],
# 	},
#  {'_organ':['processed_onehot_tum_swap2']
#  }
#  ),


# # 11 SimMAE+DAN
# 	('baselines.histopathology.jax_finetune_dan', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit_mae_dan/gc_<_gc>/nl_<_nl>/hdim_<_hdim>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune_dan.py',
# 			'config.seed': '<_seed>',
#'config.lr.base':'<_lr>',
# 			'config.model.domain_predictor.grl_coeff': '<_gc>',
#     		'config.model.domain_predictor.num_layers':'<_nl>',
# 			'config.model.domain_predictor.hid_dim':'<_hdim>', 
#    'config.model_init': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit_mae/mr_0.6/0/checkpoints/checkpoint_655360.npz'
# 	},  
#   {	'_seed': range(6),
# 	} ,
#  {	'_gc': [1] #[0.1, 0.3, 1, 3, 10],
# 	},
#  {	'_nl': [5] #[2, 3, 4, 5],
# 	},
#  {	'_hdim': [768] # [256, 512, 768],
# 	},
#  {'_organ':['processed_onehot_tum_swap2']
#  }
#  ),

# # 12. MCD+DAN
# 	('baselines.histopathology.jax_finetune_mcd_dan', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit-mcd-dan/lr_<_lr>/drp_<_drp>/grl_<_grl>/layers_<_layers>/dim_<_dim>/dlc_<_dlc>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune_dan.py',
# 			'config.seed': '<_seed>',
#'config.lr.base':'<_lr>',
# 			'config.use_test': 'True',
# 			'config.lr.base': '<_lr>',
# 			'config.model.transformer.attention_dropout_rate': '<_drp>',
# 			'config.model.transformer.dropout_rate': '<_drp>',
# 			'config.model.domain_predictor.grl_coeff': '<_grl>',
# 			'config.model.domain_predictor.num_layers': '<_layers>',
# 			'config.model.domain_predictor.hid_dim': '<_dim>',
# 			'config.dp_loss_coeff': '<_dlc>'
# 	}, {	'_lr': [1e-3],
# 	}, {	'_drp': [0.01] #[0.01, 0.03, 0.1, 0.3],
# 	}, {	'_grl': [1],
# 	}, {	'_layers': [5],
# 	}, {	'_dim': [768],
# 	}, {	'_dlc': [0.05] #[0.05, 0.1, 3],
# 	}, {	'_seed': range(6),
# 	},
# 	   {'_organ':['processed_onehot_tum_swap2']
# 			}
#  ),




# EVAL
# 0 vit-base
#  ('baselines.histopathology.jax_finetune_deterministic', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_builder_config>/eval/vit/lr_<_lr>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune_eval.py',
# 			'config.seed': '<_seed>',
#'config.lr.base':'<_lr>',
# 			'config.model_init':'gs://ue-usrl-anuj/karm/outputs/histopathology/<_builder_config>/vit/lr_<_lr>/<_seed>/checkpoints/checkpoint_<_step>.npz',
#    			
#       		'config.distribution_shift':'<_builder_config>'
   
# 	}, {	'_lr': [1e-3],
# 	}, {	'_seed': range(6),
#      '_step': [16384]*6
# 	},
#  {
#     '_builder_config':['processed_onehot_tum_swap']
#  },
#  ),
 
#   ('baselines.histopathology.jax_finetune_deterministic', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/eval/vit/lr_<_lr>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune_eval.py',
# 			'config.seed': '<_seed>',
#'config.lr.base':'<_lr>',
# 			'config.model_init':'gs://ue-usrl-anuj/karm/outputs/histopathology/processed_onehot_tum/vit/lr_<_lr>/<_seed>/checkpoints/checkpoint_<_step>.npz',
#    			'config.builder_config': 'processed_512_onehot_o_<_organ>' 
# 	}, {	'_lr': [1e-3],
# 	}, {	'_seed': range(1),
#      '_step': [7110, 35550, 9480, 7110, 7110, 7110]
# 	},
#  {
# 	 '_organ':['upper_extremity', 'lower_extremity', 'palms_soles']
#  }
#  ),
 
#  # 1 osp -- done
#  	('baselines.histopathology.jax_finetune_osp', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/eval/vit_osp/llr_<_llr>/plr_<_plr>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune_osp_eval.py',
# 			'config.seed': '<_seed>',
#'config.lr.base':'<_lr>',
# 			'config.model.lagrangian.lambda_grad':'<_llr>',
# 			'config.model.lagrangian.phi_grad':'<_plr>',
# 			'config.model_init':'gs://ue-usrl-anuj/karm/outputs/histopathology/processed_onehot_tum/vit_osp/llr_<_llr>/plr_<_plr>/<_seed>/checkpoints/checkpoint_<_step>.npz',
   			 
# 	},  
#   {	'_seed': range(6),
#    '_step': [16384, 16384, 16384, 16384, 16384, 16384],
# 	} ,
#  {	'_llr': [1],
# 	},
#  {	'_plr': [0.3],
# 	},
#   {'_organ':['processed_onehot_tum_swap2']
#  }
#  ),
  
#   # 2 mcd -- done
# 	('baselines.histopathology.jax_finetune_mcd', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/eval/vit_mcd/lr_<_lr>/dr_<_dr>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune_mcd_eval.py',
# 			'config.seed': '<_seed>',
# 			'config.lr.base':'<_lr>',
# 			'config.model.transformer.attention_dropout_rate':'<_dr>',
# 			'config.model.transformer.dropout_rate':'<_dr>',
# 			'config.model_init':'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit_mcd/lr_<_lr>/dr_<_dr>/<_seed>/checkpoints/checkpoint_<_step>.npz',
   			 
# 	},  
#   {	'_seed': range(6),
#    '_step': [15360, 40960, 15360, 46080, 15360, 66560],
# 	} ,
#  {	'_dr': [0.01],
# 	},
#    {
# 	'_lr': [1e-3]  
# 	},
#  {'_organ':['processed_onehot_tum_swap2']
#  }
#  ),
 
# #  3 dan+iw
#  	('baselines.histopathology.jax_finetune_impwt', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/processed_onehot_tum/eval/vit_dan_iw/dlc_<_dlc>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune_impwt_eval.py',
# 			'config.seed': '<_seed>',
#'config.lr.base':'<_lr>',
#    			'config.model.domain_predictor.grl_coeff':'1.0',
# 			'config.dp_loss_coeff':'<_dlc>',
# 			'config.model_init':'gs://ue-usrl-anuj/karm/outputs/histopathology/processed_onehot_tum/vit_dan_iw/dlc_<_dlc>/<_seed>/checkpoints/checkpoint_<_step>.npz'
# 	},  
#   {	'_seed': range(6),
#    '_step': [32768]*6,
# 	} ,
#  {	'_dlc': [0.1],
# 	},
#  ),
  

# #  4  iw -- done
#  	('baselines.histopathology.jax_finetune_impwt', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/processed_onehot_tum/eval/vit_iw/dlc_<_dlc>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune_impwt_eval.py',
# 			'config.seed': '<_seed>',
#'config.lr.base':'<_lr>',
#    			'config.model.domain_predictor.grl_coeff':'0',
# 			'config.dp_loss_coeff':'<_dlc>',
# 			'config.model_init':'gs://ue-usrl-anuj/karm/outputs/histopathology/processed_onehot_tum/vit_iw/dlc_<_dlc>/<_seed>/checkpoints/checkpoint_<_step>.npz'
   
# 	},  
#   {	'_seed': range(6),
#    '_step': [81920, 65536, 32768, 32768, 32768, 32768]
# ,
# 	} ,
#  {	'_dlc': [3],
# 	},
#  ),
  
# #   5 mim_finetune -- done
#   ('baselines.histopathology.jax_finetune_deterministic', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/eval/vit_mim_finetune/lr_<_lr>/mr_<_mr>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune_eval.py',
# 			'config.seed': '<_seed>',
# 			'config.lr.base':'<_lr>',
#    			'config.model_init': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit_mim_finetune/lr_<_lr>/mr_<_mr>/<_seed>/checkpoints/checkpoint_<_step>.npz'
# 	}, {	'_lr': [1e-4],
# 	}, {	'_seed': range(6),
#      '_step': [10240, 15360, 10240, 15360, 10240, 10240],
# 	},
#   {	'_mr': [0.4],
# 	},
#  ),
  

# #   5 mae_finetune -- done
#   ('baselines.histopathology.jax_finetune_deterministic', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/processed_onehot_tum/eval/vit_mae_finetune/mr_<_mr>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune_eval.py',
# 			'config.seed': '<_seed>',
#'config.lr.base':'<_lr>',
#    			'config.model_init': 'gs://ue-usrl-anuj/karm/outputs/histopathology/processed_onehot_tum/vit_mae_finetune/mr_<_mr>/<_seed>/checkpoints/checkpoint_<_step>.npz'
# 	}, {	'_lr': [1e-3],
# 	}, {	'_seed': range(6),
#      '_step': [16384]*6,
# 	},
#   {	'_mr': [0.6],
# 	},
#  ),
  
#   6. SimcLR + finetune
#   ('baselines.histopathology.jax_finetune_deterministic', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/processed_onehot_tum/eval/vit_simclr_finetune/crop_<_crop>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune_eval.py',
# 			'config.seed': '<_seed>',
#'config.lr.base':'<_lr>',
#    			'config.model_init': 'gs://ue-usrl-anuj/karm/outputs/histopathology/processed_onehot_tum/vit_simclr_finetune/crop_<_crop>/<_seed>/checkpoints/checkpoint_<_step>.npz',
# 	}, {	'_lr': [1e-3],
# 	}, 
#  {	'_seed': range(6),
#      '_step': [16384]*6,
# 	},
#   {	'_crop': [0.64],
# 	},
#  ),

# #   7. CMD finetune
#   	('baselines.histopathology.jax_finetune_cmd', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/processed_onehot_tum/eval/vit_cmd/lr_<_lr>/clc_<_clc>/cmo_<_cmo>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune_cmd_eval.py',
# 			'config.seed': '<_seed>',
# 			'config.lr.base':'<_lr>',
# 			'config.cmd_loss_coeff':'<_clc>',
# 			'config.cmd_max_order':'<_cmo>',
#    			'config.model_init': 'gs://ue-usrl-anuj/karm/outputs/histopathology/processed_onehot_tum/vit_cmd/lr_<_lr>/clc_<_clc>/cmo_<_cmo>/<_seed>/checkpoints/checkpoint_<_step>.npz',
   
# 	},  
#   {	'_seed': range(6),
#    '_step': [20480, 20480, 20480, 10240, 20480, 20480],
# 	} ,
#  {	'_clc': [0.1],
# 	},
#  {	'_cmo': [8],
# 	},
#  {	'_lr': [1e-3],
# 	}
#  )


# 	('baselines.histopathology.jax_finetune_dan', {
# 			'output_dir': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/eval/vit_dan/gc_<_gc>/nl_<_nl>/hdim_<_hdim>/<_seed>',
# 			'config': 'configs/histopathology/vit_finetune_dan_eval.py',
# 			'config.seed': '<_seed>',
#'config.lr.base':'<_lr>',
# 			'config.model.domain_predictor.grl_coeff': '<_gc>',
#     		'config.model.domain_predictor.num_layers':'<_nl>',
# 			'config.model.domain_predictor.hid_dim':'<_hdim>',
#    			'config.distribution_shift':'<_organ>',
# 			
#    			'config.model_init': 'gs://ue-usrl-anuj/karm/outputs/histopathology/<_organ>/vit_dan/gc_<_gc>/nl_<_nl>/hdim_<_hdim>/<_seed>/checkpoints/checkpoint_<_step>.npz'
# 	},  
#   {	'_seed': range(1),
# #    '_step': [8192,8192,8192,24576,163840,32768]
# 	} ,
#  {	'_gc': [0.1, 0.3, 1, 3],
#   '_step': [8192,8192,8192,24576]
# 	},
#  {	'_nl': [3] #[2, 3, 4, 5],
# 	},
#  {	'_hdim': [256] # [256, 512, 768],
# 	},
#  {'_organ':['processed_onehot_tum_swap2']
#  }
#  ),

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
