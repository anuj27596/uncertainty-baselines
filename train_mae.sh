seed=0
data_dir=~/datasets/isic_id
# output_dir=/data5/scratch/karmpatel/outputs/isic/vit
# output_dir=/data5/scratch/karmpatel/outputs/isic/head_neck/vit
output_dir=/data5/scratch/karmpatel/outputs/isic/head_neck/vit_mae
chkpt=~/datasets/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz
# chkpt=~/outputs/ext_ops/isic/head_neck/vit_simclr/0/checkpoints/checkpoint_18920.npz
CUDA_VISIBLE_DEVICES="6" \
python -m baselines.isic.jax_pretrain_mae \
	--config=configs/isic/vit_pretrain_mae.py \
	--config.data_dir=$data_dir \
	--config.model_init=$chkpt \
	--config.batch_size=16 \
	--config.log_training_steps=500 \
    --config.seed=$seed \
	--output_dir=$output_dir/$seed \