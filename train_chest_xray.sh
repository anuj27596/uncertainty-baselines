seed=0
data_dir=/data3/home/karmpatel/dsmil-wsi/datasets/Camelyon16/
output_dir=/data3/home/karmpatel/ub_karm/outputs
chkpt=/troy/anuj/gub-og/checkpoints/vit_imgnet21k/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz
# chkpt=/troy/anuj/gub-og/checkpoints/vit_dan/checkpoint_1647.npz
CUDA_VISIBLE_DEVICES="1" \
python -m baselines.cancer.jax_finetune_deterministic \
	--config=configs/cancer/vit_finetune.py \
	--config.data_dir=$data_dir \
	--config.batch_size=1 \
	--config.log_training_steps=100 \
	--config.log_eval_steps=4390 \
	--config.checkpoint_steps=4390 \
    --config.seed=$seed \
	--output_dir=$output_dir/$seed \