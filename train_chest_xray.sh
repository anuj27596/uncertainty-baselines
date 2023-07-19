seed=0
data_dir=/troy/anuj/gub-mod/uncertainty-baselines/data/downloads/manual/cancer_cam16_embs
output_dir=/troy/anuj/karm/outputs/cancer_cam16_embs/vit_simclr
chkpt=/troy/anuj/gub-og/checkpoints/vit_imgnet21k/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz
# chkpt=/troy/anuj/gub-og/checkpoints/vit_dan/checkpoint_1647.npz
CUDA_VISIBLE_DEVICES="0" \
python -m baselines.cancer.jax_finetune_deterministic \
	--config=configs/cancer/vit_finetune.py \
	--config.data_dir=$data_dir \
	--config.model_init=$chkpt \
	--config.batch_size=16 \
	--config.log_training_steps=100 \
	--config.log_eval_steps=4390 \
	--config.checkpoint_steps=4390 \
    --config.seed=$seed \
	--output_dir=$output_dir/$seed \