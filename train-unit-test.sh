
data_dir=/troy/anuj/gub-mod/uncertainty-baselines/data/downloads/manual/diabetic_retinopathy_diagnosis

output_dir=/troy/anuj/gub-og/outputs/vit/eval_results/in21k_dan/0


# chkpt=/troy/anuj/gub-og/checkpoints/vit_imgnet21k/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz
chkpt=/troy/anuj/gub-og/checkpoints/vit_dan/checkpoint_1647.npz

CUDA_VISIBLE_DEVICES="0" \
python -m baselines.diabetic_retinopathy_detection.jax_finetune_dan \
	--config=configs/drd/vit_eval.py \
	--config.data_dir="/troy/anuj/gub-mod/uncertainty-baselines/data/downloads/manual/diabetic_retinopathy_diagnosis" \
	--config.model_init=$chkpt \
	--config.batch_size=64 \
	--config.log_training_steps=100 \
	--config.log_eval_steps=4390 \
	--config.checkpoint_steps=4390 \
	--output_dir=$output_dir \

