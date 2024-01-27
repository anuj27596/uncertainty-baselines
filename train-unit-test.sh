
data_dir=/troy/anuj/gub-mod/uncertainty-baselines/data/downloads/manual/diabetic_retinopathy_diagnosis
# data_dir=/troy/anuj/gub-mod/uncertainty-baselines/data/downloads/manual/chest_xray

chkpt=/troy/anuj/gub-og/checkpoints/vit_imgnet21k/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz


if [ "$1" == "mae" ]
then
	output_dir=/troy/anuj/gub-og/outputs/local/temp
	rm -r $output_dir/*

	CUDA_VISIBLE_DEVICES="0" \
	python -m baselines.diabetic_retinopathy_detection.jax_pretrain_mae \
		--config=configs/drd/vit_pretrain_mae.py \
		--config.data_dir=$data_dir \
		--config.model_init=$chkpt \
		--config.batch_size=2 \
		--output_dir=$output_dir \

elif [ "$1" == "temp" ]
then
	output_dir=/troy/anuj/gub-og/outputs/local/temp
	rm -r $output_dir/*

	CUDA_VISIBLE_DEVICES="0" \
	python -m baselines.diabetic_retinopathy_detection.jax_finetune_deterministic \
		--config=configs/drd/vit_finetune.py \
		--config.data_dir=$data_dir \
		--config.model_init=$chkpt \
		--config.batch_size=2 \
		--output_dir=$output_dir \

elif [ "$1" == "dvp" ]
then
	output_dir=/troy/anuj/gub-og/outputs/local/temp
	rm -r $output_dir/*

	CUDA_VISIBLE_DEVICES="0" \
	python -m baselines.diabetic_retinopathy_detection.jax_finetune_dan \
		--config=configs/drd/vit_finetune_dan.py \
		--config.data_dir=$data_dir \
		--config.model_init=$chkpt \
		--config.batch_size=2 \
		--config.ood_val_percent=50 \
		--output_dir=$output_dir \

else
	echo not implemented
fi
