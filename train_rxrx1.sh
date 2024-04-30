seed=0
for seed in {0,}
do
data_dir=/troy/anuj/gub-mod/uncertainty-baselines/data/downloads/manual/rxrx1
output_dir=/troy/anuj/karm/outputs/rxrx1/vit
# output_dir=/troy/anuj/karm/outputs/rxrx1/vit_simclr_finetune
chkpt=/troy/anuj/gub-og/checkpoints/vit_imgnet21k/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz
# chkpt=~/karm/outputs/rxrx1/vit_simclr/0/checkpoints/checkpoint_4390.npz
# chkpt=/troy/anuj/gub-og/checkpoints/vit_dan/checkpoint_1647.npz
CUDA_VISIBLE_DEVICES="1" \
python -m baselines.rxrx1.jax_finetune_deterministic \
	--config=configs/rxrx1/vit_finetune.py \
	--config.data_dir=$data_dir \
	--config.model_init=$chkpt \
	--config.batch_size=16 \
	--config.log_training_steps=1000 \
    --config.seed=$seed \
	--output_dir=$output_dir/$seed 
done

# seed=0
# data_dir=/troy/anuj/gub-mod/uncertainty-baselines/data/downloads/manual/rxrx1
# output_dir=/troy/anuj/karm/outputs/rxrx1/vit_simclr
# chkpt=/troy/anuj/gub-og/checkpoints/vit_imgnet21k/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz
# # chkpt=/troy/anuj/gub-og/checkpoints/vit_dan/checkpoint_1647.npz
# CUDA_VISIBLE_DEVICES="0" \
# python -m baselines.rxrx1.jax_pretrain_simclr \
# 	--config=configs/rxrx1/vit_simclr_pretrain.py \
# 	--config.data_dir=$data_dir \
# 	--config.model_init=$chkpt \
# 	--config.batch_size=16 \
# 	--config.log_training_steps=100 \
# 	--config.log_eval_steps=4390 \
# 	--config.checkpoint_steps=4390 \
#     --config.seed=$seed \
# 	--output_dir=$output_dir/$seed \