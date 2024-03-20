# seed=0
# for seed in {0,}
# do
# data_dir=/troy/anuj/gub-mod/uncertainty-baselines/data/downloads/manual/isic_id
# # output_dir=/troy/anuj/karm/outputs/isic/vit
# output_dir=/troy/anuj/karm/outputs/isic/head_neck/vit_simclr_dan
# chkpt=/troy/anuj/gub-og/checkpoints/vit_imgnet21k/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz
# # chkpt=~/karm/outputs/isic/vit_simclr/0/checkpoints/checkpoint_4390.npz
# # chkpt=/troy/anuj/gub-og/checkpoints/vit_dan/checkpoint_1647.npz
# CUDA_VISIBLE_DEVICES="0" \
# python -m baselines.isic.jax_finetune_dan \
# 	--config=configs/isic/vit_finetune_dan.py \
# 	--config.data_dir=$data_dir \
# 	--config.model_init=$chkpt \
# 	--config.batch_size=16 \
# 	--config.log_training_steps=500 \
#     --config.seed=$seed \
# 	--output_dir=$output_dir/$seed 
# done

# for seed in {0,}
# do
# data_dir=/troy/anuj/gub-mod/uncertainty-baselines/data/downloads/manual/isic_id
# output_dir=/troy/anuj/karm/outputs/isic/head_neck/vit_simclr
# chkpt=/troy/anuj/gub-og/checkpoints/vit_imgnet21k/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz
# # chkpt=/troy/anuj/gub-og/checkpoints/vit_dan/checkpoint_1647.npz
# CUDA_VISIBLE_DEVICES="0" \
# python -m baselines.isic.jax_pretrain_simclr \
# 	--config=configs/isic/vit_simclr_pretrain.py \
# 	--config.data_dir=$data_dir \
# 	--config.model_init=$chkpt \
# 	--config.batch_size=8 \
# 	--config.log_training_steps=500 \
#     --config.seed=$seed \
# 	--output_dir=$output_dir/$seed \

# done

python -m baselines.isic.jax_finetune_deterministic \
    --output_dir=gs://ue-usrl-anuj/karm/outputs/isic/head_neck/eval/vit/lr_0.001/0 \
    --config=configs/isic/vit_finetune_eval.py \
    --config.seed=0 \
    --config.model_init=gs://ue-usrl-anuj/karm/outputs/isic/head_neck/vit/lr_0.001/0/checkpoints/checkpoint_7110.npz