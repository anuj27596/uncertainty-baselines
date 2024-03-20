seed=0
dataset="histopathology"
data_dir="/troy/anuj/gub-mod/uncertainty-baselines/data/downloads/manual/$dataset"
output_dir="/troy/anuj/karm/outputs/$dataset/processed_onehot_tum_swap2/vit_dan"
chkpt=/troy/anuj/gub-og/checkpoints/vit_imgnet21k/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz
# chkpt=/troy/anuj/gub-og/checkpoints/vit_dan/checkpoint_1647.npz
# CUDA_VISIBLE_DEVICES="0" \
# python -m baselines.histopathology.jax_finetune_dan \
#     --output_dir=$output_dir/$seed \
#     --config=configs/histopathology/vit_finetune_dan.py \
#     --config.data_dir=$data_dir \
#     --config.seed=$seed \
#     --config.model.domain_predictor.grl_coeff=0.3 \
#     --config.model.domain_predictor.num_layers=3 \
#     --config.model.domain_predictor.hid_dim=256 \
# 	--config.batch_size=8 \
# 	--config.model_init=$chkpt \

# output_dir="/troy/anuj/karm/outputs/$dataset/processed_onehot_tum_swap2/vit_iw"
# CUDA_VISIBLE_DEVICES="0" \
# python -m baselines.histopathology.jax_finetune_impwt \
#     --output_dir=$output_dir/$seed \
#     --config=configs/histopathology/vit_finetune_impwt.py \
#     --config.data_dir=$data_dir \
#     --config.seed=0 \
#     --config.lr.base=0.001 \
#     --config.model.domain_predictor.grl_coeff=0 \
#     --config.dp_loss_coeff=0.3 \
#     --config.batch_size=8 \
# 	--config.model_init=$chkpt 

output_dir="/troy/anuj/karm/outputs/$dataset/processed_onehot_tum_swap2/vit_cmd"
CUDA_VISIBLE_DEVICES="0" \
python -m baselines.histopathology.jax_finetune_cmd \
    --output_dir=$output_dir/$seed \
    --config=configs/histopathology/vit_finetune_cmd.py \
    --config.data_dir=$data_dir \
    --config.seed=0 \
    --config.lr.base=0.001 \
    --config.cmd_loss_coeff=10.0 \
    --config.cmd_max_order=8