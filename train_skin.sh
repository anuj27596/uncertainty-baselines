seed=1
dataset="histopathology"
data_dir="/troy/anuj/gub-mod/uncertainty-baselines/data/downloads/manual/$dataset"
# output_dir=/troy/anuj/karm/outputs/isic/upper_extremity/vit
output_dir="/troy/anuj/karm/outputs/$dataset/vit"
chkpt=/troy/anuj/gub-og/checkpoints/vit_imgnet21k/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz
# chkpt=/troy/anuj/gub-og/checkpoints/vit_dan/checkpoint_1647.npz
CUDA_VISIBLE_DEVICES="0" \
python -m baselines.histopathology.jax_finetune_deterministic \
	--config=configs/histopathology/vit_finetune.py \
	--config.data_dir=$data_dir \
	--config.model_init=$chkpt \
	--config.batch_size=16 \
	--config.log_training_steps=100 \
    --config.seed=$seed \
	--output_dir=$output_dir/$seed \

# seed=0
# data_dir=/troy/anuj/gub-mod/uncertainty-baselines/data/downloads/manual/isic_id
# output_dir="/troy/anuj/karm/outputs/isic/palms_soles/vit"
# chkpt=/troy/anuj/gub-og/checkpoints/vit_imgnet21k/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz
# # chkpt=/troy/anuj/gub-og/checkpoints/vit_dan/checkpoint_1647.npz
# CUDA_VISIBLE_DEVICES="0" \
# python -m baselines.isic.jax_finetune_deterministic \
# 	--config=configs/isic/vit_finetune.py \
# 	--config.data_dir=$data_dir \
# 	--config.model_init=$chkpt \
# 	--config.batch_size=16 \
# 	--config.log_training_steps=100 \
#     --config.seed=$seed \
# 	--config.builder_config="processed_512_onehot_o_palms_soles" \
# 	--output_dir=$output_dir/$seed 

# seed=0
# dataset="histopathology"
# data_dir="/troy/anuj/gub-mod/uncertainty-baselines/data/downloads/manual/$dataset"
# # output_dir="/troy/anuj/karm/outputs/isic/upper_extremity/vit_dan"
# output_dir="/troy/anuj/karm/outputs/isic/upper_extremity/vit_dan"
# chkpt=/troy/anuj/gub-og/checkpoints/vit_imgnet21k/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz
# # chkpt=/troy/anuj/gub-og/checkpoints/vit_dan/checkpoint_1647.npz
# CUDA_VISIBLE_DEVICES="0" \
# python -m baselines.isic.jax_finetune_dan \
#     --output_dir=gs://ue-usrl-anuj/karm/outputs/isic/upper_extremity/vit_dan/gc_0.3/nl_3/hdim_256/5 \
#     --config=configs/isic/vit_finetune_dan.py \
#     --config.seed=5 \
#     --config.model.domain_predictor.grl_coeff=0.3 \
#     --config.model.domain_predictor.num_layers=3 \
#     --config.model.domain_predictor.hid_dim=256 \
# 	--config.batch_size=8 \
#     --config.builder_config=processed_512_onehot_o_upper_extremity