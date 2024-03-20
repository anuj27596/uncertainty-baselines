# python -m baselines.histopathology.jax_pretrain_simclr_debiased \
#     --output_dir=gs://ue-usrl-anuj/karm/outputs/histopathology/processed_onehot_tum/vit_simclr_debiased/crop_0.64/0 \
#     --config=configs/histopathology/vit_simclr_pretrain.py \
#     --config.seed=0 \
#     --config.pp_train="histopathology_preprocess(224)|simclr_aug(0.64)" \
#     --config.pp_eval="histopathology_preprocess(224)|simclr_aug(0.64)" --config.batch_size=8

# python -m baselines.histopathology.jax_finetune_cmd \
#     --output_dir=gs://ue-usrl-anuj/karm/outputs/histopathology/processed_onehot_tum/vit_cmd/clc_0.1/cmo_2/0 \
#     --config=configs/histopathology/vit_finetune_cmd.py \
#     --config.seed=0 \
#     --config.cmd_loss_coeff=0.1 \
#     --config.cmd_max_order=2 --config.batch_size=8
python -m baselines.histopathology.jax_pretrain_simclr_debiased \
    --output_dir=gs://ue-usrl-anuj/karm/outputs/histopathology/processed_onehot_tum_swap/vit_simclr_debiased/lr_0.0001/crop_64/0 \
    --config=configs/histopathology/vit_simclr_pretrain.py \
    --config.seed=0 \
    --config.lr.base=0.0001 \
    --config.pp_train="histopathology_preprocess(224)|simclr_aug(64, 45)" \
    --config.pp_eval="histopathology_preprocess(224)|simclr_aug(64, 45)" \
    --config.batch_size=8

# python -m baselines.histopathology.jax_finetune_dan \
#     --output_dir=gs://ue-usrl-anuj/karm/outputs/histopathology/processed_onehot_tum/vit_simclr_dan/gc_1/nl_5/hdim_768/2 \
#     --config=configs/histopathology/vit_finetune_dan.py \
#     --config.seed=2 \
#     --config.model.domain_predictor.grl_coeff=1 \
#     --config.model.domain_predictor.num_layers=5 \
#     --config.model.domain_predictor.hid_dim=768 \
#     --config.model_init=gs://ue-usrl-anuj/karm/outputs/histopathology/processed_onehot_tum/vit_simclr2/crop_0.64/0/checkpoints/checkpoint_819200.npz --config.batch_size=8
# # --config.batch_size=8