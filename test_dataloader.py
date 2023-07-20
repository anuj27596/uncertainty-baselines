import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys
sys.path.append("/data3/home/karmpatel/ub_karm/uncertainty-baselines")

import uncertainty_baselines as ub
import jax
import baselines.cancer.input_utils as input_utils 
import numpy as np
train_ds_rng = jax.random.PRNGKey(0)

dataset_name = "cancer_cam16_embd"
data_dir = "/data3/home/karmpatel/dsmil-wsi/datasets/Camelyon16/"

train_base_dataset = ub.datasets.get(
  dataset_name,
  "train",
  builder_config=f'{dataset_name}/processed_swap',
  data_dir=data_dir)

train_dataset_builder = train_base_dataset._dataset_builder  # pylint: disable=protected-access
train_dataset_builder
train_dataset_builder.download_and_prepare()

train_ds = input_utils.get_data(
  dataset=train_dataset_builder,
  split="train",
  rng=train_ds_rng,
  process_batch_size=1,
  data_dir=data_dir,
  preprocess_fn=None, )