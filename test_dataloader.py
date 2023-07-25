import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
sys.path.append("uncertainty-baselines")

import uncertainty_baselines as ub
import jax
import baselines.pacs.input_utils as input_utils 
import baselines.chest_xray.preprocess_utils as preprocess_utils  # EDIT(anuj)
from clu import preprocess_spec
import numpy as np
train_ds_rng = jax.random.PRNGKey(0)

dataset_name = "pacs_ood"
#data_dir = "/data3/home/karmpatel/dsmil-wsi/datasets/Camelyon16/"
data_dir = "~/troy_anuj/gub-mod/uncertainty-baselines/data/downloads/manual/pacs"

train_base_dataset = ub.datasets.get(
  dataset_name,
  "validation",
  builder_config=f'{dataset_name}/processed_sketch', 
  data_dir=data_dir)

train_dataset_builder = train_base_dataset._dataset_builder  # pylint: disable=protected-access
train_dataset_builder
train_dataset_builder.download_and_prepare()

# train_base_dataset = ub.datasets.get(
#   dataset_name,
#   "test",
#   builder_config=f'{dataset_name}/processed',
#   data_dir=data_dir)

# train_dataset_builder = train_base_dataset._dataset_builder  # pylint: disable=protected-access
# train_dataset_builder
# train_dataset_builder.download_and_prepare()

preproc_fn = preprocess_spec.parse(
      spec="chest_xray_preprocess(256)", available_ops=preprocess_utils.all_ops())

train_ds = input_utils.get_data(
  dataset=train_dataset_builder,
  split="validation",
  rng=train_ds_rng,
  process_batch_size=1,
  data_dir=data_dir,
  preprocess_fn=preproc_fn, )

train_ds = input_utils.get_data(
  dataset=train_dataset_builder,
  split="test",
  rng=train_ds_rng,
  process_batch_size=1,
  data_dir=data_dir,
  preprocess_fn=preproc_fn, )

b = next(iter(train_ds.batch(batch_size=2)))    

import matplotlib.pyplot as plt
plt.imsave("t.jpg", b['image'][0][0][0].numpy())
