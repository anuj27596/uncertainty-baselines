import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import uncertainty_baselines as ub
# import baselines.diabetic_retinopathy_detection.input_utils as input_utils

# import jax
# from clu import preprocess_spec
# import baselines.diabetic_retinopathy_detection.preprocess_utils as preprocess_utils


os.environ['CUDA_VISIBLE_DEVICES'] = ''
	
data_dir = '/troy/anuj/gub-mod/uncertainty-baselines/data/downloads/manual/diabetic_retinopathy_diagnosis'
ds_name = 'ub_diabetic_retinopathy_detection'
split = 'train'
batch_size = 16
# local_batch_size = batch_size // jax.process_count()
# shuffle_buffer_size = 10_000

# rng = jax.random.PRNGKey(0)
# rng, train_ds_rng = jax.random.split(rng)
# train_ds_rng = jax.random.fold_in(train_ds_rng, jax.process_index())

builder = ub.datasets.get(
    ds_name,
    split = split,
    data_dir = data_dir,
    decision_threshold = 'moderate',
    cache = False,
    drop_remainder = False,
    # builder_config = f'{ds_name}/original',
    download_data = True,
)

# builder = builder._dataset_builder
dataset = builder.load(batch_size = batch_size)
batch = next(iter(dataset))

# plt.imshow(images[0])
images, labels, names = batch['features'], batch['labels'], batch['name']

# train_base_dataset = ub.datasets.get( # returns: class UBDiabeticRetinopathyDetectionDataset(base.BaseDataset)
#       dataset_names['in_domain_dataset'], # 'ub_diabetic_retinopathy_detection'
#       split=split_names['train_split'],
#       data_dir=config.get('data_dir'),
#       download_data=True, # Karm
#       builder_config='ub_diabetic_retinopathy_detection/btgraham-300-left') # Karm
# train_dataset_builder = train_base_dataset._dataset_builder  # pylint: disable=protected-access
  
# num_classes = 2
# pp_train = (f'diabetic_retinopathy_preprocess({512})' + f'|onehot({num_classes})')
# preproc_fn = preprocess_spec.parse(spec=pp_train, available_ops=preprocess_utils.all_ops())
# train_ds = input_utils.get_data(
#       dataset=train_dataset_builder,
#       split="train",
#       rng=train_ds_rng,
#       process_batch_size=local_batch_size,
#       preprocess_fn=preproc_fn,
#       shuffle_buffer_size=shuffle_buffer_size,
#     #   prefetch_size=config.get('prefetch_to_host', 2),
#       data_dir=data_dir)

# batch = next(iter(builder.load(batch_size = batch_size)))

# plt.imshow(images[0])