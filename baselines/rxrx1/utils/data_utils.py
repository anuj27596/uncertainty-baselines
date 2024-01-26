# coding=utf-8
# Copyright 2022 The Uncertainty Baselines Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Diabetic Retinopathy Data Loading utils."""
# pylint: disable=g-bare-generic
# pylint: disable=g-doc-args
# pylint: disable=g-doc-return-or-yield
# pylint: disable=g-importing-member
# pylint: disable=g-no-space-after-docstring-summary
# pylint: disable=g-short-docstring-punctuation
# pylint: disable=logging-format-interpolation
# pylint: disable=logging-fstring-interpolation
# pylint: disable=missing-function-docstring
import logging

import tensorflow_datasets as tfds

import uncertainty_baselines as ub


def load_pneumonia_dataset(train_batch_size,
                           eval_batch_size,
                           flags,
                           strategy,
                           load_for_eval=False):
  """Full Kaggle/EyePACS Diabetic Retinopathy dataset, including OOD validation/test sets (APTOS).

  Optionally exclude train split (e.g., loading for evaluation) in flags.
  See runscripts for more information on loading options.

  Args:
    train_batch_size: int.
    eval_batch_size: int.
    flags: FlagValues, runscript flags.
    strategy: tf.distribute strategy, used to distribute datasets.
    load_for_eval: Bool, if True, does not truncate the last batch.

  Returns:
    Dict of datasets, Dict of number of steps per dataset.
  """
  data_dir = flags.data_dir
  load_train_split = flags.load_train_split

  # * Load Steps Per Epoch for Each Dataset *
  split_to_steps_per_epoch = {}

  # As per the Kaggle challenge, we have split sizes for the EyePACS subsets:
  # train: 35,126
  # validation: 10,906
  # test: 42,670
  if load_train_split:
    split_to_steps_per_epoch['train']               = 4709  // train_batch_size
  split_to_steps_per_epoch['in_domain_validation']  = 523   // eval_batch_size
  split_to_steps_per_epoch['in_domain_test']        = 624   // eval_batch_size

  # ChestXray8 Evaluation Data
  split_to_steps_per_epoch['ood_validation']        = 86524 // eval_batch_size
  split_to_steps_per_epoch['ood_test']              = 25596 // eval_batch_size

  # * Load Datasets *
  split_to_dataset = {}

  # Load validation data
  dataset_validation_builder = ub.datasets.get(
      'zhang_pneumonia',
      split='validation',
      data_dir=data_dir,
      is_training=not flags.use_validation,
      # download_data=True,  # anuj
      cache=flags.cache_eval_datasets,
      drop_remainder=not load_for_eval,
      builder_config=f'zhang_pneumonia/{flags.preproc_builder_config}')
  validation_batch_size = (
      eval_batch_size if flags.use_validation else train_batch_size)
  dataset_validation = dataset_validation_builder.load(
      batch_size=validation_batch_size).repeat()  # anuj

  # If `flags.use_validation`, then we distribute the validation dataset
  # independently and add as a separate dataset.
  # Otherwise, we concatenate it with the training data below.
  if flags.use_validation:
    # Load APTOS validation dataset
    chest_xray8_validation_builder = ub.datasets.get(
        'chest_xray8',
        split='validation',
        data_dir=data_dir,
        # download_data=True,  # anuj
        cache=flags.cache_eval_datasets,
        drop_remainder=not load_for_eval,
        builder_config=f'chest_xray8/{flags.preproc_builder_config}')
    dataset_ood_validation = chest_xray8_validation_builder.load(
        batch_size=eval_batch_size).repeat()  # anuj

    if strategy is not None:
      dataset_validation = strategy.experimental_distribute_dataset(
          dataset_validation)
      dataset_ood_validation = strategy.experimental_distribute_dataset(
          dataset_ood_validation)

    split_to_dataset['in_domain_validation'] = dataset_validation
    split_to_dataset['ood_validation'] = dataset_ood_validation

  if load_train_split:
    # Load EyePACS train data
    dataset_train_builder = ub.datasets.get(
        'zhang_pneumonia',
        split='train',
        data_dir=data_dir,
        # download_data=True,  # anuj
        builder_config=f'zhang_pneumonia/{flags.preproc_builder_config}')
    dataset_train = dataset_train_builder.load(batch_size=train_batch_size).repeat()  # anuj

    if not flags.use_validation:
      # TODO(nband): investigate validation dataset concat bug
      # Note that this will not create any mixed batches of
      # train and validation images.
      # dataset_train = dataset_train.concatenate(dataset_validation)
      raise NotImplementedError(
          'Existing bug involving the number of steps not being adjusted after '
          'concatenating the validation dataset. Needs verifying.')

    if strategy is not None:
      dataset_train = strategy.experimental_distribute_dataset(dataset_train)

    split_to_dataset['train'] = dataset_train

  if flags.use_test:
    # In-Domain Test
    dataset_test_builder = ub.datasets.get(
        'zhang_pneumonia',
        split='test',
        data_dir=data_dir,
        # download_data=True,  # anuj
        cache=flags.cache_eval_datasets,
        drop_remainder=not load_for_eval,
        builder_config=f'zhang_pneumonia/{flags.preproc_builder_config}')
    dataset_test = dataset_test_builder.load(batch_size=eval_batch_size).repeat()  # anuj
    if strategy is not None:
      dataset_test = strategy.experimental_distribute_dataset(dataset_test)

    split_to_dataset['in_domain_test'] = dataset_test

    # OOD (APTOS) Test
    chest_xray8_test_builder = ub.datasets.get(
        'chest_xray8',
        split='test',
        data_dir=data_dir,
        # download_data=True,  # anuj
        cache=flags.cache_eval_datasets,
        drop_remainder=not load_for_eval,
        builder_config=f'chest_xray8/{flags.preproc_builder_config}')
    dataset_ood_test = chest_xray8_test_builder.load(batch_size=eval_batch_size).repeat()  # anuj
    if strategy is not None:
      dataset_ood_test = strategy.experimental_distribute_dataset(
          dataset_ood_test)

    split_to_dataset['ood_test'] = dataset_ood_test

  return split_to_dataset, split_to_steps_per_epoch


def load_dataset(train_batch_size,
                 eval_batch_size,
                 flags,
                 strategy,
                 load_for_eval=False):
  """Retrieve the in-domain and OOD datasets for a given distributional shift task in diabetic retinopathy.

  Optionally exclude train split (e.g., loading for evaluation) in flags.
  See runscripts for more information on loading options.

  Args:
    train_batch_size: int.
    eval_batch_size: int.
    flags: FlagValues, runscript flags.
    strategy: tf.distribute strategy, used to distribute datasets.
    load_for_eval: Bool, if True, does not truncate the last batch.

  Returns:
    Dict of datasets, Dict of number of steps per dataset.
  """
  datasets, steps = load_pneumonia_dataset(
      train_batch_size,
      eval_batch_size,
      flags=flags,
      strategy=strategy,
      load_for_eval=load_for_eval)

  logging.info(f'Datasets using builder config {flags.preproc_builder_config}.')
  logging.info(f'Successfully loaded the following dataset splits from the '
               f'pneumonia shift dataset: {list(datasets.keys())}')
  return datasets, steps
