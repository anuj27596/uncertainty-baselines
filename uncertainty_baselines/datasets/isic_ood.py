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

"""APTOS 2019 Blindness Detection dataset builder.

https://www.kaggle.com/c/aptos2019-blindness-detection/overview
"""
import os
from typing import Dict, Optional

import pandas as pd
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from uncertainty_baselines.datasets import base
from uncertainty_baselines.datasets.processing_pneumonia import _pneumonia_processing, _resize_image_if_necessary

_DESCRIPTION = """\
APTOS is a dataset containing the 3,662 high-resolution fundus images
from the APTOS 2019 Blindness Detection Kaggle competition
https://www.kaggle.com/c/aptos2019-blindness-detection.

We split ~80% (2929 examples) as a test set
and ~20% (733 examples) as a validation set.

Intended use is as a set of distributionally shifted evaluation sets
with the larger Kaggle/EyePACS Diabetic Retinopathy Detection dataset.
https://www.kaggle.com/c/diabetic-retinopathy-detection
"""

_CITATION = """\
 @misc{
   kaggle,
   title={APTOS 2019 blindness detection},
   url={https://www.kaggle.com/c/aptos2019-blindness-detection},
   journal={Kaggle}
 }
"""

_NUM_CLASSES = 2 # Karm
_BTGRAHAM_DESCRIPTION_PATTERN = (
    "Images have been preprocessed as the winner of the Kaggle competition did "
    "in 2015: first they are resized so that the radius of an eyeball is "
    "{} pixels, then they are cropped to 90% of the radius, and finally they "
    "are encoded with 72 JPEG quality.")
_BLUR_BTGRAHAM_DESCRIPTION_PATTERN = (
    "A variant of the processing method used by the winner of the 2015 Kaggle "
    "competition: images are resized so that the radius of an eyeball is "
    "{} pixels, then receive a Gaussian blur-based normalization with Kernel "
    "standard deviation along the X-axis of {}. Then they are cropped to 90% "
    "of the radius, and finally they are encoded with 72 JPEG quality.")


class IsicOodConfig(tfds.core.BuilderConfig):
  """BuilderConfig for APTOS 2019 Blindness Detection."""

  def __init__(self, target_pixels=None, blur_constant=None, **kwargs):
    """BuilderConfig for APTOS 2019 Blindness Detection.

    Args:
      target_pixels: If given, rescale the images so that the total number of
        pixels is roughly this value.
      blur_constant: Constant used to vary the Kernel standard deviation in
        smoothing the image with Gaussian blur.
      **kwargs: keyword arguments forward to super.
    """
    super(IsicOodConfig, self).__init__(
        version=tfds.core.Version("1.0.0"),
        release_notes={
            "1.0.0": "Initial release.",
        },
        **kwargs)
    self._target_pixels = target_pixels
    self._blur_constant = blur_constant

  @property
  def target_pixels(self):
    return self._target_pixels

  @property
  def blur_constant(self):
    return self._blur_constant


class IsicOod(tfds.core.GeneratorBasedBuilder):
  """APTOS 2019 Blindness Detection dataset."""
  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
    You have to download this dataset from Kaggle.
    https://www.kaggle.com/c/aptos2019-blindness-detection/data
    """
  BUILDER_CONFIGS = [
      IsicOodConfig(
          name="original",
          description="Images at their original resolution and quality."),
      IsicOodConfig(
          name="processed",
          description=_BTGRAHAM_DESCRIPTION_PATTERN.format(300),
          target_pixels=300),
      # IsicOodConfig(
      #     name="frontal",
      #     description=_BTGRAHAM_DESCRIPTION_PATTERN.format(300),
      #     target_pixels=300),
      # IsicOodConfig(
      #     name="processed_swap",
      #     description=_BTGRAHAM_DESCRIPTION_PATTERN.format(300),
      #     target_pixels=300),
  ]

  def _info(self):
    """Returns basic information of dataset.

    Returns:
      tfds.core.DatasetInfo.
    """
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "name": tfds.features.Text(),  # patient ID. eg: "000c1434d8d7".
            "image": tfds.features.Image(),
            # From 0 (no DR) to 4 (Proliferative DR). -1 if no label provided.
            # "label": tfds.features.ClassLabel(num_classes=_NUM_CLASSES), # Karm
            "label": tfds.features.ClassLabel(num_classes=_NUM_CLASSES) # Karm
        }),
        homepage="https://www.kaggle.com/c/aptos2019-blindness-detection/data",
        citation=_CITATION)

  def _split_generators(self, dl_manager):
    """We only have labels for the train images.

    We use these for evaluation (e.g., one could train with the
    Kaggle Diabetic Retinopathy Detection dataset, which is actually
    reasonably large, and evaluate with this APTOS dataset of
    3,662 examples).

    Args:
      dl_manager: download manager.

    Returns:
      A tuple (list) of tfds.core.SplitGenerator for validation and test.
    """
    # TODO(nband): implement download using kaggle API.
    # TODO(nband): implement extraction of multiple files archives.
    path = dl_manager.manual_dir
    # if 'swap' not in self.builder_config.name:
    #   splits = ('train', 'validation', 'test')
    # else:
    #   splits = ('validation', 'test')    
    splits = ('test', 'validation')
    
    return [
        tfds.core.SplitGenerator(
            name=split,
            gen_kwargs={
                "images_dir_path": os.path.join(path,"isic_ood","train"),
                "split": split,
                "csv_path": os.path.join(path,"isic_ood","ISIC_2020_Training_GroundTruth.csv"),
            })
        for split in splits
    ]

  def _generate_examples(self,
                         images_dir_path,
                         split,
                         csv_path=None,
                         csv_usage=None):
    """Yields Example instances from given CSV.

    Args:
      images_dir_path: path to dir in which images are stored.
      is_validation: bool, use validation set (20%) else use test (80%).
      csv_path: optional, path to csv file with two columns: name of image and
        label. If not provided, just scan image directory, don't set labels.
      csv_usage: optional, subset of examples from the csv file to use based on
        the "Usage" column from the csv.
    """
    if csv_path:
      with tf.io.gfile.GFile(csv_path) as csv_f:
        df = pd.read_csv(csv_f)
        df = df.sample(frac=1, random_state=1)
        organs = ["lower extremity"]

      # if 'swap' in self.builder_config.name and split == 'validation':
      #   split = ('train', 'validation') # consider train & validation as validation split
      # else:
      # split = (split,)
      
      # karm: validation = test, because test labels are not availabe
      if split == "validation":
        data = df[df["anatom_site_general_challenge"].isin(organs)]
        data['path'] = data["image_name"].apply(lambda x: os.path.join(x+".jpg"))
        data = data[["path", "target"]]
      
      else:
        data = df[df["anatom_site_general_challenge"].isin(organs)]
        data['path'] = data["image_name"].apply(lambda x: os.path.join(x+".jpg"))
        data = data[["path", "target"]]
        
    else:
      data = [(fname, [-1]*_NUM_CLASSES) for fname in tf.io.gfile.listdir(images_dir_path)] # Karm D
            # if fname.endswith(".png") # Karm

    print(f"Using BuilderConfig {self.builder_config.name}.")

    for idx, (path, label) in data.iterrows(): # Karm
      image_filepath = os.path.join(images_dir_path, path)
      record = {
          "name": path,
          "image": self._process_image(image_filepath),
          "label": int(label) # Karm
      }
      yield path, record

  def _process_image(self, filepath):
    with tf.io.gfile.GFile(filepath, mode="rb") as image_fobj:
      if self.builder_config.name.startswith("processed") \
          or self.builder_config.name.startswith("frontal"):
        return _pneumonia_processing(
            image_fobj=image_fobj,
            filepath=filepath,
            target_pixels=self.builder_config.target_pixels)
      else:
        return _resize_image_if_necessary(  # pylint: disable=protected-access
            image_fobj=image_fobj,
            target_pixels=self.builder_config.target_pixels)


class IsicOodDataset(base.BaseDataset):
  """Kaggle APTOS 2019 Blindness Detection dataset builder class."""

  def __init__(self,
               split: str,
               builder_config: str = "isic_ood/processed",
               shuffle_buffer_size: Optional[int] = None,
               num_parallel_parser_calls: int = 64,
               data_dir: Optional[str] = None,
               download_data: bool = False,
               drop_remainder: bool = True,
               is_training: Optional[bool] = None,
               cache: bool = False):
    """Create a APTOS 2019 Blindness Detection tf.data.Dataset builder.

    Args:
      split: a dataset split, either a custom tfds.Split or one of the
        tfds.Split enums [TRAIN, VALIDATION, TEST] or their lowercase string
        names.
      builder_config: a builder config contained in the APTOS dataset builder
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      data_dir: optional dir to save TFDS data to. If none then the local
        filesystem is used. Required for using TPUs on Cloud.
      download_data: Whether or not to download data before loading.
      drop_remainder: whether or not to drop the last batch of data if the
        number of points is not exactly equal to the batch size. This option
        needs to be True for running on TPUs. We probably don't want it for such
        a small eval dataset.
      decision_threshold: specifies where to binarize the labels {0, 1, 2, 3, 4}
        to create the binary classification task.
        'mild': classify {0} vs {1, 2, 3, 4}, i.e., mild DR or worse?
        'moderate': classify {0, 1} vs {2, 3, 4}, i.e., moderate DR or worse?
      cache: Whether or not to cache the dataset in memory. Can lead to OOM
        errors in host memory.
    """
    if is_training is None:
      is_training = split in ['train', tfds.Split.TRAIN]
    print(f"Using IsicOod builder config {builder_config}.")
    dataset_builder = tfds.builder(builder_config, data_dir=data_dir)
    super(IsicOodDataset, self).__init__(
        name=f"{dataset_builder.name}/{dataset_builder.builder_config.name}",
        dataset_builder=dataset_builder,
        split=split,
        is_training=is_training,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        download_data=download_data,
        drop_remainder=drop_remainder,
        cache=cache)
    print(f"Building isic ID dataset")
    if not drop_remainder:
      print("Not dropping the remainder (i.e., not truncating last batch).")

  def _create_process_example_fn(self) -> base.PreProcessFn:

    def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      """Preprocess images to range [0, 1], binarize task based on provided decision threshold, produce example `Dict`."""
      image = example["image"]
      image = tf.image.convert_image_dtype(image, tf.float32)
      image = tf.image.resize(image, size=(256, 256), method="bilinear")

      label = tf.cast(example["label"], tf.int32)

      parsed_example = {
          "features": image,
          "labels": label,
          "name": example["name"],
      }
      return parsed_example

    return _example_parser