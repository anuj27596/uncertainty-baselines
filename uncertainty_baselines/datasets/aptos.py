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

import numpy as np # Karm
from uncertainty_baselines.datasets import base
from uncertainty_baselines.datasets.diabetic_retinopathy_dataset_utils import _btgraham_processing

# By karm
from tensorflow_datasets.image_classification.diabetic_retinopathy_detection import _scale_radius_size
from tensorflow_datasets.image_classification.diabetic_retinopathy_detection import _subtract_local_average
from tensorflow_datasets.image_classification.diabetic_retinopathy_detection import _mask_and_crop_to_radius
import io

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

_NUM_EXAMPLES = 3662
_NUM_TRAINING_EXAMPLES = 2929
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


def _btgraham_processing_left(image_fobj, filepath, target_pixels, crop_to_radius=False, eye_label=np.nan # Karm
):
    """Process an image as the winner of the 2015 Kaggle competition.

    Args:
      image_fobj: File object containing the original image.
      filepath: Filepath of the image, for logging purposes only.
      target_pixels: The number of target pixels for the radius of the image.
      crop_to_radius: If True, crop the borders of the image to remove gray areas.

    Returns:
      A file object.
    """
    cv2 = tfds.core.lazy_imports.cv2
    # Decode image using OpenCV2.
    image = cv2.imdecode(
        np.frombuffer(image_fobj.read(), dtype=np.uint8), flags=3
    )
    # Process the image.
    image_name = filepath.split("/")[-1].split(".")[0] # Karm
    if eye_label == "left": # Karm
      # import pdb; pdb.set_trace()
      # breakpoint()
      # import matplotlib.pyplot as plt
      # output_dir = 'btgraham_processing_left_images_aptos'
      # plt.imsave(os.path.join(output_dir, f'{image_name}_original.png'), image)
      image = tf.image.flip_left_right(image).numpy() # Karm
      # plt.imsave(os.path.join(output_dir, f'{image_name}_flipped.png'), image)
      
    image = _scale_radius_size(image, filepath, target_radius_size=target_pixels)
    image = _subtract_local_average(image, target_radius_size=target_pixels)
    image = _mask_and_crop_to_radius(
        image,
        target_radius_size=target_pixels,
        radius_mask_ratio=0.9,
        crop_to_radius=crop_to_radius,
    )
    # Encode the image with quality=72 and store it in a BytesIO object.
    _, buff = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    return io.BytesIO(buff.tobytes())
  
class APTOSConfig(tfds.core.BuilderConfig):
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
    super(APTOSConfig, self).__init__(
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


class APTOS(tfds.core.GeneratorBasedBuilder):
  """APTOS 2019 Blindness Detection dataset."""
  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
    You have to download this dataset from Kaggle.
    https://www.kaggle.com/c/aptos2019-blindness-detection/data
    """
  BUILDER_CONFIGS = [
      APTOSConfig(
          name="original",
          description="Images at their original resolution and quality."),
      APTOSConfig(
          name="btgraham-300",
          description=_BTGRAHAM_DESCRIPTION_PATTERN.format(300),
          target_pixels=300),
      APTOSConfig( # Karm
          name="btgraham-300-left",
          description=_BTGRAHAM_DESCRIPTION_PATTERN.format(300),
          target_pixels=300),
      APTOSConfig(
          name="blur-3-btgraham-300",
          description=_BLUR_BTGRAHAM_DESCRIPTION_PATTERN.format(300, 300 / 3),
          blur_constant=3,
          target_pixels=300),
      APTOSConfig(
          name="blur-5-btgraham-300",
          description=_BLUR_BTGRAHAM_DESCRIPTION_PATTERN.format(300, 300 / 5),
          blur_constant=5,
          target_pixels=300),
      APTOSConfig(
          name="blur-10-btgraham-300",
          description=_BLUR_BTGRAHAM_DESCRIPTION_PATTERN.format(300, 300 // 10),
          blur_constant=10,
          target_pixels=300),
      APTOSConfig(
          name="blur-20-btgraham-300",
          description=_BLUR_BTGRAHAM_DESCRIPTION_PATTERN.format(300, 300 // 20),
          blur_constant=20,
          target_pixels=300)
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
            "label": tfds.features.ClassLabel(num_classes=5),
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
    return [
        tfds.core.SplitGenerator(
            name="validation",
            gen_kwargs={
                "images_dir_path": os.path.join(path, "train_images"),
                "is_validation": True,
                "csv_path": os.path.join(path, "train.csv"),
                "csv_usage": None,
            }),
        tfds.core.SplitGenerator(
            name="test",
            gen_kwargs={
                "images_dir_path": os.path.join(path, "train_images"),
                "is_validation": False,
                "csv_path": os.path.join(path, "train.csv"),
                "csv_usage": None
            })
    ] + [
        tfds.core.SplitGenerator(
            name=f"validation_{val_percent}",
            gen_kwargs={
                "images_dir_path": os.path.join(path, "train_images"),
                "is_validation": True,
                "csv_path": os.path.join(path, "train.csv"),
                "csv_usage": None,
                "val_percent": val_percent,
            })
        for val_percent in [25, 50, 75, 100]
    ]

  def _generate_examples(self,
                         images_dir_path,
                         is_validation,
                         val_percent=100,
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

      id_code_and_diagnosis = list(zip(df["id_code"], df["diagnosis"], df["eye_label"])) # Karm
      if is_validation:
        data = id_code_and_diagnosis[_NUM_TRAINING_EXAMPLES:]
        data = data[:int(val_percent * len(data) / 100)]
      else:
        data = id_code_and_diagnosis[:_NUM_TRAINING_EXAMPLES]

      # data = [(id_code, int(diagnosis)) for id_code, diagnosis in data]
      data = [(id_code, int(diagnosis), eye_label) for id_code, diagnosis, eye_label in data] # Karm
      
    else:
      
      data = [(fname[:-4], -1, np.nan) # Karm
              for fname in tf.io.gfile.listdir(images_dir_path)
              if fname.endswith(".png")]

    print(f"Using BuilderConfig {self.builder_config.name}.")

    for name, label, eye_label in data:
      image_filepath = "%s/%s.png" % (images_dir_path, name)
      record = {
          "name": name,
          "image": self._process_image(filepath = image_filepath, eye_label = eye_label),
          "label": label,
      }
      yield name, record

  def _process_image(self, filepath, eye_label=np.nan):
    # breakpoint()
    with tf.io.gfile.GFile(filepath, mode="rb") as image_fobj:
      # breakpoint()
      if self.builder_config.name.endswith("left"): # Karm: to get btgraham-300 profile
                  
        return _btgraham_processing_left(image_fobj=image_fobj, filepath=filepath, target_pixels=self.builder_config.target_pixels,
            crop_to_radius=True,
            eye_label=eye_label) # Karm
        
      elif self.builder_config.name.startswith("btgraham"):
        return tfds.image_classification.diabetic_retinopathy_detection._btgraham_processing(  # pylint: disable=protected-access
            image_fobj=image_fobj,
            filepath=filepath,
            target_pixels=self.builder_config.target_pixels,
            crop_to_radius=True)
      elif self.builder_config.name.startswith("blur"):
        return _btgraham_processing(
            image_fobj=image_fobj,
            filepath=filepath,
            target_pixels=self.builder_config.target_pixels,
            blur_constant=self.builder_config.blur_constant,
            crop_to_radius=True)
      else:
        return tfds.image_classification.diabetic_retinopathy_detection._resize_image_if_necessary(  # pylint: disable=protected-access
            image_fobj=image_fobj,
            target_pixels=self.builder_config.target_pixels)


class APTOSDataset(base.BaseDataset):
  """Kaggle APTOS 2019 Blindness Detection dataset builder class."""

  def __init__(self,
               split: str,
               builder_config: str = "aptos/btgraham-300",
               shuffle_buffer_size: Optional[int] = None,
               num_parallel_parser_calls: int = 64,
               data_dir: Optional[str] = None,
               download_data: bool = False,
               drop_remainder: bool = True,
               decision_threshold: Optional[str] = "moderate",
               val_percent: Optional[int] = None,
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
    print(f"Using APTOS builder config {builder_config}.")
    dataset_builder = tfds.builder(builder_config, data_dir=data_dir)
    super(APTOSDataset, self).__init__(
        name=f"{dataset_builder.name}/{dataset_builder.builder_config.name}",
        dataset_builder=dataset_builder,
        split=split if val_percent is None else f"{split}_{val_percent}",
        is_training=val_percent is not None,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        download_data=download_data,
        drop_remainder=drop_remainder,
        cache=cache)
    self.decision_threshold = decision_threshold
    print(f"Building APTOS OOD dataset with decision threshold: "
          f"{decision_threshold}.")
    if not drop_remainder:
      print("Not dropping the remainder (i.e., not truncating last batch).")

  def _create_process_example_fn(self) -> base.PreProcessFn:

    def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      """Preprocess images to range [0, 1], binarize task based on provided decision threshold, produce example `Dict`."""
      image = example["image"]
      image = tf.image.convert_image_dtype(image, tf.float32)
      image = tf.image.resize(image, size=(512, 512), method="bilinear")

      if self.decision_threshold == "mild":
        highest_negative_class = 0
      elif self.decision_threshold == "moderate":
        highest_negative_class = 1
      else:
        raise NotImplementedError

      # Binarize task.
      label = tf.cast(example["label"] > highest_negative_class, tf.int32)

      parsed_example = {
          "features": image,
          "labels": label,
          "name": example["name"],
      }
      return parsed_example

    return _example_parser
