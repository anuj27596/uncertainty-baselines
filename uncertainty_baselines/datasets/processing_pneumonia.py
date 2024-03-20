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

"""https://www.kaggle.com/c/diabetic-retinopathy-detection/data."""

import csv
import io
import os

from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

_CITATION = """\
@ONLINE {kaggle-diabetic-retinopathy,
    author = "Kaggle and EyePacs",
    title  = "Kaggle Diabetic Retinopathy Detection",
    month  = "jul",
    year   = "2015",
    url    = "https://www.kaggle.com/c/diabetic-retinopathy-detection/data"
}
"""
_URL_TEST_LABELS = (
    "https://storage.googleapis.com/kaggle-forum-message-attachments/"
    "90528/2877/retinopathy_solution.csv")
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


def _resize_image_if_necessary(image_fobj, target_pixels=None):
  """Resize an image to have (roughly) the given number of target pixels.

  Args:
    image_fobj: File object containing the original image.
    target_pixels: If given, number of pixels that the image must have.
  Returns:
    A file object.
  """
  if target_pixels is None:
    return image_fobj

  cv2 = tfds.core.lazy_imports.cv2
  # Decode image using OpenCV2.
  image = cv2.imdecode(
      np.frombuffer(image_fobj.read(), dtype=np.uint8), flags=3)
  # Get image height and width.
  height, width, _ = image.shape
  actual_pixels = height * width
  if actual_pixels > target_pixels:
    factor = np.sqrt(target_pixels / actual_pixels)
    image = cv2.resize(image, dsize=None, fx=factor, fy=factor)
  # Encode the image with quality=72 and store it in a BytesIO object.
  _, buff = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
  return io.BytesIO(buff.tobytes())

def _resize_image_if_necessary_pcam(image_arr, target_pixels=None):
  """Resize an image to have (roughly) the given number of target pixels.

  Args:
    image_fobj: File object containing the original image.
    target_pixels: If given, number of pixels that the image must have.
  Returns:
    A file object.
  """
  if target_pixels is None:
    return image_arr

  cv2 = tfds.core.lazy_imports.cv2
  # Decode image using OpenCV2.
  image = image_arr
  # Get image height and width.
  height, width, _ = image.shape
  actual_pixels = height * width
  if actual_pixels > target_pixels:
    factor = np.sqrt(target_pixels / actual_pixels)
    image = cv2.resize(image, dsize=None, fx=factor, fy=factor)
  # Encode the image with quality=72 and store it in a BytesIO object.
  _, buff = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
  return io.BytesIO(buff.tobytes())


def _pneumonia_processing(image_fobj,
                          filepath,
                          target_pixels):
  """Process an image as the winner of the 2015 Kaggle competition.

  Args:
    image_fobj: File object containing the original image.
    filepath: Filepath of the image, for logging purposes only.
    target_pixels: The number of target pixels for the radius of the image.
    blur_constant: Constant used to vary the Kernel standard deviation in
      smoothing the image with Gaussian blur.
    crop_to_radius: If True, crop the borders of the image to remove gray areas.

  Returns:
    A file object.
  """
  cv2 = tfds.core.lazy_imports.cv2
  # Decode image using OpenCV2.
  image = cv2.imdecode(
      np.frombuffer(image_fobj.read(), dtype=np.uint8), flags=3)

  height, width, _ = image.shape
  actual_pixels = height * width
  if actual_pixels > target_pixels:
    factor = target_pixels / np.sqrt(actual_pixels)
    image = cv2.resize(image, dsize=None, fx=factor, fy=factor)

  # Encode the image with quality=72 and store it in a BytesIO object.
  _, buff = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
  return io.BytesIO(buff.tobytes())


def _subtract_local_average(image, target_radius_size, blur_constant=30):
  cv2 = tfds.core.lazy_imports.cv2
  image_blurred = cv2.GaussianBlur(image, (0, 0),
                                   target_radius_size / blur_constant)
  image = cv2.addWeighted(image, 4, image_blurred, -4, 128)
  return image

