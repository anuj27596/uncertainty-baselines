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

"""SSVAE M1 model."""

import string
import tensorflow as tf

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def bottleneck_block(inputs, filters, stage, block, strides):
  """Residual block with 1x1 -> 3x3 -> 1x1 convs in main path.

  Note that strides appear in the second conv (3x3) rather than the first (1x1).
  This is also known as "ResNet v1.5" as it differs from He et al. (2015)
  (http://torch.ch/blog/2016/02/04/resnets.html).

  Args:
    inputs: tf.Tensor.
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    strides: Strides for the second conv layer in the block.

  Returns:
    tf.Tensor.
  """
  filters1, filters2, filters3 = filters
  conv_name_base = 'enc_clf_res' + str(stage) + block + '_branch'
  bn_name_base = 'enc_clf_bn' + str(stage) + block + '_branch'

  x = tf.keras.layers.Conv2D(
      filters1,
      kernel_size=1,
      use_bias=False,
      kernel_initializer='he_normal',
      name=conv_name_base + '2a')(inputs)
  x = tf.keras.layers.BatchNormalization(
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '2a')(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Conv2D(
      filters2,
      kernel_size=3,
      strides=strides,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal',
      name=conv_name_base + '2b')(x)
  x = tf.keras.layers.BatchNormalization(
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '2b')(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Conv2D(
      filters3,
      kernel_size=1,
      use_bias=False,
      kernel_initializer='he_normal',
      name=conv_name_base + '2c')(x)
  x = tf.keras.layers.BatchNormalization(
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '2c')(x)

  shortcut = inputs
  if not x.shape.is_compatible_with(shortcut.shape):
    shortcut = tf.keras.layers.Conv2D(
        filters3,
        kernel_size=1,
        use_bias=False,
        strides=strides,
        kernel_initializer='he_normal',
        name=conv_name_base + '1')(shortcut)
    shortcut = tf.keras.layers.BatchNormalization(
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=bn_name_base + '1')(shortcut)

  x = tf.keras.layers.add([x, shortcut])
  x = tf.keras.layers.Activation('relu')(x)
  return x


def bottleneck_transpose(inputs, filters, stage, block, strides):
  """Residual block with 1x1 -> 3x3 -> 1x1 convs in main path.

  Note that strides appear in the second conv (3x3) rather than the first (1x1).
  This is also known as "ResNet v1.5" as it differs from He et al. (2015)
  (http://torch.ch/blog/2016/02/04/resnets.html).

  Args:
    inputs: tf.Tensor.
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    strides: Strides for the second conv layer in the block.

  Returns:
    tf.Tensor.
  """
  filters1, filters2, filters3 = filters
  conv_name_base = 'dec_res' + str(stage) + block + '_branch'
  bn_name_base = 'dec_bn' + str(stage) + block + '_branch'

  x = tf.keras.layers.Conv2DTranspose(
      filters1,
      kernel_size=1,
      use_bias=False,
      kernel_initializer='he_normal',
      name=conv_name_base + '2a')(inputs)
  x = tf.keras.layers.BatchNormalization(
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '2a')(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Conv2DTranspose(
      filters2,
      kernel_size=3,
      strides=strides,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal',
      name=conv_name_base + '2b')(x)
  x = tf.keras.layers.BatchNormalization(
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '2b')(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Conv2DTranspose(
      filters3,
      kernel_size=1,
      use_bias=False,
      kernel_initializer='he_normal',
      name=conv_name_base + '2c')(x)
  x = tf.keras.layers.BatchNormalization(
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '2c')(x)

  shortcut = inputs
  if not x.shape.is_compatible_with(shortcut.shape):
    shortcut = tf.keras.layers.Conv2DTranspose(
        filters3,
        kernel_size=1,
        use_bias=False,
        strides=strides,
        kernel_initializer='he_normal',
        name=conv_name_base + '1')(shortcut)
    shortcut = tf.keras.layers.BatchNormalization(
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=bn_name_base + '1')(shortcut)

  x = tf.keras.layers.add([x, shortcut])
  x = tf.keras.layers.Activation('relu')(x)
  return x


def group(inputs, filters, num_blocks, stage, strides):
  blocks = string.ascii_lowercase
  x = bottleneck_block(inputs, filters, stage, block=blocks[0], strides=strides)
  for i in range(num_blocks - 1):
    x = bottleneck_block(x, filters, stage, block=blocks[i + 1], strides=1)
  return x


def group_transpose(x, filters, last_filter, num_blocks, stage, strides):
  blocks = string.ascii_lowercase
  for i in range(num_blocks - 1, 0, -1):
    x = bottleneck_transpose(x, filters, stage, block=blocks[i], strides=1)
  last_filters = filters[:-1] + [last_filter]
  x = bottleneck_transpose(x, last_filters, stage, block=blocks[0], strides=strides)
  return x


def reparametrize(mean, logvar):
  eps = tf.random.normal(shape=tf.shape(mean))
  return mean + tf.exp(logvar * 0.5) * eps # z: layers[196].output


def label_softify(y, logits): # TODO(anuj): change please
  y = tf.keras.layers.Reshape((1,))(y)
  p = tf.nn.sigmoid(logits)
  return tf.divide(p + y - 1, p * (2 * y - 1) - y + 1)


def conditionalify(x, y, name):
  y = tf.keras.layers.Reshape((1,))(y) # after this, y: layers[195].input
  y = tf.keras.layers.Dense(
      x.shape[-1],
      kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
      name=name)(y)
  if x.shape.ndims > 2:
    y = tf.keras.layers.Reshape([1] * (x.shape.ndims - 2) + [-1])(y)
  x = x + y
  return x


def encoder_classifier(x, num_classes):
  x = tf.keras.layers.ZeroPadding2D(padding=3, name='enc_clf_conv1_pad')(x)
  x = tf.keras.layers.Conv2D(
      64,
      kernel_size=7,
      strides=2,
      padding='valid',
      use_bias=False,
      kernel_initializer='he_normal',
      name='enc_clf_conv1')(x)
  x = tf.keras.layers.BatchNormalization(
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name='enc_clf_bn_conv1')(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)
  x = group(x, [64, 64, 256], stage=2, num_blocks=3, strides=1)
  x = group(x, [128, 128, 512], stage=3, num_blocks=4, strides=2)
  x = group(x, [256, 256, 1024], stage=4, num_blocks=6, strides=2)
  x = group(x, [512, 512, 2048], stage=5, num_blocks=3, strides=2)

  y = tf.keras.layers.GlobalAveragePooling2D(name='enc_clf_avg_pool')(x)
  y = tf.keras.layers.Dense(
      num_classes,
      activation=None,
      kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
      name='clf_fc')(y)

  mean = tf.keras.layers.Dense(
      2048,
      activation=None,
      kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
      name='enc_mean')(x)
  logvar = tf.keras.layers.Dense(
      2048,
      activation=None,
      kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
      name='enc_logvar')(x)

  return y, mean, logvar


def decoder(x, y):
  x = conditionalify(x, y, name='dec_cond')
  x = tf.keras.layers.Dense(
      2048,
      activation=None,
      kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
      name='dec_first')(x)
  x = group_transpose(x, [512, 512, 2048], 1024, stage=6, num_blocks=3, strides=2)
  x = group_transpose(x, [256, 256, 1024], 512, stage=7, num_blocks=6, strides=2)
  x = group_transpose(x, [128, 128, 512], 256, stage=8, num_blocks=4, strides=2)
  x = group_transpose(x, [64, 64, 256], 64, stage=9, num_blocks=3, strides=1)
  x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
  x = tf.keras.layers.Conv2DTranspose(
      3,
      kernel_size=7,
      strides=2,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal',
      name='dec_conv_last')(x)
  return x


def resnet50_ssvae_m2(input_shape, num_classes):
  x = tf.keras.layers.Input(shape=input_shape)
  y = tf.keras.layers.Input(shape=())

  y_hat, mean, logvar = encoder_classifier(x, num_classes)

  z = reparametrize(mean, logvar)
  y_soft = label_softify(y, y_hat)

  x_hat = decoder(z, y_soft)

  return tf.keras.Model(
      inputs=dict(
          image=x,
          label=y),
      outputs=dict(
          reconstruction=x_hat,
          logits=y_hat,
          mean=mean,
          logvar=logvar),
      name='ssvae_m2')


