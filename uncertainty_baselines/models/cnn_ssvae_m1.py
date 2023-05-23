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


class Reparametrize(tf.keras.layers.Layer):
  def call(self, *inputs):
    mean, logvar = inputs
    eps = tf.random.normal(shape=tf.shape(mean))
    return mean + tf.exp(logvar * 0.5) * eps


def encoder(x):
  num_blocks = max(x.shape[1:3]).bit_length() - 3  # pre mean,var shape be (..., 2, 2, D)
  filters_list = [64 * 2 ** (i // 2) for i in range(num_blocks)]

  x = tf.keras.layers.Conv2D(
      64,
      kernel_size=7,
      strides=2,
      padding='same',
      kernel_initializer='he_normal',
      name=f'enc_first')(x)
  x = tf.keras.layers.Activation('relu')(x)

  for idx, filters in enumerate(filters_list):
    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer='he_normal',
        name=f'enc_{idx}a')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size=3,
        strides=2,
        padding='same',
        kernel_initializer='he_normal',
        name=f'enc_{idx}b')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer='he_normal',
        name=f'enc_{idx}c')(x)
    x = tf.keras.layers.Activation('relu')(x)
  
  x = tf.keras.layers.Flatten()(x)

  mean = tf.keras.layers.Dense(
      filters_list[-1] * 4,
      kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
      name='enc_latent_mean')(x)

  logvar = tf.keras.layers.Dense(
      filters_list[-1] * 4,
      kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
      name='enc_latent_logvar')(x)

  return mean, logvar


def decoder(x, num_blocks = 8):
  x = tf.keras.layers.Reshape((2, 2, -1))(x)
  filters_list = [(64, 64)] + [(64 * 2 ** ((i + 1) // 2), 64 * 2 ** (i // 2)) for i in range(num_blocks - 2)]
  filters_list.reverse()
  for idx, filters in enumerate(filters_list):
    x = tf.keras.layers.Conv2DTranspose(
        filters[0],
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer='he_normal',
        name=f'dec_{idx}a')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2DTranspose(
        filters[0],
        kernel_size=3,
        strides=2,
        padding='same',
        kernel_initializer='he_normal',
        name=f'dec_{idx}b')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2DTranspose(
        filters[1],
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer='he_normal',
        name=f'dec_{idx}c')(x)
    x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Conv2DTranspose(
      3,
      kernel_size=7,
      strides=2,
      padding='same',
      kernel_initializer='he_normal',
      name=f'dec_final')(x)
  # x = tf.keras.layers.Activation('sigmoid')(x)

  return x


def classifier(x, num_classes):
  x = tf.stop_gradient(x)
  
  for idx, hidden_dim in enumerate([1024, 512, 256, 64]):
    x = tf.keras.layers.Dense(
      hidden_dim,
      kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
      name=f'clf_{idx}')(x)
    x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Dense(
      num_classes,
      kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
      name=f'clf_final')(x)

  return x


def ssvae_m1(input_shape, num_classes):
  x = tf.keras.layers.Input(shape=input_shape)

  mean, logvar = encoder(x)

  z = Reparametrize()(mean, logvar)

  logits = classifier(z, num_classes)
  x_rec = decoder(z)

  return tf.keras.Model(
      inputs=x,
      outputs=dict(
          x_rec=x_rec,
          logits=logits,
          mean=mean,
          logvar=logvar),
      name='ssvae_m1')

