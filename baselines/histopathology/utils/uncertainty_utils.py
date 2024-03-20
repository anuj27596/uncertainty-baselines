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

"""Uncertainty Utilities.

A set of model wrappers and evaluation utilities to determine the robustness
and quality of uncertainty estimates of the given model.

Note that the model produces a single scalar "prediction" for each example
corresponding to p(class = 1), for computational efficiency (e.g., we can
compute predictive variance for binary classification using only this).

TODO(nband): better use TensorArrays in TF methods for faster TPU execution.
"""
# pylint: disable=g-bare-generic
# pylint: disable=g-doc-args
# pylint: disable=g-doc-return-or-yield
# pylint: disable=g-importing-member
# pylint: disable=g-no-space-after-docstring-summary
# pylint: disable=g-short-docstring-punctuation
# pylint: disable=logging-format-interpolation
# pylint: disable=logging-fstring-interpolation
# pylint: disable=missing-function-docstring
import functools

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.backend as K

tfd = tfp.distributions
"""Binary classification utilities.
"""
# Retrieves all parts of (1 - p, p, Log[1 - p], Log[p])
# given either probs or logits
_probs_and_log_probs_tf = tfd.bernoulli._probs_and_log_probs  # pylint: disable=protected-access


def _probs_and_log_probs_np(probs):
  """NumPy equivalent based on tfd.bernoulli._probs_and_log_probs."""
  to_return = ()
  p = np.array(probs)
  to_return += (1 - p, p)
  to_return += (np.log1p(-p), np.log(p))
  return to_return


def binary_entropy_tf(probs):
  """Compute binary entropy, from tfp.distributions.Bernoulli.entropy."""
  probs0, probs1, log_probs0, log_probs1 = _probs_and_log_probs_tf(
      probs=probs, logits=None, return_log_probs=True)
  return -1. * (
      tf.math.multiply_no_nan(log_probs0, probs0) +
      tf.math.multiply_no_nan(log_probs1, probs1))


def binary_entropy_np(probs):
  """Compute binary entropy in NumPy."""
  probs0, probs1, log_probs0, log_probs1 = _probs_and_log_probs_np(probs)
  return -1. * (
      np.where(probs0 == 0, 0, np.multiply(log_probs0, probs0)) +
      np.where(probs1 == 0, 0, np.multiply(log_probs1, probs1)))


def binary_entropy_jax(array):
  return jax.scipy.special.entr(array) + jax.scipy.special.entr(1 - array)


def sigmoid(x):  # anuj (def)
  return 1 / (1 + np.exp(-x))

def logit(x):  # anuj (def)
  return np.log(x) - np.log(1 - x)

def average_logit(samples):  # anuj (def)
  avg_s = np.mean(sigmoid(samples), axis = 0)
  pos = avg_s > 0.5
  avg_s[pos] = np.mean(sigmoid(-samples[:, pos]), axis = 0)
  avg_l = logit(avg_s)
  avg_l[pos] *= -1
  return avg_l

def average_logit_tf(samples):  # anuj (def)
  avg_s = K.mean(tf.nn.sigmoid(samples), axis = 0)
  pos = avg_s > 0.5
  avg_s[pos] = K.mean(tf.nn.sigmoid(-samples[:, pos]), axis = 0)
  avg_l = logit(avg_s)
  avg_l[pos] *= -1
  return avg_l

def _nl_elbo(inputs, outputs):  # anuj (def)
  reconstruction, mean, logvar = outputs['reconstruction'], outputs['mean'], outputs['logvar']
  rec_nll = K.mean(K.binary_crossentropy(inputs, reconstruction, from_logits=True),
                   axis = range(1, len(inputs.shape)))
  rec_nll -= K.mean(K.binary_crossentropy(inputs, inputs),
                    axis = range(1, len(inputs.shape))) # debias
  latent_kl = K.mean(tf.square(mean) + tf.exp(logvar) - logvar,
                     axis = range(1, len(mean.shape))) / 2
  return rec_nll + latent_kl

def nll_impsample(**args):  # anuj (def)
  inputs = args['inputs']
  reconstruction = args['reconstruction']
  latent_sample = args['latent_sample']
  mean = args['mean']
  logvar = args['logvar']

  rec_nll = K.mean(K.binary_crossentropy(inputs, reconstruction, from_logits=True),
                   axis = tuple(range(1, len(inputs.shape))))
  rec_nll -= K.mean(K.binary_crossentropy(inputs, inputs),
                    axis = tuple(range(1, len(inputs.shape)))) # debias
  prior_nll = K.mean(tf.square(latent_sample), axis = tuple(range(1, len(latent_sample.shape)))) / 2
  var_posterior_nll = K.mean(tf.square(latent_sample - mean) * tf.exp(-logvar), axis = tuple(range(1, len(mean.shape)))) / 2 + K.mean(logvar, axis = tuple(range(1, len(logvar.shape)))) / 2
  return rec_nll + prior_nll - var_posterior_nll


# Model Wrappers: Using a set of MC samples, produce the prediction and
# uncertainty estimates: predictive entropy, predictive variance, epistemic
# uncertainty (MI), and aleatoric uncertainty (expected entropy).
def predict_and_decompose_uncertainty_tf(mc_samples: tf.Tensor):
  """Using a set of MC samples, produce the prediction and uncertainty

    estimates: predictive entropy, predictive variance, epistemic uncertainty
    (MI), and aleatoric uncertainty (expected entropy).

  Args:
    mc_samples: `tf.Tensor`, Monte Carlo samples from a sigmoid predictive
      distribution, shape [S, B] where S is the number of samples and B is the
      batch size.

  Returns:
    Dict: {
      prediction: `tf.Tensor`, prediction, with shape [B].
      predictive_entropy: `tf.Tensor`, predictive entropy, with shape [B].
      predictive_variance: `tf.Tensor`, predictive variance, with shape [B].
      epistemic_uncertainty: `tf.Tensor`, mutual info, with shape [B].
      aleatoric_uncertainty: `tf.Tensor`, expected entropy, with shape [B].
    }
  """
  
  # logit = average_logit_tf(mc_samples_logits)  # anuj

  # Prediction: mean of sigmoid probabilities over MC samples
  prediction = tf.reduce_mean(mc_samples, axis=0)

  # Compute predictive entropy H[p(y|x)], with predictive mean p(y|x)
  predictive_entropy = binary_entropy_tf(probs=prediction)

  # Compute per-sample entropies (for use in expected entropy)
  per_sample_entropies = binary_entropy_tf(probs=mc_samples)
  expected_entropy = tf.reduce_mean(per_sample_entropies, axis=0)

  # Take variance over MC samples
  # In binary classification, we can simply do this over the positive class
  # because 0.5 * (Var(X) + Var(1 - X)) = Var(X)
  predictive_variance = tf.math.reduce_variance(mc_samples, axis=0)

  return {
      'prediction': prediction,
      # 'logit': logit,  # anuj
      'predictive_entropy': predictive_entropy,
      'predictive_variance': predictive_variance,
      'epistemic_uncertainty': predictive_entropy - expected_entropy,  # MI
      'aleatoric_uncertainty': expected_entropy
  }


def predict_and_decompose_uncertainty_np(mc_samples: np.ndarray, mc_samples_logits: np.ndarray):  # anuj
  """Using a set of MC samples, produce the prediction and uncertainty

    estimates: predictive entropy, predictive variance, epistemic uncertainty
    (MI), and aleatoric uncertainty (expected entropy).

  Args:
    mc_samples: `np.ndarray`, Monte Carlo samples from a sigmoid predictive
      distribution, shape [S, B] where S is the number of samples and B is the
      batch size.

  Returns:
    Dict: {
      prediction: `numpy.ndarray`, prediction, with shape [B].
      predictive_entropy: `numpy.ndarray`, predictive entropy, with shape [B].
      predictive_variance: `numpy.ndarray`, predictive variance, with shape [B].
      epistemic_uncertainty: `numpy.ndarray`, mutual info, with shape [B].
      aleatoric_uncertainty: `numpy.ndarray`, expected entropy, with shape [B].
    }
  """

  logit = average_logit(mc_samples_logits)  # anuj

  # Prediction: mean of sigmoid probabilities over MC samples
  prediction = mc_samples.mean(axis=0)
  # if np.isnan(prediction).any(): np.save('/troy/anuj/gub-mod/mc_samples-B150.npy', mc_samples)  # anuj
  # Compute predictive entropy H[p(y|x)], with predictive mean p(y|x)
  predictive_entropy = binary_entropy_np(probs=prediction)

  # Compute per-sample entropies (for use in expected entropy)
  per_sample_entropies = binary_entropy_np(probs=mc_samples)
  expected_entropy = per_sample_entropies.mean(axis=0)

  # Take variance over MC samples
  # In binary classification, we can simply do this over the positive class
  # because 0.5 * (Var(X) + Var(1 - X)) = Var(X)
  predictive_variance = mc_samples.var(axis=0)

  return {
      'prediction': prediction,
      'logit': logit,  # anuj
      'predictive_entropy': predictive_entropy,
      'predictive_variance': predictive_variance,
      'epistemic_uncertainty': predictive_entropy - expected_entropy,  # MI
      'aleatoric_uncertainty': expected_entropy
  }


def variational_predict_and_decompose_uncertainty_np(x, model, training_setting,
                                                     num_samples):
  """Monte Carlo uncertainty estimator for a variational model.

  Should work for all variational methods which sample from model posterior
  in each forward pass -- e.g., MC Dropout, MFVI, Radial BNNs.

  Args:
    x: `numpy.ndarray`, datapoints from input space, with shape [B, H, W, 3],
      where B the batch size and H, W the input images height and width
      accordingly.
    model: a probabilistic model (e.g., `tensorflow.keras.model`) which accepts
      input with shape [B, H, W, 3] and outputs sigmoid probability [0.0, 1.0],
      and also accepts boolean argument `training` for disabling e.g.,
      BatchNorm, Dropout at test time.
    training_setting: bool, if True, run model prediction in training mode. See
      note in docstring at top of file.
    num_samples: `int`, number of Monte Carlo samples (i.e. forward passes from
      dropout) used for the calculation of predictive mean and uncertainty.

  Returns:
    Dict: {
      prediction: `numpy.ndarray`, prediction, with shape [B].
      predictive_entropy: `numpy.ndarray`, predictive entropy, with shape [B].
      predictive_variance: `numpy.ndarray`, predictive variance, with shape [B].
      epistemic_uncertainty: `numpy.ndarray`, mutual info, with shape [B].
      aleatoric_uncertainty: `numpy.ndarray`, expected entropy, with shape [B].
    }
  """

  # Get shapes of data
  b, _, _, _ = x.shape

  # Monte Carlo samples from different dropout mask at test time
  # See note in docstring regarding `training` mode

  # mc_samples = np.asarray([
  #     model(x, training=training_setting) for _ in range(num_samples)
  # ]).reshape(-1, b)  # anuj

  prob_list, logit_list = [], []  # anuj

  for _ in range(num_samples):  # anuj
    prob, logit = model(x, training=training_setting)  # anuj
    prob_list.append(prob)  # anuj
    logit_list.append(logit)  # anuj

  mc_samples = np.asarray(prob_list).reshape(-1, b)  # anuj
  mc_samples_logits = np.asarray(logit_list).reshape(-1, b)  # anuj

  return predict_and_decompose_uncertainty_np(mc_samples=mc_samples, mc_samples_logits=mc_samples_logits)  # anuj


def variational_predict_and_decompose_uncertainty_tf(x, model, training_setting,
                                                     num_samples):
  """Monte Carlo uncertainty estimator for a variational model.

  Should work for all variational methods which sample from model posterior
  in each forward pass -- e.g., MC Dropout, MFVI, Radial BNNs.

  Args:
    x: `tf.Tensor`, datapoints from input space, with shape [B, H, W, 3], where
      B the batch size and H, W the input images height and width accordingly.
    model: a probabilistic model (e.g., `tensorflow.keras.model`) which accepts
      input with shape [B, H, W, 3] and outputs sigmoid probability [0.0, 1.0],
      and also accepts boolean argument `training` for disabling e.g.,
      BatchNorm, Dropout at test time.
    training_setting: bool, if True, run model prediction in training mode. See
      note in docstring at top of file.
    num_samples: `int`, number of Monte Carlo samples (i.e. forward passes from
      dropout) used for the calculation of predictive mean and uncertainty.

  Returns:
    Dict: {
      prediction: `tf.Tensor`, prediction, with shape [B].
      predictive_entropy: `tf.Tensor`, predictive entropy, with shape [B].
      predictive_variance: `tf.Tensor`, predictive variance, with shape [B].
      epistemic_uncertainty: `tf.Tensor`, mutual info, with shape [B].
      aleatoric_uncertainty: `tf.Tensor`, expected entropy, with shape [B].
    }
  """

  # Get shapes of data
  b = tf.shape(x)[0]

  # prob_list, logit_list = [], []  # anuj

  # Monte Carlo samples from different dropout mask at test time
  # See note in docstring regarding `training` mode
  if num_samples > 1:
    mc_samples = tf.convert_to_tensor(
        [model(x, training=training_setting) for _ in range(num_samples)])  # EDIT(anuj)
    # prob, logit = model(x, training=training_setting)  # anuj
    # prob_list.append(prob)  # anuj
    # logit_list.append(logit)  # anuj

  else:
    mc_samples = model(x, training=training_setting)

  mc_samples = tf.reshape(mc_samples, [-1, b])  # EDIT(anuj)
  
  # mc_samples = tf.reshape(tf.convert_to_tensor(prob_list), [-1, b])  # EDIT(anuj)
  # mc_samples_logits = tf.reshape(tf.convert_to_tensor(logit_list), [-1, b])  # EDIT(anuj)

  return predict_and_decompose_uncertainty_tf(mc_samples=mc_samples)  # EDIT(anuj)


def variational_ensemble_predict_and_decompose_uncertainty_np(
    x, models, training_setting, num_samples):
  """Monte Carlo uncertainty estimator for ensembles of variational models.

  Should work for all variational methods which sample from model posterior
  in each forward pass -- MC Dropout, MFVI, Radial BNNs.
  This estimator is for ensembles of the above methods.

  Args:
    x: `numpy.ndarray`, datapoints from input space, with shape [B, H, W, 3],
      where B the batch size and H, W the input images height and width
      accordingly.
    models: `iterable` of probabilistic models (e.g., `tensorflow.keras.model`),
      each of which accepts input with shape [B, H, W, 3] and outputs sigmoid
      probability [0.0, 1.0], and also accepts boolean argument `training` for
      disabling e.g., BatchNorm, Dropout at test time.
    training_setting: bool, if True, run model prediction in training mode. See
      note in docstring at top of file.
    num_samples: `int`, number of Monte Carlo samples (i.e. forward passes from
      dropout) used for the calculation of predictive mean and uncertainty.

  Returns:
    Dict: {
      prediction: `numpy.ndarray`, prediction, with shape [B].
      predictive_entropy: `numpy.ndarray`, predictive entropy, with shape [B].
      predictive_variance: `numpy.ndarray`, predictive variance, with shape [B].
      epistemic_uncertainty: `numpy.ndarray`, mutual info, with shape [B].
      aleatoric_uncertainty: `numpy.ndarray`, expected entropy, with shape [B].
    }
  """

  # Get shapes of data
  b, _, _, _ = x.shape

  # Monte Carlo samples from different dropout mask at
  # test time from different models
  # See note in docstring regarding `training` mode
  # pylint: disable=g-complex-comprehension
  mc_samples = np.asarray([
      model(x, training=training_setting)
      for _ in range(num_samples)
      for model in models
  ]).reshape(-1, b)
  # pylint: enable=g-complex-comprehension

  return predict_and_decompose_uncertainty_np(mc_samples=mc_samples)


def variational_ensemble_predict_and_decompose_uncertainty_tf(
    x, models, training_setting, num_samples):
  """Monte Carlo uncertainty estimator for ensembles of variational models.

  Should work for all variational methods which sample from model posterior
  in each forward pass -- MC Dropout, MFVI, Radial BNNs.
  This estimator is for ensembles of the above methods.

  Args:
    x: `tf.Tensor`, datapoints from input space, with shape [B, H, W, 3], where
      B the batch size and H, W the input images height and width accordingly.
    models: `iterable` of probabilistic models (e.g., `tensorflow.keras.model`),
      each of which accepts input with shape [B, H, W, 3] and outputs sigmoid
      probability [0.0, 1.0], and also accepts boolean argument `training` for
      disabling e.g., BatchNorm, Dropout at test time.
    training_setting: bool, if True, run model prediction in training mode. See
      note in docstring at top of file.
    num_samples: `int`, number of Monte Carlo samples (i.e. forward passes from
      dropout) used for the calculation of predictive mean and uncertainty.

  Returns:
    Dict: {
      prediction: `tf.Tensor`, prediction, with shape [B].
      predictive_entropy: `tf.Tensor`, predictive entropy, with shape [B].
      predictive_variance: `tf.Tensor`, predictive variance, with shape [B].
      epistemic_uncertainty: `tf.Tensor`, mutual info, with shape [B].
      aleatoric_uncertainty: `tf.Tensor`, expected entropy, with shape [B].
    }
  """

  # Get shapes of data
  b = x.shape[0]

  # Monte Carlo samples from different dropout mask at
  # test time from different models
  # See note in docstring regarding `training` mode
  # pylint: disable=g-complex-comprehension
  mc_samples = tf.convert_to_tensor([
      model(x, training=training_setting)
      for _ in range(num_samples)
      for model in models
  ])
  # pylint: enable=g-complex-comprehension

  mc_samples = tf.reshape(mc_samples, [-1, b])
  return predict_and_decompose_uncertainty_tf(mc_samples=mc_samples)


def deterministic_predict_and_decompose_uncertainty_np(x, model,
                                                       training_setting):
  """Wrapper for simple sigmoid uncertainty estimator -- returns None for

    predictive variance, aleatoric, and epistemic uncertainty, as we cannot
    obtain these.

  Args:
    x: `numpy.ndarray`, datapoints from input space, with shape [B, H, W, 3],
      where B the batch size and H, W the input images height and width
      accordingly.
    model: a probabilistic model (e.g., `tensorflow.keras.model`) which accepts
      input with shape [B, H, W, 3] and outputs sigmoid probability [0.0, 1.0],
      and also accepts boolean argument `training` for disabling e.g.,
      BatchNorm, Dropout at test time.
    training_setting: bool, if True, run model prediction in training mode. See
      note in docstring at top of file.

  Returns:
    Dict: {
      prediction: `numpy.ndarray`, prediction, with shape [B].
      predictive_entropy: `numpy.ndarray`, predictive entropy, with shape [B].
      predictive_variance: None
      epistemic_uncertainty: None
      aleatoric_uncertainty: None
    }
  """
  prediction, logit, predictive_entropy = deterministic_predict_np(  # anuj
      x=x, model=model, training_setting=training_setting)

  return {
      'prediction': prediction,
      'logit': logit,  # anuj
      'predictive_entropy': predictive_entropy,
      'predictive_variance': None,
      'epistemic_uncertainty': None,
      'aleatoric_uncertainty': None
  }


def deterministic_predict_and_decompose_uncertainty_tf(x, model,
                                                       training_setting):
  """Wrapper for simple sigmoid uncertainty estimator -- returns None for

    predictive variance, aleatoric, and epistemic uncertainty, as we cannot
    obtain these.

  Args:
    x: `tf.Tensor`, datapoints from input space, with shape [B, H, W, 3], where
      B the batch size and H, W the input images height and width accordingly.
    model: a probabilistic model (e.g., `tensorflow.keras.model`) which accepts
      input with shape [B, H, W, 3] and outputs sigmoid probability [0.0, 1.0],
      and also accepts boolean argument `training` for disabling e.g.,
      BatchNorm, Dropout at test time.
    training_setting: bool, if True, run model prediction in training mode. See
      note in docstring at top of file.

  Returns:
    Dict: {
      prediction: `numpy.ndarray`, prediction, with shape [B].
      predictive_entropy: `numpy.ndarray`, predictive entropy, with shape [B].
      predictive_variance: None
      epistemic_uncertainty: None
      aleatoric_uncertainty: None
    }
  """
  prediction, predictive_entropy = deterministic_predict_tf(  # EDIT(anuj)
      x=x, model=model, training_setting=training_setting)

  return {
      'prediction': prediction,
      # 'logit': logit,  # EDIT(anuj)
      'predictive_entropy': predictive_entropy,
      'predictive_variance': None,
      'epistemic_uncertainty': None,
      'aleatoric_uncertainty': None
  }


def deep_ensemble_predict_and_decompose_uncertainty_np(x, models,
                                                       training_setting):
  """Monte Carlo uncertainty estimator for ensembles of deterministic models.

  For example, this method should be used with Deep Ensembles (ensembles of
    deterministic neural networks, with different data/model seeds).

  Args:
    x: `numpy.ndarray`, datapoints from input space, with shape [B, H, W, 3],
      where B the batch size and H, W the input images height and width
      accordingly.
    models: `iterable` of probabilistic models (e.g., `tensorflow.keras.model`),
      each of which accepts input with shape [B, H, W, 3] and outputs sigmoid
      probability [0.0, 1.0], and also accepts boolean argument `training` for
      disabling e.g., BatchNorm, Dropout at test time.
    training_setting: bool, if True, run model prediction in training mode. See
      note in docstring at top of file.

  Returns:
    Dict: {
      prediction: `numpy.ndarray`, prediction, with shape [B].
      predictive_entropy: `numpy.ndarray`, predictive entropy, with shape [B].
      predictive_variance: `numpy.ndarray`, predictive variance, with shape [B].
      epistemic_uncertainty: `numpy.ndarray`, mutual info, with shape [B].
      aleatoric_uncertainty: `numpy.ndarray`, expected entropy, with shape [B].
    }
  """

  # Get shapes of data
  b, _, _, _ = x.shape

  # Monte Carlo samples from different deterministic models
  mc_samples = np.asarray(
      [model(x, training=training_setting) for model in models]).reshape(-1, b)

  return predict_and_decompose_uncertainty_np(mc_samples=mc_samples)


def deep_ensemble_predict_and_decompose_uncertainty_tf(x, models,
                                                       training_setting):
  """Monte Carlo uncertainty estimator for ensembles of deterministic models.

  For example, this method should be used with Deep Ensembles (ensembles of
    deterministic neural networks, with different data/model seeds).

  Args:
    x: `tf.Tensor`, datapoints from input space, with shape [B, H, W, 3], where
      B the batch size and H, W the input images height and width accordingly.
    models: `iterable` of probabilistic models (e.g., `tensorflow.keras.model`),
      each of which accepts input with shape [B, H, W, 3] and outputs sigmoid
      probability [0.0, 1.0], and also accepts boolean argument `training` for
      disabling e.g., BatchNorm, Dropout at test time.
    training_setting: bool, if True, run model prediction in training mode. See
      note in docstring at top of file.

  Returns:
    Dict: {
      prediction: `tf.Tensor`, prediction, with shape [B].
      predictive_entropy: `tf.Tensor`, predictive entropy, with shape [B].
      predictive_variance: `tf.Tensor`, predictive variance, with shape [B].
      epistemic_uncertainty: `tf.Tensor`, mutual info, with shape [B].
      aleatoric_uncertainty: `tf.Tensor`, expected entropy, with shape [B].
    }
  """

  # Get shapes of data
  b = x.shape[0]

  # Monte Carlo samples from different deterministic models
  mc_samples = tf.convert_to_tensor(
      [model(x, training=training_setting) for model in models])
  mc_samples = tf.reshape(mc_samples, [-1, b])
  return predict_and_decompose_uncertainty_tf(mc_samples=mc_samples)


# Model Wrappers: obtain predictive entropy or predictive stddev along with the
# predictive mean.
def deterministic_predict_np(x, model, training_setting):
  """Simple sigmoid uncertainty estimator.

  Args:
    x: `numpy.ndarray`, datapoints from input space, with shape [B, H, W, 3],
      where B the batch size and H, W the input images height and width
      accordingly.
    model: a probabilistic model (e.g., `tensorflow.keras.model`) which accepts
      input with shape [B, H, W, 3] and outputs sigmoid probability [0.0, 1.0],
      and also accepts boolean argument `training` for disabling e.g.,
      BatchNorm, Dropout at test time.
    training_setting: bool, if True, run model prediction in training mode. See
      note in docstring at top of file.

  Returns:
    prediction: `numpy.ndarray`, prediction, with shape [B].
    predictive_entropy: `numpy.ndarray`, predictive entropy, with shape [B].
  """
  # Single forward pass from the deterministic model
  prediction, logit = model(x, training=training_setting)  # anuj

  # Compute predictive entropy H[p(y|x)], with predictive mean p(y|x)
  predictive_entropy = binary_entropy_np(probs=prediction)

  return prediction, logit, predictive_entropy  # anuj


def deterministic_predict_tf(x, model, training_setting):
  """Simple sigmoid uncertainty estimator.

  Args:
    x: `tf.Tensor`, datapoints from input space, with shape [B, H, W, 3], where
      B the batch size and H, W the input images height and width accordingly.
    model: a probabilistic model (e.g., `tensorflow.keras.model`) which accepts
      input with shape [B, H, W, 3] and outputs sigmoid probability [0.0, 1.0],
      and also accepts boolean argument `training` for disabling e.g.,
      BatchNorm, Dropout at test time.
    training_setting: bool, if True, run model prediction in training mode. See
      note in docstring at top of file.

  Returns:
    prediction: `tf.Tensor`, prediction, with shape [B].
    predictive_entropy: `tf.Tensor`, predictive entropy, with shape [B].
  """
  # Single forward pass from the deterministic model
  prediction = model(x, training=training_setting)  # EDIT(anuj)

  # Compute predictive entropy H[p(y|x)], with predictive mean p(y|x)
  predictive_entropy = binary_entropy_tf(probs=prediction)

  return prediction, predictive_entropy  # EDIT(anuj)


def fsvi_predict_and_decompose_uncertainty_jax(
    x,
    model,
    rng_key,
    training_setting,
    num_samples,
    params,
    state,
):
  """Args:

    x: `numpy.ndarray`, datapoints from input space, with shape [B, H, W, 3],
    where B the batch size and H, W the input images height and width
    accordingly.
    model: a probabilistic model (e.g., `tensorflow.keras.model`) which accepts
    input with shape [B, H, W, 3] and outputs sigmoid probability [0.0, 1.0],
    and also accepts boolean argument `training` for disabling e.g., BatchNorm,
    Dropout at test time, as well as rng_key as random key for the forward
    passes.
    rng_key: `jax.numpy.ndarray`, jax random key for the forward passes.
    training_setting: bool, if True, run model prediction in training mode. See
    note in docstring at top of file.
    num_samples: int, the number of MC samples for each member of ensenble
    params: parameters of haiku model
    state: state of haiku model

  Returns:
    Dict: {
      prediction: `numpy.ndarray`, prediction, with shape [B].
      predictive_entropy: `numpy.ndarray`, predictive entropy, with shape [B].
      predictive_variance: `numpy.ndarray`, predictive variance, with shape [B].
      epistemic_uncertainty: `numpy.ndarray`, mutual info, with shape [B].
      aleatoric_uncertainty: `numpy.ndarray`, expected entropy, with shape [B].
    }
  """
  # mc_samples has shape [S, B]
  preds_y_samples, _, _ = model.predict_y_multisample_jitted(
      params=params,
      state=state,
      inputs=x,
      rng_key=rng_key,
      n_samples=num_samples,
      is_training=training_setting,
  )
  mc_samples = preds_y_samples[:, :, 1]

  return predict_and_decompose_uncertainty_jax(mc_samples=mc_samples)


def fsvi_ensemble_predict_and_decompose_uncertainty_jax(
    x,
    model,
    rng_key,
    training_setting,
    num_samples,
    params,
    state,
):
  """Args:

    x: `numpy.ndarray`, datapoints from input space, with shape [B, H, W, 3],
    where B the batch size and H, W the input images height and width
    accordingly.
    model: a list of FSVI Model objects
    rng_key: `jax.numpy.ndarray`, jax random key for the forward passes.
    training_setting: bool, if True, run model prediction in training mode. See
    note in docstring at top of file.
    num_samples: int, the number of MC samples for each member of ensenble
    params: parameters of haiku model
    state: state of haiku model

  Returns:
    Dict: {
      prediction: `numpy.ndarray`, prediction, with shape [B].
      predictive_entropy: `numpy.ndarray`, predictive entropy, with shape [B].
      predictive_variance: `numpy.ndarray`, predictive variance, with shape [B].
      epistemic_uncertainty: `numpy.ndarray`, mutual info, with shape [B].
      aleatoric_uncertainty: `numpy.ndarray`, expected entropy, with shape [B].
    }
  """
  # mc_samples has shape [S, B]
  list_mc_samples = []
  for i, m in enumerate(model):
    preds_y_samples, _, _ = m.predict_y_multisample_jitted(
        params=params[i],
        state=state[i],
        inputs=x,
        rng_key=rng_key,
        n_samples=num_samples,
        is_training=training_setting,
    )
    list_mc_samples.append(preds_y_samples[:, :, 1])

  mc_samples = jnp.concatenate(list_mc_samples)

  return predict_and_decompose_uncertainty_jax(mc_samples=mc_samples)


def predict_and_decompose_uncertainty_jax(mc_samples):
  """Using a set of MC samples, produce the prediction and uncertainty

    estimates: predictive entropy, predictive variance, epistemic uncertainty
    (MI), and aleatoric uncertainty (expected entropy).

  Args:
    mc_samples: `np.ndarray`, Monte Carlo samples from a sigmoid predictive
      distribution, shape [S, B] where S is the number of samples and B is the
      batch size.

  Returns:
    Dict: {
      prediction: `numpy.ndarray`, prediction, with shape [B].
      predictive_entropy: `numpy.ndarray`, predictive entropy, with shape [B].
      predictive_variance: `numpy.ndarray`, predictive variance, with shape [B].
      epistemic_uncertainty: `numpy.ndarray`, mutual info, with shape [B].
      aleatoric_uncertainty: `numpy.ndarray`, expected entropy, with shape [B].
    }
  """
  expected_entropy = binary_entropy_jax(mc_samples).mean(axis=0)

  prediction = mc_samples.mean(axis=0)
  predictive_entropy = binary_entropy_jax(prediction)
  predictive_variance = mc_samples.var(axis=0)

  return {
      'prediction': prediction,
      'predictive_entropy': predictive_entropy,
      'predictive_variance': predictive_variance,
      'epistemic_uncertainty': predictive_entropy - expected_entropy,  # MI
      'aleatoric_uncertainty': expected_entropy
  }


def ssvae_m1_predict_and_decompose_uncertainty_tf(x, model, training_setting):
  outputs = model(x, training=training_setting)

  prediction = tf.squeeze(tf.nn.sigmoid(outputs['logits']))
  predictive_entropy = binary_entropy_tf(probs=prediction)

  return {
      'prediction': prediction,
      'predictive_entropy': predictive_entropy,
      'predictive_variance': None,
      'epistemic_uncertainty': None,
      'aleatoric_uncertainty': None
  }


def ssvae_m2_predict_and_decompose_uncertainty_tf(x, model, training_setting):
  y_placeholder = 0.5 * tf.ones(x.shape[0])
  outputs = model(dict(image=x, label=y_placeholder), training=training_setting)

  prediction = tf.squeeze(tf.nn.sigmoid(outputs['logits']))
  predictive_entropy = binary_entropy_tf(probs=prediction)

  return {
      'prediction': prediction,
      'predictive_entropy': predictive_entropy,
      'predictive_variance': None,
      'epistemic_uncertainty': None,
      'aleatoric_uncertainty': None
  }


def ssvae_m2_predict_and_decompose_uncertainty_np(x, model, training_setting, num_samples):
  enc_input_nodes = model.input['image']
  enc_output_nodes = dict(
    logit=model.output['logits'],
    mean=model.output['mean'],
    logvar=model.output['logvar'],
  )
  dec_input_nodes = dict(
    mean=model.output['mean'],
    logvar=model.output['logvar'],
    label=model.layers[195].input,
  )
  dec_output_nodes = dict(
    reconstruction=model.output['reconstruction'],
    latent_sample=model.layers[196].output,
  )
  encoder_classifier = tf.keras.Model(inputs=enc_input_nodes, outputs=enc_output_nodes)
  decoder = tf.keras.Model(inputs=dec_input_nodes, outputs=dec_output_nodes)

  enc_outputs = encoder_classifier(x)
  logit = enc_outputs['logit'].numpy()
  mean, logvar = enc_outputs['mean'], enc_outputs['logvar']

  y_zero = -tf.ones(x.shape[0])
  y_one = tf.ones(x.shape[0])
  nl_prior_y = [0.2177419503550854, 1.6313409048600758] # -log([28253/35126, 6873/35126]) : from train set

  input_nll_zero, input_nll_one = [], []

  for _ in range(num_samples):
    outputs_zero = decoder(dict(mean=mean, logvar=logvar, label=y_zero))
    input_nll_zero.append(nll_impsample(inputs=x, **enc_outputs, **outputs_zero))

    outputs_one = decoder(dict(mean=mean, logvar=logvar, label=y_one))
    input_nll_one.append(nll_impsample(inputs=x, **enc_outputs, **outputs_one))

  input_nll_zero = tf.reduce_logsumexp(input_nll_zero, axis=0).numpy()
  input_nll_one = tf.reduce_logsumexp(input_nll_one, axis=0).numpy()

  prediction = sigmoid(logit)
  predictive_entropy = binary_entropy_np(probs=prediction)

  return {
      'logit': logit,
      'prediction': prediction,
      'predictive_entropy': predictive_entropy,
      'predictive_variance': None,
      'epistemic_uncertainty': None,
      'aleatoric_uncertainty': None,
      'input_nll_zero': input_nll_zero,
      'input_nll_one': input_nll_one,
      # 'rec_nll_0': rec_nll_0,
      # 'rec_nll_1': rec_nll_1,
      # 'lp_nll_0': lp_nll_0,
      # 'lp_nll_1': lp_nll_1,
      # 'vp_nll_0': vp_nll_0,
      # 'vp_nll_1': vp_nll_1,
  }


def cvae_predict_and_decompose_uncertainty_tf(x, model, training_setting):
  outputs = model(x, training=training_setting)
  latent_means = outputs['mean']
  logit = tf.reduce_mean(latent_means, axis = tuple(range(1, len(latent_means.shape))))

  prediction = tf.squeeze(tf.nn.sigmoid(logit))
  predictive_entropy = binary_entropy_tf(probs=prediction)

  return {
      'prediction': prediction,
      'predictive_entropy': predictive_entropy,
      'predictive_variance': None,
      'epistemic_uncertainty': None,
      'aleatoric_uncertainty': None,
  }


def cvae_predict_and_decompose_uncertainty_np(x, model, training_setting, num_samples):

  input_nll_zero, input_nll_one = [], []

  for _ in range(num_samples):
    outputs = model(x, training=training_setting)
    input_nll_zero.append(nll_impsample(
       inputs=x,
       mean=outputs['mean'] - outputs['mean_zero'],
       latent_sample=outputs['latent_sample'] - outputs['mean_zero'],
       logvar=outputs['logvar'],
       reconstruction=outputs['reconstruction'],
    ))
    input_nll_one.append(nll_impsample(
       inputs=x,
       mean=outputs['mean'] - outputs['mean_one'],
       latent_sample=outputs['latent_sample'] - outputs['mean_one'],
       logvar=outputs['logvar'],
       reconstruction=outputs['reconstruction'],
    ))

  logit = outputs['logit'].numpy()
  prediction = sigmoid(logit)
  predictive_entropy = binary_entropy_np(probs=prediction)

  input_nll_zero = tf.reduce_logsumexp(input_nll_zero, axis=0).numpy()
  input_nll_one = tf.reduce_logsumexp(input_nll_one, axis=0).numpy()

  return {
      'prediction': prediction,
      'predictive_entropy': predictive_entropy,
      'predictive_variance': None,
      'epistemic_uncertainty': None,
      'aleatoric_uncertainty': None,
      'logit': logit,
      'input_nll_zero': input_nll_zero,
      'input_nll_one': input_nll_one,
  }


def cvaex_predict_and_decompose_uncertainty_tf(x, model, training_setting, num_samples=1):
  y_zero = tf.zeros(x.shape[0])
  y_one = tf.ones(x.shape[0])

  input_nll_zero, input_nll_one = [], []

  for _ in range(num_samples):
    input_nll_zero.append(nll_impsample(inputs=x, **model(dict(image=x, label=y_zero), training=training_setting)))
    input_nll_one.append(nll_impsample(inputs=x, **model(dict(image=x, label=y_one), training=training_setting)))

  input_nll_zero = tf.reduce_logsumexp(input_nll_zero, axis=0)
  input_nll_one = tf.reduce_logsumexp(input_nll_one, axis=0)

  logit = input_nll_zero - input_nll_one
  prediction = tf.squeeze(tf.nn.sigmoid(logit))
  predictive_entropy = binary_entropy_tf(probs=prediction)

  return {
      'prediction': prediction,
      'predictive_entropy': predictive_entropy,
      'predictive_variance': None,
      'epistemic_uncertainty': None,
      'aleatoric_uncertainty': None,
      'logit': logit,
      'input_nll_zero': input_nll_zero,
      'input_nll_one': input_nll_one,
  }


def cvaex_predict_and_decompose_uncertainty_np(x, model, training_setting, num_samples):
  y_zero = tf.zeros(x.shape[0])
  y_one = tf.ones(x.shape[0])

  input_nll_zero, input_nll_one = [], []

  for _ in range(num_samples):
    input_nll_zero.append(nll_impsample(inputs=x, **model(dict(image=x, label=y_zero), training=training_setting)))
    input_nll_one.append(nll_impsample(inputs=x, **model(dict(image=x, label=y_one), training=training_setting)))

  input_nll_zero = tf.reduce_logsumexp(input_nll_zero, axis=0).numpy()
  input_nll_one = tf.reduce_logsumexp(input_nll_one, axis=0).numpy()

  logit = input_nll_zero - input_nll_one
  prediction = sigmoid(logit)
  predictive_entropy = binary_entropy_np(probs=prediction)

  return {
      'prediction': prediction,
      'predictive_entropy': predictive_entropy,
      'predictive_variance': None,
      'epistemic_uncertainty': None,
      'aleatoric_uncertainty': None,
      'logit': logit,
      'input_nll_zero': input_nll_zero,
      'input_nll_one': input_nll_one,
  }


# Format:
# (model_type, use_ensemble): predict_and_decompose_uncertainty_fn
RETINOPATHY_MODEL_TO_DECOMPOSED_UNCERTAINTY_ESTIMATOR = {
    ('deterministic', False):
        deterministic_predict_and_decompose_uncertainty_np,
    ('deterministic', True):
        deep_ensemble_predict_and_decompose_uncertainty_np,
    ('dropout', False):
        variational_predict_and_decompose_uncertainty_np,
    ('dropout', True):
        variational_ensemble_predict_and_decompose_uncertainty_np,
    ('radial', False):
        variational_predict_and_decompose_uncertainty_np,
    ('radial', True):
        variational_ensemble_predict_and_decompose_uncertainty_np,
    ('variational_inference', False):
        (variational_predict_and_decompose_uncertainty_np),
    ('variational_inference', True):
        (variational_ensemble_predict_and_decompose_uncertainty_np),
    ('rank1', False):
        variational_predict_and_decompose_uncertainty_np,
    ('rank1', True):
        variational_ensemble_predict_and_decompose_uncertainty_np,
    ('swag', False):
        None,  # SWAG requires sampling outside the dataset loop
    ('swag', True):
        None,
    ('fsvi', False):
        fsvi_predict_and_decompose_uncertainty_jax,
    ('fsvi', True):
        fsvi_ensemble_predict_and_decompose_uncertainty_jax,
    # ('ssvae_m1', False):
    #     ssvae_m1_predict_and_decompose_uncertainty_np,
    ('ssvae_m2', False):
        ssvae_m2_predict_and_decompose_uncertainty_np,
    ('resnet50_ssvae_m2', False):
        ssvae_m2_predict_and_decompose_uncertainty_np,
    ('cvae', False):
        cvae_predict_and_decompose_uncertainty_np,
    ('cvaex', False):
        cvaex_predict_and_decompose_uncertainty_np,
}

# (model_type, use_ensemble): predict_and_decompose_uncertainty_fn
# Need these for use with TensorFlow TPU and GPU strategies
RETINOPATHY_MODEL_TO_TF_DECOMPOSED_UNCERTAINTY_ESTIMATOR = {
    ('deterministic', False):
        deterministic_predict_and_decompose_uncertainty_tf,
    ('deterministic', True):
        deep_ensemble_predict_and_decompose_uncertainty_tf,
    ('dropout', False):
        variational_predict_and_decompose_uncertainty_tf,
    ('dropout', True):
        variational_ensemble_predict_and_decompose_uncertainty_tf,
    ('radial', False):
        variational_predict_and_decompose_uncertainty_tf,
    ('radial', True):
        variational_ensemble_predict_and_decompose_uncertainty_tf,
    ('variational_inference', False):
        (variational_predict_and_decompose_uncertainty_tf),
    ('variational_inference', True):
        (variational_ensemble_predict_and_decompose_uncertainty_tf),
    # Rank 1 BNNs also have default functionality for mixture posteriors
    ('rank1', False): (variational_predict_and_decompose_uncertainty_tf),
    ('rank1', True):
        (variational_ensemble_predict_and_decompose_uncertainty_tf),
    ('swag', False):
        None,  # SWAG requires sampling outside the dataset loop
    ('swag', True):
        None,
    ('ssvae_m1', False):
        ssvae_m1_predict_and_decompose_uncertainty_tf,
    ('ssvae_m2', False):
        ssvae_m2_predict_and_decompose_uncertainty_tf,
    ('cvae', False):
        cvae_predict_and_decompose_uncertainty_tf,
    ('cvaex', False):
        cvaex_predict_and_decompose_uncertainty_tf,
}
"""Prediction and Loss computation.
"""


def wrap_retinopathy_estimator(estimator,
                               use_mixed_precision,
                               return_logits=False,
                               numpy_outputs=True):
  """Models used in the Diabetic Retinopathy baseline output logits by default.

  Apply conversion if necessary based on mixed precision setting, and apply
  a sigmoid to obtain sigmoid probability [0.0, 1.0] for the model.

  Args:
    estimator: a `tensorflow.keras.model` probabilistic model, that accepts
      input with shape [B, H, W, 3] and outputs logits
    use_mixed_precision: bool, whether to use mixed precision.
    return_logits: bool, optionally return logits.
    numpy_outputs: bool, convert outputs to numpy.

  Returns:
     wrapped estimator, outputting sigmoid probabilities.
  """

  def estimator_wrapper(inputs, training, estimator):
    logits = estimator(inputs, training=training)
    if use_mixed_precision:
      logits = tf.cast(logits, tf.float32)
    probs = tf.squeeze(tf.nn.sigmoid(logits))

    if numpy_outputs and return_logits:
      return probs.numpy(), logits.numpy()
    elif return_logits:
      return probs, logits
    elif numpy_outputs:
      return probs.numpy()
    else:
      return probs

  return functools.partial(estimator_wrapper, estimator=estimator)


def wrap_retinopathy_cvae_estimator(estimator, coeff, dims, numpy_outputs=False):

  mean_zero = tf.constant(- coeff * (np.arange(2048) < dims)) # TODO(anuj): remove hardcode
  mean_one = tf.constant(coeff * (np.arange(2048) < dims)) # TODO(anuj): remove hardcode

  def estimator_wrapper(inputs, training, estimator):
    outputs = estimator(inputs, training=training)
    outputs['logit'] = tf.reduce_mean(outputs['mean'][..., :dims], axis = tuple(range(1, len(outputs['mean'].shape))))
    outputs['mean_zero'] = tf.cast(mean_zero, outputs['latent_sample'].dtype)
    outputs['mean_one'] = tf.cast(mean_one, outputs['latent_sample'].dtype)
    if numpy_outputs:
      outputs = {key: value.numpy() for key, value in outputs.items()}
    return outputs

  return functools.partial(estimator_wrapper, estimator=estimator)


def negative_log_likelihood_metric(labels, probs):
  """Wrapper computing NLL for the Diabetic Retinopathy classification task.

  Args:
    labels: the ground truth labels, with shape `[batch_size, d0, .., dN]`.
    probs: the predicted values, with shape `[batch_size, d0, .., dN]`.

  Returns:
    Binary NLL.
  """
  return tf.reduce_mean(
      tf.keras.losses.binary_crossentropy(
          y_true=tf.expand_dims(labels, axis=-1),
          y_pred=tf.expand_dims(probs, axis=-1),
          from_logits=False))


def get_uncertainty_estimator(model_type, use_ensemble, use_tf):
  if model_type == 'swag':
    raise NotImplementedError  # Special eval loop
  try:
    if use_tf:
      uncertainty_estimator_fn = (
          RETINOPATHY_MODEL_TO_TF_DECOMPOSED_UNCERTAINTY_ESTIMATOR[(
              model_type, use_ensemble)])
    else:
      uncertainty_estimator_fn = (
          RETINOPATHY_MODEL_TO_DECOMPOSED_UNCERTAINTY_ESTIMATOR[(model_type,
                                                                 use_ensemble)])
  except KeyError:
    raise NotImplementedError(
        'Unsupported model type. Try implementing a wrapper to retrieve '
        'aleatoric, epistemic, and total uncertainty in X.py.')

  return uncertainty_estimator_fn
