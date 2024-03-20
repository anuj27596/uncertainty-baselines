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

"""Vision Transformer (ViT) model."""
from typing import Any, Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import uncertainty_baselines.models.vit as vit

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


class GradientReversalLayer(nn.Module):
  """Identity layer, convenient for giving a name to an array."""

  grad_coeff: float
  
  @nn.compact
  def __call__(self, x):
    sgx = jax.lax.stop_gradient(x)
    return sgx + self.grad_coeff * (sgx - x)


class DomainPredictor(nn.Module):
  """Transformer MLP / feed-forward block."""

  hid_dim: int
  grl_coeff: float
  num_layers: int = 2
  dtype: Dtype = jnp.float32
  kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.xavier_uniform()
  bias_init: Callable[[PRNGKey, Shape, Dtype],
                      Array] = nn.initializers.normal(stddev=1e-6)

  @nn.compact
  def __call__(self, inputs):
    x = GradientReversalLayer(self.grl_coeff)(inputs)

    for _ in range(self.num_layers - 1):
      x = nn.Dense(
          features=self.hid_dim,
          dtype=self.dtype,
          kernel_init=self.kernel_init,
          bias_init=self.bias_init)(  # pytype: disable=wrong-arg-types
              x)
      x = nn.gelu(x)
    
    output = nn.Dense(
        features=1,
        dtype=self.dtype,
        kernel_init=nn.initializers.zeros)(  # pytype: disable=wrong-arg-types
            x)
    return output


class ReverseClassifier(nn.Module):
  """Transformer MLP / feed-forward block."""

  dtype: Dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs):
    x = jax.lax.stop_gradient(inputs)
    output = nn.Dense(
        features=1,
        dtype=self.dtype,
        kernel_init=nn.initializers.zeros)(  # pytype: disable=wrong-arg-types
            x)
    return output


class VisionTransformerDan(nn.Module):
  """Vision Transformer model."""

  num_classes: int
  patches: Any
  transformer: Any
  domain_predictor: Any
  hidden_size: int
  representation_size: Optional[int] = None
  classifier: str = 'token'
  fix_base_model: bool = False

  @nn.compact
  def __call__(self, inputs, *, train):
    out = {}

    x = inputs 
    n, h, w, c = x.shape

    # We can merge s2d+emb into a single conv; it's the same.
    x = nn.Conv(
        features=self.hidden_size,
        kernel_size=self.patches.size,
        strides=self.patches.size,
        padding='VALID',
        name='embedding')(
            x)

    # Here, x is a grid of embeddings.
    # TODO(dusenberrymw): Switch to self.sow(.).
    out['stem'] = x

    # Transformer.
    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    # If we want to add a class token, add it here.
    if self.classifier == 'token':
      cls = self.param('cls', nn.initializers.zeros, (1, 1, c))
      cls = jnp.tile(cls, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)

    x = vit.Encoder(name='Transformer', **self.transformer)(x, train=train)
    out['transformed'] = x

    if self.classifier == 'token':
      x = x[:, 0]
    elif self.classifier == 'gap':
      x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))  # (1,) or (1,2)
    else:
      raise ValueError(f'Invalid classifier={self.classifier}')

    out['head_input'] = x

    if self.representation_size is not None:
      x = nn.Dense(features=self.representation_size, name='pre_logits')(x)
      out['pre_logits'] = x
      x = nn.tanh(x)
    else:
      x = vit.IdentityLayer(name='pre_logits')(x)
      out['pre_logits'] = x

    # TODO(markcollier): Fix base model without using stop_gradient.
    if self.fix_base_model:
      x = jax.lax.stop_gradient(x)

    out['domain_pred'] = DomainPredictor(
        **self.domain_predictor)(
            x)

    out['reverse_pred'] = ReverseClassifier()(x)

    x = nn.Dense(
        features=self.num_classes,
        name='head',
        kernel_init=nn.initializers.zeros)(
            x)
    out['logits'] = x
    return x, out


def vision_transformer_dan(num_classes: int,
                       patches: Any,
                       transformer: Any,
                       domain_predictor: Any,
                       hidden_size: int,
                       representation_size: Optional[int] = None,
                       classifier: str = 'token',
                       fix_base_model: bool = False):
  """Builds a Vision Transformer (ViT) model."""
  # TODO(dusenberrymw): Add API docs once config dict in VisionTransformer is
  # cleaned up.
  return VisionTransformerDan(
      num_classes=num_classes,
      patches=patches,
      transformer=transformer,
      domain_predictor=domain_predictor,
      hidden_size=hidden_size,
      representation_size=representation_size,
      classifier=classifier,
      fix_base_model=fix_base_model)
