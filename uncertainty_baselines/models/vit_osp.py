# coding=utf-8
# Copyright 2023 The Uncertainty Baselines Authors.
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


class GradientMultiplier(nn.Module):
  grad_coeff: float

  @nn.compact
  def __call__(self, x):
    sgx = jax.lax.stop_gradient(x)
    return sgx + self.grad_coeff * (x - sgx)


class Lagrangian(nn.Module):
  num_classes: int
  mu: float = 1
  lambda_grad: float = 1
  phi_grad: float = 1

  @nn.compact
  def __call__(self, x, y):
    loglambdas = self.param('lambda', nn.initializers.zeros, self.num_classes)
    lambdas = jax.nn.softplus(loglambdas)
    lambdas = GradientMultiplier(-self.lambda_grad)(lambdas)
    phis = self.param('phi', nn.initializers.zeros, self.num_classes)
    phis = GradientMultiplier(self.phi_grad)(phis)

    x = jnp.reshape(x, (-1, self.num_classes))
    y = jnp.reshape(y, (-1, self.num_classes))

    nl_p = -jax.nn.log_sigmoid(x)
    nl_not_p = -jax.nn.log_sigmoid(-x)
    loss = jnp.sum(nl_p * y, axis=0) / (jnp.sum(y, axis=0) + 1e-16)
    constraint = jnp.sum(nl_not_p * (1 - y), axis=0) / (jnp.sum((1 - y), axis=0) + 1e-16)
    
    lagrangian = loss + lambdas * (constraint - phis) + self.mu * phis
    return jnp.mean(lagrangian)


class VisionTransformerOsp(nn.Module):
  """Vision Transformer model."""

  num_classes: int
  patches: Any
  transformer: Any
  lagrangian: Any
  hidden_size: int
  representation_size: Optional[int] = None
  classifier: str = 'token'
  fix_base_model: bool = False

  @nn.compact
  def __call__(self, inputs, labels, *, train):
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

    x = nn.Dense(
        features=self.num_classes,
        name='head',
        kernel_init=nn.initializers.zeros)(
            x)
    out['logits'] = x

    out['lagrangian'] = Lagrangian(num_classes=self.num_classes, **self.lagrangian)(x, labels)

    return x, out


def vision_transformer_osp(num_classes: int,
                       patches: Any,
                       transformer: Any,
                       lagrangian: Any,
                       hidden_size: int,
                       representation_size: Optional[int] = None,
                       classifier: str = 'token',
                       fix_base_model: bool = False):
  """Builds a Vision Transformer (ViT) model."""
  # TODO(dusenberrymw): Add API docs once config dict in VisionTransformer is
  # cleaned up.
  return VisionTransformerOsp(
      num_classes=num_classes,
      patches=patches,
      transformer=transformer,
      lagrangian=lagrangian,
      hidden_size=hidden_size,
      representation_size=representation_size,
      classifier=classifier,
      fix_base_model=fix_base_model)
