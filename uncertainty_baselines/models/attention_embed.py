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

""" Attention based multiple instance learning """
from typing import Any, Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

class GatedAttentionEmbd(nn.Module):
    output_dim : int
    @nn.compact
    def __call__(self, x):
        # x.shape = B X P X 512
        # B = No of batches (i.e = 1)
        # P = No of patches (i.e = 5000)
        # 512 is embeddings     
        K = 1
        att_v = nn.Dense(self.output_dim)(x)
        att_v = nn.tanh(att_v) # P X 128 
        
        att_u = nn.Dense(self.output_dim)(x)
        att_u = nn.sigmoid(att_u) # P X 128 
                
        att = nn.Dense(K)(att_v * att_u) # P X K
        att = jnp.transpose(att) # K X P
        
        att = nn.softmax(att, axis=1) # K X P - Softmax over P
        
        weighted_sm = jnp.matmul(att, x) # K X 512 weighted summation
        
        out = {}
        out["y_logits"] = nn.Dense(1)(weighted_sm)
        out["y_prob"] = nn.sigmoid(out["y_logits"])
        out["y_pred"] = (out["y_prob"] >= 0.5).astype(int)
        return out["y_logits"], out


def gated_attention_embed(output_dim = 128):
    return GatedAttentionEmbd(output_dim)
    