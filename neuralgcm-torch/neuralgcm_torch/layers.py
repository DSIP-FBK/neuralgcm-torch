# Copyright 2024 Google LLC
# Copyright 2026 Fondazione Bruno Kessler
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Basic neural network layers, as standard torch modules.

Unlike the original haiku-style modules, modules here take explicit input sizes
(no shape inference at first call) and hold real `nn.Parameter`s. Each
module provides `import_haiku(params, prefix)` which copies weights from a
converted checkpoint's parameter bundles (see
`neuralgcm_torch.checkpoint`); the haiku path layout of its children
is the module's own responsibility, mirroring the legacy naming exactly.

Weight layout conversions from haiku:
  Linear:  w (in, out)          -> nn.Linear.weight (out, in)
  Conv1D:  w (kernel, in, out)  -> nn.Conv1d.weight (out, in, kernel)
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn


def relu(x: torch.Tensor) -> torch.Tensor:
  return F.relu(x)


def gelu(x: torch.Tensor) -> torch.Tensor:
  # jax.nn.gelu defaults to the tanh approximation.
  return F.gelu(x, approximate='tanh')


def silu(x: torch.Tensor) -> torch.Tensor:
  return F.silu(x)


def _copy_linear(linear: nn.Linear, bundle: dict) -> None:
  """Copies a haiku Linear bundle {'w': (in, out), 'b': (out,)}."""
  w = bundle['w']
  if w.shape != (linear.in_features, linear.out_features):
    raise ValueError(
        f'haiku Linear w has shape {tuple(w.shape)}; expected '
        f'({linear.in_features}, {linear.out_features})'
    )
  with torch.no_grad():
    linear.weight.copy_(w.t())
    if linear.bias is not None:
      linear.bias.copy_(bundle['b'])
    elif 'b' in bundle:
      raise ValueError('bundle has a bias but the module does not')


class MlpUniform(nn.Module):
  """MLP with the same output size in each hidden layer.

  Operates over the last dimension (features); all leading dimensions are
  batch. Mirrors the legacy `MlpUniform` (an `hk.nets.MLP` with an explicit
  final layer): haiku children are `<prefix>/~/linear_{i}` for
  i in [0, num_hidden_layers].
  """

  HAIKU_NAME = 'mlp_uniform'

  def __init__(
      self,
      input_size: int,
      output_size: int,
      num_hidden_units: int,
      num_hidden_layers: int,
      *,
      with_bias: bool = True,
      activation: Callable[[torch.Tensor], torch.Tensor] = relu,
      activate_final: bool = False,
      device=None,
      dtype: torch.dtype = torch.float32,
  ):
    super().__init__()
    self.activation = activation
    self.activate_final = activate_final
    sizes = (
        [input_size] + [num_hidden_units] * num_hidden_layers + [output_size]
    )
    self.linears = nn.ModuleList(
        nn.Linear(
            sizes[i], sizes[i + 1], bias=with_bias, device=device, dtype=dtype
        )
        for i in range(len(sizes) - 1)
    )

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    out = inputs
    num_layers = len(self.linears)
    for i, linear in enumerate(self.linears):
      out = linear(out)
      if i < num_layers - 1 or self.activate_final:
        out = self.activation(out)
    return out

  def import_haiku(self, params: dict, prefix: str) -> None:
    for i, linear in enumerate(self.linears):
      _copy_linear(linear, params[f'{prefix}/~/linear_{i}'])


class ConvLevel(nn.Module):
  """1D convolution in the vertical (convolution on atmospheric columns).

  Input layout `(..., channels, levels)`; leading dimensions are batch.
  Uses haiku's 'SAME' padding convention (`(k-1)*dilation` total, split
  with the extra element on the right).
  """

  HAIKU_NAME = 'conv_level'

  def __init__(
      self,
      input_size: int,
      output_size: int,
      kernel_shape: int,
      *,
      dilation_rate: int = 1,
      with_bias: bool = True,
      device=None,
      dtype: torch.dtype = torch.float32,
  ):
    super().__init__()
    pad_total = (kernel_shape - 1) * dilation_rate
    self._pad = (pad_total // 2, pad_total - pad_total // 2)
    self.conv = nn.Conv1d(
        input_size,
        output_size,
        kernel_shape,
        dilation=dilation_rate,
        bias=with_bias,
        device=device,
        dtype=dtype,
    )

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    x = F.pad(inputs, self._pad)
    batch = x.shape[:-2]
    if len(batch) != 1:
      x = x.reshape(-1, *x.shape[-2:])
    out = self.conv(x)
    if len(batch) != 1:
      out = out.reshape(*batch, *out.shape[-2:])
    return out

  def import_haiku(self, params: dict, prefix: str) -> None:
    bundle = params[prefix]
    w = bundle['w']  # (kernel, in, out)
    expected = (
        self.conv.kernel_size[0],
        self.conv.in_channels,
        self.conv.out_channels,
    )
    if tuple(w.shape) != expected:
      raise ValueError(
          f'haiku Conv1D w has shape {tuple(w.shape)}; expected {expected}'
      )
    with torch.no_grad():
      self.conv.weight.copy_(w.permute(2, 1, 0))
      if self.conv.bias is not None:
        # haiku Conv1D bias has shape (out, 1), broadcasting over width.
        self.conv.bias.copy_(bundle['b'].reshape(-1))
      elif 'b' in bundle:
        raise ValueError('bundle has a bias but the module does not')
