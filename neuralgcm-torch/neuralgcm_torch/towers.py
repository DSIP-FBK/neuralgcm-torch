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
"""Neural network towers: networks mapped over (longitude, latitude).

A tower operates identically over the trailing two spatial dimensions.
Sub-modules are passed in already constructed (standard torch composition)
rather than via gin factories; `import_haiku` methods reproduce the legacy
haiku path layout, including the gin-bound child names used by the
published checkpoints (`encode_tower`, `process_tower`, `process_tower_N`,
`decode_tower`).

Only the towers exercised by published NeuralGCM checkpoints are ported:
`ColumnTower`, `VerticalConvTower` and `EpdTower`.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
from torch import nn

from neuralgcm_torch import layers


class ColumnTower(nn.Module):
  """Applies a column network independently at every (lon, lat) point.

  Input layout `(channels, lon, lat)`; the column net sees `channels` as
  its feature dimension. As in the original JAX implementation, the columns are moved to
  leading batch dimensions instead of using a nested vmap, so the
  column-net matmuls lower to single GEMMs.
  """

  HAIKU_NAME = 'column_tower'

  def __init__(self, column_net: nn.Module):
    super().__init__()
    self.column_net = column_net

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    moved = torch.movedim(inputs, (-2, -1), (0, 1))
    out = self.column_net(moved)
    return torch.movedim(out, (0, 1), (-2, -1))

  def import_haiku(self, params: dict, prefix: str) -> None:
    # the column net is created in the legacy tower's __init__.
    self.column_net.import_haiku(
        params, f'{prefix}/~/{self.column_net.HAIKU_NAME}'
    )


class VerticalConvTower(nn.Module):
  """A stack of vertical 1D convolutions applied at every (lon, lat) point.

  Input shape `(in_channels, level, lon, lat)`; output
  `(output_size, level, lon, lat)`.
  """

  HAIKU_NAME = 'vertical_conv_tower'

  def __init__(
      self,
      input_size: int,
      output_size: int,
      channels: Sequence[int],
      kernel_shape: int,
      *,
      with_bias: bool = True,
      activation: Callable[[torch.Tensor], torch.Tensor] = layers.relu,
      activate_final: bool = False,
      device=None,
      dtype: torch.dtype = torch.float32,
  ):
    super().__init__()
    self.activation = activation
    self.activate_final = activate_final
    sizes = [input_size] + list(channels) + [output_size]
    self.layers = nn.ModuleList(
        layers.ConvLevel(
            sizes[i],
            sizes[i + 1],
            kernel_shape,
            with_bias=with_bias,
            device=device,
            dtype=dtype,
        )
        for i in range(len(sizes) - 1)
    )

  def _net(self, inputs: torch.Tensor) -> torch.Tensor:
    out = inputs
    num_layers = len(self.layers)
    for i, layer in enumerate(self.layers):
      out = layer(out)
      if i < num_layers - 1 or self.activate_final:
        out = self.activation(out)
    return out

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    moved = torch.movedim(inputs, (-2, -1), (0, 1))
    out = self._net(moved)
    return torch.movedim(out, (0, 1), (-2, -1))

  def import_haiku(self, params: dict, prefix: str) -> None:
    # ConvLevel children are created in the legacy tower's __init__ with
    # automatic names conv_level, conv_level_1, ...
    for i, layer in enumerate(self.layers):
      suffix = 'conv_level' if i == 0 else f'conv_level_{i}'
      layer.import_haiku(params, f'{prefix}/~/{suffix}')


class EpdTower(nn.Module):
  """Encode-process-decode tower with residual process blocks.

  `process_towers` are applied as residual updates between the encode and
  decode towers. In the published checkpoints the sub-towers carry the
  gin-bound haiku names `encode_tower`, `process_tower`, `process_tower_1`,
  ..., `decode_tower`, created inside the legacy `__call__` (so they nest
  under `<prefix>/<name>` without a `~` component).
  """

  HAIKU_NAME = 'epd_tower'

  def __init__(
      self,
      encode_tower: nn.Module,
      process_towers: Sequence[nn.Module],
      decode_tower: nn.Module,
      *,
      post_encode_activation: Optional[Callable] = None,
      pre_decode_activation: Optional[Callable] = None,
      final_activation: Optional[Callable] = None,
      child_names: Sequence[str] = (
          'encode_tower', 'process_tower', 'decode_tower'
      ),
  ):
    super().__init__()
    self.encode_tower = encode_tower
    self.process_towers = nn.ModuleList(process_towers)
    self.decode_tower = decode_tower
    self.post_encode_activation = post_encode_activation
    self.pre_decode_activation = pre_decode_activation
    self.final_activation = final_activation
    # haiku names of the sub-towers (gin `name=` bindings in the published
    # checkpoints, e.g. 'surface_model_encode_tower').
    self.child_names = tuple(child_names)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    encoded = self.encode_tower(inputs)
    if self.post_encode_activation is not None:
      encoded = self.post_encode_activation(encoded)
    current = encoded
    for process_tower in self.process_towers:
      current = current + process_tower(current)
    if self.pre_decode_activation is not None:
      current = self.pre_decode_activation(current)
    out = self.decode_tower(current)
    if self.final_activation is not None:
      return self.final_activation(out)
    return out

  def import_haiku(
      self,
      params: dict,
      prefix: str,
      *,
      encode_name: Optional[str] = None,
      process_name: Optional[str] = None,
      decode_name: Optional[str] = None,
  ) -> None:
    encode_name = encode_name or self.child_names[0]
    process_name = process_name or self.child_names[1]
    decode_name = decode_name or self.child_names[2]
    self.encode_tower.import_haiku(params, f'{prefix}/{encode_name}')
    for i, process_tower in enumerate(self.process_towers):
      suffix = process_name if i == 0 else f'{process_name}_{i}'
      process_tower.import_haiku(params, f'{prefix}/{suffix}')
    self.decode_tower.import_haiku(params, f'{prefix}/{decode_name}')
