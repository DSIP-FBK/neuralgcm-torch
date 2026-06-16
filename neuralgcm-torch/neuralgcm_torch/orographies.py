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
"""Modules responsible for orography processing and initialization.

Orography modules are called with no arguments and return the modal
orography. Data loading and nondimensionalization happen in the
checkpoint/model builder; modules receive ready nodal arrays.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

from dinosaur_torch import spherical_harmonic


def _interpolate_modal(
    x: torch.Tensor,
    source_grid: spherical_harmonic.Grid,
    target_grid: spherical_harmonic.Grid,
) -> torch.Tensor:
  """Slices or pads modal coefficients from one grid onto another."""
  source_shape = source_grid.modal_shape
  target_shape = target_grid.modal_shape
  if all(s >= t for s, t in zip(source_shape, target_shape)):
    return x[..., : target_shape[0], : target_shape[1]]
  if all(s <= t for s, t in zip(source_shape, target_shape)):
    pad = (0, target_shape[1] - source_shape[1],
           0, target_shape[0] - source_shape[0])
    return torch.nn.functional.pad(x, pad)
  raise ValueError(
      f'incompatible modal shapes {source_shape} and {target_shape}'
  )


class ClippedOrography(nn.Module):
  """Converts nodal orography to modal representation with clipping."""

  HAIKU_NAME = 'clipped_orography'

  def __init__(
      self,
      grid: spherical_harmonic.Grid,
      nodal_orography: np.ndarray,
      wavenumbers_to_clip: int = 1,
  ):
    super().__init__()
    self.grid = grid
    self.wavenumbers_to_clip = wavenumbers_to_clip
    ref = grid.cos_lat
    self.register_buffer(
        'nodal_orography',
        torch.as_tensor(
            np.asarray(nodal_orography), dtype=ref.dtype, device=ref.device
        ),
        persistent=False,
    )

  def forward(self) -> torch.Tensor:
    return self.grid.clip_wavenumbers(
        self.grid.to_modal(self.nodal_orography), self.wavenumbers_to_clip
    )


class FilteredCustomOrography(nn.Module):
  """Initializes orography from external data, interpolated and filtered.

  `nodal_orography` is given on `input_grid` (typically the linear-
  truncation grid of the source dataset), converted to modal there,
  interpolated onto `grid`, then passed through `filters` in order.
  """

  HAIKU_NAME = 'filtered_custom_orography'

  def __init__(
      self,
      grid: spherical_harmonic.Grid,
      input_grid: spherical_harmonic.Grid,
      nodal_orography: np.ndarray,
      filters: Sequence[nn.Module] = tuple(),
  ):
    super().__init__()
    self.grid = grid
    self.input_grid = input_grid
    self.filters = nn.ModuleList(filters)
    ref = input_grid.cos_lat
    self.register_buffer(
        'nodal_orography',
        torch.as_tensor(
            np.asarray(nodal_orography), dtype=ref.dtype, device=ref.device
        ),
        persistent=False,
    )

  def forward(self) -> torch.Tensor:
    modal = _interpolate_modal(
        self.input_grid.to_modal(self.nodal_orography),
        self.input_grid,
        self.grid,
    )
    for filter_module in self.filters:
      modal = filter_module(modal)
    return modal


class LearnedOrography(nn.Module):
  """Adds a learned modal correction to a base orography module."""

  HAIKU_NAME = 'learned_orography'

  def __init__(
      self,
      grid: spherical_harmonic.Grid,
      base_orography: nn.Module,
      correction_scale: float,
  ):
    super().__init__()
    self.grid = grid
    self.base_orography = base_orography
    self.scale = correction_scale
    ref = grid.cos_lat
    mask = torch.as_tensor(np.asarray(grid.mask), device=ref.device)
    self.register_buffer('mask', mask, persistent=False)
    self.correction = nn.Parameter(
        torch.zeros(
            int(mask.sum()), dtype=ref.dtype, device=ref.device
        )
    )

  def forward(self) -> torch.Tensor:
    correction_2d = torch.zeros(
        self.grid.modal_shape,
        dtype=self.correction.dtype,
        device=self.correction.device,
    )
    correction_2d = correction_2d.masked_scatter(self.mask, self.correction)
    return self.base_orography() + correction_2d * self.scale

  def import_haiku(self, params: dict, prefix: str) -> None:
    bundle = params[prefix]
    with torch.no_grad():
      self.correction.copy_(bundle['orography'])
    # the base orography module is created in the legacy __init__.
    if hasattr(self.base_orography, 'import_haiku'):
      self.base_orography.import_haiku(
          params, f'{prefix}/~/{self.base_orography.HAIKU_NAME}'
      )
