# Copyright 2023 Google LLC
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

"""`CoordinateSystem`: a horizontal `Grid` plus vertical `SigmaLevels`."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from dinosaur_torch import pytree
from dinosaur_torch import sigma_coordinates
from dinosaur_torch import spherical_harmonic


class CoordinateSystem(nn.Module):
  """Bundles the horizontal and vertical discretizations of a model.

  Attributes:
    horizontal: `spherical_harmonic.Grid` for the horizontal plane.
    vertical: `sigma_coordinates.SigmaLevels` for the vertical coordinate.
  """

  def __init__(
      self,
      horizontal: spherical_harmonic.Grid,
      vertical: sigma_coordinates.SigmaLevels,
  ):
    super().__init__()
    self.horizontal = horizontal
    self.vertical = vertical

  @property
  def nodal_shape(self) -> tuple[int, int, int]:
    """3d nodal grid shape: (layers,) + horizontal nodal shape."""
    return (self.vertical.layers,) + self.horizontal.nodal_shape

  @property
  def modal_shape(self) -> tuple[int, int, int]:
    """3d modal grid shape: (layers,) + horizontal modal shape."""
    return (self.vertical.layers,) + self.horizontal.modal_shape

  @property
  def surface_nodal_shape(self) -> tuple[int, int, int]:
    return (1,) + self.horizontal.nodal_shape

  @property
  def surface_modal_shape(self) -> tuple[int, int, int]:
    return (1,) + self.horizontal.modal_shape

  def maybe_to_nodal(self, fields):
    """Transforms tensor leaves of `fields` that are not yet nodal."""
    nodal = self.horizontal.nodal_shape

    def fn(x):
      if tuple(x.shape[-2:]) == nodal:
        return x
      return self.horizontal.to_nodal(x)

    return pytree.map_fields(fn, fields)

  def maybe_to_modal(self, fields):
    """Transforms tensor leaves of `fields` that are not yet modal."""
    modal = self.horizontal.modal_shape

    def fn(x):
      if tuple(x.shape[-2:]) == modal:
        return x
      return self.horizontal.to_modal(x)

    return pytree.map_fields(fn, fields)


def get_spectral_downsample_fn(
    coords: CoordinateSystem,
    save_coords: CoordinateSystem,
    expect_same_vertical: bool = True,
) -> Callable:
  """Returns a function that downsamples modal state to `save_coords`."""
  if expect_same_vertical and (
      coords.vertical.coordinates != save_coords.vertical.coordinates
  ):
    raise ValueError('downsampling vertical resolution is not supported.')
  lon_slice = slice(0, save_coords.horizontal.modal_shape[0])
  total_slice = slice(0, save_coords.horizontal.modal_shape[1])
  if (
      coords.horizontal.spec.total_wavenumbers
      < save_coords.horizontal.spec.total_wavenumbers
  ) or (
      coords.horizontal.spec.longitude_wavenumbers
      < save_coords.horizontal.spec.longitude_wavenumbers
  ):
    raise ValueError('save_coords.horizontal larger than coords.horizontal')

  def downsample_fn(state):
    return pytree.map_fields(lambda x: x[..., lon_slice, total_slice], state)

  return downsample_fn


def get_spectral_upsample_fn(
    coords: CoordinateSystem,
    save_coords: CoordinateSystem,
    expect_same_vertical: bool = True,
) -> Callable:
  """Returns a function that upsamples modal state to `save_coords`."""
  if expect_same_vertical and (
      coords.vertical.coordinates != save_coords.vertical.coordinates
  ):
    raise ValueError('upsampling vertical resolution is not supported.')
  save_shape = save_coords.horizontal.modal_shape
  coords_shape = coords.horizontal.modal_shape
  lon_pad = save_shape[0] - coords_shape[0]
  total_pad = save_shape[1] - coords_shape[1]
  if lon_pad < 0 or total_pad < 0:
    raise ValueError('save_coords.horizontal smaller than coords.horizontal')

  def upsample_fn(state):
    # F.pad pads from the last dimension backwards.
    pad = (0, total_pad, 0, lon_pad)
    return pytree.map_fields(
        lambda x: torch.nn.functional.pad(x, pad), state
    )

  return upsample_fn


def get_spectral_interpolate_fn(
    source_coords: CoordinateSystem,
    target_coords: CoordinateSystem,
    expect_same_vertical: bool = True,
) -> Callable:
  """Modal interpolation from `source_coords` to `target_coords`."""
  source_spec = source_coords.horizontal.spec
  target_spec = target_coords.horizontal.spec
  if (source_spec.total_wavenumbers < target_spec.total_wavenumbers) and (
      source_spec.longitude_wavenumbers < target_spec.longitude_wavenumbers
  ):
    return get_spectral_upsample_fn(
        source_coords, target_coords, expect_same_vertical
    )
  elif (source_spec.total_wavenumbers >= target_spec.total_wavenumbers) and (
      source_spec.longitude_wavenumbers >= target_spec.longitude_wavenumbers
  ):
    return get_spectral_downsample_fn(
        source_coords, target_coords, expect_same_vertical
    )
  else:
    raise ValueError(
        'Incompatible horizontal coordinates with shapes '
        f'{source_coords.horizontal.modal_shape}, '
        f'{target_coords.horizontal.modal_shape}'
    )
