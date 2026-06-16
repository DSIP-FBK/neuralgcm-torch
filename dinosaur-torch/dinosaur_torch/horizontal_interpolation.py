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

"""Routines for horizontal regridding between lat/lon grids.

Regridders are `nn.Module`s built from a pair of `GridSpec`s; interpolation
weights are computed once (NumPy, float64) and stored as non-persistent
buffers. Conservative regridding schemes are adapted from:
https://gist.github.com/shoyer/c0f1ddf409667650a076c058f9a17276
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from sklearn import neighbors

from dinosaur_torch import spherical_harmonic


def _assert_increasing(x: np.ndarray):
  if not (np.diff(x) > 0).all():
    raise ValueError(f'array is not increasing: {x}')


def _latitude_cell_bounds(x: np.ndarray) -> np.ndarray:
  return np.concatenate([[-np.pi / 2], (x[:-1] + x[1:]) / 2, [np.pi / 2]])


def _latitude_overlap(
    source_points: np.ndarray, target_points: np.ndarray
) -> np.ndarray:
  """Calculate the area overlap as a function of latitude."""
  source_bounds = _latitude_cell_bounds(source_points)
  target_bounds = _latitude_cell_bounds(target_points)
  upper = np.minimum(
      target_bounds[1:, np.newaxis], source_bounds[np.newaxis, 1:]
  )
  lower = np.maximum(
      target_bounds[:-1, np.newaxis], source_bounds[np.newaxis, :-1]
  )
  # normalized cell area: integral from lower to upper of cos(latitude)
  return (upper > lower) * (np.sin(upper) - np.sin(lower))


def conservative_latitude_weights(
    source_points: np.ndarray, target_points: np.ndarray
) -> np.ndarray:
  """Create a weight matrix for conservative regridding along latitude.

  Args:
    source_points: 1D latitude coordinates in units of radians for centers
      of source cells.
    target_points: 1D latitude coordinates in units of radians for centers
      of target cells.

  Returns:
    Array with shape (target, source). Rows sum to 1.
  """
  _assert_increasing(source_points)
  _assert_increasing(target_points)
  weights = _latitude_overlap(source_points, target_points)
  weights /= np.sum(weights, axis=1, keepdims=True)
  assert weights.shape == (len(target_points), len(source_points))
  return weights


def _align_phase_with(x, target, period):
  """Aligns the phase of a periodic number to match another."""
  shift_down = x > target + period / 2
  shift_up = x < target - period / 2
  return x + period * shift_up - period * shift_down


def _periodic_upper_bounds(x, period):
  x_plus = _align_phase_with(np.roll(x, -1), x, period)
  return (x + x_plus) / 2


def _periodic_lower_bounds(x, period):
  x_minus = _align_phase_with(np.roll(x, +1), x, period)
  return (x_minus + x) / 2


def _periodic_overlap(x0, x1, y0, y1, period):
  # valid as long as no intervals are larger than period/2
  y0 = _align_phase_with(y0, x0, period)
  y1 = _align_phase_with(y1, x0, period)
  upper = np.minimum(x1, y1)
  lower = np.maximum(x0, y0)
  return np.clip(upper - lower, 0, None)


def _longitude_overlap(
    first_points: np.ndarray,
    second_points: np.ndarray,
    period: float = 2 * np.pi,
) -> np.ndarray:
  """Calculate the area overlap as a function of longitude."""
  first_points = first_points % period
  first_upper = _periodic_upper_bounds(first_points, period)
  first_lower = _periodic_lower_bounds(first_points, period)

  second_points = second_points % period
  second_upper = _periodic_upper_bounds(second_points, period)
  second_lower = _periodic_lower_bounds(second_points, period)

  return _periodic_overlap(
      first_lower[:, np.newaxis],
      first_upper[:, np.newaxis],
      second_lower[np.newaxis, :],
      second_upper[np.newaxis, :],
      period=period,
  )


def conservative_longitude_weights(
    source_points: np.ndarray, target_points: np.ndarray
) -> np.ndarray:
  """Create a weight matrix for conservative regridding along longitude.

  Args:
    source_points: 1D longitude coordinates in units of radians for centers
      of source cells.
    target_points: 1D longitude coordinates in units of radians for centers
      of target cells.

  Returns:
    Array with shape (new_size, old_size). Rows sum to 1.
  """
  _assert_increasing(source_points)
  _assert_increasing(target_points)
  weights = _longitude_overlap(target_points, source_points)
  weights /= np.sum(weights, axis=1, keepdims=True)
  assert weights.shape == (len(target_points), len(source_points))
  return weights


def nearest_neighbor_indices(
    source_grid: spherical_harmonic.GridSpec,
    target_grid: spherical_harmonic.GridSpec,
) -> np.ndarray:
  """Haversine nearest neighbor indices from source_grid to target_grid."""
  lon_source, sin_lat_source = source_grid.nodal_mesh
  lat_source = np.arcsin(sin_lat_source)

  lon_target, sin_lat_target = target_grid.nodal_mesh
  lat_target = np.arcsin(sin_lat_target)

  # construct a BallTree to find nearest neighbor on the surface of a sphere
  index_coords = np.stack([lat_source.ravel(), lon_source.ravel()], axis=-1)
  query_coords = np.stack([lat_target.ravel(), lon_target.ravel()], axis=-1)
  tree = neighbors.BallTree(index_coords, metric='haversine')
  return tree.query(query_coords, return_distance=False).squeeze(axis=-1)


def _interp_along_last(
    x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor
) -> torch.Tensor:
  """np.interp along the last dim of `fp` (constant extrapolation)."""
  n = xp.shape[0]
  u = torch.searchsorted(xp, x.contiguous(), right=True).clamp(1, n - 1)
  w = ((x - xp[u - 1]) / (xp[u] - xp[u - 1])).clamp(0, 1)
  return fp[..., u - 1] + w * (fp[..., u] - fp[..., u - 1])


class Regridder(nn.Module):
  """Base class: maps fields (..., lon, lat) between two grids."""

  def __init__(
      self,
      source_grid: spherical_harmonic.GridSpec,
      target_grid: spherical_harmonic.GridSpec,
      *,
      device: torch.device | str | None = None,
      dtype: torch.dtype = torch.float32,
  ):
    super().__init__()
    self.source_grid = source_grid
    self.target_grid = target_grid

  def _buffer(self, name, array, dtype, device):
    self.register_buffer(
        name,
        torch.as_tensor(np.asarray(array, np.float64), dtype=dtype,
                        device=device),
        persistent=False,
    )

  def forward(self, field: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


class BilinearRegridder(Regridder):
  """Regrid with bilinear interpolation."""

  def __init__(self, source_grid, target_grid, *, device=None,
               dtype=torch.float32):
    super().__init__(source_grid, target_grid, device=device, dtype=dtype)
    self._buffer('lat_source', source_grid.latitudes, dtype, device)
    self._buffer('lat_target', target_grid.latitudes, dtype, device)
    self._buffer('lon_source', source_grid.longitudes, dtype, device)
    self._buffer('lon_target', target_grid.longitudes, dtype, device)

  def forward(self, field: torch.Tensor) -> torch.Tensor:
    # interpolate latitude (the last axis)
    field = _interp_along_last(self.lat_target, self.lat_source, field)
    # interpolate longitude (the second-to-last axis)
    field = _interp_along_last(
        self.lon_target, self.lon_source, field.transpose(-1, -2)
    ).transpose(-1, -2)
    return field


class NearestRegridder(Regridder):
  """Regrid with nearest neighbor interpolation."""

  def __init__(self, source_grid, target_grid, *, device=None,
               dtype=torch.float32):
    super().__init__(source_grid, target_grid, device=device, dtype=dtype)
    self.register_buffer(
        'indices',
        torch.as_tensor(
            nearest_neighbor_indices(source_grid, target_grid),
            dtype=torch.long,
            device=device,
        ),
        persistent=False,
    )

  def forward(self, field: torch.Tensor) -> torch.Tensor:
    if tuple(field.shape[-2:]) != self.source_grid.nodal_shape:
      raise ValueError(
          f'expected {tuple(field.shape[-2:])=} to match '
          f'{self.source_grid.nodal_shape=}'
      )
    batch = field.shape[:-2]
    flat = field.reshape(*batch, -1)[..., self.indices]
    return flat.reshape(*batch, *self.target_grid.nodal_shape)


class ConservativeRegridder(Regridder):
  """Regrid with linear conservative regridding.

  Args:
    skipna: whether to ignore NaN values when interpolating. If True, acts
      like numpy nanmean over neighboring points (NaN only where all
      neighbors are NaN). If False, cells are NaN wherever any neighboring
      point is NaN.
  """

  def __init__(self, source_grid, target_grid, skipna: bool = False, *,
               device=None, dtype=torch.float32):
    super().__init__(source_grid, target_grid, device=device, dtype=dtype)
    self.skipna = skipna
    self._buffer(
        'lon_weights',
        conservative_longitude_weights(
            source_grid.longitudes, target_grid.longitudes
        ),
        dtype,
        device,
    )
    self._buffer(
        'lat_weights',
        conservative_latitude_weights(
            source_grid.latitudes, target_grid.latitudes
        ),
        dtype,
        device,
    )

  def _mean(self, field: torch.Tensor) -> torch.Tensor:
    """Computes cell-averages of field on the target grid."""
    # Note: any NaN in input produces all NaN in output.
    return torch.einsum(
        'ab,cd,...bd->...ac', self.lon_weights, self.lat_weights, field
    )

  def forward(self, field: torch.Tensor) -> torch.Tensor:
    not_nulls = torch.logical_not(torch.isnan(field))
    mean = self._mean(torch.where(not_nulls, field, torch.zeros_like(field)))
    not_null_fraction = self._mean(not_nulls.to(field.dtype))
    if self.skipna:
      return mean / not_null_fraction  # intended NaN if fraction == 0
    return torch.where(
        torch.isclose(
            not_null_fraction,
            torch.ones_like(not_null_fraction),
            rtol=1e-3,
        ),
        mean / not_null_fraction,
        torch.full_like(mean, float('nan')),
    )
