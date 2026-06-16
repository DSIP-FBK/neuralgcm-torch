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

"""Routines for regridding between sigma and pressure levels.

Unlike the original JAX implementation (which vectorized a scalar interpolant with nested
`vmap`s), interpolation here is written directly in batched form with
`searchsorted` + `gather` over the level dimension at `dim=-3`.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Sequence, Union

import numpy as np
import torch
from torch import nn

from dinosaur_torch import pytree
from dinosaur_torch import sigma_coordinates


@dataclasses.dataclass(frozen=True)
class PressureCoordinates:
  """Specifies the vertical coordinate with pressure levels.

  Attributes:
    centers: center of each pressure level, starting at the level closest to
      the top of the atmosphere. Must be monotonic increasing.
    layers: number of vertical layers.
  """

  centers: np.ndarray

  def __init__(self, centers: Union[Sequence[float], np.ndarray]):
    object.__setattr__(self, 'centers', np.asarray(centers))
    if not all(np.diff(self.centers) > 0):
      raise ValueError(
          'Expected `centers` to be monotonically increasing, '
          f'got centers = {self.centers}'
      )

  @property
  def layers(self) -> int:
    return len(self.centers)

  def asdict(self) -> Dict[str, Any]:
    return {k: v.tolist() for k, v in dataclasses.asdict(self).items()}

  def __hash__(self):
    return hash(tuple(self.centers.tolist()))

  def __eq__(self, other):
    return isinstance(other, PressureCoordinates) and np.array_equal(
        self.centers, other.centers
    )


def _expand_targets(x: torch.Tensor, spatial: tuple) -> torch.Tensor:
  """Broadcasts 1-D targets `x` to (a,) + spatial."""
  if x.ndim == 1 and spatial:
    x = x.reshape((-1,) + (1,) * len(spatial)).expand((-1,) + spatial)
  return x


def _interp_core(
    x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor, clamp: bool
) -> torch.Tensor:
  """Interpolates along the level axis.

  The level axis is dim 0 for 1-D `fp` and dim -3 otherwise (fields are
  `(..., level, x, y)`); extra leading dims of `x` and `fp` are batch
  dims and broadcast against each other.
  """
  spatial = tuple(fp.shape[-2:]) if fp.ndim > 1 else ()
  x = _expand_targets(x, spatial)
  b = xp.shape[0]
  u = torch.searchsorted(xp, x.contiguous(), right=True).clamp(1, b - 1)
  x_lo, x_hi = xp[u - 1], xp[u]
  w = (x - x_lo) / (x_hi - x_lo)
  if clamp:
    w = w.clamp(0, 1)  # constant extrapolation
  if fp.ndim == 1:
    f_lo, f_hi = fp[u - 1], fp[u]
  else:
    if u.ndim < fp.ndim:
      u = u.expand(fp.shape[:-3] + u.shape)
    elif fp.ndim < u.ndim:
      fp = fp.expand(u.shape[:-3] + fp.shape)
    f_lo, f_hi = torch.gather(fp, -3, u - 1), torch.gather(fp, -3, u)
  return f_lo + w * (f_hi - f_lo)


def interp(
    x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor
) -> torch.Tensor:
  """Like `np.interp` (linear interpolation with constant extrapolation).

  Args:
    x: target coordinates, any shape; if `fp` has spatial dimensions
      (b, *spatial), `x` may be either 1-D of shape (a,) or per-column of
      shape (a, *spatial).
    xp: 1-D increasing source coordinates of shape (b,).
    fp: source values of shape (b,) or (b, *spatial).

  Returns:
    Interpolated values of shape `x.shape` or (a, *spatial).
  """
  return _interp_core(x, xp, fp, clamp=True)


def interp_linear_extrap(
    x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor
) -> torch.Tensor:
  """Linear interpolation with unlimited linear extrapolation at the ends."""
  return _interp_core(x, xp, fp, clamp=False)


vertical_interpolation = interp


def _extrapolate_both(y: torch.Tensor, dim: int = 0) -> torch.Tensor:
  """Extends `y` by one linear cell at each end of `dim`."""
  n = y.shape[dim]
  first, second = y.narrow(dim, 0, 1), y.narrow(dim, 1, 1)
  last, second_last = y.narrow(dim, n - 1, 1), y.narrow(dim, n - 2, 1)
  left = (2 * first - second).clone()
  right = (2 * last - second_last).clone()
  return torch.cat([left, y, right], dim=dim)


def interp_with_safe_extrap(
    x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor, n: int = 1
) -> torch.Tensor:
  """Linear interpolation, extrapolating `n` grid cells; NaN beyond."""
  spatial = tuple(fp.shape[-2:]) if fp.ndim > 1 else ()
  x = _expand_targets(x, spatial)
  for _ in range(n):
    xp = _extrapolate_both(xp)
    fp = _extrapolate_both(fp, dim=-3 if fp.ndim > 1 else 0)
  out = interp(x, xp, fp)
  nan = torch.full_like(out, torch.nan)
  out = torch.where(x < xp[0], nan, out)
  out = torch.where(x > xp[-1], nan, out)
  return out


def get_surface_pressure(
    pressure_levels: torch.Tensor,
    geopotential: torch.Tensor,
    orography: torch.Tensor,
    gravity_acceleration: float,
) -> torch.Tensor:
  """Calculate surface pressure from geopotential on pressure levels.

  Args:
    pressure_levels: tensor of shape [level] with increasing pressure levels.
    geopotential: tensor with dimensions [..., level, x, y].
    orography: tensor with dimensions [1, x, y].
    gravity_acceleration: acceleration due to gravity.

  Returns:
    Tensor with dimensions [..., 1, x, y].
  """
  # note: relative height must be an increasing function along the level
  # axis, which is why we subtract geopotential (which decreases as you get
  # closer to the surface of the Earth).
  relative_height = orography * gravity_acceleration - geopotential
  rh = relative_height.movedim(-3, -1).contiguous()  # [..., x, y, level]
  b = rh.shape[-1]
  zero = torch.zeros(rh.shape[:-1] + (1,), dtype=rh.dtype, device=rh.device)
  u = torch.searchsorted(rh, zero, right=True).clamp(1, b - 1)
  rh_lo = torch.gather(rh, -1, u - 1)
  rh_hi = torch.gather(rh, -1, u)
  w = (zero - rh_lo) / (rh_hi - rh_lo)  # unclamped: linear extrapolation
  p_lo, p_hi = pressure_levels[u - 1], pressure_levels[u]
  out = p_lo + w * (p_hi - p_lo)
  return out.movedim(-1, -3)


def interp_pressure_to_sigma(
    fields,
    pressure_centers: torch.Tensor,
    sigma_centers: torch.Tensor,
    surface_pressure: torch.Tensor,
    extrapolate: str = 'nan',
):
  """Interpolate 3D fields from pressure to sigma levels.

  Fields whose level dimension (dim -3) does not match
  `len(pressure_centers)` pass through unchanged. With
  `extrapolate='nan'` (the default, matching the legacy default), values
  more than one grid cell outside the source levels become NaN; with
  `'constant'` they are held at the boundary values.
  """
  desired = sigma_centers[:, None, None] * surface_pressure
  if extrapolate == 'nan':
    interp_fn = interp_with_safe_extrap
  elif extrapolate == 'constant':
    interp_fn = interp
  else:
    raise ValueError(f'unknown {extrapolate=}')

  def regrid(x):
    if x.ndim < 3 or x.shape[-3] != pressure_centers.shape[0]:
      return x
    return interp_fn(desired, pressure_centers, x)

  return pytree.map_fields(regrid, fields)


def interp_sigma_to_pressure(
    fields,
    pressure_centers: torch.Tensor,
    sigma_centers: torch.Tensor,
    surface_pressure: torch.Tensor,
    extrapolate: str = 'nan',
):
  """Interpolate 3D fields from sigma to pressure levels.

  `extrapolate` is one of 'nan' (NaN beyond one cell, the legacy default),
  'constant', or 'linear'.
  """
  desired = pressure_centers[:, None, None] / surface_pressure
  interp_fn = {
      'nan': interp_with_safe_extrap,
      'constant': interp,
      'linear': interp_linear_extrap,
  }[extrapolate]
  return pytree.map_fields(
      lambda x: interp_fn(desired, sigma_centers, x), fields
  )


def interp_centers_to_centers(
    fields, source_centers: torch.Tensor, target_centers: torch.Tensor
):
  """Interpolate 3D fields between center coordinates (sigma or pressure).

  Uses constant extrapolation outside the source levels; fields with fewer
  than 3 dimensions pass through unchanged.
  """

  def regrid(x):
    if x.ndim < 3:
      return x
    return interp(target_centers, source_centers, x)

  return pytree.map_fields(regrid, fields)


interp_sigma_to_sigma = interp_centers_to_centers
interp_pressure_to_pressure = interp_centers_to_centers


class PressureLevels(nn.Module):
  """Pressure-level regridding for a `PressureCoordinates` on a device.

  Bundles the level-center buffer with the standard conversions to and from
  sigma coordinates.
  """

  def __init__(
      self,
      coordinates: PressureCoordinates,
      *,
      device: torch.device | str | None = None,
      dtype: torch.dtype = torch.float32,
  ):
    super().__init__()
    self.coordinates = coordinates
    self.register_buffer(
        'centers',
        torch.as_tensor(
            np.asarray(coordinates.centers, np.float64),
            dtype=dtype,
            device=device,
        ),
        persistent=False,
    )

  @property
  def layers(self) -> int:
    return self.coordinates.layers

  def to_sigma(
      self,
      fields,
      sigma: sigma_coordinates.SigmaLevels,
      surface_pressure: torch.Tensor,
  ):
    """Interpolates fields from pressure levels to sigma levels."""
    return interp_pressure_to_sigma(
        fields, self.centers, sigma.centers, surface_pressure
    )

  def from_sigma(
      self,
      fields,
      sigma: sigma_coordinates.SigmaLevels,
      surface_pressure: torch.Tensor,
  ):
    """Interpolates fields from sigma levels to pressure levels."""
    return interp_sigma_to_pressure(
        fields, self.centers, sigma.centers, surface_pressure
    )

  def get_surface_pressure(
      self,
      geopotential: torch.Tensor,
      orography: torch.Tensor,
      gravity_acceleration: float,
  ) -> torch.Tensor:
    return get_surface_pressure(
        self.centers, geopotential, orography, gravity_acceleration
    )
