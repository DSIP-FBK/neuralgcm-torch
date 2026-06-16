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

"""A vertical coordinate system based on normalized pressure.

See https://en.wikipedia.org/wiki/Sigma_coordinate_system

`SigmaCoordinates` is the static (NumPy, hashable) description of the levels;
`SigmaLevels` is the `nn.Module` holding the per-level constants as buffers
and implementing the vertical finite-difference / integral operators.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Callable

import numpy as np
import torch
from torch import nn


def _with_f64_math(
    f: Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
  """Returns a function that uses float64 internally."""
  return lambda x: f(x.astype(np.float64)).astype(x.dtype)


@dataclasses.dataclass(frozen=True)
class SigmaCoordinates:
  """A static description of a discrete vertical coordinate system.

  Layers are indexed from the "top" of the atmosphere (𝜎 = 0) to the surface
  of the earth (𝜎 = 1).

  Attributes:
    boundaries: sigma values of the boundaries of horizontal layers. Must be
      an increasing array of values beginning with zero and ending with one.
      For `n` layers, `boundaries` has length `n + 1`.
    internal_boundaries: sigma values of the boundaries _between_ layers. For
      `n` layers, `internal_boundaries` has length `n - 1`.
    centers: sigma values of the centers of horizontal layers. For `n`
      layers, `centers` has length `n`
    layer_thickness: the thickness of each layer in sigma coordinates. For
      `n` layers, `layer_thickness` has length `n`.
    center_to_center: the distances between the centers of each layer in
      sigma coordinates. For `n` layers, `center_to_center` has length
      `n - 1`.
    layers: the number of layers.
  """

  boundaries: np.ndarray

  def __init__(self, boundaries):
    boundaries = np.asarray(boundaries)
    if not (np.isclose(boundaries[0], 0) and np.isclose(boundaries[-1], 1)):
      raise ValueError(
          'Expected boundaries[0] = 0, boundaries[-1] = 1, '
          f'got boundaries = {boundaries}'
      )
    if not all(np.diff(boundaries) > 0):
      raise ValueError(
          'Expected `boundaries` to be monotonically increasing, '
          f'got boundaries = {boundaries}'
      )
    object.__setattr__(self, 'boundaries', boundaries)

  @functools.cached_property
  def internal_boundaries(self) -> np.ndarray:
    return self.boundaries[1:-1]

  @functools.cached_property
  def centers(self) -> np.ndarray:
    # Use float64 internally so we can convert float32 boundaries to centers
    # and back without any loss of precision.
    return _with_f64_math(lambda x: (x[1:] + x[:-1]) / 2)(self.boundaries)

  @functools.cached_property
  def layer_thickness(self) -> np.ndarray:
    return _with_f64_math(np.diff)(self.boundaries)

  @functools.cached_property
  def center_to_center(self) -> np.ndarray:
    return _with_f64_math(np.diff)(self.centers)

  @property
  def layers(self) -> int:
    return len(self.boundaries) - 1

  @classmethod
  def equidistant(cls, layers: int, dtype=np.float32) -> SigmaCoordinates:
    boundaries = np.linspace(0, 1, layers + 1, dtype=dtype)
    return cls(boundaries)

  @classmethod
  def from_centers(cls, centers) -> SigmaCoordinates:
    """Create sigma coordinates from the centers of each layer."""
    # The relationship between cell centers and boundaries is given by:
    #   centers[i] = 0.5 * (boundaries[i] + boundaries[i + 1])
    # Writing this as a matrix and dropping the column corresponding to
    # boundaries[0] (fixed at zero), we have a linear system of N equations
    # and N unknowns that we can solve to obtain cell boundaries.

    def centers_to_boundaries(centers):
      layers = len(centers)
      bounds_to_centers = 0.5 * (np.eye(layers) + np.eye(layers, k=-1))
      unpadded_bounds = np.linalg.solve(bounds_to_centers, centers)
      return np.pad(unpadded_bounds, [(1, 0)])

    boundaries = _with_f64_math(centers_to_boundaries)(centers)
    return cls(boundaries)

  def asdict(self):
    return {k: v.tolist() for k, v in dataclasses.asdict(self).items()}

  def __hash__(self):
    return hash(tuple(self.centers.tolist()))

  def __eq__(self, other):
    return isinstance(other, SigmaCoordinates) and np.array_equal(
        self.centers, other.centers
    )


def _per_level(x: torch.Tensor, weights: torch.Tensor, dim: int):
  """Multiplies `x` by a per-level 1-D `weights` along `dim`."""
  shape = [1] * x.ndim
  shape[dim % x.ndim] = -1
  return x * weights.view(shape)


def _zeros_slice(x: torch.Tensor, dim: int) -> torch.Tensor:
  return torch.zeros_like(x.narrow(dim % x.ndim, 0, 1))


def _reverse_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
  """out[i] = sum_{j >= i} x[j] along `dim`."""
  return torch.flip(torch.cumsum(torch.flip(x, [dim]), dim), [dim])


# For consistency with commonly accepted notation, we use Greek letters
# within some of the functions below.
# pylint: disable=invalid-name


class SigmaLevels(nn.Module):
  """Vertical operators for a `SigmaCoordinates` on a fixed device/dtype.

  All per-level constants are precomputed non-persistent buffers. Operators
  default to `dim=-3`, the conventional (level, longitude, latitude) layout.
  """

  def __init__(
      self,
      coordinates: SigmaCoordinates,
      *,
      device: torch.device | str | None = None,
      dtype: torch.dtype = torch.float32,
  ):
    super().__init__()
    self.coordinates = coordinates

    def buffer(name, array):
      self.register_buffer(
          name,
          torch.as_tensor(np.asarray(array, np.float64), dtype=dtype,
                          device=device),
          persistent=False,
      )

    buffer('boundaries', coordinates.boundaries)
    buffer('centers', coordinates.centers)
    buffer('layer_thickness', coordinates.layer_thickness)
    buffer('inv_center_to_center', 1 / coordinates.center_to_center)
    log_centers = np.log(coordinates.centers.astype(np.float64))
    buffer('dlog_sigma', np.diff(log_centers, append=0))

  @property
  def layers(self) -> int:
    return self.coordinates.layers

  def _check_layers(self, x: torch.Tensor, dim: int):
    if x.shape[dim] != self.layers:
      raise ValueError(
          '`x.shape[dim]` must be equal to `coordinates.layers`; '
          f'got {x.shape[dim]} and {self.layers}.'
      )

  def centered_difference(self, x: torch.Tensor, dim: int = -3):
    """Derivative of `x` with respect to `sigma` along `dim`.

    The derivative is approximated as

      (∂x / ∂𝜎)[n + ½] ≈ (x[n + 1] - x[n]) / (𝜎[n + 1] - 𝜎[n])

    so the result is located on the internal boundaries between layers and
    has one fewer entries than `x` along `dim`.
    """
    self._check_layers(x, dim)
    d = dim % x.ndim
    dx = x.narrow(d, 1, x.shape[d] - 1) - x.narrow(d, 0, x.shape[d] - 1)
    return _per_level(dx, self.inv_center_to_center, dim)

  def sigma_integral(
      self, x: torch.Tensor, dim: int = -3, keepdims: bool = True
  ) -> torch.Tensor:
    """The full integral of `x` with respect to 𝜎 (midpoint rule)."""
    self._check_layers(x, dim)
    xd𝜎 = _per_level(x, self.layer_thickness, dim)
    return xd𝜎.sum(dim=dim, keepdim=keepdims)

  def cumulative_sigma_integral(
      self, x: torch.Tensor, dim: int = -3, downward: bool = True
  ) -> torch.Tensor:
    """The cumulative integral of `x` with respect to 𝜎 (midpoint rule).

    Integrates from the top of the atmosphere down to each layer's lower
    boundary when `downward` is True, and up from the surface (𝜎 = 1)
    otherwise.
    """
    self._check_layers(x, dim)
    xd𝜎 = _per_level(x, self.layer_thickness, dim)
    if downward:
      return torch.cumsum(xd𝜎, dim=dim % x.ndim)
    return _reverse_cumsum(xd𝜎, dim % x.ndim)

  def cumulative_log_sigma_integral(
      self, x: torch.Tensor, dim: int = -3, downward: bool = True
  ) -> torch.Tensor:
    """The cumulative integral of `x` with respect to log(𝜎).

    Uses the trapezoid rule between layer centers; between the surface
    (𝜎 = 1) and the center of the last layer a constant value of `x[-1]` is
    assumed.
    """
    self._check_layers(x, dim)
    d = dim % x.ndim
    n = x.shape[d]
    x_last = x.narrow(d, n - 1, 1)
    x_interpolated = (x.narrow(d, 1, n - 1) + x.narrow(d, 0, n - 1)) / 2
    integrand = torch.cat([x_interpolated, x_last], dim=d)
    xd𝜎 = _per_level(integrand, self.dlog_sigma, dim)
    if downward:
      return torch.cumsum(xd𝜎, dim=d)
    return _reverse_cumsum(xd𝜎, d)

  def centered_vertical_advection(
      self,
      w: torch.Tensor,
      x: torch.Tensor,
      dim: int = -3,
      w_boundary_values: tuple[torch.Tensor, torch.Tensor] | None = None,
      dx_dsigma_boundary_values: (
          tuple[torch.Tensor, torch.Tensor] | None
      ) = None,
  ) -> torch.Tensor:
    """Vertical advection `-(w ∂x/∂𝜎)` using 2nd order finite differences.

    Computes the expression at layer centers via the averaging approximation

      -(w ∂x/∂𝜎)[n] ≈ -½ (w[n+½] (∂x/∂𝜎)[n+½] + w[n-½] (∂x/∂𝜎)[n-½])

    `w` is given at `coordinates.internal_boundaries`; boundary values of
    both `w` and `∂x/∂𝜎` default to zero.

    `w` and `x` may have different ranks (e.g. a batched velocity against
    a static reference profile) as long as they broadcast; `dim` must be
    negative so the level axis is found in both.
    """
    if dim >= 0:
      raise ValueError(f'level axis must be counted from the end, got {dim=}')
    if w_boundary_values is None:
      w_boundary_values = (_zeros_slice(w, dim), _zeros_slice(w, dim))
    if dx_dsigma_boundary_values is None:
      dx_dsigma_boundary_values = (_zeros_slice(x, dim), _zeros_slice(x, dim))

    w_top, w_bot = w_boundary_values
    w = torch.cat([w_top, w, w_bot], dim=dim)

    x_diff = self.centered_difference(x, dim)
    x_diff_top, x_diff_bot = dx_dsigma_boundary_values
    x_diff = torch.cat([x_diff_top, x_diff, x_diff_bot], dim=dim)

    w_times_x_diff = w * x_diff
    n = w_times_x_diff.shape[dim]
    return -0.5 * (
        w_times_x_diff.narrow(dim, 1, n - 1)
        + w_times_x_diff.narrow(dim, 0, n - 1)
    )

  def upwind_vertical_advection(
      self, w: torch.Tensor, x: torch.Tensor, dim: int = -3
  ) -> torch.Tensor:
    """Vertical advection `-(w ∂x/∂𝜎)` using 1st order upwinding."""
    if dim >= 0:
      raise ValueError(f'level axis must be counted from the end, got {dim=}')
    w_zeros = _zeros_slice(w, dim)
    x_zeros = _zeros_slice(x, dim)

    # https://en.wikipedia.org/wiki/Upwind_scheme#Compact_form
    x_diff = self.centered_difference(x, dim)

    w_up = torch.cat([w_zeros, w], dim=dim)
    w_down = torch.cat([w, w_zeros], dim=dim)
    x_diff_up = torch.cat([x_zeros, x_diff], dim=dim)
    x_diff_down = torch.cat([x_diff, x_zeros], dim=dim)
    # tendency (i.e. r.h.s. has a negative sign).
    return -(
        torch.clamp(w_up, min=0) * x_diff_up
        + torch.clamp(w_down, max=0) * x_diff_down
    )


# pylint: enable=invalid-name
