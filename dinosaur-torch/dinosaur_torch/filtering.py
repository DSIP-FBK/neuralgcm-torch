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

"""Spectral filters for GCM models.

Filter factories take a `spherical_harmonic.Grid` module and build the
per-wavenumber scaling as a tensor on the grid's device once, up front.
"""
from typing import Callable

import torch

from dinosaur_torch import pytree
from dinosaur_torch import spherical_harmonic


def _preserves_shape(target: torch.Tensor, scaling_shape: tuple) -> bool:
  target_shape = tuple(target.shape)
  if len(scaling_shape) > len(target_shape):
    return False
  return all(
      s == 1 or s == t
      for s, t in zip(scaling_shape, target_shape[-len(scaling_shape):])
  )


def _make_filter_fn(scaling: torch.Tensor) -> Callable:
  """Returns a pytree filter multiplying broadcast-compatible leaves."""
  scaling_shape = tuple(scaling.shape)

  def rescale(x):
    if not _preserves_shape(x, scaling_shape):
      return x
    return scaling * x

  return lambda state: pytree.map_fields(rescale, state)


def exponential_filter(
    grid: spherical_harmonic.Grid,
    attenuation: float | torch.Tensor = 16,
    order: int | torch.Tensor = 18,
    cutoff: float = 0,
) -> Callable:
  """Returns a filter that attenuates modes with high total wavenumber.

  Components with `k > cutoff` are damped by a factor of:

    exp(-attenuation * ((k - cutoff) / (1 - cutoff)) ** (2 * order))

  where `k = total_wavenumber / maximum_total_wavenumber`.

  Args:
    grid: the `spherical_harmonic.Grid` to use for the computation.
    attenuation: controls the steepness of the attenuation above the cutoff
      frequency. Typically attenuation is chosen as -log(epsilon), so the max
      frequency components are multiplied by floating point epsilon. Tensor
      values enable variable filter parameters for different levels/times.
    order: controls the polynomial order of the exponential filter. A higher
      order filter is smoother, and starts attenuating at a higher frequency.
    cutoff: a hard threshold with which to start attenuation, expressed as a
      proportion of maximum total wavenumber.

  Returns:
    A function that accepts a state and returns a filtered state.
  """
  ref = grid.laplacian_eigenvalues  # reference buffer for device/dtype
  _, total_wavenumber = grid.modal_axes
  k = torch.as_tensor(
      total_wavenumber / total_wavenumber.max(),
      dtype=ref.dtype,
      device=ref.device,
  )
  a, c, p = attenuation, cutoff, order
  scaling = torch.exp((k > c) * (-a * (((k - c) / (1 - c)) ** (2 * p))))
  return _make_filter_fn(scaling)


def horizontal_diffusion_filter(
    grid: spherical_harmonic.Grid,
    scale: float | torch.Tensor,
    order: int = 1,
) -> Callable:
  """Returns a filter that applies a horizontal diffusion step."""
  eigenvalues = grid.laplacian_eigenvalues
  scaling = torch.exp(-scale * (-eigenvalues) ** order)
  return _make_filter_fn(scaling)
