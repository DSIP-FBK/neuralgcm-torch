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
"""Filters that improve the stability of time integration.

Step filters are modules called as `filter(u, u_next) -> u_next`; data
filters act on modal fields without time-step context. Constructors take a
`spherical_harmonic.Grid` (and nondimensional scalars) directly. `tau`
arguments accept either a nondimensional float or a string/pint quantity
together with `physics_specs` for conversion.

Only the filters referenced by published checkpoint configs are ported
(plus `FixGlobalMeanFilter`, used by checkpoint modifications).
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Sequence, Union

from torch import nn

from dinosaur_torch import filtering
from dinosaur_torch import scales
from dinosaur_torch import spherical_harmonic
from dinosaur_torch import time_integration
from dinosaur_torch import units


def _nondimensionalize_tau(
    tau: Union[float, str, scales.Quantity],
    physics_specs: Optional[units.SimUnits],
) -> float:
  if isinstance(tau, (int, float)):
    return float(tau)
  if physics_specs is None:
    raise ValueError(f'physics_specs required to nondimensionalize {tau!r}')
  return float(physics_specs.nondimensionalize(scales.Quantity(tau)))


class NoFilter(nn.Module):
  """Step filter that performs no filtering."""

  def forward(self, u, u_next):
    del u  # unused
    return u_next


class ClipFilter(nn.Module):
  """Step filter that clips the highest total wavenumbers of u_next."""

  def __init__(self, grid: spherical_harmonic.Grid,
               wavenumbers_to_clip: int = 1):
    super().__init__()
    self.grid = grid
    self.wavenumbers_to_clip = wavenumbers_to_clip

  def forward(self, u, u_next):
    del u  # unused
    return self.grid.clip_wavenumbers(u_next, self.wavenumbers_to_clip)


class ExponentialFilter(nn.Module):
  """Step filter removing high-frequency components from a spectral state.

  See `dinosaur_torch.time_integration.exponential_step_filter`.
  """

  def __init__(
      self,
      grid: spherical_harmonic.Grid,
      dt: float,
      tau: Union[float, str, scales.Quantity] = '0.010938',
      order: int = 18,
      cutoff: float = 0,
      *,
      physics_specs: Optional[units.SimUnits] = None,
  ):
    super().__init__()
    tau = _nondimensionalize_tau(tau, physics_specs)
    self.filter_fn = time_integration.exponential_step_filter(
        grid, dt, tau, order, cutoff
    )

  def forward(self, u, u_next):
    return self.filter_fn(u, u_next)


class HorizontalDiffusionFilter(nn.Module):
  """Step filter applying an implicit diffusion operator to u_next."""

  def __init__(
      self,
      grid: spherical_harmonic.Grid,
      dt: float,
      tau: Union[float, str, scales.Quantity],
      order: int = 1,
      *,
      physics_specs: Optional[units.SimUnits] = None,
  ):
    super().__init__()
    tau = _nondimensionalize_tau(tau, physics_specs)
    self.filter_fn = time_integration.horizontal_diffusion_step_filter(
        grid, dt, tau, order
    )

  def forward(self, u, u_next):
    return self.filter_fn(u, u_next)


class SequentialStepFilter(nn.Module):
  """Combines multiple step filters applied sequentially."""

  def __init__(self, filters: Sequence[nn.Module]):
    super().__init__()
    self.filters = nn.ModuleList(filters)

  def forward(self, u, u_next):
    for filter_module in self.filters:
      u_next = filter_module(u, u_next)
    return u_next


class FixGlobalMeanFilter(nn.Module):
  """Removes the change in the global mean of certain (modal) keys.

  Works on dicts and on dataclass states (e.g.
  `dinosaur_torch.primitive_equations.State`).
  """

  def __init__(self, keys: tuple[str, ...] = ('log_surface_pressure',)):
    super().__init__()
    self.keys = keys

  def forward(self, u, u_next):
    is_dataclass = dataclasses.is_dataclass(u_next)
    get = (lambda s, k: getattr(s, k)) if is_dataclass else (
        lambda s, k: s[k])

    replacements = {}
    for key in self.keys:
      global_mean = get(u, key)[..., 0]
      fixed = get(u_next, key).clone()
      fixed[..., 0] = global_mean
      replacements[key] = fixed

    if is_dataclass:
      return dataclasses.replace(u_next, **replacements)
    return {**u_next, **replacements}


#  ===========================================================================
#  Filters that act on modal variables without time-step context.
#  ===========================================================================


class DataNoFilter(nn.Module):
  """Data filter that performs no filtering."""

  def forward(self, inputs):
    return inputs


class DataExponentialFilter(nn.Module):
  """Removes high-frequency components from modal data.

  See `dinosaur_torch.filtering.exponential_filter`.
  """

  def __init__(
      self,
      grid: spherical_harmonic.Grid,
      attenuation: float = 16,
      order: int = 18,
      cutoff: float = 0,
  ):
    super().__init__()
    self.filter_fn = filtering.exponential_filter(
        grid, attenuation, order, cutoff
    )

  def forward(self, inputs):
    return self.filter_fn(inputs)


class PerVariableDataFilter(nn.Module):
  """Applies a different data filter to each state field.

  `filters` mirrors the (possibly nested dict) structure of the filtered
  state; fields without an entry — and `None` leaves — pass through
  unchanged.
  """

  def __init__(self, filters: dict):
    super().__init__()

    def to_module_dict(tree):
      return nn.ModuleDict({
          k: to_module_dict(v) if isinstance(v, dict) else v
          for k, v in tree.items()
      })

    self.filters = to_module_dict(filters)

  def _apply(self, values: dict, filters) -> dict:
    out = {}
    for k, v in values.items():
      filter_module = filters[k] if k in filters else None
      if isinstance(v, dict):
        out[k] = self._apply(v, filter_module or {})
      elif filter_module is None or v is None:
        out[k] = v
      else:
        out[k] = filter_module(v)
    return out

  def forward(self, inputs):
    if dataclasses.is_dataclass(inputs):
      values = {
          f.name: getattr(inputs, f.name)
          for f in dataclasses.fields(inputs)
      }
      return dataclasses.replace(inputs, **self._apply(values, self.filters))
    return self._apply(inputs, self.filters)
