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
"""Forcing modules that produce time-dependent forcing values.

A forcing module is called as `forcing_fn(forcing_data, sim_time)` where
`forcing_data` is a dict of tensors with a leading time axis (plus a
`sim_time` vector) in the units given by `inputs_to_units_mapping`, and
returns the nondimensionalized forcing dict at the requested time.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import torch
from torch import nn

from dinosaur_torch import scales
from dinosaur_torch import units as units_lib


class NoForcing(nn.Module):
  """Module that returns an empty forcing dict."""

  def forward(self, forcing_data, sim_time):
    del forcing_data, sim_time  # unused
    return {}


class DynamicDataForcing(nn.Module):
  """Returns forcing values by querying time-varying data.

  `sim_time` must match a value in `forcing_data['sim_time']` within
  `dt_tolerance` (after nondimensionalization), or all returned values are
  NaN. Nondimensionalization multiplies each key by a scale factor computed
  once from `inputs_to_units_mapping` (linear units only, which covers all
  checkpoint configurations).
  """

  def __init__(
      self,
      physics_specs: units_lib.SimUnits,
      inputs_to_units_mapping: Dict[str, str],
      forcing_transform: Optional[nn.Module] = None,
      time_axis: int = 0,
      dt_tolerance: Union[float, str, scales.Quantity] = '1 hour',
  ):
    super().__init__()
    if time_axis != 0:
      raise NotImplementedError('only time_axis=0 is supported')
    self.forcing_transform = forcing_transform
    self.scale_factors = {
        key: float(
            physics_specs.nondimensionalize(
                1.0 * scales.parse_units(unit_str)
            )
        )
        for key, unit_str in inputs_to_units_mapping.items()
    }
    if isinstance(dt_tolerance, (str, scales.Quantity)):
      dt_tolerance = float(
          physics_specs.nondimensionalize(scales.Quantity(dt_tolerance))
      )
    self.dt_tolerance = dt_tolerance

  def forward(self, forcing_data: dict, sim_time) -> dict:
    """Returns forcings at the specified sim_time."""
    forcing_data = {
        key: value * self.scale_factors[key]
        for key, value in forcing_data.items()
    }

    times = forcing_data['sim_time']
    if not isinstance(sim_time, torch.Tensor):
      sim_time = torch.as_tensor(
          sim_time, dtype=times.dtype, device=times.device
      )
    # nearest-index lookup via linear interpolation of the index vector
    n = times.shape[0]
    if n == 1:
      index = torch.zeros((), dtype=torch.long, device=times.device)
    else:
      u = torch.searchsorted(
          times, sim_time.reshape(1).contiguous(), right=True
      ).clamp(1, n - 1)
      lo, hi = times[u - 1], times[u]
      w = ((sim_time - lo) / (hi - lo)).clamp(0, 1)
      approx_index = (u - 1) + w
      index = torch.round(approx_index).to(torch.long).squeeze(0)

    forcing = {
        key: torch.index_select(value, 0, index.reshape(1)).squeeze(0)
        for key, value in forcing_data.items()
    }

    # Replace values with NaN if the matched time is outside the tolerance.
    abs_error = torch.abs(forcing['sim_time'] - sim_time)
    is_valid = abs_error < self.dt_tolerance
    forcing = {
        key: torch.where(is_valid, value, torch.nan)
        for key, value in forcing.items()
    }
    if self.forcing_transform is not None:
      forcing = self.forcing_transform(forcing)
    return forcing
