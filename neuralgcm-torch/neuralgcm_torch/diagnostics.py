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
"""Diagnostics modules computing extra outputs from the model state.

A step diagnostics module is called as `module(model_state,
physics_tendencies, forcing)` after each advance substep (and once at
encoding time) and returns a dict stored on `ModelState.diagnostics`; a
decoder diagnostics module has the same signature and returns the dict
merged into the decoded outputs.

The v1 weather checkpoints use no diagnostics (the `diagnostics` field
stays an empty dict; `SurfacePressureDiagnostics` is opt-in via checkpoint
config modifications — see the checkpoint_modifications notebook). The
v1_precip checkpoints use `PrecipitationDiagnosticsConstrained` to predict
precipitation and evaporation.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from dinosaur_torch import coordinate_systems
from dinosaur_torch import scales
from dinosaur_torch import units as units_lib


class SurfacePressureDiagnostics(nn.Module):
  """Returns the nodal surface pressure of the (modal) model state."""

  def __init__(self, coords: coordinate_systems.CoordinateSystem):
    super().__init__()
    self.coords = coords

  def forward(self, model_state, physics_tendencies, forcing=None) -> dict:
    del physics_tendencies, forcing  # unused
    lsp = model_state.state.log_surface_pressure
    surface_pressure = torch.squeeze(
        torch.exp(self.coords.horizontal.to_nodal(lsp)), 0
    )
    return {'surface_pressure': surface_pressure}


class NodalModelDiagnosticsDecoder(nn.Module):
  """Decoder diagnostics returning `model_state.diagnostics` (nodal)."""

  def __init__(self, coords: coordinate_systems.CoordinateSystem):
    super().__init__()
    self.coords = coords

  def forward(self, model_state, physics_tendencies, forcing=None) -> dict:
    del physics_tendencies, forcing  # unused
    return self.coords.maybe_to_nodal(model_state.diagnostics)


PRECIPITATION = 'precipitation'
EVAPORATION = 'evaporation'


class PrecipitationDiagnosticsConstrained(nn.Module):
  """Predicts one of precipitation/evaporation; diagnoses the other.

  The total moisture budget `E - P` is computed by vertically integrating
  the physics moisture tendencies; a learned embedding predicts one of the
  two terms (`is_precipitation` selects which) and the other follows from
  the constraint. Precipitation is reported in length units (cumulative,
  under `field_name`) or as a rate; evaporation in `kg m**-2 s**-1` (or
  cumulative length units), following the legacy module.
  """

  HAIKU_NAME = 'precipitation_diagnostics_constrained'

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: units_lib.SimUnits,
      embedding: nn.Module,
      moisture_species: Sequence[str] = (
          'specific_humidity',
          'specific_cloud_ice_water_content',
          'specific_cloud_liquid_water_content',
      ),
      is_precipitation: bool = True,
      method_precipitation: str = 'cumulative',
      method_evaporation: str = 'rate',
      field_name: str = 'total_precipitation',
  ):
    super().__init__()
    if method_precipitation not in ('rate', 'cumulative'):
      raise ValueError(f'unknown {method_precipitation=}')
    if method_evaporation not in ('rate', 'cumulative'):
      raise ValueError(f'unknown {method_evaporation=}')
    self.coords = coords
    self.dt = dt
    self.physics_specs = physics_specs
    self.embedding = embedding
    self.moisture_species = tuple(moisture_species)
    self.is_precipitation = is_precipitation
    self.method_precipitation = method_precipitation
    self.method_evaporation = method_evaporation
    self.field_name = field_name
    self.predicted_name = PRECIPITATION if is_precipitation else EVAPORATION
    self.diagnosed_name = EVAPORATION if is_precipitation else PRECIPITATION
    self.water_density = float(
        physics_specs.nondimensionalize(scales.WATER_DENSITY)
    )

  def _evaporation_minus_precipitation(
      self, model_state, physics_tendencies
  ) -> torch.Tensor:
    to_nodal = self.coords.horizontal.to_nodal
    lsp = model_state.state.log_surface_pressure
    p_surface = torch.squeeze(torch.exp(to_nodal(lsp)), 0)
    moisture_tendencies = sum(
        to_nodal(v)
        for tracer, v in physics_tendencies.tracers.items()
        if tracer in self.moisture_species
    )
    scale = p_surface / self.physics_specs.g
    return scale * self.coords.vertical.sigma_integral(
        moisture_tendencies, keepdims=False
    )

  def _previous(self, model_state, key: str, like: torch.Tensor):
    previous = model_state.diagnostics.get(key)
    if previous is None:
      return torch.zeros(
          self.coords.horizontal.nodal_shape,
          dtype=like.dtype, device=like.device,
      )
    return previous

  def forward(self, model_state, physics_tendencies, forcing=None) -> dict:
    e_minus_p = self._evaporation_minus_precipitation(
        model_state, physics_tendencies
    )
    water_budget = dict(
        self.embedding(
            model_state.state,
            model_state.memory,
            model_state.diagnostics,
            model_state.randomness.nodal_value,
            forcing,
        )
    )
    water_budget[self.diagnosed_name] = (
        -e_minus_p - water_budget[self.predicted_name]
    )

    # sign conventions follow the legacy module: `e_minus_p` is positive
    # for evaporation; precipitation is positive.
    outputs = {}
    precipitation = water_budget[PRECIPITATION]
    if self.method_precipitation == 'rate':  # units: length / time
      outputs[PRECIPITATION + '_rate'] = precipitation / self.water_density
    else:  # cumulative; units: length
      previous = self._previous(model_state, self.field_name, precipitation)
      outputs[self.field_name] = previous + (
          precipitation / self.water_density * self.dt
      )
    evaporation = water_budget[EVAPORATION]
    if self.method_evaporation == 'rate':  # units: mass length^-2 time^-1
      outputs[EVAPORATION] = evaporation
    else:  # cumulative; units: length
      previous = self._previous(
          model_state, EVAPORATION + '_cumulative', evaporation
      )
      outputs[EVAPORATION + '_cumulative'] = previous + (
          evaporation / self.water_density * self.dt
      )
    return outputs

  def import_haiku(self, params: dict, prefix: str) -> None:
    self.embedding.import_haiku(
        params, f'{prefix}/~/{self.embedding.HAIKU_NAME}'
    )
