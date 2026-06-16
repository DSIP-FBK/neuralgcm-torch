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
"""Decoders that map model states to WeatherBench-style outputs.

Decoders are called as `decoder(inputs, forcing=None)` where `inputs` is a
`model_states.ModelState`, and return a dict of fields on pressure levels.

Only the decoders referenced by published checkpoint configs are ported.
Perturbation and diagnostics modules are not: both checkpoints use the
no-op defaults.
"""

from __future__ import annotations

import dataclasses
import zlib
from typing import Optional

import torch
from torch import nn

from dinosaur_torch import coordinate_systems
from dinosaur_torch import primitive_equations
from dinosaur_torch import vertical_interpolation
from neuralgcm_torch import model_states
from neuralgcm_torch import stochastic
from neuralgcm_torch import transforms

ModelState = model_states.ModelState
WeatherbenchState = model_states.WeatherbenchState

_DECODER_SALT = zlib.crc32(b'decoder')  # arbitrary uint32 value


class DecoderIdentityTransform(nn.Module):
  """Returns inputs without modification."""

  def forward(self, inputs):
    return inputs


class PrimitiveToWeatherbenchDecoder(nn.Module):
  """Converts a primitive-equations state to WeatherBench outputs.

  Args:
    coords: model coordinate system (sigma levels).
    output_coords: data coordinate system; `output_coords.vertical` must be
      a `vertical_interpolation.PressureLevels` with nondimensional centers.
    ref_temperatures: reference temperature per model level, shape [layers].
    orography_module: module returning the modal orography on the model
      grid.
    geopotential: a `primitive_equations.Geopotential` module for the model
      sigma levels (with water-vapor effects).
    transform: optional transform applied to the final output dict.
  """

  HAIKU_NAME = 'primitive_to_weatherbench_decoder'

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      output_coords,
      ref_temperatures,
      orography_module: nn.Module,
      geopotential: primitive_equations.Geopotential,
      transform: Optional[nn.Module] = None,
  ):
    super().__init__()
    self.coords = coords
    self.output_coords = output_coords
    self.orography_module = orography_module
    self.geopotential = geopotential
    self.transform = transform or DecoderIdentityTransform()
    ref = coords.horizontal.cos_lat
    self.register_buffer(
        'ref_temps',
        torch.as_tensor(
            ref_temperatures, dtype=ref.dtype, device=ref.device
        )[:, None, None],
        persistent=False,
    )

  def primitive_to_weatherbench(
      self, inputs: primitive_equations.State
  ) -> WeatherbenchState:
    """Converts a modal PE state to a WB state on pressure levels."""
    # as in the legacy decoder: output-grid transforms for the state (the
    # grids are identical in published checkpoints), model grid for the
    # orography.
    to_nodal = self.output_coords.horizontal.to_nodal
    u, v = self.output_coords.horizontal.vor_div_to_uv_nodal(
        inputs.vorticity, inputs.divergence
    )
    t = to_nodal(inputs.temperature_variation) + self.ref_temps
    tracers = to_nodal(inputs.tracers)
    nodal_orography = self.coords.horizontal.to_nodal(self.orography_module())
    z = self.geopotential(
        t, nodal_orography, tracers['specific_humidity']
    )
    surface_pressure = torch.exp(to_nodal(inputs.log_surface_pressure))

    pressure_centers = self.output_coords.vertical.centers
    sigma_centers = self.coords.vertical.centers
    regrid = lambda tree, mode: (
        vertical_interpolation.interp_sigma_to_pressure(
            tree, pressure_centers, sigma_centers, surface_pressure,
            extrapolate=mode,
        )
    )
    # regridding choices follow the legacy decoder: constant extrapolation
    # for u, v and tracers; linear extrapolation for z and t.
    return WeatherbenchState(
        u=regrid(u, 'constant'),
        v=regrid(v, 'constant'),
        t=regrid(t, 'linear'),
        z=regrid(z, 'linear'),
        sim_time=inputs.sim_time,
        tracers=regrid(tracers, 'constant'),
    )

  def forward(self, inputs: ModelState, forcing=None) -> dict:
    del forcing  # unused
    wb_state = self.primitive_to_weatherbench(inputs.state)
    out = {
        f.name: getattr(wb_state, f.name)
        for f in dataclasses.fields(wb_state)
        if f.name != 'diagnostics'
    }
    return self.transform(out)


class LearnedPrimitiveToWeatherbenchDecoder(PrimitiveToWeatherbenchDecoder):
  """Decoder with learned corrections on the pressure-level outputs.

  Args:
    model_features: features computed on the model grid from the modal
      primitive-equations state.
    data_features: features computed on the output grid from the modal WB
      state.
    mapping: a `NodalMapping` whose output_shapes cover exactly the masked
      prediction fields, inserted in legacy flatten order — the WB output
      is a plain dict, so SORTED key order (sim_time, t, tracers, u, v, z
      for the masked subset).
    correction_transform: transform applied to the raw nodal corrections.
    prediction_mask: dict of bools marking which outputs get corrections.
    randomness_module: random field sampled at decoding time.
    diagnostics_module: optional module called as `module(inputs, None)`;
      its output dict is added to the outputs under `diagnostics` (after
      the modal features are computed, before corrections and the final
      transform — matching the legacy decoder).
  """

  HAIKU_NAME = 'learned_primitive_to_weatherbench_decoder'

  def __init__(
      self,
      coords,
      output_coords,
      ref_temperatures,
      orography_module,
      geopotential,
      model_features: nn.Module,
      data_features: nn.Module,
      mapping: nn.Module,
      correction_transform: Optional[nn.Module],
      prediction_mask: dict,
      transform: Optional[nn.Module] = None,
      randomness_module: Optional[nn.Module] = None,
      diagnostics_module: Optional[nn.Module] = None,
  ):
    super().__init__(
        coords,
        output_coords,
        ref_temperatures,
        orography_module,
        geopotential,
        transform=transform,
    )
    self.model_features = model_features
    self.data_features = data_features
    self.mapping = mapping
    self.correction_transform = (
        correction_transform or transforms.IdentityTransform()
    )
    self.prediction_mask = prediction_mask
    self.randomness_module = randomness_module
    self.diagnostics_module = diagnostics_module

  def _add_corrections(self, outputs: dict, corrections: dict) -> dict:
    out = {}
    for k, v in outputs.items():
      if isinstance(v, dict):
        out[k] = self._add_corrections(v, corrections.get(k, {}))
      elif k in corrections:
        out[k] = v + corrections[k]
      else:
        out[k] = v
    return out

  def forward(self, inputs: ModelState, forcing=None) -> dict:
    if self.randomness_module is not None:
      salt = _DECODER_SALT + int(inputs.randomness.prng_step)
      prng_key = inputs.randomness.prng_key
      if isinstance(prng_key, (tuple, list)):
        # member-batched state: draw each member's decoder noise from the
        # key its sequential decode would use, then stack
        from neuralgcm_torch import ensembles

        randomness = ensembles._stack_randomness([  # pylint: disable=protected-access
            self.randomness_module.unconditional_sample(
                stochastic.fold_in(key, salt)
            )
            for key in prng_key
        ])
      else:
        randomness = self.randomness_module.unconditional_sample(
            stochastic.fold_in(prng_key, salt)
        )
    else:
      randomness = stochastic.RandomnessState()

    prognostics = inputs.state
    wb_state = self.primitive_to_weatherbench(prognostics)
    wb_dict = {
        f.name: getattr(wb_state, f.name)
        for f in dataclasses.fields(wb_state)
        if f.name != 'diagnostics'
    }
    wb_modal = self.output_coords.maybe_to_modal(wb_dict)
    if self.diagnostics_module is not None:
      # diagnostics enter the outputs (and the final transform) but not the
      # modal features computed above, matching the legacy decoder.
      wb_dict['diagnostics'] = self.diagnostics_module(inputs, None)

    def as_dict(state):
      return {
          f.name: getattr(state, f.name)
          for f in dataclasses.fields(state)
      }

    model_features = self.model_features(
        as_dict(prognostics),
        forcing=forcing,
        randomness=randomness.nodal_value,
    )
    data_features = self.data_features(wb_modal, forcing=forcing)
    all_features = {f'data_{k}': v for k, v in data_features.items()}
    all_features |= {f'model_{k}': v for k, v in model_features.items()}

    nodal_outputs = self.correction_transform(self.mapping(all_features))
    return self.transform(self._add_corrections(wb_dict, nodal_outputs))

  def import_haiku(self, params: dict, prefix: str) -> None:
    if hasattr(self.orography_module, 'import_haiku'):
      self.orography_module.import_haiku(
          params, f'{prefix}/~/{self.orography_module.HAIKU_NAME}'
      )
    if hasattr(self.model_features, 'import_haiku'):
      self.model_features.import_haiku(
          params, f'{prefix}/~/{self.model_features.HAIKU_NAME}'
      )
    if hasattr(self.data_features, 'import_haiku'):
      n = 1 if (
          self.data_features.HAIKU_NAME == self.model_features.HAIKU_NAME
      ) else 0
      name = self.data_features.HAIKU_NAME
      suffix = name if n == 0 else f'{name}_{n}'
      self.data_features.import_haiku(params, f'{prefix}/~/{suffix}')
    self.mapping.import_haiku(params, f'{prefix}/{self.mapping.HAIKU_NAME}')
    if self.randomness_module is not None and hasattr(
        self.randomness_module, 'import_haiku'
    ):
      self.randomness_module.import_haiku(
          params, f'{prefix}/~/{self.randomness_module.HAIKU_NAME}'
      )


class DimensionalLearnedPrimitiveToWeatherbenchDecoder(
    LearnedPrimitiveToWeatherbenchDecoder
):
  """Learned decoder producing dimensional (SI) outputs."""

  HAIKU_NAME = 'dimensional_learned_primitive_to_weatherbench_decoder'

  def __init__(self, *args, physics_specs, inputs_to_units_mapping, **kwargs):
    super().__init__(*args, **kwargs)
    self.redim_transform = transforms.RedimensionalizeTransform(
        physics_specs, inputs_to_units_mapping
    )

  def forward(self, inputs: ModelState, forcing=None) -> dict:
    return self.redim_transform(super().forward(inputs, forcing))
