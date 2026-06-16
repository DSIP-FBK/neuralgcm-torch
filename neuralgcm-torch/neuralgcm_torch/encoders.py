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
"""Encoders that map WeatherBench-style inputs to model states.

Encoders are called as `encoder(inputs, forcing, rng=None)` where `inputs`
is a dict of arrays with a leading time axis (already nondimensionalized
except for the `Dimensional*` variants) and return a
`model_states.ModelState`.

Only the encoders referenced by published checkpoint configs are ported
(the primitive-equations / WeatherBench path; shallow-water and leapfrog
encoders are research-only upstream). Perturbation modules are not ported:
both checkpoints use the default no-op perturbation.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Sequence

import torch
from torch import nn

from dinosaur_torch import coordinate_systems
from dinosaur_torch import primitive_equations
from dinosaur_torch import pytree
from dinosaur_torch import vertical_interpolation
from neuralgcm_torch import model_states
from neuralgcm_torch import stochastic
from neuralgcm_torch import transforms

ModelState = model_states.ModelState
WeatherbenchState = model_states.WeatherbenchState


def slice_time(inputs: dict, idx: int = -1) -> dict:
  """Selects time index `idx` (axis 0) from every tensor leaf."""
  return pytree.map_fields(lambda x: x[idx], inputs)


#  ===========================================================================
#  Encoder input transforms.
#  ===========================================================================


class EncoderIdentityTransform(nn.Module):
  """Returns inputs without modification."""

  def forward(self, inputs):
    return inputs


class InputClipTransform(nn.Module):
  """Clips the highest total wavenumbers of the (modal) input state."""

  def __init__(self, input_grid, wavenumbers_to_clip: int = 1):
    super().__init__()
    self.input_grid = input_grid
    self.wavenumbers_to_clip = wavenumbers_to_clip

  def forward(self, inputs):
    return self.input_grid.clip_wavenumbers(inputs, self.wavenumbers_to_clip)


class EncoderFilterTransform(nn.Module):
  """Applies data filters to the (modal) input state in order."""

  def __init__(self, filters: Sequence[nn.Module]):
    super().__init__()
    self.filters = nn.ModuleList(filters)

  def forward(self, inputs):
    for filter_module in self.filters:
      inputs = filter_module(inputs)
    return inputs


class EncoderCombinedTransform(nn.Module):
  """Applies multiple encoder transforms sequentially."""

  def __init__(self, transforms_seq: Sequence[nn.Module]):
    super().__init__()
    self.transforms = nn.ModuleList(transforms_seq)

  def forward(self, inputs):
    for transform in self.transforms:
      inputs = transform(inputs)
    return inputs


#  ===========================================================================
#  WeatherBench -> primitive equations encoders.
#  ===========================================================================


class WeatherbenchToPrimitiveEncoder(nn.Module):
  """Extracts a primitive-equations state from WeatherBench inputs.

  Args:
    coords: model coordinate system (sigma levels).
    input_coords: data coordinate system; `input_coords.vertical` must be a
      `vertical_interpolation.PressureLevels` with nondimensional centers.
    ref_temperatures: reference temperature per model level, shape [layers].
    orography_module: module returning the modal orography on the model
      grid (its output is interpolated to the input grid for the surface
      pressure solve).
    gravity_acceleration: nondimensional g.
    transform: optional transform applied to the final state.
  """

  HAIKU_NAME = 'weatherbench_to_primitive_encoder'

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      input_coords,
      ref_temperatures,
      orography_module: nn.Module,
      gravity_acceleration: float,
      transform: Optional[nn.Module] = None,
      time_axis: int = 0,
  ):
    super().__init__()
    if time_axis != 0:
      raise NotImplementedError('only time_axis=0 is supported')
    self.coords = coords
    self.input_coords = input_coords
    self.orography_module = orography_module
    self.gravity_acceleration = gravity_acceleration
    self.transform = transform or EncoderIdentityTransform()
    ref = coords.horizontal.cos_lat
    self.register_buffer(
        'ref_temps',
        torch.as_tensor(
            ref_temperatures, dtype=ref.dtype, device=ref.device
        )[:, None, None],
        persistent=False,
    )

  def _input_nodal_orography(self) -> torch.Tensor:
    """The model orography interpolated onto the input grid (nodal)."""
    from neuralgcm_torch import orographies  # avoid import cycle

    modal = self.orography_module()
    modal = orographies._interpolate_modal(
        modal, self.coords.horizontal, self.input_coords.horizontal
    )
    return self.input_coords.horizontal.to_nodal(modal)

  def weatherbench_to_primitive(
      self, wb_state_nodal: WeatherbenchState
  ) -> primitive_equations.State:
    """Converts a nodal WB state on pressure levels to primitive on sigma."""
    # Note: the returned values have mixed nodal/modal representations.
    pressure_levels = self.input_coords.vertical
    surface_pressure = vertical_interpolation.get_surface_pressure(
        pressure_levels.centers,
        wb_state_nodal.z,
        self._input_nodal_orography(),
        self.gravity_acceleration,
    )
    regrid = lambda tree: vertical_interpolation.interp_pressure_to_sigma(
        tree,
        pressure_levels.centers,
        self.coords.vertical.centers,
        surface_pressure,
        extrapolate='constant',
    )
    wb_on_sigma = regrid(wb_state_nodal)
    vorticity, divergence = (
        self.input_coords.horizontal.uv_nodal_to_vor_div_modal(
            wb_on_sigma.u, wb_on_sigma.v
        )
    )
    return primitive_equations.State(
        vorticity=vorticity,
        divergence=divergence,
        temperature_variation=wb_on_sigma.t - self.ref_temps,
        log_surface_pressure=torch.log(surface_pressure),
        tracers=wb_on_sigma.tracers,
        sim_time=wb_on_sigma.sim_time,
    )

  def _interpolate_to_model(self, state):
    """Interpolates a modal state from the input grid to the model grid."""
    from neuralgcm_torch import orographies  # avoid import cycle

    return pytree.map_fields(
        lambda x: orographies._interpolate_modal(
            x, self.input_coords.horizontal, self.coords.horizontal
        ),
        state,
    )

  def forward(self, inputs: dict, forcing=None, rng=None) -> ModelState:
    del forcing, rng  # unused
    wb_state = WeatherbenchState(**slice_time(inputs))
    wb_state = self.input_coords.maybe_to_nodal(wb_state)
    pe_state = self.weatherbench_to_primitive(wb_state)
    pe_state = self.input_coords.maybe_to_modal(pe_state)
    pe_state = self._interpolate_to_model(pe_state)
    return ModelState(state=self.transform(pe_state))


class LearnedWeatherbenchToPrimitiveEncoder(WeatherbenchToPrimitiveEncoder):
  """WeatherBench encoder with learned corrections.

  Args:
    data_features: features computed on the input grid from the modal WB
      state.
    model_features: features computed on the model grid from the modal
      primitive-equations state.
    mapping: a `NodalMapping` whose output_shapes cover exactly the masked
      prediction fields (nested `tracers` included).
    correction_transform: transform applied to the raw nodal corrections.
    prediction_mask: dict of bools marking which state fields receive
      corrections.
    randomness_module: random field sampled at encoding time.
  """

  HAIKU_NAME = 'learned_weatherbench_to_primitive_encoder'

  def __init__(
      self,
      coords,
      input_coords,
      ref_temperatures,
      orography_module,
      gravity_acceleration,
      data_features: nn.Module,
      model_features: nn.Module,
      mapping: nn.Module,
      correction_transform: Optional[nn.Module],
      prediction_mask: dict,
      transform: Optional[nn.Module] = None,
      randomness_module: Optional[nn.Module] = None,
      time_axis: int = 0,
  ):
    super().__init__(
        coords,
        input_coords,
        ref_temperatures,
        orography_module,
        gravity_acceleration,
        transform=transform,
        time_axis=time_axis,
    )
    self.data_features = data_features
    self.model_features = model_features
    self.mapping = mapping
    self.correction_transform = (
        correction_transform or transforms.IdentityTransform()
    )
    self.prediction_mask = prediction_mask
    self.randomness_module = randomness_module

  def _add_corrections(self, state_dict: dict, corrections: dict) -> dict:
    out = {}
    for k, v in state_dict.items():
      if isinstance(v, dict):
        out[k] = self._add_corrections(v, corrections.get(k, {}))
      elif k in corrections:
        out[k] = v + corrections[k]
      else:
        out[k] = v
    return out

  def forward(self, inputs: dict, forcing=None, rng=None) -> ModelState:
    if self.randomness_module is not None:
      randomness = self.randomness_module.unconditional_sample(
          rng if rng is not None else 0
      )
    else:
      randomness = stochastic.RandomnessState()
    wb_state = WeatherbenchState(**slice_time(inputs))
    wb_state_nodal = self.input_coords.maybe_to_nodal(wb_state)
    wb_state_modal = self.input_coords.maybe_to_modal(wb_state)
    pe_state = self.weatherbench_to_primitive(wb_state_nodal)
    pe_state_modal = self._interpolate_to_model(
        self.input_coords.maybe_to_modal(pe_state)
    )
    pe_state_nodal = self.coords.maybe_to_nodal(pe_state_modal)

    def as_dict(state):
      return {
          f.name: getattr(state, f.name)
          for f in dataclasses.fields(state)
      }

    data_features = self.data_features(
        as_dict(wb_state_modal), forcing=forcing
    )
    model_features = self.model_features(
        as_dict(pe_state_modal),
        forcing=forcing,
        randomness=randomness.nodal_value,
    )
    all_features = {f'data_{k}': v for k, v in data_features.items()}
    all_features |= {f'model_{k}': v for k, v in model_features.items()}

    nodal_corrections = self.correction_transform(
        self.mapping(all_features)
    )
    modal_corrections = self.coords.maybe_to_modal(nodal_corrections)
    corrected = self._add_corrections(
        as_dict(self.coords.maybe_to_modal(pe_state_nodal)),
        modal_corrections,
    )
    state = primitive_equations.State(**corrected)
    return ModelState(
        state=self.transform(state), randomness=randomness
    )

  def import_haiku(self, params: dict, prefix: str) -> None:
    # feature modules are created in the legacy __init__; the mapping is
    # created inside the legacy __call__ (no '~').
    if hasattr(self.orography_module, 'import_haiku'):
      self.orography_module.import_haiku(
          params, f'{prefix}/~/{self.orography_module.HAIKU_NAME}'
      )
    if hasattr(self.data_features, 'import_haiku'):
      self.data_features.import_haiku(
          params, f'{prefix}/~/{self.data_features.HAIKU_NAME}'
      )
    if hasattr(self.model_features, 'import_haiku'):
      counts = 1 if (
          self.model_features.HAIKU_NAME == self.data_features.HAIKU_NAME
      ) else 0
      name = self.model_features.HAIKU_NAME
      suffix = name if counts == 0 else f'{name}_{counts}'
      self.model_features.import_haiku(params, f'{prefix}/~/{suffix}')
    self.mapping.import_haiku(params, f'{prefix}/{self.mapping.HAIKU_NAME}')
    if self.randomness_module is not None and hasattr(
        self.randomness_module, 'import_haiku'
    ):
      self.randomness_module.import_haiku(
          params, f'{prefix}/~/{self.randomness_module.HAIKU_NAME}'
      )


class DimensionalLearnedWeatherbenchToPrimitiveEncoder(
    LearnedWeatherbenchToPrimitiveEncoder
):
  """Learned WeatherBench encoder accepting dimensional (SI) inputs."""

  HAIKU_NAME = 'dimensional_learned_weatherbench_to_primitive_encoder'

  def __init__(self, *args, physics_specs, inputs_to_units_mapping, **kwargs):
    super().__init__(*args, **kwargs)
    self.nondim_transform = transforms.NondimensionalizeTransform(
        physics_specs, inputs_to_units_mapping
    )

  def forward(self, inputs: dict, forcing=None, rng=None) -> ModelState:
    return super().forward(self.nondim_transform(inputs), forcing, rng)


class DimensionalLearnedWeatherbenchToPrimitiveWithMemoryEncoder(nn.Module):
  """Dimensional learned encoder that also encodes a memory state.

  Holds two `LearnedWeatherbenchToPrimitiveEncoder`s: a deterministic one
  for the memory and a (possibly stochastic) one for the state.
  """

  HAIKU_NAME = (
      'dimensional_learned_weatherbench_to_primitive_with_memory_encoder'
  )

  def __init__(
      self,
      memory_encoder: LearnedWeatherbenchToPrimitiveEncoder,
      state_encoder: LearnedWeatherbenchToPrimitiveEncoder,
      physics_specs,
      inputs_to_units_mapping: Dict[str, str],
  ):
    super().__init__()
    self.memory_encoder = memory_encoder
    self.state_encoder = state_encoder
    self.nondim_transform = transforms.NondimensionalizeTransform(
        physics_specs, inputs_to_units_mapping
    )

  def forward(self, inputs: dict, forcing=None, rng=None) -> ModelState:
    nondim_inputs = self.nondim_transform(inputs)
    memory = self.memory_encoder(nondim_inputs, forcing)
    model_state = self.state_encoder(nondim_inputs, forcing, rng)
    return ModelState(
        state=model_state.state,
        memory=memory.state,
        randomness=model_state.randomness,
    )

  def import_haiku(self, params: dict, prefix: str) -> None:
    # both children share the auto name with haiku _N numbering, memory
    # encoder first (legacy construction order).
    child = 'learned_weatherbench_to_primitive_encoder'
    self.memory_encoder.import_haiku(params, f'{prefix}/~/{child}')
    self.state_encoder.import_haiku(params, f'{prefix}/~/{child}_1')
