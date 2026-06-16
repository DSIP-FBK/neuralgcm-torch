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

"""The converted-checkpoint format for neuralgcm-torch.

Original NeuralGCM checkpoints are pickles holding a gin config string plus
a haiku parameter tree. This package does not depend on gin or implement
haiku semantics; instead, checkpoints are converted **once**, offline, from
the original JAX/haiku checkpoint into a plain dictionary saved with
`torch.save`:

  {
    'format_version': 1,
    'config': {
        'model_grid': {...GridSpec fields..., 'spherical_harmonics': 'real'|'fast'},
        'data_grid': {...} | None,
        'model_sigma_boundaries': [float, ...],
        'data_pressure_levels': [float, ...] | None,
        'dt': float,                      # nondimensional model time step
        'timestep_seconds': float,
        'reference_datetime': str,        # ISO format
        'physics': {...nondimensional constants for units.SimUnits...},
        'scale_si': {'length_m', 'time_s', 'mass_kg', 'temperature_K'},
        'input_variables': [...],
        'forcing_variables': [...],
        'tracer_variables': [...],
        'gin_config_str': str,            # original config, for reference only
        'model': {                        # parsed gin bindings as plain data
            'scope/ClassName': {param: value | {'__ref__': name,
                                                '__call__': bool}, ...},
            ...,                          # macros inlined; see model_builder
        },
        'data': {
            'orography_input_grid': {...grid fields...},
            'covariate_units': {name: unit_str},
        },
    },
    'aux_features': {name: np.ndarray, ...},   # e.g. nodal_orography_m,
                                               # land_sea_mask, covariate_*
    'params': {bundle_path: {param_name: cpu tensor, ...}, ...},
  }

Use `model_builder.from_checkpoint` to construct a ready-to-run model.

`params` preserves the original haiku bundle paths verbatim; mapping them
onto this package's module tree is the responsibility of the model loaders,
next to the module definitions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from dinosaur_torch import scales
from dinosaur_torch import sigma_coordinates
from dinosaur_torch import spherical_harmonic
from dinosaur_torch import units
from dinosaur_torch import vertical_interpolation

FORMAT_VERSION = 1

_SPHERICAL_HARMONICS_IMPLS = {
    'real': spherical_harmonic.RealSphericalHarmonics,
    'fast': spherical_harmonic.FastSphericalHarmonics,
}


def save(checkpoint: dict[str, Any], path) -> None:
  """Saves a converted checkpoint."""
  if checkpoint.get('format_version') != FORMAT_VERSION:
    raise ValueError(f'expected format_version {FORMAT_VERSION}')
  torch.save(checkpoint, path)


def load(path) -> dict[str, Any]:
  """Loads a converted checkpoint (see the module docstring for the format)."""
  checkpoint = torch.load(path, map_location='cpu', weights_only=False)
  version = checkpoint.get('format_version')
  if version != FORMAT_VERSION:
    raise ValueError(
        f'unsupported checkpoint format_version {version}; '
        f'expected {FORMAT_VERSION}'
    )
  return checkpoint


def grid_spec_from_config(grid_config: dict) -> spherical_harmonic.GridSpec:
  """Builds a `GridSpec` from the checkpoint's grid description."""
  fields = {
      k: v for k, v in grid_config.items() if k != 'spherical_harmonics'
  }
  return spherical_harmonic.GridSpec(**fields)


def spherical_harmonics_impl_from_config(grid_config: dict):
  """Returns the transform class named by the checkpoint's grid config."""
  return _SPHERICAL_HARMONICS_IMPLS[grid_config['spherical_harmonics']]


def sigma_coordinates_from_config(config: dict) -> (
    sigma_coordinates.SigmaCoordinates):
  return sigma_coordinates.SigmaCoordinates(
      np.asarray(config['model_sigma_boundaries'])
  )


def pressure_coordinates_from_config(config: dict) -> (
    vertical_interpolation.PressureCoordinates | None):
  levels = config.get('data_pressure_levels')
  if levels is None:
    return None
  return vertical_interpolation.PressureCoordinates(np.asarray(levels))


def sim_units_from_config(config: dict) -> units.SimUnits:
  """Reconstructs `SimUnits` with the checkpoint's nondimensionalization."""
  scale_si = config['scale_si']
  scale = scales.Scale(
      scale_si['length_m'] * scales.units.m,
      scale_si['time_s'] * scales.units.s,
      scale_si['mass_kg'] * scales.units.kilogram,
      scale_si['temperature_K'] * scales.units.degK,
  )
  physics = config['physics']
  return units.SimUnits(scale=scale, **physics)


def num_params(checkpoint: dict[str, Any]) -> int:
  """Total number of scalar parameters in the checkpoint."""
  return sum(
      v.numel()
      for bundle in checkpoint['params'].values()
      for v in bundle.values()
  )
