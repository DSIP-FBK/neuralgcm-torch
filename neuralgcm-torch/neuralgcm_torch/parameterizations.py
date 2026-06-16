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
"""Physics parameterization modules that compute non-dynamical tendencies.

Only `DivCurlNeuralParameterization` is ported; it is the parameterization
used by all published checkpoint configs.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import torch
from torch import nn

from dinosaur_torch import primitive_equations
from dinosaur_torch import spherical_harmonic
from neuralgcm_torch import filters
from neuralgcm_torch import transforms


def _as_dict(state) -> dict:
  if isinstance(state, dict):
    return state
  return {f.name: getattr(state, f.name) for f in dataclasses.fields(state)}


class DivCurlNeuralParameterization(nn.Module):
  """Computes modal physics tendencies via `u, v` -> `delta, zeta`.

  The neural network predicts nodal tendencies for the masked state fields,
  with velocity tendencies `u, v` in place of `divergence, vorticity`; the
  output is converted to modal space with the velocity pair mapped through
  the divergence/curl operators.

  Args:
    grid: model horizontal grid.
    features: feature module computing nodal network inputs from the modal
      state.
    mapping: a `NodalMapping` whose `output_shapes` cover exactly the masked
      prediction fields in legacy pytree flatten order (sorted keys of the
      state dict, with `u`/`v` in place of `divergence`/`vorticity`).
    tendency_transform: transform applied to the raw nodal predictions.
    prediction_mask: nested dict of bools marking the predicted fields;
      non-predicted fields appear as `None` leaves in the output state.
    filter_module: optional step filter applied to the modal tendencies.
  """

  HAIKU_NAME = 'div_curl_neural_parameterization'

  def __init__(
      self,
      grid: spherical_harmonic.Grid,
      features: nn.Module,
      mapping: nn.Module,
      tendency_transform: Optional[nn.Module],
      prediction_mask: dict,
      filter_module: Optional[nn.Module] = None,
  ):
    super().__init__()
    self.grid = grid
    self.features = features
    self.mapping = mapping
    self.tendency_transform = (
        tendency_transform or transforms.IdentityTransform()
    )
    self.prediction_mask = prediction_mask
    self.filter = filter_module or filters.NoFilter()
    self.register_buffer('_sec_lat', 1 / grid.cos_lat, persistent=False)

  def _to_modal_with_div_curl(self, nodal: dict) -> dict:
    """Converts nodal tendencies to modal, replacing u/v with div/curl."""
    if 'u' not in nodal or 'v' not in nodal:
      raise ValueError(
          f'predictions must include `u, v`, got keys: {nodal.keys()}'
      )
    nodal = dict(nodal)
    # u, v stand for velocity / cos(lat); the cos(lat) factor cancels in the
    # divergence and curl operators below.
    u = self.grid.to_modal(nodal.pop('u') * self._sec_lat)
    v = self.grid.to_modal(nodal.pop('v') * self._sec_lat)
    modal = {
        k: self.grid.to_modal(val) if val is not None else None
        for k, val in nodal.items()
    }
    modal['divergence'] = self.grid.div_cos_lat((u, v))
    modal['vorticity'] = self.grid.curl_cos_lat((u, v))
    return modal

  def _zero_filled(self, tendencies: dict, inputs: dict) -> dict:
    """Returns a full state-field dict with zero tendencies where nothing is
    predicted.

    The original JAX implementation used `None` leaves instead; torch's registered-dataclass
    pytrees move `None` fields into the treespec context, which would make
    the tendency state structurally incompatible with the dycore state when
    they are composed.
    """
    out = {}
    for field, mask in self.prediction_mask.items():
      if field == 'sim_time':
        out[field] = None if inputs[field] is None else 0.0
      elif isinstance(mask, dict):
        out[field] = {
            k: tendencies.get(field, {}).get(k)
            if tendencies.get(field, {}).get(k) is not None
            else torch.zeros_like(inputs[field][k])
            for k in mask
        }
      elif field in tendencies:
        out[field] = tendencies[field]
      else:
        out[field] = torch.zeros_like(inputs[field])
    return out

  def forward(
      self,
      inputs: primitive_equations.State,
      memory=None,
      diagnostics=None,
      randomness=None,
      forcing=None,
  ) -> primitive_equations.State:
    inputs_dict = _as_dict(inputs)
    memory_dict = None if memory is None else _as_dict(memory)
    nodal_inputs = self.features(
        inputs_dict, memory_dict, diagnostics, randomness, forcing
    )
    nodal_tendencies = self.mapping(nodal_inputs)
    nodal_tendencies = self.tendency_transform(nodal_tendencies)
    modal_tendencies = self._zero_filled(
        self._to_modal_with_div_curl(nodal_tendencies), inputs_dict
    )
    modal_tendencies = self.filter(inputs_dict, modal_tendencies)
    return primitive_equations.State(**modal_tendencies)

  def import_haiku(self, params: dict, prefix: str) -> None:
    # the features module is created in the legacy __init__ ('~'); the
    # mapping is created inside the legacy __call__ (no '~').
    if hasattr(self.features, 'import_haiku'):
      self.features.import_haiku(
          params, f'{prefix}/~/{self.features.HAIKU_NAME}'
      )
    self.mapping.import_haiku(params, f'{prefix}/{self.mapping.HAIKU_NAME}')
