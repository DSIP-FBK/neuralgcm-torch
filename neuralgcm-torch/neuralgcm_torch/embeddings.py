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
"""Modules that predict an embedding from the model state.

Embeddings share the feature-module call signature
`forward(state, memory=None, diagnostics=None, randomness=None,
forcing=None)` and return dictionaries of nodal fields.

Only the embeddings referenced by published checkpoint configs are ported:
`ModalToNodalEmbedding` (features -> nodal mapping -> transform) and
`NodalLandSeaIceEmbedding` (separate land/sea/sea-ice embeddings blended by
surface-type fractions).
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import torch
from torch import nn

from neuralgcm_torch import transforms

# forcing dictionary key (matches dinosaur xarray_utils.SEA_ICE_COVER)
SEA_ICE_COVER = 'sea_ice_cover'


def _as_dict(state) -> dict:
  if isinstance(state, dict):
    return state
  if dataclasses.is_dataclass(state):
    # shallow conversion: keep tensor leaves as-is
    return {
        f.name: getattr(state, f.name) for f in dataclasses.fields(state)
    }
  raise TypeError(f'unsupported state type {type(state)}')


class ModalToNodalEmbedding(nn.Module):
  """Embedding that expects modal state input and returns nodal output."""

  HAIKU_NAME = 'modal_to_nodal_embedding'

  def __init__(
      self,
      features_module: nn.Module,
      mapping: nn.Module,
      output_transform: Optional[nn.Module] = None,
  ):
    super().__init__()
    self.features_module = features_module
    self.mapping = mapping
    self.output_transform = output_transform or transforms.IdentityTransform()

  def forward(self, state, memory=None, diagnostics=None, randomness=None,
              forcing=None):
    """Returns the embedding output on nodal locations."""
    nodal_inputs = self.features_module(
        _as_dict(state), memory, diagnostics, randomness, forcing
    )
    return self.output_transform(self.mapping(nodal_inputs))

  def import_haiku(self, params: dict, prefix: str) -> None:
    # the features module is created in the legacy __init__ ('~'); the
    # mapping is created inside the legacy __call__ (no '~').
    if hasattr(self.features_module, 'import_haiku'):
      self.features_module.import_haiku(
          params, f'{prefix}/~/{self.features_module.HAIKU_NAME}'
      )
    self.mapping.import_haiku(params, f'{prefix}/{self.mapping.HAIKU_NAME}')


class UniformParameterEmbedding(nn.Module):
  """Learned constants broadcast over the surface grid.

  The legacy `NodalLandSeaIceEmbedding` falls back to this when one of its
  sub-embeddings is not configured: a `(output_size, 1, 1)` parameter
  vector (named `<param_name>_params` in the parent's haiku bundle) is
  broadcast over the nodal grid and unpacked to `output_shapes`.
  """

  def __init__(
      self,
      output_shapes: dict,
      param_name: str,
      *,
      device=None,
      dtype: torch.dtype = torch.float32,
  ):
    super().__init__()
    self.output_shapes = {k: tuple(v) for k, v in output_shapes.items()}
    self.param_name = param_name
    output_size = sum(s[-3] for s in self.output_shapes.values())
    self.params = nn.Parameter(
        torch.zeros(output_size, 1, 1, device=device, dtype=dtype)
    )

  def forward(self, state, memory=None, diagnostics=None, randomness=None,
              forcing=None):
    del state, memory, diagnostics, randomness, forcing  # unused
    outputs = {}
    offset = 0
    # split in sorted-key order, matching the legacy (jax pytree) unpack
    for key in sorted(self.output_shapes):
      shape = self.output_shapes[key]
      channels = shape[-3]
      outputs[key] = self.params[offset:offset + channels].expand(shape)
      offset += channels
    return outputs

  def import_haiku(self, params: dict, prefix: str) -> None:
    # the parameters live directly in the parent embedding's bundle.
    source = params[prefix][f'{self.param_name}_params']
    with torch.no_grad():
      self.params.copy_(source.reshape(self.params.shape))


class NodalLandSeaIceEmbedding(nn.Module):
  """Embedding representing a nodal land/sea/sea-ice surface.

  Outputs of the three sub-embeddings are blended with weights derived from
  the static land/sea mask and the `sea_ice_cover` forcing.
  """

  HAIKU_NAME = 'nodal_land_sea_ice_embedding'

  def __init__(
      self,
      land_embedding: nn.Module,
      sea_embedding: nn.Module,
      sea_ice_embedding: nn.Module,
      land_sea_mask: np.ndarray,
      output_transform: Optional[nn.Module] = None,
      *,
      device=None,
      dtype: torch.dtype = torch.float32,
  ):
    super().__init__()
    self.land_embedding = land_embedding
    self.sea_embedding = sea_embedding
    self.sea_ice_embedding = sea_ice_embedding
    self.output_transform = output_transform or transforms.IdentityTransform()
    self.register_buffer(
        'land_fraction',
        torch.as_tensor(
            np.asarray(land_sea_mask), dtype=dtype, device=device
        ),
        persistent=False,
    )

  def forward(self, state, memory=None, diagnostics=None, randomness=None,
              forcing=None):
    """Returns the embedding output on nodal locations."""
    args = (state, memory, diagnostics, randomness, forcing)
    land_outputs = self.land_embedding(*args)
    sea_outputs = self.sea_embedding(*args)
    sea_ice_outputs = self.sea_ice_embedding(*args)

    # masks with fractional values in [0, 1]
    land_fraction = self.land_fraction
    sea_fraction = 1 - land_fraction
    sea_ice_fraction = forcing[SEA_ICE_COVER]

    land_weight = land_fraction
    sea_ice_weight = sea_ice_fraction * sea_fraction  # ice covered sea
    sea_weight = (1 - sea_ice_fraction) * sea_fraction  # sea without ice

    surface_outputs = {
        k: (
            land_weight * land_outputs[k]
            + sea_weight * sea_outputs[k]
            + sea_ice_weight * sea_ice_outputs[k]
        )
        for k in land_outputs
    }
    return self.output_transform(surface_outputs)

  def import_haiku(self, params: dict, prefix: str) -> None:
    # children are created in the legacy __init__ in land/sea/sea-ice order,
    # with haiku's per-parent _N numbering for repeated class names. Slots
    # configured as uniform parameters create no child module in the legacy
    # code (their parameters live in this module's own bundle), so they do
    # not participate in the numbering.
    counts: dict[str, int] = {}
    for child in (
        self.land_embedding,
        self.sea_embedding,
        self.sea_ice_embedding,
    ):
      if isinstance(child, UniformParameterEmbedding):
        child.import_haiku(params, prefix)
        continue
      name = child.HAIKU_NAME
      n = counts.get(name, 0)
      counts[name] = n + 1
      if hasattr(child, 'import_haiku'):
        suffix = name if n == 0 else f'{name}_{n}'
        child.import_haiku(params, f'{prefix}/~/{suffix}')
