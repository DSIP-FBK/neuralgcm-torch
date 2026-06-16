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
"""Modules that compute state features consumed by ML components.

Feature modules share the call signature
`forward(inputs, memory=None, diagnostics=None, randomness=None,
forcing=None) -> dict[str, Tensor]`. `inputs` is the modal model state as a
dictionary (with `tracers` and `sim_time` entries).

Only the feature modules referenced by published checkpoint configs are
ported. `LearnedPositionalFeatures` is the only one holding parameters of
its own; `Embedding*Features` delegate to an embedding module supplied at
construction.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn

from dinosaur_torch import sigma_coordinates
from dinosaur_torch import spherical_harmonic
from dinosaur_torch import radiation as radiation_lib
from neuralgcm_torch import transforms

KeyWithCosLatFactor = transforms.KeyWithCosLatFactor


class FeatureModule(nn.Module):
  """Base class fixing the feature-module call signature."""

  def forward(
      self,
      inputs: Optional[dict] = None,
      memory: Optional[dict] = None,
      diagnostics: Optional[dict] = None,
      randomness=None,
      forcing: Optional[dict] = None,
  ) -> Dict[str, torch.Tensor]:
    raise NotImplementedError


class NullFeatures(FeatureModule):
  """Placeholder features module that returns an empty dict."""

  HAIKU_NAME = 'null_features'

  def forward(self, inputs=None, memory=None, diagnostics=None,
              randomness=None, forcing=None):
    return {}


class VelocityAndPrognostics(FeatureModule):
  """Returns prognostics + u,v (from vorticity/divergence) and gradients.

  All features are returned in nodal space with cos-lat factors removed.
  """

  HAIKU_NAME = 'velocity_and_prognostics'

  def __init__(
      self,
      grid: spherical_harmonic.Grid,
      fields_to_include: Optional[Sequence[str]] = None,
      features_transform: Optional[nn.Module] = None,
      compute_gradients: Optional[nn.Module] = None,
  ):
    super().__init__()
    self.grid = grid
    self.fields_to_include = fields_to_include
    self.features_transform = features_transform or transforms.IdentityTransform()
    self.compute_gradients = compute_gradients or transforms.EmptyTransform()
    # precomputed reciprocal (legacy multiplies by 1/cos_lat, not divides)
    self.register_buffer(
        '_sec_lat', 1 / grid.cos_lat, persistent=False
    )

  def _sec_lat_scale(self, factor_order: int):
    if factor_order == 0:
      return None
    if factor_order == 1:
      return self._sec_lat
    if factor_order == 2:
      return self.grid.sec2_lat
    raise ValueError(f'unsupported {factor_order=}')

  def _extract_features(self, inputs: dict, prefix: str = '') -> dict:
    """Returns nodal velocity and prognostic features."""
    # Note: intermediate features carry explicit cos-lat factors in the key;
    # the factors are removed before returning.
    if {'vorticity', 'divergence'}.issubset(inputs.keys()) and not (
        {'u', 'v'} & set(inputs.keys())
    ):
      cos_lat_u, cos_lat_v = self.grid.cos_lat_vector(
          inputs['vorticity'], inputs['divergence']
      )
      modal_features = {
          KeyWithCosLatFactor(prefix + 'u', 1): cos_lat_u,
          KeyWithCosLatFactor(prefix + 'v', 1): cos_lat_v,
      }
    else:
      modal_features = {}
    prognostics_keys = [
        k for k in inputs.keys() if k not in ('tracers', 'sim_time')
    ]
    for k in prognostics_keys:
      if self.fields_to_include is None or k in self.fields_to_include:
        modal_features[KeyWithCosLatFactor(prefix + k, 0)] = inputs[k]
    for k, v in inputs.get('tracers', {}).items():
      if self.fields_to_include is None or k in self.fields_to_include:
        modal_features[KeyWithCosLatFactor(prefix + k, 0)] = v

    diff_operator_features = self.compute_gradients(modal_features)
    features = {}
    for k, v in (diff_operator_features | modal_features).items():
      scale = self._sec_lat_scale(k.factor_order)
      nodal = self.grid.to_nodal(v)
      if scale is not None:
        nodal = nodal * scale
      features[k.name] = nodal
    return features

  def forward(self, inputs=None, memory=None, diagnostics=None,
              randomness=None, forcing=None):
    del memory, diagnostics, randomness, forcing  # unused
    return self.features_transform(self._extract_features(inputs))


class MemoryVelocityAndValues(VelocityAndPrognostics):
  """Like `VelocityAndPrognostics`, but operates on the memory state.

  Features are prefixed with 'memory_'.
  """

  HAIKU_NAME = 'memory_velocity_and_values'

  def forward(self, inputs=None, memory=None, diagnostics=None,
              randomness=None, forcing=None):
    del inputs, diagnostics, randomness, forcing  # unused
    return self.features_transform(
        self._extract_features(memory, 'memory_')
    )


class RadiationFeatures(FeatureModule):
  """Computes the incident (normalized) solar radiation flux."""

  HAIKU_NAME = 'radiation_features'

  def __init__(
      self,
      solar_radiation: radiation_lib.SolarRadiation,
      features_transform: Optional[nn.Module] = None,
  ):
    super().__init__()
    self.solar_radiation = solar_radiation
    self.features_transform = features_transform or transforms.IdentityTransform()

  def forward(self, inputs=None, memory=None, diagnostics=None,
              randomness=None, forcing=None):
    del memory, diagnostics, randomness, forcing  # unused
    flux = self.solar_radiation.radiation_flux(inputs['sim_time'])
    return self.features_transform({'radiation': flux[None]})


class ForcingFeatures(FeatureModule):
  """Provides forcing values as features."""

  HAIKU_NAME = 'forcing_features'

  def __init__(
      self,
      forcing_to_include: Sequence[str] = tuple(),
      features_transform: Optional[nn.Module] = None,
  ):
    super().__init__()
    self.forcing_to_include = tuple(forcing_to_include)
    self.features_transform = features_transform or transforms.IdentityTransform()

  def forward(self, inputs=None, memory=None, diagnostics=None,
              randomness=None, forcing=None):
    del inputs, memory, diagnostics, randomness  # unused
    features = {}
    for key in self.forcing_to_include:
      value = forcing[key]
      # Expect singleton "level" dimension for surface forcings.
      if value.ndim > 3:
        raise ValueError(
            f'Expected forcing "{key}" to have ndim <= 3, got {value.ndim}'
        )
      if value.ndim == 2:
        value = value[None]
      if value.shape[0] != 1:
        raise ValueError(
            f'Expected forcing "{key}" to have leading dimension 1 '
            f'for level, got {value.shape}'
        )
      features[key] = value
    return self.features_transform(features)


class LatitudeFeatures(FeatureModule):
  """Provides cos and sin of latitude as features."""

  HAIKU_NAME = 'latitude_features'

  def __init__(
      self,
      grid: spherical_harmonic.Grid,
      features_transform: Optional[nn.Module] = None,
  ):
    super().__init__()
    self.features_transform = features_transform or transforms.IdentityTransform()
    ref = grid.cos_lat
    _, sin_lat = grid.spec.nodal_mesh
    sin_features = sin_lat[np.newaxis, ...]
    self.register_buffer(
        'sin_latitude',
        torch.as_tensor(sin_features, dtype=ref.dtype, device=ref.device),
        persistent=False,
    )
    self.register_buffer(
        'cos_latitude',
        torch.as_tensor(
            np.cos(np.arcsin(sin_features)), dtype=ref.dtype, device=ref.device
        ),
        persistent=False,
    )

  def forward(self, inputs=None, memory=None, diagnostics=None,
              randomness=None, forcing=None):
    del inputs, memory, diagnostics, randomness, forcing  # unused
    return self.features_transform({
        'cos_latitude': self.cos_latitude,
        'sin_latitude': self.sin_latitude,
    })


class RandomnessFeatures(FeatureModule):
  """Returns fields from `randomness` as features."""

  HAIKU_NAME = 'randomness_features'

  def __init__(self, features_transform: Optional[nn.Module] = None):
    super().__init__()
    self.features_transform = features_transform or transforms.IdentityTransform()

  def forward(self, inputs=None, memory=None, diagnostics=None,
              randomness=None, forcing=None):
    del inputs, memory, diagnostics, forcing  # unused
    if randomness is None:
      random_features = {}
    elif isinstance(randomness, dict):
      random_features = transforms._flatten_keys(randomness)
    elif isinstance(randomness, torch.Tensor):
      random_features = {'randomness': randomness}
    else:
      raise ValueError(f'randomness has unsupported {type(randomness)=}.')
    # random fields are 2D by construction; add a feature/level dimension.
    # ndim 4 is a member-batched (member, fields, lon, lat) value.
    def make_3d(x):
      if x.ndim in (3, 4):
        return x
      if x.ndim == 2:
        return x[None]
      raise ValueError(f'Random fields expected 2D or 3D, got {x.ndim=}')

    random_features = {k: make_3d(v) for k, v in random_features.items()}
    return self.features_transform(random_features)


class PressureFeatures(FeatureModule):
  """Computes the nodal pressure (sigma x surface pressure)."""

  HAIKU_NAME = 'pressure_features'

  def __init__(
      self,
      grid: spherical_harmonic.Grid,
      vertical: sigma_coordinates.SigmaLevels,
      features_transform: Optional[nn.Module] = None,
  ):
    super().__init__()
    self.grid = grid
    self.vertical = vertical
    self.features_transform = features_transform or transforms.IdentityTransform()

  def _nodal_pressure(self, inputs: dict, prefix: str = '') -> dict:
    surface_pressure = torch.exp(
        self.grid.to_nodal(inputs['log_surface_pressure'])
    )
    pressure = surface_pressure * self.vertical.centers[:, None, None]
    return {prefix + 'pressure': pressure}

  def forward(self, inputs=None, memory=None, diagnostics=None,
              randomness=None, forcing=None):
    del memory, diagnostics, randomness, forcing  # unused
    return self.features_transform(self._nodal_pressure(inputs))


class LearnedPositionalFeatures(FeatureModule):
  """Feature module with learned parameters at surface nodal locations."""

  HAIKU_NAME = 'learned_positional_features'

  def __init__(
      self,
      latent_size: int,
      nodal_shape: tuple[int, int],
      scale: float = 1.0,
      *,
      device=None,
      dtype: torch.dtype = torch.float32,
  ):
    super().__init__()
    self.scale = scale
    self.positional_features = nn.Parameter(
        torch.zeros((latent_size,) + tuple(nodal_shape), device=device,
                    dtype=dtype)
    )

  def forward(self, inputs=None, memory=None, diagnostics=None,
              randomness=None, forcing=None):
    del inputs, memory, diagnostics, randomness, forcing  # unused
    return {
        'learned_positional_features': self.scale * self.positional_features
    }

  def import_haiku(self, params: dict, prefix: str) -> None:
    bundle = params[prefix]
    with torch.no_grad():
      self.positional_features.copy_(bundle['learned_positional_features'])


class EmbeddingSurfaceFeatures(FeatureModule):
  """Returns embedding surface outputs `{feature_name: (size, lon, lat)}`.

  The embedding module is constructed by the model builder with
  `output_shapes = {feature_name: (output_size, lon, lat)}`.
  """

  HAIKU_NAME = 'embedding_surface_features'

  def __init__(self, embedding: nn.Module,
               features_transform: Optional[nn.Module] = None):
    super().__init__()
    self.embedding = embedding
    self.features_transform = features_transform or transforms.IdentityTransform()

  def forward(self, inputs=None, memory=None, diagnostics=None,
              randomness=None, forcing=None):
    features = self.embedding(inputs, memory, diagnostics, randomness,
                              forcing)
    return self.features_transform(features)

  def import_haiku(self, params: dict, prefix: str) -> None:
    self.embedding.import_haiku(
        params, f'{prefix}/~/{self.embedding.HAIKU_NAME}'
    )


class EmbeddingVolumeFeatures(EmbeddingSurfaceFeatures):
  """Returns embedding volume outputs unpacked per output channel.

  The embedding module is constructed with
  `output_shapes = {f'{feature_name}_{i}': (level, lon, lat) ...}`.
  """

  HAIKU_NAME = 'embedding_volume_features'


class FloatDataFeatures(FeatureModule):
  """Supplies floating-point covariates (e.g. static maps) as features.

  Unlike the original JAX implementation (which read an xarray dataset at construction),
  `covariates` are passed in already nondimensionalized and transposed to
  (level?, lon, lat); the checkpoint/model builder handles data loading.
  """

  HAIKU_NAME = 'float_data_features'

  def __init__(
      self,
      grid: spherical_harmonic.Grid,
      covariates: Dict[str, np.ndarray],
      compute_gradients: Optional[nn.Module] = None,
  ):
    super().__init__()
    self.grid = grid
    self.compute_gradients = compute_gradients or transforms.EmptyTransform()
    ref = grid.cos_lat
    self.register_buffer(
        '_sec_lat', 1 / grid.cos_lat, persistent=False
    )
    self._covariate_keys = []
    for key, data in covariates.items():
      data = np.asarray(data)
      if data.ndim != 3:
        data = data[np.newaxis, ...]
      self.register_buffer(
          f'_covariate_{key}',
          torch.as_tensor(data, dtype=ref.dtype, device=ref.device),
          persistent=False,
      )
      self._covariate_keys.append(key)

  def _sec_lat_scale(self, factor_order: int):
    if factor_order == 0:
      return None
    if factor_order == 1:
      return self._sec_lat
    if factor_order == 2:
      return self.grid.sec2_lat
    raise ValueError(f'unsupported {factor_order=}')

  def forward(self, inputs=None, memory=None, diagnostics=None,
              randomness=None, forcing=None):
    del inputs, memory, diagnostics, randomness, forcing  # unused
    features = {
        k: getattr(self, f'_covariate_{k}') for k in self._covariate_keys
    }
    modal_features = {
        KeyWithCosLatFactor(k, 0): self.grid.to_modal(v)
        for k, v in features.items()
    }
    modal_gradient_features = self.compute_gradients(modal_features)
    for k, v in modal_gradient_features.items():
      scale = self._sec_lat_scale(k.factor_order)
      nodal = self.grid.to_nodal(v)
      if scale is not None:
        nodal = nodal * scale
      features[k.name] = nodal
    return features


class CombinedFeatures(FeatureModule):
  """Combines multiple feature modules together."""

  HAIKU_NAME = 'combined_features'

  def __init__(
      self,
      feature_modules: Sequence[nn.Module],
      features_to_exclude: Sequence[str] = tuple(),
      features_transform: Optional[nn.Module] = None,
  ):
    super().__init__()
    self.feature_modules = nn.ModuleList(feature_modules)
    self.features_to_exclude = tuple(features_to_exclude)
    self.features_transform = features_transform or transforms.IdentityTransform()

  def forward(self, inputs=None, memory=None, diagnostics=None,
              randomness=None, forcing=None):
    all_features = {}
    for feature_module in self.feature_modules:
      features = feature_module(inputs, memory, diagnostics, randomness,
                                forcing)
      for k, v in features.items():
        if k in all_features:
          raise ValueError(f'Encountered duplicate feature {k}')
        all_features[k] = v
    all_features = self.features_transform(all_features)
    for k in self.features_to_exclude:
      all_features.pop(k, None)
    return all_features

  def import_haiku(self, params: dict, prefix: str) -> None:
    # children are created in the legacy __init__ (hence the '~'); repeated
    # module classes get haiku's per-parent _N numbering.
    counts: dict[str, int] = {}
    for child in self.feature_modules:
      name = child.HAIKU_NAME
      n = counts.get(name, 0)
      counts[name] = n + 1
      if hasattr(child, 'import_haiku'):
        suffix = name if n == 0 else f'{name}_{n}'
        child.import_haiku(params, f'{prefix}/~/{suffix}')
