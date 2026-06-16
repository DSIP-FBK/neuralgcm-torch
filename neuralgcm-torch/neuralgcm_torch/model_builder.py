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
"""Builds a `StochasticModularStepModel` from a converted checkpoint.

The converted checkpoint (see `neuralgcm_torch.checkpoint`) carries the
original gin bindings as plain data under `config['model']`: a dictionary
mapping `scope/ClassName` to bound parameters, where references appear as
`{'__ref__': name, '__call__': bool}` and macros are already inlined. This
module interprets those bindings with one small builder function per
configurable class, constructing this package's modules and threading the
haiku parameter paths down the tree (the paths serve double duty: network
input sizes are read from the parameter shapes, and `import_haiku` loads
the weights along the same paths, so any mismatch fails loudly).

Gin scoping is resolved with single-component scope chains, which is exact
for all published checkpoint configs: a reference `@Name` seen while
expanding a binding in scope `s` resolves to `s/Name` bindings when they
exist, else to the unscoped `Name`; parameters missing from a scoped
binding fall back to the unscoped one.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import numpy as np
import torch
from torch import nn

from dinosaur_torch import coordinate_systems
from dinosaur_torch import primitive_equations
from dinosaur_torch import radiation as radiation_lib
from dinosaur_torch import scales
from dinosaur_torch import sigma_coordinates
from dinosaur_torch import spherical_harmonic
from dinosaur_torch import time_integration
from dinosaur_torch import units as units_lib
from dinosaur_torch import vertical_interpolation
from neuralgcm_torch import checkpoint as checkpoint_lib
from neuralgcm_torch import correctors
from neuralgcm_torch import decoders
from neuralgcm_torch import embeddings
from neuralgcm_torch import encoders
from neuralgcm_torch import features
from neuralgcm_torch import filters
from neuralgcm_torch import forcings
from neuralgcm_torch import layers
from neuralgcm_torch import mappings
from neuralgcm_torch import models
from neuralgcm_torch import orographies
from neuralgcm_torch import parameterizations
from neuralgcm_torch import steps
from neuralgcm_torch import stochastic
from neuralgcm_torch import towers
from neuralgcm_torch import transforms

_REQUIRED = object()

_SPHERICAL_HARMONICS_IMPLS = {
    'RealSphericalHarmonics': spherical_harmonic.RealSphericalHarmonics,
    'RealSphericalHarmonicsWithZeroImag': (
        spherical_harmonic.FastSphericalHarmonics
    ),
    'FastSphericalHarmonics': spherical_harmonic.FastSphericalHarmonics,
}

_ACTIVATIONS = {
    'gelu': layers.gelu,
    'relu': layers.relu,
    'silu': layers.silu,
}

_TIME_INTEGRATORS = {
    'imex_rk_sil3': time_integration.imex_rk_sil3,
    'backward_forward_euler': time_integration.backward_forward_euler,
    'crank_nicolson_rk2': time_integration.crank_nicolson_rk2,
}

_EQUATIONS = {
    'PrimitiveEquations': {},
    'PrimitiveEquationsWithTime': {},
    'MoistPrimitiveEquations': {'humidity_key': 'specific_humidity'},
    'MoistPrimitiveEquationsWithCloudMoisture': {
        'humidity_key': 'specific_humidity',
        'cloud_keys': (
            'specific_cloud_liquid_water_content',
            'specific_cloud_ice_water_content',
        ),
    },
}


@dataclasses.dataclass
class _Context:
  """Everything the builder functions need, in one bag."""

  bindings: dict
  params: dict
  aux: dict
  data_config: dict
  coords: coordinate_systems.CoordinateSystem
  input_coords: coordinate_systems.CoordinateSystem
  output_coords: coordinate_systems.CoordinateSystem
  physics: units_lib.SimUnits
  dt: float
  ref_datetime: np.datetime64
  device: Any
  dtype: torch.dtype

  # -- binding lookup --------------------------------------------------------

  def get(self, key: str, param: str, default=_REQUIRED):
    """Returns the bound value of `key.param` with unscoped fallback."""
    scoped = self.bindings.get(key, {})
    if param in scoped:
      return scoped[param]
    base = key.split('/')[-1]
    if base != key and param in self.bindings.get(base, {}):
      return self.bindings[base][param]
    if default is _REQUIRED:
      raise KeyError(f'no binding for {key}.{param}')
    return default

  def resolve_key(self, ref: dict, scopes: tuple) -> str:
    """Returns the bindings key for a reference seen within `scopes`."""
    name = ref['__ref__']
    if '/' in name:
      return name
    for scope in scopes:
      if f'{scope}/{name}' in self.bindings:
        return f'{scope}/{name}'
    return name

  def nondim(self, value):
    return self.physics.nondimensionalize(value)


def _class_name(key: str) -> str:
  return key.split('/')[-1]


def _child_scopes(key: str, scopes: tuple) -> tuple:
  """Scope chain for references inside the binding `key`."""
  if '/' in key:
    return (key.rsplit('/', 1)[0],) + scopes
  return scopes


def _activation(ctx: _Context, ref, scopes: tuple):
  if ref is None:
    return None
  key = ctx.resolve_key(ref, scopes)
  return _ACTIVATIONS[_class_name(key)]


#  ===========================================================================
#  Coordinates.
#  ===========================================================================


def _build_grid(ctx: _Context, key: str, scopes: tuple):
  name = _class_name(key)
  impl_ref = ctx.get(key, 'spherical_harmonics_impl', None)
  if impl_ref is None:
    impl = spherical_harmonic.RealSphericalHarmonics
  else:
    impl = _SPHERICAL_HARMONICS_IMPLS[
        _class_name(ctx.resolve_key(impl_ref, scopes))
    ]
  radius = ctx.get(key, 'radius', None)
  kwargs = {} if radius is None else {'radius': float(radius)}
  if name == 'GridWithWavenumbers':
    spec = spherical_harmonic.GridSpec.with_wavenumbers(
        longitude_wavenumbers=ctx.get(key, 'longitude_wavenumbers'),
        dealiasing=ctx.get(key, 'dealiasing', 'quadratic'),
        latitude_spacing=ctx.get(key, 'latitude_spacing', 'gauss'),
        longitude_offset=ctx.get(key, 'longitude_offset', 0.0),
        **kwargs,
    )
  elif name.startswith('Grid'):
    spec = getattr(spherical_harmonic.GridSpec, name[len('Grid'):])(**kwargs)
  else:
    raise ValueError(f'unsupported grid constructor {name}')
  return spherical_harmonic.Grid(
      spec, impl=impl, device=ctx.device, dtype=ctx.dtype
  )


def _build_vertical(ctx: _Context, key: str, scopes: tuple):
  name = _class_name(key)
  if name == 'SigmaCoordinatesEquidistant':
    coordinates = sigma_coordinates.SigmaCoordinates.equidistant(
        ctx.get(key, 'layers')
    )
  elif name == 'SigmaCoordinates':
    coordinates = sigma_coordinates.SigmaCoordinates(
        np.asarray(ctx.get(key, 'boundaries'))
    )
  else:
    raise ValueError(f'unsupported vertical constructor {name}')
  # reuse the model vertical when identical, so identity checks (e.g. in
  # spectral interpolation) hold.
  if np.array_equal(
      coordinates.boundaries, ctx.coords.vertical.coordinates.boundaries
  ):
    return ctx.coords.vertical
  return sigma_coordinates.SigmaLevels(
      coordinates, device=ctx.device, dtype=ctx.dtype
  )


def _build_coordinate_system(ctx: _Context, key: str, scopes: tuple):
  child_scopes = _child_scopes(key, scopes)
  grid = _build_grid(
      ctx, ctx.resolve_key(ctx.get(key, 'horizontal'), child_scopes),
      child_scopes,
  )
  vertical = _build_vertical(
      ctx, ctx.resolve_key(ctx.get(key, 'vertical'), child_scopes),
      child_scopes,
  )
  return coordinate_systems.CoordinateSystem(grid, vertical)


#  ===========================================================================
#  Transforms and filters.
#  ===========================================================================


def _build_transform(
    ctx: _Context, ref, scopes: tuple, grid: spherical_harmonic.Grid
) -> Optional[nn.Module]:
  """Builds a (data) transform module from a reference; None passes through."""
  if ref is None:
    return None
  key = ctx.resolve_key(ref, scopes)
  child_scopes = _child_scopes(key, scopes)
  name = _class_name(key)
  if name == 'IdentityTransform' or name == 'EncoderIdentityTransform':
    return transforms.IdentityTransform()
  if name == 'EmptyTransform':
    return transforms.EmptyTransform()
  if name == 'SequentialTransform':
    return transforms.SequentialTransform([
        _build_transform(ctx, t, child_scopes, grid)
        for t in ctx.get(key, 'transform_modules')
    ])
  if name in ('ShiftAndNormalize', 'InverseShiftAndNormalize'):
    if ctx.get(key, 'features_to_exclude', []):
      raise NotImplementedError('features_to_exclude is not supported')
    cls = getattr(transforms, name)
    return cls(
        shifts=ctx.get(key, 'shifts'),
        scales=ctx.get(key, 'scales'),
        global_scale=ctx.get(key, 'global_scale', None),
    )
  if name in ('LevelScale', 'InverseLevelScale'):
    cls = getattr(transforms, name)
    return cls(
        ctx.get(key, 'scales'),
        keys_to_scale=ctx.get(key, 'keys_to_scale', ()),
        device=ctx.device,
        dtype=ctx.dtype,
    )
  if name == 'SoftClip':
    return transforms.SoftClip(
        ctx.get(key, 'max_value'),
        ctx.get(key, 'hinge_softness', 1.0),
    )
  if name == 'HardClip':
    return transforms.HardClip(ctx.get(key, 'max_value'))
  if name == 'TruncateSigmaLevels':
    sigma_ranges = {
        k: tuple(v) for k, v in ctx.get(key, 'sigma_ranges').items()
    }
    return transforms.TruncateSigmaLevels(
        ctx.coords.vertical.coordinates, sigma_ranges
    )
  if name == 'TakeSurfaceAdjacentSigmaLevel':
    return transforms.TakeSurfaceAdjacentSigmaLevel()
  if name == 'ToModalDiffOperators':
    return transforms.ToModalDiffOperators(grid)
  if name == 'ClipTransform':
    return transforms.ClipTransform(
        grid, ctx.get(key, 'wavenumbers_to_clip', 1)
    )
  raise ValueError(f'unsupported transform {name}')


def _build_step_filter(
    ctx: _Context, ref, scopes: tuple, grid, dt: float
) -> Optional[nn.Module]:
  if ref is None:
    return None
  key = ctx.resolve_key(ref, scopes)
  child_scopes = _child_scopes(key, scopes)
  name = _class_name(key)
  if name == 'NoFilter':
    return filters.NoFilter()
  if name == 'SequentialStepFilter':
    return filters.SequentialStepFilter([
        _build_step_filter(ctx, f, child_scopes, grid, dt)
        for f in ctx.get(key, 'filter_modules')
    ])
  if name == 'ExponentialFilter':
    return filters.ExponentialFilter(
        grid,
        dt,
        tau=ctx.get(key, 'tau'),
        order=ctx.get(key, 'order'),
        cutoff=ctx.get(key, 'cutoff', 0),
        physics_specs=ctx.physics,
    )
  if name == 'ClipFilter':
    return filters.ClipFilter(grid, ctx.get(key, 'wavenumbers_to_clip', 1))
  if name == 'FixGlobalMeanFilter':
    return filters.FixGlobalMeanFilter(
        keys=tuple(ctx.get(key, 'keys', ('log_surface_pressure',)))
    )
  raise ValueError(f'unsupported step filter {name}')


def _build_data_filter(ctx: _Context, ref, scopes: tuple, grid) -> nn.Module:
  key = ctx.resolve_key(ref, scopes)
  name = _class_name(key)
  if name == 'DataNoFilter':
    return filters.DataNoFilter()
  if name == 'DataExponentialFilter':
    return filters.DataExponentialFilter(
        grid,
        attenuation=ctx.get(key, 'attenuation', 16),
        order=ctx.get(key, 'order', 18),
        cutoff=ctx.get(key, 'cutoff', 0),
    )
  if name == 'PerVariableDataFilter':
    child_scopes = _child_scopes(key, scopes)

    def build_tree(tree):
      if isinstance(tree, dict) and '__ref__' not in tree:
        return {k: build_tree(v) for k, v in tree.items()}
      return _build_data_filter(ctx, tree, child_scopes, grid)

    return filters.PerVariableDataFilter(
        build_tree(ctx.get(key, 'per_variable_filters'))
    )
  raise ValueError(f'unsupported data filter {name}')


#  ===========================================================================
#  Orographies.
#  ===========================================================================


def _build_orography(ctx: _Context, ref, scopes: tuple, grid) -> nn.Module:
  key = ctx.resolve_key(ref, scopes)
  child_scopes = _child_scopes(key, scopes)
  name = _class_name(key)
  if name == 'FilteredCustomOrography':
    if ctx.get(key, 'orography_data_path', None) is not None:
      raise NotImplementedError('orography_data_path is not supported')
    input_grid_config = ctx.data_config['orography_input_grid']
    input_grid = spherical_harmonic.Grid(
        checkpoint_lib.grid_spec_from_config(input_grid_config),
        impl=checkpoint_lib.spherical_harmonics_impl_from_config(
            input_grid_config
        ),
        device=ctx.device,
        dtype=ctx.dtype,
    )
    nodal_orography = ctx.nondim(
        ctx.aux['nodal_orography_m'] * scales.units.meter
    )
    return orographies.FilteredCustomOrography(
        grid,
        input_grid,
        np.asarray(nodal_orography, np.float32),
        filters=[
            _build_data_filter(ctx, f, child_scopes, grid)
            for f in ctx.get(key, 'filter_modules', [])
        ],
    )
  if name == 'LearnedOrography':
    base = _build_orography(
        ctx, ctx.get(key, 'base_orography_module'), child_scopes, grid
    )
    return orographies.LearnedOrography(
        grid, base, ctx.get(key, 'correction_scale')
    )
  if name == 'ClippedOrography':
    nodal_orography = ctx.nondim(
        ctx.aux['nodal_orography_m'] * scales.units.meter
    )
    return orographies.ClippedOrography(
        grid, np.asarray(nodal_orography, np.float32)
    )
  raise ValueError(f'unsupported orography {name}')


#  ===========================================================================
#  Networks: layers, towers, mappings.
#  ===========================================================================


def _param_input_size(ctx: _Context, bundle_path: str, axis: int) -> int:
  """Reads a network input size from a checkpoint parameter shape."""
  if bundle_path not in ctx.params:
    raise KeyError(
        f'cannot infer input size: no parameter bundle {bundle_path!r}'
    )
  return int(ctx.params[bundle_path]['w'].shape[axis])


def _build_mlp(
    ctx: _Context, key: str, scopes: tuple, input_size: int, output_size: int
) -> layers.MlpUniform:
  return layers.MlpUniform(
      input_size,
      output_size,
      num_hidden_units=ctx.get(key, 'num_hidden_units'),
      num_hidden_layers=ctx.get(key, 'num_hidden_layers'),
      with_bias=ctx.get(key, 'with_bias', True),
      activation=_activation(
          ctx, ctx.get(key, 'activation', None), _child_scopes(key, scopes)
      ) or layers.relu,
      activate_final=ctx.get(key, 'activate_final', False),
      device=ctx.device,
      dtype=ctx.dtype,
  )


def _build_column_tower(
    ctx: _Context, ref, scopes: tuple, input_size: int, output_size: int
) -> tuple[towers.ColumnTower, str]:
  """Builds a ColumnTower; returns it with its haiku child name."""
  key = ctx.resolve_key(ref, scopes)
  child_scopes = _child_scopes(key, scopes)
  net_key = ctx.resolve_key(ctx.get(key, 'column_net_factory'), child_scopes)
  if _class_name(net_key) != 'MlpUniform':
    raise NotImplementedError(f'unsupported column net {net_key}')
  net = _build_mlp(ctx, net_key, child_scopes, input_size, output_size)
  name = ctx.get(key, 'name', None) or 'column_tower'
  return towers.ColumnTower(net), name


def _build_tower(
    ctx: _Context, ref, scopes: tuple, prefix: str, output_size: int
) -> nn.Module:
  """Builds a tower for a mapping; `prefix` is the tower's haiku path."""
  key = ctx.resolve_key(ref, scopes)
  child_scopes = _child_scopes(key, scopes)
  name = _class_name(key)
  if name == 'EpdTower':
    latent_size = ctx.get(key, 'latent_size')
    num_blocks = ctx.get(key, 'num_process_blocks')
    encode_ref = ctx.get(key, 'encode_tower_factory')
    encode_name_key = ctx.resolve_key(encode_ref, child_scopes)
    encode_name = ctx.get(encode_name_key, 'name', None) or 'encode_tower'
    input_size = _param_input_size(
        ctx, f'{prefix}/{encode_name}/~/mlp_uniform/~/linear_0', axis=0
    )
    encode_tower, encode_name = _build_column_tower(
        ctx, encode_ref, child_scopes, input_size, latent_size
    )
    process_towers = []
    process_name = 'process_tower'
    for _ in range(num_blocks):
      tower, process_name = _build_column_tower(
          ctx,
          ctx.get(key, 'process_tower_factory'),
          child_scopes,
          latent_size,
          latent_size,
      )
      process_towers.append(tower)
    decode_tower, decode_name = _build_column_tower(
        ctx,
        ctx.get(key, 'decode_tower_factory'),
        child_scopes,
        latent_size,
        output_size,
    )
    get_act = lambda p: _activation(ctx, ctx.get(key, p, None), child_scopes)
    return towers.EpdTower(
        encode_tower,
        process_towers,
        decode_tower,
        post_encode_activation=get_act('post_encode_activation'),
        pre_decode_activation=get_act('pre_decode_activation'),
        final_activation=get_act('final_activation'),
        child_names=(encode_name, process_name, decode_name),
    )
  if name == 'VerticalConvTower':
    input_size = _param_input_size(
        ctx, f'{prefix}/~/conv_level', axis=1
    )
    return towers.VerticalConvTower(
        input_size,
        output_size,
        channels=ctx.get(key, 'channels'),
        kernel_shape=ctx.get(key, 'kernel_shape'),
        with_bias=ctx.get(key, 'with_bias', True),
        activation=_activation(
            ctx, ctx.get(key, 'activation', None), child_scopes
        ) or layers.relu,
        activate_final=ctx.get(key, 'activate_final', False),
        device=ctx.device,
        dtype=ctx.dtype,
    )
  raise ValueError(f'unsupported tower {name}')


def _flat_channel_sizes(output_shapes: dict, dim: int) -> int:
  total = 0
  for v in output_shapes.values():
    if isinstance(v, dict):
      total += _flat_channel_sizes(v, dim)
    else:
      total += v[dim]
  return total


def _count_leaves(output_shapes: dict) -> int:
  return sum(
      _count_leaves(v) if isinstance(v, dict) else 1
      for v in output_shapes.values()
  )


def _build_mapping(
    ctx: _Context, ref, scopes: tuple, output_shapes: dict, prefix: str
) -> nn.Module:
  """Builds a (volume) mapping; `prefix` is the mapping's haiku path."""
  key = ctx.resolve_key(ref, scopes)
  child_scopes = _child_scopes(key, scopes)
  name = _class_name(key)
  tower_ref = ctx.get(key, 'tower_factory')
  tower_key = ctx.resolve_key(tower_ref, child_scopes)
  tower_haiku = {
      'EpdTower': 'epd_tower',
      'VerticalConvTower': 'vertical_conv_tower',
  }[_class_name(tower_key)]
  if name == 'NodalMapping':
    output_size = _flat_channel_sizes(output_shapes, -3)
    tower = _build_tower(
        ctx, tower_ref, child_scopes, f'{prefix}/~/{tower_haiku}', output_size
    )
    return mappings.NodalMapping(tower, output_shapes)
  if name == 'NodalVolumeMapping':
    output_size = _count_leaves(output_shapes)
    tower = _build_tower(
        ctx, tower_ref, child_scopes, f'{prefix}/~/{tower_haiku}', output_size
    )
    return mappings.NodalVolumeMapping(tower, output_shapes)
  raise ValueError(f'unsupported mapping {name}')


#  ===========================================================================
#  Features and embeddings.
#  ===========================================================================

# class name -> haiku module name, for prefix threading.
_FEATURE_HAIKU_NAMES = {
    'CombinedFeatures': 'combined_features',
    'VelocityAndPrognostics': 'velocity_and_prognostics',
    'MemoryVelocityAndValues': 'memory_velocity_and_values',
    'RadiationFeatures': 'radiation_features',
    'LatitudeFeatures': 'latitude_features',
    'RandomnessFeatures': 'randomness_features',
    'PressureFeatures': 'pressure_features',
    'LearnedPositionalFeatures': 'learned_positional_features',
    'FloatDataFeatures': 'float_data_features',
    'ForcingFeatures': 'forcing_features',
    'EmbeddingSurfaceFeatures': 'embedding_surface_features',
    'EmbeddingVolumeFeatures': 'embedding_volume_features',
    'NullFeatures': 'null_features',
}

_EMBEDDING_HAIKU_NAMES = {
    'ModalToNodalEmbedding': 'modal_to_nodal_embedding',
    'NodalLandSeaIceEmbedding': 'nodal_land_sea_ice_embedding',
}


def _build_features(
    ctx: _Context, ref, scopes: tuple, grid, prefix: str
) -> nn.Module:
  """Builds a feature module; `prefix` is the module's own haiku path."""
  key = ctx.resolve_key(ref, scopes)
  child_scopes = _child_scopes(key, scopes)
  name = _class_name(key)
  transform = lambda: _build_transform(
      ctx, ctx.get(key, 'features_transform_module', None), child_scopes, grid
  )
  if name == 'NullFeatures':
    return features.NullFeatures()
  if name == 'CombinedFeatures':
    if ctx.get(key, 'feature_module_names_to_exclude', []):
      raise NotImplementedError(
          'feature_module_names_to_exclude is not supported'
      )
    children = []
    counts: dict[str, int] = {}
    for child_ref in ctx.get(key, 'feature_modules'):
      child_key = ctx.resolve_key(child_ref, child_scopes)
      haiku_name = _FEATURE_HAIKU_NAMES[_class_name(child_key)]
      n = counts.get(haiku_name, 0)
      counts[haiku_name] = n + 1
      suffix = haiku_name if n == 0 else f'{haiku_name}_{n}'
      children.append(
          _build_features(
              ctx, child_ref, child_scopes, grid, f'{prefix}/~/{suffix}'
          )
      )
    return features.CombinedFeatures(
        children,
        features_to_exclude=ctx.get(key, 'features_to_exclude', ()),
        features_transform=transform(),
    )
  if name in ('VelocityAndPrognostics', 'MemoryVelocityAndValues'):
    cls = getattr(features, name)
    return cls(
        grid,
        fields_to_include=ctx.get(key, 'fields_to_include', None),
        features_transform=transform(),
        compute_gradients=_build_transform(
            ctx, ctx.get(key, 'compute_gradients_module', None),
            child_scopes, grid,
        ),
    )
  if name == 'RadiationFeatures':
    solar = radiation_lib.SolarRadiation.normalized(
        grid.spec,
        ctx.physics,
        ctx.ref_datetime,
        device=ctx.device,
        dtype=ctx.dtype,
    )
    return features.RadiationFeatures(solar, features_transform=transform())
  if name == 'LatitudeFeatures':
    return features.LatitudeFeatures(grid, features_transform=transform())
  if name == 'RandomnessFeatures':
    return features.RandomnessFeatures(features_transform=transform())
  if name == 'PressureFeatures':
    return features.PressureFeatures(
        grid, ctx.coords.vertical, features_transform=transform()
    )
  if name == 'LearnedPositionalFeatures':
    return features.LearnedPositionalFeatures(
        ctx.get(key, 'latent_size'),
        grid.spec.nodal_shape,
        scale=ctx.get(key, 'scale', 1.0),
        device=ctx.device,
        dtype=ctx.dtype,
    )
  if name == 'FloatDataFeatures':
    covariates = {}
    for covariate_key in ctx.get(key, 'covariate_keys'):
      units_str = ctx.data_config['covariate_units'][covariate_key]
      factor = float(ctx.nondim(1.0 * scales.parse_units(units_str)))
      covariates[covariate_key] = (
          ctx.aux[f'covariate_{covariate_key}'] * factor
      ).astype(np.float32)
    return features.FloatDataFeatures(
        grid,
        covariates,
        compute_gradients=_build_transform(
            ctx, ctx.get(key, 'compute_gradients_module', None),
            child_scopes, grid,
        ),
    )
  if name == 'ForcingFeatures':
    return features.ForcingFeatures(
        forcing_to_include=ctx.get(key, 'forcing_to_include', ()),
        features_transform=transform(),
    )
  if name in ('EmbeddingSurfaceFeatures', 'EmbeddingVolumeFeatures'):
    feature_name = ctx.get(key, 'feature_name')
    output_size = ctx.get(key, 'output_size')
    if name == 'EmbeddingSurfaceFeatures':
      output_shapes = {
          feature_name: (output_size,) + grid.spec.nodal_shape
      }
    else:
      # insertion order must match the legacy (jax pytree) flatten order,
      # which sorts the generated names LEXICOGRAPHICALLY: with more than
      # ten outputs, 'CNN1D_10' sorts before 'CNN1D_2'.
      output_shapes = {
          key: (ctx.coords.vertical.layers,) + grid.spec.nodal_shape
          for key in sorted(
              f'{feature_name}_{i}' for i in range(output_size)
          )
      }
    embedding_ref = ctx.get(key, 'embedding_module')
    embedding_key = ctx.resolve_key(embedding_ref, child_scopes)
    haiku_name = _EMBEDDING_HAIKU_NAMES[_class_name(embedding_key)]
    embedding = _build_embedding(
        ctx, embedding_ref, child_scopes, grid, output_shapes,
        f'{prefix}/~/{haiku_name}',
    )
    cls = getattr(features, name)
    return cls(embedding)
  raise ValueError(f'unsupported feature module {name}')


def _build_embedding(
    ctx: _Context, ref, scopes: tuple, grid, output_shapes: dict, prefix: str
) -> nn.Module:
  key = ctx.resolve_key(ref, scopes)
  child_scopes = _child_scopes(key, scopes)
  name = _class_name(key)
  if name == 'ModalToNodalEmbedding':
    features_ref = ctx.get(key, 'modal_to_nodal_features_module')
    features_key = ctx.resolve_key(features_ref, child_scopes)
    features_haiku = _FEATURE_HAIKU_NAMES[_class_name(features_key)]
    features_module = _build_features(
        ctx, features_ref, child_scopes, grid, f'{prefix}/~/{features_haiku}'
    )
    mapping_ref = ctx.get(key, 'nodal_mapping_module')
    mapping_key = ctx.resolve_key(mapping_ref, child_scopes)
    mapping_haiku = {
        'NodalMapping': 'nodal_mapping',
        'NodalVolumeMapping': 'nodal_volume_mapping',
    }[_class_name(mapping_key)]
    mapping = _build_mapping(
        ctx, mapping_ref, child_scopes, output_shapes,
        f'{prefix}/{mapping_haiku}',
    )
    return embeddings.ModalToNodalEmbedding(
        features_module,
        mapping,
        output_transform=_build_transform(
            ctx, ctx.get(key, 'output_transform_module', None),
            child_scopes, grid,
        ),
    )
  if name == 'NodalLandSeaIceEmbedding':
    if ctx.get(key, 'static_vars_ds_path', None) is not None:
      raise NotImplementedError('static_vars_ds_path is not supported')
    children = []
    built = 0  # only real sub-embeddings take part in haiku's numbering
    for child_param, param_name in (
        ('land_embedding', 'land'),
        ('sea_embedding', 'sea'),
        ('sea_ice_embedding', 'sea_ice'),
    ):
      child_ref = ctx.get(key, child_param, None)
      if child_ref is None:
        # the legacy module falls back to learned uniform constants
        children.append(
            embeddings.UniformParameterEmbedding(
                output_shapes, param_name,
                device=ctx.device, dtype=ctx.dtype,
            )
        )
        continue
      suffix = (
          'modal_to_nodal_embedding' if built == 0
          else f'modal_to_nodal_embedding_{built}'
      )
      built += 1
      children.append(
          _build_embedding(
              ctx, child_ref, child_scopes, grid,
              output_shapes, f'{prefix}/~/{suffix}',
          )
      )
    return embeddings.NodalLandSeaIceEmbedding(
        *children,
        land_sea_mask=ctx.aux['land_sea_mask'].astype(np.float32),
        device=ctx.device,
        dtype=ctx.dtype,
    )
  raise ValueError(f'unsupported embedding {name}')


#  ===========================================================================
#  Stochastic fields.
#  ===========================================================================


def _build_randomness(
    ctx: _Context, ref, scopes: tuple, grid, dt: float
) -> Optional[nn.Module]:
  if ref is None:
    return None
  key = ctx.resolve_key(ref, scopes)
  name = _class_name(key)
  if name == 'NoRandomField':
    return stochastic.NoRandomField()
  if name == 'ZerosRandomField':
    return stochastic.ZerosRandomField(grid)
  if name == 'BatchGaussianRandomFieldModule':
    return stochastic.BatchGaussianRandomFieldModule(
        grid,
        dt,
        initial_correlation_times=ctx.get(key, 'initial_correlation_times'),
        initial_correlation_lengths=ctx.get(
            key, 'initial_correlation_lengths'
        ),
        variances=ctx.get(key, 'variances'),
        field_subset=ctx.get(key, 'field_subset', None),
        n_fixed_fields=ctx.get(key, 'n_fixed_fields', None),
        clip=ctx.get(key, 'clip', 6.0),
        physics_specs=ctx.physics,
    )
  if name == 'GaussianRandomField':
    return stochastic.GaussianRandomField(
        grid,
        dt,
        correlation_time=ctx.get(key, 'correlation_time'),
        correlation_length=ctx.get(key, 'correlation_length'),
        variance=ctx.get(key, 'variance'),
        clip=ctx.get(key, 'clip', 6.0),
        physics_specs=ctx.physics,
    )
  if name == 'DictOfGaussianRandomFieldModules':
    times = list(ctx.get(key, 'initial_correlation_times'))
    lengths = list(ctx.get(key, 'initial_correlation_lengths'))
    variances = list(ctx.get(key, 'variances'))
    field_names = ctx.get(key, 'field_names', None)
    field_names = (
        [f'GRF{i}' for i in range(len(times))] if field_names is None
        else list(field_names)
    )
    subset = ctx.get(key, 'field_subset', None)
    if subset is not None:
      pick = lambda seq: [seq[i] for i in subset]
      times, lengths, variances, field_names = (
          pick(times), pick(lengths), pick(variances), pick(field_names)
      )
    clip = ctx.get(key, 'clip', 6.0)
    return stochastic.DictOfGaussianRandomFieldModules({
        field_name: stochastic.GaussianRandomFieldModule(
            grid, dt, time, length, variance, clip,
            physics_specs=ctx.physics,
        )
        for field_name, time, length, variance in zip(
            field_names, times, lengths, variances
        )
    })
  raise ValueError(f'unsupported randomness module {name}')


#  ===========================================================================
#  Parameterization / corrector / step / model.
#  ===========================================================================


def _state_field_shape(
    ctx: _Context, field: str, grid
) -> tuple[int, int, int]:
  layers = 1 if field == 'log_surface_pressure' else (
      ctx.coords.vertical.layers
  )
  return (layers,) + grid.spec.nodal_shape


def _div_curl_output_shapes(ctx: _Context, prediction_mask: dict, grid):
  """Mapping output shapes for DivCurl tendencies, in sorted key order."""
  renames = {'divergence': 'u', 'vorticity': 'v'}
  shapes = {}
  for field in sorted(
      prediction_mask, key=lambda f: renames.get(f, f)
  ):
    mask = prediction_mask[field]
    if isinstance(mask, dict):
      tracers = {
          k: _state_field_shape(ctx, field, grid)
          for k in sorted(mask) if mask[k]
      }
      if tracers:
        shapes[field] = tracers
    elif mask:
      shapes[renames.get(field, field)] = _state_field_shape(
          ctx, field, grid
      )
  return shapes


def _build_parameterization(
    ctx: _Context, ref, scopes: tuple, dt: float, prefix: str
) -> nn.Module:
  key = ctx.resolve_key(ref, scopes)
  child_scopes = _child_scopes(key, scopes)
  name = _class_name(key)
  if name != 'DivCurlNeuralParameterization':
    raise ValueError(f'unsupported parameterization {name}')
  grid = ctx.coords.horizontal
  prediction_mask = ctx.get(key, 'prediction_mask')
  features_ref = ctx.get(key, 'modal_to_nodal_features_module')
  features_key = ctx.resolve_key(features_ref, child_scopes)
  features_haiku = _FEATURE_HAIKU_NAMES[_class_name(features_key)]
  features_module = _build_features(
      ctx, features_ref, child_scopes, grid, f'{prefix}/~/{features_haiku}'
  )
  mapping = _build_mapping(
      ctx,
      ctx.get(key, 'nodal_mapping_module'),
      child_scopes,
      _div_curl_output_shapes(ctx, prediction_mask, grid),
      f'{prefix}/nodal_mapping',
  )
  return parameterizations.DivCurlNeuralParameterization(
      grid,
      features_module,
      mapping,
      tendency_transform=_build_transform(
          ctx, ctx.get(key, 'tendency_transform_module', None),
          child_scopes, grid,
      ),
      prediction_mask=prediction_mask,
      filter_module=_build_step_filter(
          ctx, ctx.get(key, 'filter_module', None), child_scopes, grid, dt
      ),
  )


def _build_corrector(
    ctx: _Context,
    ref,
    scopes: tuple,
    coords: coordinate_systems.CoordinateSystem,
    dt: float,
    prefix: str,
) -> nn.Module:
  key = ctx.resolve_key(ref, scopes)
  child_scopes = _child_scopes(key, scopes)
  name = _class_name(key)
  if name == 'CustomCoordsCorrector':
    custom_coords_ref = ctx.get(key, 'custom_coords')
    custom_coords = _build_coordinate_system(
        ctx, ctx.resolve_key(custom_coords_ref, child_scopes), child_scopes
    )
    inner = _build_corrector(
        ctx,
        ctx.get(key, 'corrector_module'),
        child_scopes,
        custom_coords,
        dt,
        f'{prefix}/~/dycore_with_physics_corrector',
    )
    return correctors.CustomCoordsCorrector(coords, custom_coords, inner)
  if name == 'DycoreWithPhysicsCorrector':
    equation_ref = ctx.get(key, 'dycore_equation_module')
    equation_key = ctx.resolve_key(equation_ref, child_scopes)
    equation_name = _class_name(equation_key)
    if equation_name not in _EQUATIONS:
      raise ValueError(f'unsupported equation {equation_name}')
    orography_module = _build_orography(
        ctx,
        ctx.get(equation_key, 'orography_module'),
        _child_scopes(equation_key, child_scopes),
        coords.horizontal,
    )
    with torch.no_grad():
      orography = orography_module()
    equation = primitive_equations.PrimitiveEquations(
        reference_temperature=ctx.aux['ref_temperatures'],
        orography=orography,
        coords=coords,
        physics_specs=ctx.physics,
        include_vertical_advection=ctx.get(
            equation_key, 'include_vertical_advection', True
        ),
        **_EQUATIONS[equation_name],
    )
    integrator_ref = ctx.get(key, 'time_integrator', None)
    if integrator_ref is None:
      integrator = time_integration.imex_rk_sil3
    else:
      integrator = _TIME_INTEGRATORS[
          _class_name(ctx.resolve_key(integrator_ref, child_scopes))
      ]
    return correctors.DycoreWithPhysicsCorrector(
        equation,
        dt,
        substeps=ctx.get(key, 'dycore_substeps'),
        time_integrator=integrator,
        filter_module=_build_step_filter(
            ctx, ctx.get(key, 'filter_module', None), child_scopes,
            coords.horizontal, dt,
        ),
        orography_module=orography_module,
    )
  raise ValueError(f'unsupported corrector {name}')


def _build_diagnostics(
    ctx: _Context, ref, scopes: tuple, prefix: str, dt: float
) -> Optional[nn.Module]:
  """Builds a (step or decoder) diagnostics module; None when unset."""
  if ref is None:
    return None
  from neuralgcm_torch import diagnostics  # deferred: rarely used

  key = ctx.resolve_key(ref, scopes)
  child_scopes = _child_scopes(key, scopes)
  name = _class_name(key)
  if name == 'SurfacePressureDiagnostics':
    return diagnostics.SurfacePressureDiagnostics(ctx.coords)
  if name == 'NodalModelDiagnosticsDecoder':
    return diagnostics.NodalModelDiagnosticsDecoder(ctx.coords)
  if name == 'PrecipitationDiagnosticsConstrained':
    is_precipitation = ctx.get(key, 'is_precipitation', True)
    predicted = (
        diagnostics.PRECIPITATION if is_precipitation
        else diagnostics.EVAPORATION
    )
    output_shapes = {
        predicted: (1,) + ctx.coords.horizontal.spec.nodal_shape
    }
    embedding_ref = ctx.get(key, 'embedding_module')
    embedding_key = ctx.resolve_key(embedding_ref, child_scopes)
    embedding_haiku = _EMBEDDING_HAIKU_NAMES[_class_name(embedding_key)]
    diagnostics_prefix = (
        f'{prefix}/~/'
        f'{diagnostics.PrecipitationDiagnosticsConstrained.HAIKU_NAME}'
    )
    embedding = _build_embedding(
        ctx, embedding_ref, child_scopes, ctx.coords.horizontal,
        output_shapes, f'{diagnostics_prefix}/~/{embedding_haiku}',
    )
    return diagnostics.PrecipitationDiagnosticsConstrained(
        ctx.coords,
        dt,
        ctx.physics,
        embedding,
        moisture_species=tuple(
            ctx.get(key, 'moisture_species', (
                'specific_humidity',
                'specific_cloud_ice_water_content',
                'specific_cloud_liquid_water_content',
            ))
        ),
        is_precipitation=is_precipitation,
        method_precipitation=ctx.get(
            key, 'method_precipitation', 'cumulative'
        ),
        method_evaporation=ctx.get(key, 'method_evaporation', 'rate'),
        field_name=ctx.get(key, 'field_name', 'total_precipitation'),
    )
  raise ValueError(f'unsupported diagnostics module {name}')


def _build_step(
    ctx: _Context, ref, scopes: tuple, prefix: str
) -> nn.Module:
  key = ctx.resolve_key(ref, scopes)
  child_scopes = _child_scopes(key, scopes)
  name = _class_name(key)
  if name != 'StochasticPhysicsParameterizationStep':
    raise ValueError(f'unsupported step {name}')
  num_substeps = ctx.get(key, 'num_substeps', 1)
  inner_dt = ctx.dt / num_substeps
  corrector_haiku = {
      'CustomCoordsCorrector': correctors.CustomCoordsCorrector.HAIKU_NAME,
      'DycoreWithPhysicsCorrector': (
          correctors.DycoreWithPhysicsCorrector.HAIKU_NAME
      ),
  }[_class_name(ctx.resolve_key(ctx.get(key, 'corrector_module'),
                                child_scopes))]
  corrector = _build_corrector(
      ctx,
      ctx.get(key, 'corrector_module'),
      child_scopes,
      ctx.coords,
      inner_dt,
      f'{prefix}/~/{corrector_haiku}',
  )
  parameterization = _build_parameterization(
      ctx,
      ctx.get(key, 'physics_parameterization_module'),
      child_scopes,
      inner_dt,
      f'{prefix}/~/'
      f'{parameterizations.DivCurlNeuralParameterization.HAIKU_NAME}',
  )
  randomness = _build_randomness(
      ctx,
      ctx.get(key, 'randomness_module', None),
      child_scopes,
      ctx.coords.horizontal,
      ctx.dt,
  ) or stochastic.ZerosRandomField(ctx.coords.horizontal)
  return steps.StochasticPhysicsParameterizationStep(
      corrector,
      parameterization,
      randomness,
      num_substeps=num_substeps,
      # the legacy BaseStep builds its diagnostics with the OUTER dt
      diagnostics_module=_build_diagnostics(
          ctx, ctx.get(key, 'diagnostics_module', None), child_scopes,
          prefix, ctx.dt,
      ),
  )


#  ===========================================================================
#  Encoders and decoders.
#  ===========================================================================


def _state_output_shapes(ctx: _Context, prediction_mask: dict, grid):
  """Encoder mapping output shapes, in State field order."""
  field_order = (
      'vorticity',
      'divergence',
      'temperature_variation',
      'log_surface_pressure',
      'tracers',
      'sim_time',
  )
  shapes = {}
  for field in field_order:
    mask = prediction_mask.get(field, False)
    if isinstance(mask, dict):
      tracers = {
          k: _state_field_shape(ctx, field, grid)
          for k in sorted(mask) if mask[k]
      }
      if tracers:
        shapes[field] = tracers
    elif mask:
      shapes[field] = _state_field_shape(ctx, field, grid)
  return shapes


def _wb_output_shapes(ctx: _Context, prediction_mask: dict):
  """Decoder mapping output shapes, in sorted (plain-dict) key order."""
  grid = ctx.output_coords.horizontal
  levels = ctx.output_coords.vertical.layers
  shapes = {}
  for field in sorted(prediction_mask):
    mask = prediction_mask[field]
    if isinstance(mask, dict):
      tracers = {
          k: (levels,) + grid.spec.nodal_shape
          for k in sorted(mask) if mask[k]
      }
      if tracers:
        shapes[field] = tracers
    elif mask:
      shapes[field] = (levels,) + grid.spec.nodal_shape
  return shapes


def _build_encoder_transform(
    ctx: _Context, ref, scopes: tuple
) -> Optional[nn.Module]:
  """Builds the encoder state transform (applied on the model grid)."""
  if ref is None:
    return None
  key = ctx.resolve_key(ref, scopes)
  child_scopes = _child_scopes(key, scopes)
  name = _class_name(key)
  grid = ctx.coords.horizontal
  if name == 'EncoderIdentityTransform':
    return encoders.EncoderIdentityTransform()
  if name == 'EncoderCombinedTransform':
    return encoders.EncoderCombinedTransform([
        _build_encoder_transform(ctx, t, child_scopes)
        for t in ctx.get(key, 'transforms')
    ])
  if name == 'InputClipTransform':
    return encoders.InputClipTransform(
        grid, ctx.get(key, 'wavenumbers_to_clip', 1)
    )
  if name == 'EncoderFilterTransform':
    return encoders.EncoderFilterTransform([
        _build_data_filter(ctx, f, child_scopes, grid)
        for f in ctx.get(key, 'filter_modules')
    ])
  raise ValueError(f'unsupported encoder transform {name}')


def _build_learned_encoder(
    ctx: _Context,
    key: str,
    scopes: tuple,
    prefix: str,
    randomness_ref,
    dimensional: bool,
) -> nn.Module:
  """Builds a (Dimensional)LearnedWeatherbenchToPrimitiveEncoder."""
  child_scopes = _child_scopes(key, scopes)
  prediction_mask = ctx.get(key, 'prediction_mask')
  orography = _build_orography(
      ctx, ctx.get(key, 'orography_module'), child_scopes,
      ctx.coords.horizontal,
  )
  data_ref = ctx.get(key, 'modal_to_nodal_data_features_module')
  model_ref = ctx.get(key, 'modal_to_nodal_model_features_module')
  data_key = ctx.resolve_key(data_ref, child_scopes)
  model_key = ctx.resolve_key(model_ref, child_scopes)
  data_haiku = _FEATURE_HAIKU_NAMES[_class_name(data_key)]
  model_haiku = _FEATURE_HAIKU_NAMES[_class_name(model_key)]
  model_suffix = (
      f'{model_haiku}_1' if model_haiku == data_haiku else model_haiku
  )
  # data features live on the input grid, model features on the model grid.
  data_features = _build_features(
      ctx, data_ref, child_scopes, ctx.input_coords.horizontal,
      f'{prefix}/~/{data_haiku}',
  )
  model_features = _build_features(
      ctx, model_ref, child_scopes, ctx.coords.horizontal,
      f'{prefix}/~/{model_suffix}',
  )
  mapping = _build_mapping(
      ctx,
      ctx.get(key, 'nodal_mapping_module'),
      child_scopes,
      _state_output_shapes(ctx, prediction_mask, ctx.coords.horizontal),
      f'{prefix}/nodal_mapping',
  )
  kwargs = dict(
      data_features=data_features,
      model_features=model_features,
      mapping=mapping,
      correction_transform=_build_transform(
          ctx, ctx.get(key, 'correction_transform_module', None),
          child_scopes, ctx.input_coords.horizontal,
      ),
      prediction_mask=prediction_mask,
      transform=_build_encoder_transform(
          ctx, ctx.get(key, 'transform_module', None), child_scopes
      ),
      randomness_module=_build_randomness(
          ctx, randomness_ref, child_scopes, ctx.coords.horizontal, ctx.dt
      ),
  )
  args = (
      ctx.coords,
      ctx.input_coords,
      ctx.aux['ref_temperatures'],
      orography,
      ctx.physics.gravity_acceleration,
  )
  if dimensional:
    return encoders.DimensionalLearnedWeatherbenchToPrimitiveEncoder(
        *args,
        physics_specs=ctx.physics,
        inputs_to_units_mapping=ctx.get(key, 'inputs_to_units_mapping'),
        **kwargs,
    )
  return encoders.LearnedWeatherbenchToPrimitiveEncoder(*args, **kwargs)


def _build_encoder(
    ctx: _Context, ref, scopes: tuple, prefix_root: str
) -> nn.Module:
  key = ctx.resolve_key(ref, scopes)
  name = _class_name(key)
  if name == 'DimensionalLearnedWeatherbenchToPrimitiveEncoder':
    prefix = f'{prefix_root}/~/{encoders.DimensionalLearnedWeatherbenchToPrimitiveEncoder.HAIKU_NAME}'
    return _build_learned_encoder(
        ctx, key, scopes, prefix,
        ctx.get(key, 'randomness_module', None),
        dimensional=True,
    )
  if name == 'DimensionalLearnedWeatherbenchToPrimitiveWithMemoryEncoder':
    cls = (
        encoders.DimensionalLearnedWeatherbenchToPrimitiveWithMemoryEncoder
    )
    prefix = f'{prefix_root}/~/{cls.HAIKU_NAME}'
    child = 'learned_weatherbench_to_primitive_encoder'
    # the memory encoder is deterministic; the state encoder uses the bound
    # randomness module (default ZerosRandomField, as in the legacy class).
    memory_encoder = _build_learned_encoder(
        ctx, key, scopes, f'{prefix}/~/{child}',
        randomness_ref=None, dimensional=False,
    )
    randomness_ref = ctx.get(
        key, 'randomness_module', {'__ref__': 'ZerosRandomField',
                                   '__call__': False}
    )
    state_encoder = _build_learned_encoder(
        ctx, key, scopes, f'{prefix}/~/{child}_1',
        randomness_ref=randomness_ref, dimensional=False,
    )
    return cls(
        memory_encoder,
        state_encoder,
        physics_specs=ctx.physics,
        inputs_to_units_mapping=ctx.get(key, 'inputs_to_units_mapping'),
    )
  raise ValueError(f'unsupported encoder {name}')


def _build_decoder(
    ctx: _Context, ref, scopes: tuple, prefix_root: str
) -> nn.Module:
  key = ctx.resolve_key(ref, scopes)
  child_scopes = _child_scopes(key, scopes)
  name = _class_name(key)
  if name != 'DimensionalLearnedPrimitiveToWeatherbenchDecoder':
    raise ValueError(f'unsupported decoder {name}')
  cls = decoders.DimensionalLearnedPrimitiveToWeatherbenchDecoder
  prefix = f'{prefix_root}/~/{cls.HAIKU_NAME}'
  prediction_mask = ctx.get(key, 'prediction_mask')
  orography = _build_orography(
      ctx, ctx.get(key, 'orography_module'), child_scopes,
      ctx.coords.horizontal,
  )
  geopotential = primitive_equations.Geopotential(
      ctx.coords.vertical.coordinates,
      gravity_acceleration=ctx.physics.gravity_acceleration,
      ideal_gas_constant=ctx.physics.ideal_gas_constant,
      water_vapor_gas_constant=ctx.physics.water_vapor_gas_constant,
      device=ctx.device,
      dtype=ctx.dtype,
  )
  model_ref = ctx.get(key, 'modal_to_nodal_model_features_module')
  data_ref = ctx.get(key, 'modal_to_nodal_data_features_module')
  model_key = ctx.resolve_key(model_ref, child_scopes)
  data_key = ctx.resolve_key(data_ref, child_scopes)
  model_haiku = _FEATURE_HAIKU_NAMES[_class_name(model_key)]
  data_haiku = _FEATURE_HAIKU_NAMES[_class_name(data_key)]
  data_suffix = (
      f'{data_haiku}_1' if data_haiku == model_haiku else data_haiku
  )
  model_features = _build_features(
      ctx, model_ref, child_scopes, ctx.coords.horizontal,
      f'{prefix}/~/{model_haiku}',
  )
  data_features = _build_features(
      ctx, data_ref, child_scopes, ctx.output_coords.horizontal,
      f'{prefix}/~/{data_suffix}',
  )
  mapping = _build_mapping(
      ctx,
      ctx.get(key, 'nodal_mapping_module'),
      child_scopes,
      _wb_output_shapes(ctx, prediction_mask),
      f'{prefix}/nodal_mapping',
  )
  return cls(
      ctx.coords,
      ctx.output_coords,
      ctx.aux['ref_temperatures'],
      orography,
      geopotential,
      model_features=model_features,
      data_features=data_features,
      mapping=mapping,
      correction_transform=_build_transform(
          ctx, ctx.get(key, 'correction_transform_module', None),
          child_scopes, ctx.output_coords.horizontal,
      ),
      prediction_mask=prediction_mask,
      randomness_module=_build_randomness(
          ctx, ctx.get(key, 'randomness_module', None), child_scopes,
          ctx.coords.horizontal, ctx.dt,
      ),
      diagnostics_module=_build_diagnostics(
          ctx, ctx.get(key, 'diagnostics_module', None), child_scopes,
          prefix, ctx.dt,
      ),
      physics_specs=ctx.physics,
      inputs_to_units_mapping=ctx.get(key, 'inputs_to_units_mapping'),
  )


def _build_forcing(ctx: _Context, ref, scopes: tuple) -> nn.Module:
  if ref is None:
    return forcings.NoForcing()
  key = ctx.resolve_key(ref, scopes)
  name = _class_name(key)
  if name == 'NoForcing':
    return forcings.NoForcing()
  if name == 'DynamicDataForcing':
    return forcings.DynamicDataForcing(
        ctx.physics,
        inputs_to_units_mapping=ctx.get(key, 'inputs_to_units_mapping'),
        time_axis=ctx.get(key, 'time_axis', 0),
        dt_tolerance=ctx.get(key, 'dt_tolerance', '1 hour'),
    )
  raise ValueError(f'unsupported forcing {name}')


def _build_model(ctx: _Context) -> models.StochasticModularStepModel:
  key = 'StochasticModularStepModel'
  if key not in ctx.bindings:
    raise ValueError('checkpoint does not configure a '
                     'StochasticModularStepModel')
  scopes = ()
  prefix_root = models.StochasticModularStepModel.HAIKU_NAME
  advance = _build_step(
      ctx,
      ctx.get(key, 'advance_module'),
      scopes,
      f'{prefix_root}/~/'
      f'{steps.StochasticPhysicsParameterizationStep.HAIKU_NAME}',
  )
  encoder = _build_encoder(
      ctx, ctx.get(key, 'encoder_module'), scopes, prefix_root
  )
  decoder = _build_decoder(
      ctx, ctx.get(key, 'decoder_module'), scopes, prefix_root
  )
  forcing = _build_forcing(
      ctx, ctx.get(key, 'forcing_module', None), scopes
  )
  return models.StochasticModularStepModel(encoder, decoder, advance, forcing)


#  ===========================================================================
#  Entry point.
#  ===========================================================================


def from_checkpoint(
    checkpoint: dict,
    device: Any = None,
    dtype: torch.dtype = torch.float32,
) -> models.StochasticModularStepModel:
  """Builds the model described by a converted checkpoint, weights loaded.

  Args:
    checkpoint: a converted checkpoint dict (see
      `neuralgcm_torch.checkpoint.load`).
    device: device for all module buffers and parameters.
    dtype: floating dtype for all module buffers and parameters.

  Returns:
    A `StochasticModularStepModel` with the checkpoint weights imported.
  """
  config = checkpoint['config']
  if 'model' not in config:
    raise ValueError(
        'checkpoint lacks the model bindings; re-convert it from the '
        'original checkpoint'
    )

  grid = spherical_harmonic.Grid(
      checkpoint_lib.grid_spec_from_config(config['model_grid']),
      impl=checkpoint_lib.spherical_harmonics_impl_from_config(
          config['model_grid']
      ),
      device=device,
      dtype=dtype,
  )
  sigma = sigma_coordinates.SigmaLevels(
      checkpoint_lib.sigma_coordinates_from_config(config),
      device=device,
      dtype=dtype,
  )
  coords = coordinate_systems.CoordinateSystem(grid, sigma)
  physics = checkpoint_lib.sim_units_from_config(config)

  data_grid_config = config['data_grid'] or config['model_grid']
  data_grid = spherical_harmonic.Grid(
      checkpoint_lib.grid_spec_from_config(data_grid_config),
      impl=checkpoint_lib.spherical_harmonics_impl_from_config(
          data_grid_config
      ),
      device=device,
      dtype=dtype,
  )
  nondim_centers = physics.nondimensionalize(
      np.asarray(config['data_pressure_levels']) * scales.units.millibar
  )
  pressure = vertical_interpolation.PressureLevels(
      vertical_interpolation.PressureCoordinates(np.asarray(nondim_centers)),
      device=device,
      dtype=dtype,
  )
  data_coords = coordinate_systems.CoordinateSystem(data_grid, pressure)

  ctx = _Context(
      bindings=config['model'],
      params=checkpoint['params'],
      aux=checkpoint['aux_features'],
      data_config=config['data'],
      coords=coords,
      input_coords=data_coords,
      output_coords=data_coords,
      physics=physics,
      dt=float(config['dt']),
      ref_datetime=np.datetime64(config['reference_datetime']),
      device=device,
      dtype=dtype,
  )
  model = _build_model(ctx)
  model.import_haiku(checkpoint['params'])
  return model
