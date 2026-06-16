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
"""Transformations that pre/post-process dictionaries of fields.

Transforms operate on (possibly nested) dictionaries of tensors. Unlike the
original JAX implementation, constructors take only what each transform actually uses (the
gin factory protocol passed `coords, dt, physics_specs, aux_features` to
every module); the checkpoint/model builder supplies these explicitly.

Only the transforms referenced by published checkpoint configs are ported.
None of them hold learned parameters.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dinosaur_torch import sigma_coordinates
from dinosaur_torch import spherical_harmonic


class KeyWithCosLatFactor(NamedTuple):
  """Feature key carrying the number of accumulated 1/cos(lat) factors."""

  name: str
  factor_order: int
  attenuation: Optional[float] = None


def _map_leaves(fn: Callable, tree):
  """Applies `fn` to non-dict values of a nested dict (None passes through)."""
  if isinstance(tree, dict):
    return {k: _map_leaves(fn, v) for k, v in tree.items()}
  if tree is None:
    return None
  return fn(tree)


def _map_matching_keys(inputs: dict, fn: Callable, keys: Sequence[str]):
  """Applies `fn` to values (or sub-dict values) whose key is in `keys`."""
  outputs = {}
  for k, v in inputs.items():
    if isinstance(v, dict):
      outputs[k] = _map_matching_keys(v, fn, keys)
    else:
      outputs[k] = fn(v) if k in keys else v
  return outputs


class EmptyTransform(nn.Module):
  """Transform returns an empty dict."""

  def forward(self, inputs) -> dict:
    del inputs  # unused
    return {}


class IdentityTransform(nn.Module):
  """Transform does not modify inputs."""

  def forward(self, inputs):
    return inputs


def _flatten_keys(tree: dict, prefix: str = '', sep: str = '&') -> dict:
  """Flattens a nested dict into `sep`-joined path keys (legacy convention)."""
  flat = {}
  for k, v in tree.items():
    path = f'{prefix}{sep}{k}' if prefix else k
    if isinstance(v, dict):
      flat.update(_flatten_keys(v, path, sep))
    else:
      flat[path] = v
  return flat


class ShiftAndNormalize(nn.Module):
  """Shifts and normalizes values: `(x - shift) / scale` per key.

  `shifts`/`scales` map variable names to scalars; like the original JAX implementation,
  keys for values inside nested sub-dicts are matched by their '&'-joined
  path (or equivalently by nesting `shifts`/`scales` the same way). Extra
  keys are ignored; missing keys raise.
  """

  def __init__(self, shifts: dict, scales: dict,
               global_scale: Optional[float] = None):
    super().__init__()
    self.shifts = _flatten_keys(shifts)
    scales = _flatten_keys(scales)
    if global_scale is not None:
      scales = {k: v * global_scale for k, v in scales.items()}
    self.scales = scales

  def _transform_leaf(self, key, x):
    return (x - self.shifts[key]) / self.scales[key]

  def _apply(self, inputs: dict, prefix: str = '') -> dict:
    outputs = {}
    for k, v in inputs.items():
      path = f'{prefix}&{k}' if prefix else k
      if isinstance(v, dict):
        outputs[k] = self._apply(v, path)
      elif v is None:
        outputs[k] = None
      else:
        outputs[k] = self._transform_leaf(path, v)
    return outputs

  def forward(self, inputs: dict) -> dict:
    return self._apply(inputs)


class InverseShiftAndNormalize(ShiftAndNormalize):
  """Inverse of `ShiftAndNormalize` for the same `shifts`/`scales`."""

  def _transform_leaf(self, key, x):
    return x * self.scales[key] + self.shifts[key]


class NondimensionalizeTransform(nn.Module):
  """Nondimensionalizes inputs by per-key unit scale factors.

  Factors are computed once from `inputs_to_units_mapping` (linear units
  only, which covers all checkpoint configurations); keys follow the same
  '&'-joined flattened-path matching as `ShiftAndNormalize`.
  """

  def __init__(self, physics_specs, inputs_to_units_mapping: dict):
    super().__init__()
    from dinosaur_torch import scales  # local to avoid cycle at import

    self.scale_factors = {
        key: float(
            physics_specs.nondimensionalize(1.0 * scales.parse_units(unit))
        )
        for key, unit in _flatten_keys(inputs_to_units_mapping).items()
    }

  def _apply(self, inputs: dict, prefix: str = '') -> dict:
    outputs = {}
    for k, v in inputs.items():
      path = f'{prefix}&{k}' if prefix else k
      if isinstance(v, dict):
        outputs[k] = self._apply(v, path)
      else:
        outputs[k] = v * self.scale_factors[path]
    return outputs

  def forward(self, inputs: dict) -> dict:
    return self._apply(inputs)


class RedimensionalizeTransform(NondimensionalizeTransform):
  """Redimensionalizes inputs (inverse of `NondimensionalizeTransform`)."""

  def __init__(self, physics_specs, inputs_to_units_mapping: dict):
    super().__init__(physics_specs, inputs_to_units_mapping)
    self.scale_factors = {
        k: 1.0 / v for k, v in self.scale_factors.items()
    }


class SequentialTransform(nn.Module):
  """Applies multiple transforms sequentially."""

  def __init__(self, transforms: Sequence[nn.Module]):
    super().__init__()
    self.transforms = nn.ModuleList(transforms)

  def forward(self, inputs):
    for transform in self.transforms:
      inputs = transform(inputs)
    return inputs


class LevelScale(nn.Module):
  """Scales selected variables by a per-level factor."""

  def __init__(
      self,
      scales: Sequence[float],
      keys_to_scale: Sequence[str] = tuple(),
      *,
      inverse: bool = False,
      device=None,
      dtype: torch.dtype = torch.float32,
  ):
    super().__init__()
    self.keys_to_scale = tuple(keys_to_scale)
    scales = np.asarray(scales, np.float64)
    if inverse:
      scales = 1 / scales
    self.register_buffer(
        '_scales',
        torch.as_tensor(scales, dtype=dtype, device=device)[:, None, None],
        persistent=False,
    )

  def forward(self, inputs: dict) -> dict:
    return _map_matching_keys(
        inputs, lambda x: x * self._scales, self.keys_to_scale
    )


class InverseLevelScale(LevelScale):
  """Scales selected variables by the inverse per-level factor."""

  def __init__(self, scales, keys_to_scale=tuple(), **kwargs):
    super().__init__(scales, keys_to_scale, inverse=True, **kwargs)


class HardClip(nn.Module):
  """Hard-clips values to (-max_value, max_value)."""

  def __init__(self, max_value: float):
    super().__init__()
    self.max_value = max_value

  def forward(self, inputs):
    clip = lambda x: torch.clamp(x, min=-self.max_value, max=self.max_value)
    return _map_leaves(clip, inputs)


class SoftClip(nn.Module):
  """Clips values to (-max_value, max_value) with smooth boundaries.

  Values outside the range are mapped into intervals of width approximately
  `log(2) * hinge_softness` on the interior of each boundary.
  """

  def __init__(self, max_value: float, hinge_softness: float = 1.0):
    super().__init__()
    if max_value < 0 or hinge_softness < 0:
      raise ValueError(
          'max_value and hinge_softness must be positive, '
          f'{max_value=}, {hinge_softness=}'
      )
    self.low = -max_value
    self.high = max_value
    self.hinge = hinge_softness
    # softplus of the scalar span, kept as a python float.
    self.span_softplus = float(
        hinge_softness * np.logaddexp(0.0, 2 * max_value / hinge_softness)
    )

  def _clip(self, x: torch.Tensor) -> torch.Tensor:
    softplus = lambda v: self.hinge * F.softplus(v / self.hinge)
    span = self.high - self.low
    return (
        -softplus(span - softplus(x - self.low)) * span / self.span_softplus
        + self.high
    )

  def forward(self, inputs):
    return _map_leaves(self._clip, inputs)


class ClipTransform(nn.Module):
  """Clips the highest total wavenumbers of (modal) inputs."""

  def __init__(self, grid: spherical_harmonic.Grid,
               wavenumbers_to_clip: int = 1):
    super().__init__()
    self.grid = grid
    self.wavenumbers_to_clip = wavenumbers_to_clip

  def forward(self, inputs):
    return self.grid.clip_wavenumbers(inputs, self.wavenumbers_to_clip)


class ModalToNodalTransform(nn.Module):
  """Converts modal inputs to nodal representation."""

  def __init__(self, grid: spherical_harmonic.Grid):
    super().__init__()
    self.grid = grid

  def forward(self, inputs):
    return self.grid.to_nodal(inputs)


class NodalToModalTransform(nn.Module):
  """Converts nodal inputs to modal representation."""

  def __init__(self, grid: spherical_harmonic.Grid):
    super().__init__()
    self.grid = grid

  def forward(self, inputs):
    return self.grid.to_modal(inputs)


class ToModalDiffOperators(nn.Module):
  """Returns gradient and Laplacian features of (modal) input fields.

  To avoid accidental accumulation of cos(lat) factors, features are keyed
  by `KeyWithCosLatFactor`.
  """

  def __init__(self, grid: spherical_harmonic.Grid):
    super().__init__()
    self.grid = grid

  def forward(self, inputs: dict) -> dict:
    features = {}
    for k, value in inputs.items():
      name, cos_lat_order = k.name, k.factor_order
      d_value_dlon, d_value_dlat = self.grid.cos_lat_grad(value)
      laplacian_value = self.grid.laplacian(value)
      dlon_key = KeyWithCosLatFactor(name + '_dlon', cos_lat_order + 1)
      dlat_key = KeyWithCosLatFactor(name + '_dlat', cos_lat_order + 1)
      del2_key = KeyWithCosLatFactor(name + '_del2', cos_lat_order)
      features[dlon_key] = d_value_dlon
      features[dlat_key] = d_value_dlat
      features[del2_key] = laplacian_value
    return features


class TruncateSigmaLevels(nn.Module):
  """Truncates vertical levels for specified variables.

  `sigma_ranges` maps variable names to `(sigma_min, sigma_max)`; variables
  not listed keep all levels.
  """

  def __init__(
      self,
      coordinates: sigma_coordinates.SigmaCoordinates,
      sigma_ranges: Dict[str, Tuple[float, float]],
  ):
    super().__init__()
    self.sigma_ranges = dict(sigma_ranges)
    self.sigma_levels = coordinates.centers
    self._slices: dict[tuple, slice] = {}

  def _slice_for(self, sigma_range: Tuple[float, float]) -> slice:
    key = tuple(sigma_range)
    cached = self._slices.get(key)
    if cached is None:
      sigma_min, sigma_max = sigma_range
      lower_index = int(np.argmax((self.sigma_levels - sigma_min) > 0))
      if sigma_max > np.max(self.sigma_levels):
        upper_index = len(self.sigma_levels)
      else:
        upper_index = int(np.argmin((self.sigma_levels - sigma_max) < 0))
      cached = slice(lower_index, upper_index)
      self._slices[key] = cached
    return cached

  def forward(self, inputs: dict) -> dict:
    outputs = {}
    for k, v in inputs.items():
      if isinstance(v, dict):
        outputs[k] = self.forward(v)
      else:
        sigma_range = self.sigma_ranges.get(k, (0, 1))
        outputs[k] = v[..., self._slice_for(sigma_range), :, :]
    return outputs


class TakeSurfaceAdjacentSigmaLevel(nn.Module):
  """Retains only the vertical level nearest to the Earth's surface."""

  def forward(self, inputs):
    # the level axis is third from the end: leaves are (level, lon, lat)
    # with an optional leading batch (member) dimension
    return _map_leaves(lambda x: x[..., -1:, :, :], inputs)


class FeatureSelector(nn.Module):
  """Retains items whose keys fully match a regex pattern."""

  def __init__(self, regex_patterns: str):
    super().__init__()
    self.regex_patterns = regex_patterns

  def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v
        for k, v in inputs.items()
        if re.fullmatch(self.regex_patterns, k)
    }
