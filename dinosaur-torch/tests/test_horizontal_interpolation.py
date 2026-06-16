# Copyright 2023 Google LLC
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

"""Tests for horizontal_interpolation (ported from the original Dinosaur, pytest style)."""

import functools

import numpy as np
import pytest
import torch

from dinosaur_torch import horizontal_interpolation
from dinosaur_torch import spherical_harmonic


def _to_numpy(x):
  if isinstance(x, torch.Tensor):
    return x.detach().cpu().numpy()
  return np.asarray(x)


def test_conservative_latitude_weights():
  source_lat = np.pi / 180 * np.array([-75, -45, -15, 15, 45, 75])
  target_lat = np.pi / 180 * np.array([-45, 45])
  expected = np.array([
      [1 - np.sqrt(3) / 2, (np.sqrt(3) - 1) / 2, 1 / 2, 0, 0, 0],
      [0, 0, 0, 1 / 2, (np.sqrt(3) - 1) / 2, 1 - np.sqrt(3) / 2],
  ])
  actual = _to_numpy(
      horizontal_interpolation.conservative_latitude_weights(source_lat, target_lat)
  )
  np.testing.assert_almost_equal(expected, actual, decimal=6)


@pytest.mark.parametrize('x,y,expected', [
    (1, 0, 1),
    (-1, 0, -1),
    (5, 0, 5),
    (6, 0, -4),
    (1, 9, 11),
    (5, 9, 5),
])
def test_align_phase_with(x, y, expected):
  actual = horizontal_interpolation._align_phase_with(x, y, period=10)
  assert actual == expected


def test_conservative_longitude_weights():
  source_lon = np.pi / 180 * np.array([0, 60, 120, 180, 240, 300])
  target_lon = np.pi / 180 * np.array([0, 90, 180, 270])
  expected = np.array([
      [4, 1, 0, 0, 0, 1],
      [0, 3, 3, 0, 0, 0],
      [0, 0, 1, 4, 1, 0],
      [0, 0, 0, 0, 3, 3],
  ]) / 6
  actual = _to_numpy(
      horizontal_interpolation.conservative_longitude_weights(source_lon, target_lon)
  )
  np.testing.assert_allclose(expected, actual, atol=1e-5)


@pytest.mark.parametrize('regridder_cls', [
    horizontal_interpolation.BilinearRegridder,
    horizontal_interpolation.ConservativeRegridder,
    horizontal_interpolation.NearestRegridder,
])
def test_regridding_shape(regridder_cls):
  source_grid = spherical_harmonic.GridSpec.T85()
  target_grid = spherical_harmonic.GridSpec.T21()
  regridder = regridder_cls(source_grid, target_grid)

  inputs = torch.zeros(source_grid.nodal_shape)
  outputs = regridder(inputs)
  assert tuple(outputs.shape) == target_grid.nodal_shape

  batch_inputs = torch.zeros((2,) + source_grid.nodal_shape)
  batch_outputs = regridder(batch_inputs)
  assert tuple(batch_outputs.shape) == (2,) + target_grid.nodal_shape


@pytest.mark.parametrize('regridder_cls', [
    horizontal_interpolation.BilinearRegridder,
    horizontal_interpolation.NearestRegridder,
    functools.partial(horizontal_interpolation.ConservativeRegridder, skipna=True),
    functools.partial(horizontal_interpolation.ConservativeRegridder, skipna=False),
])
def test_regridding_nans(regridder_cls):
  # Use small grids to stay within CI memory limits (16 GB einsum with TL255→TL127).
  source_grid = spherical_harmonic.GridSpec.TL63(latitude_spacing='equiangular')
  target_grid = spherical_harmonic.GridSpec.TL31()
  regridder = regridder_cls(source_grid, target_grid)

  in_valid = (
      source_grid.latitudes[np.newaxis, :] ** 2
      + (source_grid.longitudes[:, np.newaxis] - np.pi) ** 2
      < (np.pi / 2) ** 2
  )
  inputs = torch.where(
      torch.as_tensor(in_valid),
      torch.ones(source_grid.nodal_shape),
      torch.full(source_grid.nodal_shape, float('nan')),
  )
  outputs = _to_numpy(regridder(inputs))

  out_valid = ~np.isnan(outputs)
  np.testing.assert_allclose(out_valid.mean(), in_valid.mean(), atol=0.1)
  np.testing.assert_allclose(outputs[out_valid], 1.0)
