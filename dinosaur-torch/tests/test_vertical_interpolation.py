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

"""Tests for vertical_interpolation."""

import numpy as np
import pytest
import torch

from dinosaur_torch import sigma_coordinates
from dinosaur_torch import vertical_interpolation as vi


def test_pressure_coordinates_validation():
  coords = vi.PressureCoordinates([100, 500, 1000])
  assert coords.layers == 3
  with pytest.raises(ValueError, match='monotonically increasing'):
    vi.PressureCoordinates([100, 100, 1000])


def test_interp_matches_numpy(device):
  rs = np.random.RandomState(0)
  xp = np.sort(rs.rand(10))
  fp = rs.randn(10)
  x = np.concatenate([rs.rand(20), [-0.5, 1.5]])  # includes out-of-range
  expected = np.interp(x, xp, fp)
  actual = vi.interp(
      torch.as_tensor(x, dtype=torch.float64, device=device),
      torch.as_tensor(xp, dtype=torch.float64, device=device),
      torch.as_tensor(fp, dtype=torch.float64, device=device),
  )
  np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=1e-12)


def test_interp_columns_match_per_column_numpy(device):
  rs = np.random.RandomState(1)
  xp = np.sort(rs.rand(7))
  fp = rs.randn(7, 4, 5)
  x = rs.rand(3, 4, 5)
  actual = vi.interp(
      torch.as_tensor(x, dtype=torch.float64, device=device),
      torch.as_tensor(xp, dtype=torch.float64, device=device),
      torch.as_tensor(fp, dtype=torch.float64, device=device),
  ).cpu().numpy()
  for i in range(4):
    for j in range(5):
      expected = np.interp(x[:, i, j], xp, fp[:, i, j])
      np.testing.assert_allclose(actual[:, i, j], expected, atol=1e-12)


def test_safe_extrap_nan_pattern(device):
  xp = torch.linspace(0, 1, 5, dtype=torch.float64, device=device)
  fp = 2 * xp
  x = torch.tensor(
      [-0.5, -0.2, 0.5, 1.2, 1.5], dtype=torch.float64, device=device
  )
  out = vi.interp_with_safe_extrap(x, xp, fp).cpu().numpy()
  # one cell (0.25) of linear extrapolation on each side, NaN beyond
  assert np.isnan(out[0]) and np.isnan(out[4])
  np.testing.assert_allclose(out[1:4], [-0.4, 1.0, 2.4], atol=1e-12)


def test_round_trip_pressure_sigma(device):
  pressure = vi.PressureLevels(
      vi.PressureCoordinates([100, 200, 300, 500, 700, 850, 1000]),
      device=device,
      dtype=torch.float64,
  )
  sigma = sigma_coordinates.SigmaLevels(
      sigma_coordinates.SigmaCoordinates.equidistant(20),
      device=device,
      dtype=torch.float64,
  )
  shape = (8, 4)
  surface_pressure = torch.full(
      (1,) + shape, 1000.0, dtype=torch.float64, device=device
  )
  # a smooth field linear in pressure interpolates exactly
  p = pressure.centers[:, None, None].expand(-1, *shape)
  fields = {'linear': 2.0 * p + 1.0}
  on_sigma = pressure.to_sigma(fields, sigma, surface_pressure)
  assert on_sigma['linear'].shape == (20,) + shape
  back = pressure.from_sigma(on_sigma, sigma, surface_pressure)
  np.testing.assert_allclose(
      back['linear'].cpu().numpy(),
      fields['linear'].cpu().numpy(),
      rtol=1e-10,
  )


def test_get_surface_pressure_exact_for_linear_geopotential(device):
  levels = torch.tensor(
      [100.0, 300.0, 500.0, 700.0, 900.0], dtype=torch.float64, device=device
  )
  g = 9.81
  shape = (6, 3)
  # geopotential height decreasing linearly with pressure: z = (1000 - p)
  z = (1000.0 - levels)[:, None, None].expand(-1, *shape)
  geopotential = g * z
  orography = torch.zeros((1,) + shape, dtype=torch.float64, device=device)
  # surface (z = 0) is reached exactly at p = 1000 (linear extrapolation)
  sp = vi.get_surface_pressure(levels, geopotential, orography, g)
  np.testing.assert_allclose(sp.cpu().numpy(), 1000.0, rtol=1e-12)
