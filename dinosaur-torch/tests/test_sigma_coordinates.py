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

"""Tests for sigma_coordinates (ported from the original Dinosaur, pytest style)."""

import numpy as np
import pytest
import torch

from dinosaur_torch import sigma_coordinates
from dinosaur_torch import spherical_harmonic

SigmaCoordinates = sigma_coordinates.SigmaCoordinates
SigmaLevels = sigma_coordinates.SigmaLevels


def _broadcast(*args):
  """Reshapes `args` so that they will broadcast over `len(args)` dims."""
  broadcasted = []
  for j, arg in enumerate(args):
    shape = [1] * len(args)
    shape[j] = -1
    broadcasted.append(arg.reshape(shape))
  return broadcasted


def quadratic_function(sigma, lon, lat):
  sigma, lon, lat = _broadcast(sigma, lon, lat)
  return sigma**2 * (1 + np.cos(lon) * np.cos(lat))


def quadratic_derivative(sigma, lon, lat):
  sigma, lon, lat = _broadcast(sigma, lon, lat)
  return 2 * sigma * (1 + np.cos(lon) * np.cos(lat))


def quadratic_integral(sigma, lon, lat):
  sigma, lon, lat = _broadcast(sigma, lon, lat)
  return sigma**3 / 3 * (1 + np.cos(lon) * np.cos(lat))


def exponential_function(sigma, lon, lat):
  sigma, lon, lat = _broadcast(sigma, lon, lat)
  return np.exp(sigma) * np.cos(lon) * np.sin(lat)


TEST_CASES = [
    pytest.param(
        quadratic_function,
        quadratic_derivative,
        quadratic_integral,
        np.array([10, 20, 40, 80, 160, 320]),
        8,
        id='quadratic',
    ),
    pytest.param(
        exponential_function,
        exponential_function,
        exponential_function,
        np.array([10, 20, 40, 80, 160]),
        16,
        id='exponential',
    ),
]


def _test_error_scaling(layers, errors, error_scaling):
  """Checks that `errors` scales with `layers` per `error_scaling`."""
  log_error_ratios = np.diff(np.log(errors))
  log_expected_ratios = np.diff(np.log(error_scaling(layers)))
  np.testing.assert_allclose(log_error_ratios, log_expected_ratios, atol=0.075)


def make_levels(nlayers, device) -> SigmaLevels:
  coordinates = SigmaCoordinates.equidistant(nlayers)
  return SigmaLevels(coordinates, device=device, dtype=torch.float64)


def nodal_axes(grid_resolution):
  spec = spherical_harmonic.GridSpec.with_wavenumbers(grid_resolution)
  return spec.nodal_axes


def to_tensor(x, device):
  return torch.as_tensor(x, dtype=torch.float64, device=device)


@pytest.mark.parametrize(
    'boundaries',
    [[0.0, 0.5, 1.0], (0.0, 0.5, 1.0), np.array([0.0, 0.5, 1.0])],
)
def test_initialization_casting(boundaries):
  coordinates = SigmaCoordinates(boundaries)
  assert isinstance(coordinates.boundaries, np.ndarray)
  assert isinstance(coordinates.asdict()['boundaries'], list)
  np.testing.assert_array_equal(coordinates.asdict()['boundaries'], boundaries)


def test_initialization_raises():
  with pytest.raises(ValueError, match=r'Expected boundaries\[0\] = 0'):
    SigmaCoordinates([0.2, 0.5, 1])
  with pytest.raises(ValueError, match=r'Expected boundaries\[0\] = 0'):
    SigmaCoordinates([0.0, 0.5, 0.9])
  with pytest.raises(ValueError, match='monotonically increasing'):
    SigmaCoordinates([0.0, 0.5, 0.5, 1])


def test_from_centers():
  centers = np.array([0.1, 0.35, 0.75], dtype=np.float32)
  coordinates = SigmaCoordinates.from_centers(centers)
  expected_boundaries = np.array([0.0, 0.2, 0.5, 1.0], dtype=np.float32)
  np.testing.assert_array_equal(coordinates.boundaries, expected_boundaries)
  np.testing.assert_array_equal(coordinates.centers, centers)


@pytest.mark.parametrize(
    'test_function,derivative_function,integral_function,layers,'
    'grid_resolution',
    TEST_CASES,
)
def test_centered_difference(
    test_function,
    derivative_function,
    integral_function,
    layers,
    grid_resolution,
    device,
):
  """`centered_difference` matches the closed form derivative."""
  del integral_function
  lon, lat = nodal_axes(grid_resolution)
  levels = make_levels(layers[-1], device)
  coordinates = levels.coordinates
  x = to_tensor(test_function(coordinates.centers, lon, lat), device)
  expected = derivative_function(coordinates.internal_boundaries, lon, lat)
  computed = levels.centered_difference(x).cpu().numpy()
  np.testing.assert_allclose(expected, computed, atol=1e-3)


@pytest.mark.parametrize('downward', [True, False])
@pytest.mark.parametrize(
    'test_function,derivative_function,integral_function,layers,'
    'grid_resolution',
    TEST_CASES,
)
def test_cumulative_sigma_integral(
    test_function,
    derivative_function,
    integral_function,
    layers,
    grid_resolution,
    downward,
    device,
):
  """Cumulative integrals converge at 1/layers² and match `sigma_integral`."""
  del derivative_function
  lon, lat = nodal_axes(grid_resolution)
  total_errors = []
  for nlayers in layers:
    levels = make_levels(nlayers, device)
    coordinates = levels.coordinates
    x = to_tensor(test_function(coordinates.centers, lon, lat), device)
    indefinite = integral_function(coordinates.boundaries, lon, lat)
    if downward:
      expected = (indefinite[1:] - indefinite[0])[-1]
      edge = -1
    else:
      expected = -(indefinite[:-1] - indefinite[-1])[0]
      edge = 0
    computed = levels.cumulative_sigma_integral(x, downward=downward)
    computed = computed.cpu().numpy()
    computed_all = levels.sigma_integral(x, keepdims=False).cpu().numpy()
    np.testing.assert_allclose(computed[edge], computed_all, atol=1e-6)
    total_errors.append(np.abs(expected - computed[edge]).max())
  # Midpoint rule: errors scale as 1 / layers².
  _test_error_scaling(layers, total_errors, lambda l: 1 / l**2)
  np.testing.assert_allclose(expected, computed[edge], atol=1e-2)


@pytest.mark.parametrize('downward', [True, False])
@pytest.mark.parametrize(
    'test_function,derivative_function,integral_function,layers,'
    'grid_resolution',
    TEST_CASES,
)
def test_cumulative_log_sigma_integral(
    test_function,
    derivative_function,
    integral_function,
    layers,
    grid_resolution,
    downward,
    device,
):
  """Log-sigma integrals converge with the log-space spacing squared."""
  del derivative_function
  lon, lat = nodal_axes(grid_resolution)
  total_errors = []
  for nlayers in layers:
    levels = make_levels(nlayers, device)
    coordinates = levels.coordinates
    centers = coordinates.centers
    # We integrate ∫f(x) 𝜎 d(log𝜎) = ∫f(x) d𝜎
    x = test_function(centers, lon, lat) * centers[:, np.newaxis, np.newaxis]
    x = to_tensor(x, device)
    indefinite = integral_function(centers, lon, lat)
    if downward:
      boundary = integral_function(np.zeros(1), lon, lat)
      expected = (indefinite - boundary)[-1]
      edge = -1
    else:
      boundary = integral_function(np.array([centers[-1]]), lon, lat)
      expected = -(indefinite - boundary)[0]
      edge = 0
    computed = levels.cumulative_log_sigma_integral(x, downward=downward)
    computed = computed.cpu().numpy()
    total_errors.append(np.abs(expected - computed[edge]).max())

  def error_scaling(layers):
    expected_scaling = []
    for l in layers:
      centers = SigmaCoordinates.equidistant(l).centers
      log_space_widths = np.diff(np.log(centers))
      expected_scaling.append(np.square(log_space_widths).mean())
    return np.array(expected_scaling)

  _test_error_scaling(layers, total_errors, error_scaling)
  np.testing.assert_allclose(expected, computed[edge], atol=1e-2)


@pytest.mark.parametrize(
    'test_function,derivative_function,integral_function,layers,'
    'grid_resolution',
    TEST_CASES,
)
def test_vertical_advection(
    test_function,
    derivative_function,
    integral_function,
    layers,
    grid_resolution,
    device,
):
  """centered/upwind advection match the closed form in the bulk."""
  del integral_function
  lon, lat = nodal_axes(grid_resolution)
  levels = make_levels(layers[-1], device)
  coordinates = levels.coordinates
  centers = coordinates.centers
  internal = coordinates.internal_boundaries

  x = to_tensor(test_function(centers, lon, lat), device)
  dx_dsigma = derivative_function(centers, lon, lat)
  w = to_tensor(test_function(internal, lon, lat), device)

  # Default (zero) boundary conditions modify only the edge layers:
  # -(w[inner] * ∂x/∂𝜎[inner] + 0) / 2
  w_np = w.cpu().numpy()
  edge = -0.5 * (
      w_np[[0, -1], ...] * derivative_function(internal, lon, lat)[[0, -1]]
  )
  expected = -dx_dsigma * test_function(centers, lon, lat)
  expected[[0, -1], ...] = edge
  actual = levels.centered_vertical_advection(w, x).cpu().numpy()
  np.testing.assert_allclose(actual, expected, atol=1e-3)

  # Upwinding is only 1st order accurate.
  actual = levels.upwind_vertical_advection(w, x).cpu().numpy()
  np.testing.assert_allclose(actual[1:-1], expected[1:-1], atol=5e-2)

  # Custom boundary values keep the edges unmodified.
  wb = test_function(coordinates.boundaries[[0, -1]], lon, lat)
  db = derivative_function(coordinates.boundaries[[0, -1]], lon, lat)
  expected = -dx_dsigma * test_function(centers, lon, lat)
  actual = levels.centered_vertical_advection(
      w,
      x,
      w_boundary_values=(
          to_tensor(wb[[0]], device),
          to_tensor(wb[[1]], device),
      ),
      dx_dsigma_boundary_values=(
          to_tensor(db[[0]], device),
          to_tensor(db[[1]], device),
      ),
  ).cpu().numpy()
  np.testing.assert_allclose(actual, expected, atol=1e-3)
