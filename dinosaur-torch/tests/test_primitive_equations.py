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

"""Tests for primitive_equations (ported from the original Dinosaur, pytest)."""

import numpy as np
import pytest
import torch

from dinosaur_torch import coordinate_systems
from dinosaur_torch import primitive_equations
from dinosaur_torch import pytree
from dinosaur_torch import scales
from dinosaur_torch import sigma_coordinates
from dinosaur_torch import spherical_harmonic
from dinosaur_torch import time_integration
from dinosaur_torch import units

GridSpec = spherical_harmonic.GridSpec
SigmaCoordinates = sigma_coordinates.SigmaCoordinates
s_units = scales.units


def make_coords(wavenumbers, layers, device, dtype=torch.float64):
  grid = spherical_harmonic.Grid(
      GridSpec.with_wavenumbers(wavenumbers), device=device, dtype=dtype
  )
  vertical = sigma_coordinates.SigmaLevels(
      SigmaCoordinates.equidistant(layers), device=device, dtype=dtype
  )
  return coordinate_systems.CoordinateSystem(grid, vertical)


def random_state(coords, device, seed=0, tracers=()):
  """A random modal state with damped high wavenumbers."""
  grid = coords.horizontal
  rs = np.random.RandomState(seed)
  _, l = np.meshgrid(*grid.modal_axes, indexing='ij')
  layers = coords.vertical.layers

  def field(num_layers):
    array = rs.normal(size=(num_layers,) + grid.modal_shape)
    array *= grid.mask / (l + 1) ** 2
    return torch.as_tensor(
        array, dtype=grid.cos_lat.dtype, device=device
    )

  return primitive_equations.State(
      vorticity=field(layers),
      divergence=field(layers),
      temperature_variation=field(layers),
      log_surface_pressure=field(1),
      tracers={name: field(layers) for name in tracers},
  )


@pytest.mark.parametrize(
    'wavenumbers,test_m_fn,test_n_fn',
    [
        (
            256,
            lambda lon, lat: torch.sin(lon) * torch.cos(lat) ** 2,
            lambda lon, lat: torch.cos(lat) ** 2,
        ),
        (
            128,
            lambda lon, lat: 2.3 * torch.cos(lon) ** 2 * torch.cos(lat),
            lambda lon, lat: 3.6 * torch.cos(lat) * torch.sin(2 * lat),
        ),
    ],
)
def test_div_sec_lat(wavenumbers, test_m_fn, test_n_fn, device):
  """The helper function div_sec_lat returns expected values."""
  spec = GridSpec.with_wavenumbers(wavenumbers)
  grid = spherical_harmonic.Grid(spec, device=device, dtype=torch.float64)
  lon_np, sin_lat = spec.nodal_mesh
  lon = torch.as_tensor(lon_np, dtype=torch.float64, device=device)
  lat = torch.as_tensor(
      np.arcsin(sin_lat), dtype=torch.float64, device=device
  )
  m = test_m_fn(lon, lat)
  n = test_n_fn(lon, lat)
  # should equal H(M, N) = (1 / cos²θ) ∂M/∂λ + (1 / cosθ) ∂N/∂θ
  grad, vmap = torch.func.grad, torch.func.vmap
  dm_dlon = vmap(vmap(grad(test_m_fn)))(lon, lat)
  dn_dlat = vmap(vmap(grad(test_n_fn, argnums=1)))(lon, lat)
  cos_lat = torch.cos(lat)
  expected = grid.to_modal(dm_dlon / cos_lat**2 + dn_dlat / cos_lat)
  actual = primitive_equations.div_sec_lat(m, n, grid)
  np.testing.assert_allclose(
      actual.cpu().numpy(), expected.cpu().numpy(), atol=1e-3
  )


@pytest.mark.parametrize('layers', [10, 111])
def test_get_sigma_ratios(layers):
  coordinates = SigmaCoordinates.equidistant(layers)
  alpha = primitive_equations.get_sigma_ratios(coordinates)
  assert alpha.shape == (layers,)
  sigma = coordinates.centers
  for j in range(layers):
    if j == layers - 1:
      expected = -np.log(sigma[j])
    else:
      expected = (np.log(sigma[j + 1]) - np.log(sigma[j])) / 2
    np.testing.assert_almost_equal(expected, alpha[j])


@pytest.mark.parametrize(
    'layers,ideal_gas_constant', [(10, 1), (21, 12.3)]
)
def test_get_geopotential_weights(layers, ideal_gas_constant):
  coordinates = SigmaCoordinates.equidistant(layers)
  g = primitive_equations.get_geopotential_weights(
      coordinates, ideal_gas_constant
  )
  assert g.shape == (layers, layers)
  alpha = primitive_equations.get_sigma_ratios(coordinates)
  for i in range(layers):
    for j in range(layers):
      if i > j:
        expected = 0
      elif i == j:
        expected = ideal_gas_constant * alpha[j]
      else:
        expected = ideal_gas_constant * (alpha[j] + alpha[j - 1])
      np.testing.assert_almost_equal(
          expected, g[i, j], err_msg=f'Mismatch on entry {[i, j]}.'
      )


@pytest.mark.parametrize(
    'layers,reference_temperature,kappa',
    [
        (5, np.linspace(100, 200, 5), 0.5),
        (23, np.linspace(250, 300, 23), 0.2857),
    ],
)
def test_get_temperature_implicit_weights(
    layers, reference_temperature, kappa
):
  coordinates = SigmaCoordinates.equidistant(layers, dtype=np.float64)
  h = primitive_equations.get_temperature_implicit_weights(
      coordinates, reference_temperature, kappa
  )
  assert h.shape == (layers, layers)
  alpha = primitive_equations.get_sigma_ratios(coordinates)
  thickness = coordinates.layer_thickness

  def k(r, s):
    if r < 0 or r == layers - 1:
      return 0
    return (
        ((r - s >= 0) - thickness[: r + 1].sum())
        * (reference_temperature[r + 1] - reference_temperature[r])
        / (thickness[r + 1] + thickness[r])
    )

  for r in range(layers):
    for s in range(layers):
      expected = thickness[s] * (
          kappa
          * reference_temperature[r]
          * ((r - s >= 0) * alpha[r] + (r - s - 1 >= 0) * alpha[r - 1])
          / thickness[r]
          - k(r, s)
          - k(r - 1, s)
      )
      np.testing.assert_almost_equal(
          expected, h[r, s], err_msg=f'Mismatch in entry {[r, s]}.'
      )


@pytest.mark.parametrize('wavenumbers,layers', [(32, 4), (64, 10)])
def test_explicit_terms_shapes(wavenumbers, layers, device):
  coords = make_coords(wavenumbers, layers, device)
  reference_temperature = 300 * np.ones(layers)
  modal_orography = torch.zeros(
      coords.horizontal.modal_shape, dtype=torch.float64, device=device
  )
  ones = lambda n: torch.ones(
      (n,) + coords.horizontal.modal_shape, dtype=torch.float64, device=device
  )
  state = primitive_equations.State(
      vorticity=ones(layers),
      divergence=ones(layers),
      temperature_variation=ones(layers),
      log_surface_pressure=ones(1),
  )
  physics_specs = units.SimUnits.from_si()
  primitive = primitive_equations.PrimitiveEquations(
      reference_temperature, modal_orography, coords, physics_specs
  )
  output = primitive.explicit_terms(state)
  assert output.divergence.shape == state.divergence.shape
  assert output.vorticity.shape == state.vorticity.shape
  assert (
      output.temperature_variation.shape == state.temperature_variation.shape
  )
  assert (
      output.log_surface_pressure.shape == state.log_surface_pressure.shape
  )


@pytest.mark.parametrize(
    'wavenumbers,layers,reference_temperature,kappa_si,gas_const_si,'
    'step_size,seed',
    [
        (
            16, 5, np.linspace(100, 200, 5),
            1.4 * s_units.dimensionless,
            33 * s_units.J / s_units.kilogram / s_units.degK,
            0.3, 0,
        ),
        (
            128, 23, np.linspace(250, 300, 23),
            111 * s_units.dimensionless,
            1 * s_units.J / s_units.kilogram / s_units.degK,
            0.1, 1,
        ),
    ],
)
def test_implicit_inverse(
    wavenumbers,
    layers,
    reference_temperature,
    kappa_si,
    gas_const_si,
    step_size,
    seed,
    device,
):
  """`implicit_inverse` computes (1 - step_size * implicit_terms)⁻¹."""
  coords = make_coords(wavenumbers, layers, device)
  physics_specs = units.SimUnits.from_si(
      ideal_gas_constant_si=gas_const_si, kappa_si=kappa_si
  )
  state = random_state(coords, device, seed=seed)
  modal_orography = torch.zeros(
      coords.horizontal.modal_shape, dtype=torch.float64, device=device
  )
  primitive = primitive_equations.PrimitiveEquations(
      reference_temperature, modal_orography, coords, physics_specs
  )
  implicit_terms = primitive.implicit_terms(state)
  primitive_equations.validate_state_shape(implicit_terms, coords)

  shifted = time_integration.linear_combination(
      [state, implicit_terms], [1, -step_size]
  )
  inverted = primitive.implicit_inverse(shifted, step_size)
  primitive_equations.validate_state_shape(inverted, coords)
  for name in (
      'vorticity',
      'divergence',
      'temperature_variation',
      'log_surface_pressure',
  ):
    np.testing.assert_allclose(
        getattr(inverted, name).cpu().numpy(),
        getattr(state, name).cpu().numpy(),
        atol=1e-5,
        err_msg=name,
    )


def test_moist_equals_dry_for_zero_humidity(device):
  """Primitive equations + humidity reduces to the dry case for q=0."""
  coords = make_coords(21, 4, device)
  physics_specs = units.SimUnits.from_si()
  state = random_state(coords, device, seed=0)
  state.tracers = {
      'specific_humidity': torch.zeros_like(state.temperature_variation)
  }
  modal_orography = torch.zeros(
      coords.horizontal.modal_shape, dtype=torch.float64, device=device
  )
  ref_temps = np.linspace(250, 300, 4)
  dry = primitive_equations.PrimitiveEquations(
      ref_temps, modal_orography, coords, physics_specs
  )
  moist = primitive_equations.MoistPrimitiveEquations(
      ref_temps, modal_orography, coords, physics_specs
  )
  tendencies_dry = dry.explicit_terms(state)
  tendencies_moist = moist.explicit_terms(state)
  pytree.tree_map(
      lambda x, y: None
      if x is None
      else np.testing.assert_allclose(
          x.cpu().numpy(), y.cpu().numpy(), atol=1e-7
      ),
      tendencies_dry,
      tendencies_moist,
  )


def test_rest_atmosphere_is_stationary(device):
  """An isothermal rest atmosphere with no orography stays at rest."""
  wavenumbers, layers = 31, 8
  coords = make_coords(wavenumbers, layers, device)
  physics_specs = units.SimUnits.from_si()
  dt = physics_specs.nondimensionalize(600 * s_units.s)

  grid = coords.horizontal
  modal_shape = grid.modal_shape
  zeros = lambda n: torch.zeros(
      (n,) + modal_shape, dtype=torch.float64, device=device
  )
  p0 = physics_specs.nondimensionalize(100e3 * s_units.pascal)
  nodal_log_sp = np.log(p0) * torch.ones(
      (1,) + tuple(grid.nodal_shape), dtype=torch.float64, device=device
  )

  # a gaussian tracer blob (advected passively; velocity is zero).
  lon, sin_lat = coords.horizontal.spec.nodal_mesh
  blob = np.exp(
      -((lon - np.pi) ** 2 + (np.arcsin(sin_lat) - 0.4) ** 2) / (2 * 0.2**2)
  )
  tracer_nodal = torch.as_tensor(
      np.broadcast_to(blob, (layers,) + blob.shape).copy(),
      dtype=torch.float64,
      device=device,
  )

  state = primitive_equations.State(
      vorticity=zeros(layers),
      divergence=zeros(layers),
      temperature_variation=zeros(layers),
      log_surface_pressure=grid.to_modal(nodal_log_sp),
      tracers={'blob': grid.to_modal(tracer_nodal)},
  )
  primitive = primitive_equations.PrimitiveEquations(
      288 * np.ones(layers),
      zeros(1)[0],
      coords,
      physics_specs,
  )
  step_fn = time_integration.step_with_filters(
      time_integration.imex_rk_sil3(primitive, dt),
      [time_integration.exponential_step_filter(grid, dt)],
  )
  final, _ = time_integration.trajectory_from_step(step_fn, 10, 2)(state)

  np.testing.assert_array_less(
      final.divergence.abs().cpu().numpy().max(), 1e-10
  )
  np.testing.assert_array_less(
      final.vorticity.abs().cpu().numpy().max(), 1e-10
  )
  np.testing.assert_allclose(
      final.temperature_variation.cpu().numpy(),
      state.temperature_variation.cpu().numpy(),
      atol=1e-8,
  )
  np.testing.assert_allclose(
      final.log_surface_pressure.cpu().numpy(),
      state.log_surface_pressure.cpu().numpy(),
      atol=1e-8,
  )

  # tracer integral is preserved (the filter only damps high wavenumbers).
  def tracer_integral(tracer):
    tracer_nodal = grid.to_nodal(tracer)
    columns = coords.vertical.sigma_integral(tracer_nodal, keepdims=False)
    return float(grid.integrate(columns))

  np.testing.assert_allclose(
      tracer_integral(final.tracers['blob']),
      tracer_integral(state.tracers['blob']),
      rtol=1e-10,
  )


def test_compiled_step_matches_eager(device):
  """A full SIL3 step compiles as a single graph and matches eager."""
  coords = make_coords(21, 4, device, dtype=torch.float32)
  physics_specs = units.SimUnits.from_si()
  dt = float(physics_specs.nondimensionalize(600 * s_units.s))
  state = random_state(coords, device, seed=3, tracers=('blob',))
  modal_orography = torch.zeros(
      coords.horizontal.modal_shape, dtype=torch.float32, device=device
  )
  primitive = primitive_equations.PrimitiveEquations(
      300 * np.ones(4), modal_orography, coords, physics_specs
  )
  step_fn = time_integration.step_with_filters(
      time_integration.imex_rk_sil3(primitive, dt),
      [time_integration.exponential_step_filter(coords.horizontal, dt)],
  )
  expected = step_fn(state)  # also warms the implicit-inverse cache
  compiled = torch.compile(step_fn, fullgraph=True)
  actual = compiled(state)
  pytree.tree_map(
      lambda x, y: None
      if x is None
      else np.testing.assert_allclose(
          x.cpu().numpy(), y.cpu().numpy(), atol=1e-6
      ),
      expected,
      actual,
  )
