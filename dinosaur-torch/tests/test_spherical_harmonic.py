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

"""Tests for spherical_harmonic (ported from the original Dinosaur, pytest style)."""

import numpy as np
import pytest
import torch

from dinosaur_torch import spherical_harmonic

Grid = spherical_harmonic.Grid
GridSpec = spherical_harmonic.GridSpec

IMPLS = [
    spherical_harmonic.RealSphericalHarmonics,
    spherical_harmonic.FastSphericalHarmonics,
]


def _function_0(lat, lon):
  return torch.cos(lat) ** 4 * torch.sin(3 * lon)


def _function_1(lat, lon):
  return torch.cos(lat) ** 4 * torch.sin(5 * lon) * torch.cos(5 * lat)


def _elementwise_grad(f, argnums=0):
  return torch.func.vmap(torch.func.vmap(torch.func.grad(f, argnums=argnums)))


def random_modal_state(grid: Grid, seed=0) -> torch.Tensor:
  """Random valid modal coefficients with damped high wavenumbers."""
  _, l = np.meshgrid(*grid.modal_axes, indexing='ij')
  rs = np.random.RandomState(seed)
  array = rs.normal(size=grid.mask.shape)
  array *= grid.mask / (l + 1) ** 2
  ref = grid.cos_lat
  return torch.as_tensor(array, dtype=ref.dtype, device=ref.device)


def make_grid(spec, impl, device, dtype=torch.float64) -> Grid:
  return Grid(spec, impl=impl, device=device, dtype=dtype)


def assert_allclose(actual, desired, atol=1e-12, rtol=1e-7):
  np.testing.assert_allclose(
      actual.detach().cpu().numpy(),
      desired.detach().cpu().numpy() if isinstance(desired, torch.Tensor)
      else desired,
      atol=atol,
      rtol=rtol,
  )


@pytest.mark.parametrize(
    'params',
    [
        dict(
            longitude_nodes=64,
            latitude_nodes=32,
            longitude_wavenumbers=32,
            total_wavenumbers=32,
            latitude_spacing='gauss',
        ),
        dict(
            longitude_nodes=117,
            latitude_nodes=13,
            longitude_wavenumbers=45,
            total_wavenumbers=123,
            latitude_spacing='equiangular',
        ),
        dict(
            longitude_nodes=117,
            latitude_nodes=13,
            longitude_wavenumbers=45,
            total_wavenumbers=123,
            latitude_spacing='equiangular_with_poles',
        ),
    ],
)
def test_basis_shapes(params, device):
  """The precomputed basis buffers have the expected shapes."""
  spec = GridSpec(**params)
  sh = spherical_harmonic.RealSphericalHarmonics(spec, device=device)
  lon_waves, tot_waves = sh.modal_shape
  assert sh.f.shape == (params['longitude_nodes'], lon_waves)
  assert sh.p.shape == (lon_waves, params['latitude_nodes'], tot_waves)
  assert sh.w.shape == (params['latitude_nodes'],)


@pytest.mark.parametrize('wavenumbers', [32, 137])
@pytest.mark.parametrize('latitude_spacing', ['gauss', 'equiangular'])
@pytest.mark.parametrize('impl', IMPLS)
def test_grid_shape(wavenumbers, latitude_spacing, impl, device):
  spec = GridSpec.with_wavenumbers(
      wavenumbers, latitude_spacing=latitude_spacing
  )
  grid = make_grid(spec, impl, device)
  assert spec.nodal_shape == spec.nodal_mesh[0].shape
  assert spec.nodal_shape == spec.nodal_mesh[1].shape
  assert spec.nodal_shape == (len(spec.nodal_axes[0]), len(spec.nodal_axes[1]))
  m, l = grid.modal_axes
  assert grid.modal_shape == (len(m), len(l))
  assert grid.mask.shape == grid.modal_shape


def test_modal_axes(device):
  spec = GridSpec(
      longitude_wavenumbers=4,
      total_wavenumbers=4,
      longitude_nodes=8,
      latitude_nodes=4,
  )
  grid = make_grid(spec, spherical_harmonic.RealSphericalHarmonics, device)
  m_actual, l_actual = grid.modal_axes
  np.testing.assert_array_equal([0, 1, -1, 2, -2, 3, -3], m_actual)
  np.testing.assert_array_equal([0, 1, 2, 3], l_actual)


@pytest.mark.parametrize(
    'longitude_offset', [0.0, np.pi / 180, -np.pi / 180]
)
def test_longitudes(longitude_offset):
  spec = GridSpec(
      longitude_wavenumbers=4,
      total_wavenumbers=4,
      longitude_nodes=8,
      latitude_nodes=4,
      longitude_offset=longitude_offset,
  )
  expected = np.linspace(0, 2 * np.pi, 8, endpoint=False) + longitude_offset
  np.testing.assert_array_equal(expected, spec.nodal_axes[0])


def test_constructors(device):
  spec = GridSpec.T21()
  grid = make_grid(spec, spherical_harmonic.RealSphericalHarmonics, device)
  assert spec.nodal_shape == (64, 32)
  assert grid.modal_shape == (43, 23)

  spec = GridSpec.TL31()
  grid = make_grid(spec, spherical_harmonic.FastSphericalHarmonics, device)
  assert spec.nodal_shape == (64, 32)
  assert grid.modal_shape == (64, 33)


@pytest.mark.parametrize(
    'longitude_wavenumbers,total_wavenumbers,latitude_spacing,impl',
    [
        (32, 32, 'gauss', spherical_harmonic.RealSphericalHarmonics),
        (64, 64, 'equiangular', spherical_harmonic.FastSphericalHarmonics),
    ],
)
def test_round_trip(
    longitude_wavenumbers, total_wavenumbers, latitude_spacing, impl, device
):
  """The modal -> nodal -> modal round trip is the identity."""
  spec = GridSpec(
      longitude_wavenumbers=longitude_wavenumbers,
      total_wavenumbers=total_wavenumbers,
      longitude_nodes=4 * longitude_wavenumbers,
      latitude_nodes=2 * total_wavenumbers,
      latitude_spacing=latitude_spacing,
  )
  grid = make_grid(spec, impl, device)
  modal = random_modal_state(grid)
  modal[0, 0] = 0
  nodal = grid.to_nodal(modal)
  reconstructed = grid.to_modal(nodal)
  assert_allclose(modal, reconstructed, atol=1e-5)


@pytest.mark.parametrize('wavenumbers', [32, 137, 255])
@pytest.mark.parametrize('latitude_spacing', ['gauss', 'equiangular'])
@pytest.mark.parametrize('impl', IMPLS)
def test_laplacian_round_trip(wavenumbers, latitude_spacing, impl, device):
  """`inverse_laplacian` is the inverse of `laplacian`."""
  spec = GridSpec.with_wavenumbers(
      wavenumbers, latitude_spacing=latitude_spacing
  )
  grid = make_grid(spec, impl, device)
  x = random_modal_state(grid)
  x[0, 0] = 0
  y = grid.inverse_laplacian(grid.laplacian(x))
  assert_allclose(x, y)


@pytest.mark.parametrize('latitude_spacing', ['gauss', 'equiangular'])
@pytest.mark.parametrize('wavenumbers', [64, 128])
@pytest.mark.parametrize('test_function', [_function_0, _function_1])
@pytest.mark.parametrize('impl', IMPLS)
def test_derivatives(latitude_spacing, wavenumbers, test_function, impl,
                     device):
  """`Grid` accurately computes derivatives."""
  atol = 1e-5
  spec = GridSpec.with_wavenumbers(
      wavenumbers, latitude_spacing=latitude_spacing
  )
  grid = make_grid(spec, impl, device)
  lon_np, sin_lat = spec.nodal_mesh
  lat = torch.as_tensor(np.arcsin(sin_lat), dtype=torch.float64, device=device)
  lon = torch.as_tensor(lon_np, dtype=torch.float64, device=device)
  fx = test_function(lat, lon)

  # sec_lat_d_dlat_cos2
  def cos2latf(lat, lon):
    return torch.cos(lat) ** 2 * test_function(lat, lon)

  expected = _elementwise_grad(cos2latf)(lat, lon) / grid.cos_lat
  actual = grid.to_nodal(grid.sec_lat_d_dlat_cos2(grid.to_modal(fx)))
  assert_allclose(expected, actual, atol=atol)

  # cos_lat_d_dlat
  expected = _elementwise_grad(test_function)(lat, lon) * grid.cos_lat
  actual = grid.to_nodal(grid.cos_lat_d_dlat(grid.to_modal(fx)))
  assert_allclose(expected, actual, atol=atol)

  # d_dlon
  expected = _elementwise_grad(test_function, argnums=1)(lat, lon)
  actual = grid.to_nodal(grid.d_dlon(grid.to_modal(fx)))
  assert_allclose(expected, actual, atol=atol)


@pytest.mark.parametrize(
    'wavenumbers,latitude_spacing,acceptable_norm_diff',
    [
        (85, 'equiangular', 10),
        (85, 'gauss', 10),
        (42, 'gauss', 5),
    ],
)
def test_derivative_artifacts(
    wavenumbers, latitude_spacing, acceptable_norm_diff, device
):
  """Clipping in cos_lat_grad removes the top-wavenumber artifact."""

  def test_function(lat, lon):
    """A hand-picked function that exposes derivative artifacts."""
    xs = torch.linspace(0, 1.9, wavenumbers, dtype=lat.dtype,
                        device=lat.device)
    ys = torch.exp(torch.sin(5 * xs) * 1.1 * xs**2 - 0.8 * xs**4)
    return sum(
        y * torch.cos(lat * 4) ** 2 * torch.cos(lon * (n % 4))
        for n, y in zip(np.arange(wavenumbers), ys)
    )

  spec = GridSpec.with_wavenumbers(
      wavenumbers, latitude_spacing=latitude_spacing
  )
  grid = make_grid(spec, spherical_harmonic.RealSphericalHarmonics, device)
  lon_np, sin_lat = spec.nodal_mesh
  lat = torch.as_tensor(np.arcsin(sin_lat), dtype=torch.float64, device=device)
  lon = torch.as_tensor(lon_np, dtype=torch.float64, device=device)
  fx = test_function(lat, lon)
  expected = _elementwise_grad(test_function)(lat, lon) * grid.cos_lat

  actual = grid.to_nodal(grid.cos_lat_d_dlat(grid.to_modal(fx)))
  error_norm = np.linalg.norm((actual - expected).cpu().numpy())
  assert error_norm > 10 * acceptable_norm_diff

  _, actual = grid.to_nodal(grid.cos_lat_grad(grid.to_modal(fx)))
  error_norm = np.linalg.norm((actual - expected).cpu().numpy())
  assert error_norm < acceptable_norm_diff


@pytest.mark.parametrize(
    'spec',
    [
        GridSpec.with_wavenumbers(128),
        GridSpec(
            longitude_wavenumbers=64,
            total_wavenumbers=64,
            longitude_nodes=192,
            latitude_nodes=128,
            radius=2.6,
            latitude_spacing='equiangular',
        ),
        GridSpec.with_wavenumbers(128, radius=0.3),
    ],
)
def test_laplacian_consistency(spec, device):
  """Computing the Laplacian in 2 ways gives identical results."""
  grid = make_grid(spec, spherical_harmonic.RealSphericalHarmonics, device)
  x = random_modal_state(grid)
  x[0, 0] = 0
  # Taking the derivative twice is inaccurate in the highest total
  # wavenumber, so trim it.
  x[:, -1] = 0

  laplacian0 = grid.laplacian(x)

  # Δx = ∇ · [cosθ ((cosθ ∇x) / cos²θ)], θ = latitude. `x` has no top
  # wavenumbers, so it's safe to skip clipping once.
  cos_lat_grad = grid.cos_lat_grad(x, clip=False)
  sec_lat_grad = grid.to_modal(
      tuple(v * grid.sec2_lat for v in grid.to_nodal(cos_lat_grad))
  )
  laplacian1 = grid.div_cos_lat(sec_lat_grad)

  assert_allclose(laplacian0, laplacian1, atol=1e-10)


@pytest.mark.parametrize(
    'spec,impl,atol',
    [
        (
            GridSpec.with_wavenumbers(128),
            spherical_harmonic.RealSphericalHarmonics,
            1e-10,
        ),
        (
            GridSpec.with_wavenumbers(128),
            spherical_harmonic.FastSphericalHarmonics,
            1e-11,
        ),
        (
            GridSpec(
                longitude_wavenumbers=64,
                total_wavenumbers=64,
                longitude_nodes=192,
                latitude_nodes=128,
                radius=3.2,
                latitude_spacing='equiangular',
            ),
            spherical_harmonic.RealSphericalHarmonics,
            1e-5,
        ),
        (
            GridSpec.with_wavenumbers(128, radius=0.54),
            spherical_harmonic.RealSphericalHarmonics,
            1e-10,
        ),
    ],
)
def test_vorticity_stream_velocity_round_trip(spec, impl, atol, device):
  """The vorticity -> stream -> velocity round trip is the identity."""
  grid = make_grid(spec, impl, device)
  vorticity = random_modal_state(grid)
  vorticity[0, 0] = 0
  vorticity[:, -1] = 0

  # Solve for the stream function ∇²ѱ = 𝜻, compute v = k x 𝝯ѱ.
  stream = grid.inverse_laplacian(vorticity)
  cos_lat_v = torch.stack(grid.k_cross(grid.cos_lat_grad(stream, clip=False)))
  sec_lat_v = grid.to_modal(grid.sec2_lat * grid.to_nodal(cos_lat_v))
  div_v = grid.div_cos_lat(sec_lat_v)

  # The velocity should be divergence-free.
  assert_allclose(div_v, torch.zeros_like(div_v), atol=atol)

  # Reconstruct the vorticity 𝜻 = ∇ x v.
  reconstructed = grid.curl_cos_lat(sec_lat_v)
  assert_allclose(vorticity, reconstructed, atol=atol)


@pytest.mark.parametrize(
    'spec,atol',
    [
        (GridSpec.with_wavenumbers(128), 1e-10),
        (
            GridSpec(
                longitude_wavenumbers=64,
                total_wavenumbers=64,
                longitude_nodes=192,
                latitude_nodes=128,
                radius=3.2,
                latitude_spacing='equiangular',
            ),
            1e-5,
        ),
        (GridSpec.with_wavenumbers(128, radius=0.54), 1e-10),
    ],
)
def test_divergence_potential_velocity_round_trip(spec, atol, device):
  """The div -> potential -> velocity round trip is the identity."""
  grid = make_grid(spec, spherical_harmonic.RealSphericalHarmonics, device)
  divergence = random_modal_state(grid)
  divergence[0, 0] = 0
  divergence[:, -1] = 0

  # Solve for the velocity potential ∇²ɸ = D, compute v = 𝝯ɸ.
  potential = grid.inverse_laplacian(divergence)
  cos_lat_v = grid.cos_lat_grad(potential, clip=False)
  sec_lat_v = grid.to_modal(
      tuple(grid.sec2_lat * v for v in grid.to_nodal(cos_lat_v))
  )
  curl_v = grid.curl_cos_lat(sec_lat_v)

  # The velocity should be curl-free.
  assert_allclose(curl_v, torch.zeros_like(curl_v), atol=atol)

  # Reconstruct the divergence D = ∇ · v.
  reconstructed = grid.div_cos_lat(sec_lat_v)
  reconstructed[:, -1] = 0
  assert_allclose(divergence, reconstructed, atol=atol)


@pytest.mark.parametrize(
    'wavenumbers,latitude_spacing,radius',
    [
        (64, 'gauss', 1.3),
        (128, 'equiangular', 2.5),
        (256, 'gauss', 1.0),
    ],
)
def test_integration_surface_area(
    wavenumbers, latitude_spacing, radius, device
):
  spec = GridSpec.with_wavenumbers(
      wavenumbers, latitude_spacing=latitude_spacing, radius=radius
  )
  grid = make_grid(spec, spherical_harmonic.RealSphericalHarmonics, device)
  ones = torch.ones(spec.nodal_shape, dtype=torch.float64, device=device)
  quadrature_surface_area = grid.integrate(ones)
  expected = 4 * np.pi * radius**2  # A = 4πr²
  np.testing.assert_almost_equal(float(quadrature_surface_area), expected)


@pytest.mark.parametrize(
    'wavenumbers,l,m',
    [(64, 0, 0), (128, 32, 17), (256, 174, 95)],
)
@pytest.mark.parametrize('impl', IMPLS)
def test_integration_spherical_harmonics(wavenumbers, l, m, impl, device):
  """Each basis function has unit L² norm on the sphere."""
  spec = GridSpec.with_wavenumbers(wavenumbers)
  grid = make_grid(spec, impl, device)
  x = torch.zeros(grid.modal_shape, dtype=torch.float64, device=device)
  x[m, l] = 1
  z = grid.to_nodal(x)
  integral = grid.integrate(z**2)
  assert_allclose(integral, torch.ones_like(integral), atol=1e-10)


def test_clip_wavenumbers(device):
  spec = GridSpec(
      longitude_wavenumbers=2,
      total_wavenumbers=3,
      longitude_nodes=8,
      latitude_nodes=4,
  )
  grid = make_grid(spec, spherical_harmonic.RealSphericalHarmonics, device)
  state = {
      'u': torch.ones((4, 3), dtype=torch.float64, device=device),
      'time': 1.0,
  }
  clipped = grid.clip_wavenumbers(state)
  expected_u = np.array([[1, 1, 0]] * 4)
  np.testing.assert_array_equal(clipped['u'].cpu().numpy(), expected_u)
  assert clipped['time'] == 1.0


def test_add_constant(device):
  spec = GridSpec.T21()
  grid = make_grid(spec, spherical_harmonic.RealSphericalHarmonics, device)
  x = torch.zeros(grid.modal_shape, dtype=torch.float64, device=device)
  y = grid.to_nodal(spherical_harmonic.add_constant(x, 1.5))
  assert_allclose(y, 1.5 * torch.ones_like(y), atol=1e-6)


# -- new-design tests: device movement, state_dict, torch.compile -------------


def test_module_to_device_and_dtype(device):
  """`.to()` moves all precomputed buffers like any other nn.Module."""
  spec = GridSpec.T21()
  grid = Grid(spec, device='cpu', dtype=torch.float64)
  grid = grid.to(device)
  x = random_modal_state(grid)
  assert x.device.type == device.type
  nodal = grid.to_nodal(x)
  assert nodal.device.type == device.type

  grid32 = grid.float()
  nodal32 = grid32.to_nodal(x.float())
  assert nodal32.dtype == torch.float32
  np.testing.assert_allclose(
      nodal32.cpu().numpy(), nodal.cpu().numpy(), atol=1e-4
  )


def test_state_dict_is_empty(device):
  """Precomputed constants are non-persistent: nothing to checkpoint."""
  grid = Grid(GridSpec.T21(), device=device)
  assert not grid.state_dict()


@pytest.mark.parametrize('impl', IMPLS)
def test_compile_round_trip_fullgraph(impl, device):
  """to_modal/to_nodal compile as a single graph (no shims, no breaks)."""
  spec = GridSpec.T21()
  grid = Grid(spec, impl=impl, device=device)

  def round_trip(x):
    return grid.to_modal(grid.to_nodal(x))

  compiled = torch.compile(round_trip, fullgraph=True)
  x = random_modal_state(grid).float()
  expected = round_trip(x)
  actual = compiled(x)
  np.testing.assert_allclose(
      actual.cpu().numpy(), expected.cpu().numpy(), rtol=1e-6, atol=1e-6
  )
