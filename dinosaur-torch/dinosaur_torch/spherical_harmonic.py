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

"""Spherical-harmonic grids, transforms and differential operators.

The static description of a grid (truncation, node counts, radius) is a
frozen dataclass, `GridSpec`. The tensor-holding objects are `nn.Module`s
constructed from a spec on an explicit device/dtype:

  spec = GridSpec.T21()
  grid = Grid(spec, device='cuda')
  modal = grid.to_modal(nodal)

Precomputed basis matrices and operator coefficients are non-persistent
buffers, so `.to()` works as usual and `state_dict()` stays empty.

Two modal layouts are implemented, matching upstream dinosaur:

- `RealSphericalHarmonics`: modal shape `(2M - 1, L)` with wavenumbers
  ordered `[0, +1, -1, +2, -2, ...]`.
- `FastSphericalHarmonics`: modal shape `(2M, L)` with cos/sin interleaved
  and a structural zero at index 1 (named `RealSphericalHarmonicsWithZeroImag`
  upstream). Halves the size of the Legendre coefficient array.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import Callable

import numpy as np
import torch
from torch import nn

from dinosaur_torch import associated_legendre
from dinosaur_torch import fourier
from dinosaur_torch import pytree


LATITUDE_SPACINGS = dict(
    gauss=associated_legendre.gauss_legendre_nodes,
    equiangular=associated_legendre.equiangular_nodes,
    equiangular_with_poles=associated_legendre.equiangular_nodes_with_poles,
)


def get_latitude_nodes(n: int, spacing: str) -> tuple[np.ndarray, np.ndarray]:
  """Computes latitude nodes using the given spacing."""
  get_nodes = LATITUDE_SPACINGS.get(spacing)
  if get_nodes is None:
    raise ValueError(
        f'Unknown spacing: {spacing}; '
        f'available spacings are {list(LATITUDE_SPACINGS.keys())}'
    )
  return get_nodes(n)


@dataclasses.dataclass(frozen=True)
class GridSpec:
  """Static description of real-space and spectral grids over the sphere.

  Hashable and cheap; holds no tensors. Tensor-holding transform modules
  (`Grid`, the `SphericalHarmonics` implementations) are constructed from a
  spec. NumPy coordinate metadata (nodal axes, latitudes, weights) lives
  here.

  Attributes:
    longitude_wavenumbers: the maximum (exclusive) wavenumber in the
      longitudinal direction, typically denoted `m`. Must satisfy
      `longitude_wavenumbers <= total_wavenumbers`.
    total_wavenumbers: the maximum (exclusive) sum of the latitudinal and
      longitudinal wavenumbers, typically denoted `l`.
    longitude_nodes: the number of nodes in the longitudinal direction,
      equally spaced in [0, 2π) starting at `longitude_offset`.
    latitude_nodes: the number of nodes in the latitudinal direction.
    latitude_spacing: 'gauss', 'equiangular' or 'equiangular_with_poles'.
    longitude_offset: the value of the first longitude node, in radians.
    radius: radius of the sphere.
  """

  longitude_wavenumbers: int
  total_wavenumbers: int
  longitude_nodes: int
  latitude_nodes: int
  latitude_spacing: str = 'gauss'
  longitude_offset: float = 0.0
  radius: float = 1.0

  def __post_init__(self):
    if self.latitude_spacing not in LATITUDE_SPACINGS:
      raise ValueError(
          f'Unsupported `latitude_spacing` "{self.latitude_spacing}". '
          f'Supported values are: {list(LATITUDE_SPACINGS)}.'
      )

  @classmethod
  def with_wavenumbers(
      cls,
      longitude_wavenumbers: int,
      dealiasing: str = 'quadratic',
      latitude_spacing: str = 'gauss',
      longitude_offset: float = 0.0,
      radius: float = 1.0,
  ) -> GridSpec:
    """Construct a `GridSpec` by specifying only wavenumbers."""
    # The number of nodes is chosen for de-aliasing.
    order = {'linear': 2, 'quadratic': 3, 'cubic': 4}[dealiasing]
    longitude_nodes = order * longitude_wavenumbers + 1
    latitude_nodes = math.ceil(longitude_nodes / 2)
    return cls(
        longitude_wavenumbers=longitude_wavenumbers,
        total_wavenumbers=longitude_wavenumbers + 1,
        longitude_nodes=longitude_nodes,
        latitude_nodes=latitude_nodes,
        latitude_spacing=latitude_spacing,
        longitude_offset=longitude_offset,
        radius=radius,
    )

  @classmethod
  def construct(
      cls,
      max_wavenumber: int,
      gaussian_nodes: int,
      latitude_spacing: str = 'gauss',
      longitude_offset: float = 0.0,
      radius: float = 1.0,
  ) -> GridSpec:
    """Construct a `GridSpec` from max wavenumber & number of nodes.

    Args:
      max_wavenumber: maximum wavenumber to resolve.
      gaussian_nodes: number of nodes on the Gaussian grid between the equator
        and a pole.
      latitude_spacing: either 'gauss' or 'equiangular'.
      longitude_offset: the value of the first longitude node, in radians.
      radius: radius of the sphere.

    Returns:
      Constructed GridSpec object.
    """
    return cls(
        longitude_wavenumbers=max_wavenumber + 1,
        total_wavenumbers=max_wavenumber + 2,
        longitude_nodes=4 * gaussian_nodes,
        latitude_nodes=2 * gaussian_nodes,
        latitude_spacing=latitude_spacing,
        longitude_offset=longitude_offset,
        radius=radius,
    )

  # Standard grids from the literature; see
  # https://www.ecmwf.int/en/forecasts/documentation-and-support/data-spatial-coordinate-systems
  # T* grids can represent quadratic terms without aliasing; TL* grids
  # resolve all wavenumbers and alias quadratic terms ("linear truncation").
  # pylint:disable=invalid-name

  @classmethod
  def T21(cls, **kwargs) -> GridSpec:
    return cls.construct(max_wavenumber=21, gaussian_nodes=16, **kwargs)

  @classmethod
  def T31(cls, **kwargs) -> GridSpec:
    return cls.construct(max_wavenumber=31, gaussian_nodes=24, **kwargs)

  @classmethod
  def T42(cls, **kwargs) -> GridSpec:
    return cls.construct(max_wavenumber=42, gaussian_nodes=32, **kwargs)

  @classmethod
  def T85(cls, **kwargs) -> GridSpec:
    return cls.construct(max_wavenumber=85, gaussian_nodes=64, **kwargs)

  @classmethod
  def T106(cls, **kwargs) -> GridSpec:
    return cls.construct(max_wavenumber=106, gaussian_nodes=80, **kwargs)

  @classmethod
  def T119(cls, **kwargs) -> GridSpec:
    return cls.construct(max_wavenumber=119, gaussian_nodes=90, **kwargs)

  @classmethod
  def T170(cls, **kwargs) -> GridSpec:
    return cls.construct(max_wavenumber=170, gaussian_nodes=128, **kwargs)

  @classmethod
  def T213(cls, **kwargs) -> GridSpec:
    return cls.construct(max_wavenumber=213, gaussian_nodes=160, **kwargs)

  @classmethod
  def T340(cls, **kwargs) -> GridSpec:
    return cls.construct(max_wavenumber=340, gaussian_nodes=256, **kwargs)

  @classmethod
  def T425(cls, **kwargs) -> GridSpec:
    return cls.construct(max_wavenumber=425, gaussian_nodes=320, **kwargs)

  @classmethod
  def TL31(cls, **kwargs) -> GridSpec:
    return cls.construct(max_wavenumber=31, gaussian_nodes=16, **kwargs)

  @classmethod
  def TL47(cls, **kwargs) -> GridSpec:
    return cls.construct(max_wavenumber=47, gaussian_nodes=24, **kwargs)

  @classmethod
  def TL63(cls, **kwargs) -> GridSpec:
    return cls.construct(max_wavenumber=63, gaussian_nodes=32, **kwargs)

  @classmethod
  def TL95(cls, **kwargs) -> GridSpec:
    return cls.construct(max_wavenumber=95, gaussian_nodes=48, **kwargs)

  @classmethod
  def TL127(cls, **kwargs) -> GridSpec:
    return cls.construct(max_wavenumber=127, gaussian_nodes=64, **kwargs)

  @classmethod
  def TL159(cls, **kwargs) -> GridSpec:
    return cls.construct(max_wavenumber=159, gaussian_nodes=80, **kwargs)

  @classmethod
  def TL179(cls, **kwargs) -> GridSpec:
    return cls.construct(max_wavenumber=179, gaussian_nodes=90, **kwargs)

  @classmethod
  def TL255(cls, **kwargs) -> GridSpec:
    return cls.construct(max_wavenumber=255, gaussian_nodes=128, **kwargs)

  @classmethod
  def TL639(cls, **kwargs) -> GridSpec:
    return cls.construct(max_wavenumber=639, gaussian_nodes=320, **kwargs)

  @classmethod
  def TL1279(cls, **kwargs) -> GridSpec:
    return cls.construct(max_wavenumber=1279, gaussian_nodes=640, **kwargs)

  # pylint:enable=invalid-name

  @functools.cached_property
  def nodal_axes(self) -> tuple[np.ndarray, np.ndarray]:
    """Longitude (radians) and sin(latitude) coordinates of the nodal grid."""
    longitude, _ = fourier.quadrature_nodes(self.longitude_nodes)
    sin_latitude, _ = get_latitude_nodes(
        self.latitude_nodes, self.latitude_spacing
    )
    return longitude + self.longitude_offset, sin_latitude

  @property
  def longitudes(self) -> np.ndarray:
    return self.nodal_axes[0]

  @property
  def latitudes(self) -> np.ndarray:
    return np.arcsin(self.nodal_axes[1])

  @property
  def nodal_shape(self) -> tuple[int, int]:
    return (self.longitude_nodes, self.latitude_nodes)

  @functools.cached_property
  def nodal_mesh(self) -> tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(*self.nodal_axes, indexing='ij')

  @functools.cached_property
  def cos_lat(self) -> np.ndarray:
    _, sin_lat = self.nodal_axes
    return np.sqrt(1 - sin_lat**2)

  @functools.cached_property
  def sec2_lat(self) -> np.ndarray:
    _, sin_lat = self.nodal_axes
    return 1 / (1 - sin_lat**2)

  @functools.cached_property
  def quadrature_weights(self) -> np.ndarray:
    """Nodal quadrature weights over sin(latitude), shape (latitude_nodes,)."""
    _, wf = fourier.quadrature_nodes(self.longitude_nodes)
    _, wp = get_latitude_nodes(self.latitude_nodes, self.latitude_spacing)
    return wf * wp

  @functools.cached_property
  def laplacian_eigenvalues(self) -> np.ndarray:
    l = np.arange(self.total_wavenumbers)
    return -l * (l + 1) / (self.radius**2)


def _buffer(module: nn.Module, name: str, array: np.ndarray, dtype, device):
  """Registers a float64 NumPy array as a non-persistent buffer."""
  module.register_buffer(
      name, torch.as_tensor(array, dtype=dtype, device=device),
      persistent=False,
  )


class SphericalHarmonics(nn.Module):
  """Base class for spherical-harmonics transform implementations.

  Subclasses precompute the Fourier matrix `f`, Legendre coefficients `p` and
  quadrature weights `w` as buffers, and define the modal layout.
  """

  def __init__(
      self,
      spec: GridSpec,
      *,
      device: torch.device | str | None = None,
      dtype: torch.dtype = torch.float32,
  ):
    super().__init__()
    self.spec = spec

  @property
  def modal_shape(self) -> tuple[int, int]:
    """Shape of the (m, l) modal representation."""
    raise NotImplementedError

  @property
  def modal_axes(self) -> tuple[np.ndarray, np.ndarray]:
    """Longitudinal and total wavenumbers (m, l) of the modal basis."""
    raise NotImplementedError

  @functools.cached_property
  def mask(self) -> np.ndarray:
    """Boolean mask of valid (non-structural-zero) modal entries."""
    raise NotImplementedError

  def transform(self, x: torch.Tensor) -> torch.Tensor:
    """Maps `x` from a nodal to a modal representation."""
    raise NotImplementedError

  def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
    """Maps `x` from a modal to a nodal representation."""
    raise NotImplementedError

  def longitudinal_derivative(self, x: torch.Tensor) -> torch.Tensor:
    """Computes `∂x/∂λ` in the modal basis, where λ denotes longitude."""
    raise NotImplementedError


class RealSphericalHarmonics(SphericalHarmonics):
  """Spherical harmonics with modal layout `m = [0, +1, -1, ..., +M, -M]`.

  Modal shape is `(2 * longitude_wavenumbers - 1, total_wavenumbers)`;
  entries with `abs(m) > l` are structural zeros. This is the layout used by
  the published 2.8° NeuralGCM checkpoints.
  """

  def __init__(self, spec, *, device=None, dtype=torch.float32):
    super().__init__(spec, device=device, dtype=dtype)
    # The product of `f` and `p` gives the real normalized spherical harmonic
    # basis evaluated on a grid of longitudes λ and latitudes θ:
    #
    #   f[i, 0]      p[0     , j, l] = cₗ₀ P⁰ₗ(sin θⱼ)
    #   f[i, 2m - 1] p[2m - 1, j, l] = cₗₘ cos(m λᵢ) Pᵐₗ(sin θⱼ)
    #   f[i, 2m]     p[2m,     j, l] = cₗₘ sin(m λᵢ) Pᵐₗ(sin θⱼ)
    #
    # with cₗₘ chosen so each function has unit L² norm on the unit sphere.
    f = fourier.real_basis(
        wavenumbers=spec.longitude_wavenumbers, nodes=spec.longitude_nodes
    )
    p = associated_legendre.evaluate(
        n_m=spec.longitude_wavenumbers,
        n_l=spec.total_wavenumbers,
        x=spec.nodal_axes[1],
    )
    # Pᵐₗ with m > 0 pairs with both the sin and cos Fourier components, so
    # rows are duplicated; the m = 0 row pairs only with the constant.
    p = np.repeat(p, 2, axis=0)[1:]
    _buffer(self, 'f', f, dtype, device)
    _buffer(self, 'p', p, dtype, device)
    _buffer(self, 'w', spec.quadrature_weights, dtype, device)

  @property
  def modal_shape(self) -> tuple[int, int]:
    return (2 * self.spec.longitude_wavenumbers - 1, self.spec.total_wavenumbers)

  @functools.cached_property
  def modal_axes(self) -> tuple[np.ndarray, np.ndarray]:
    m_pos = np.arange(1, self.spec.longitude_wavenumbers)
    m_pos_neg = np.stack([m_pos, -m_pos], axis=1).ravel()
    lon_wavenumbers = np.concatenate([[0], m_pos_neg])
    tot_wavenumbers = np.arange(self.spec.total_wavenumbers)
    return lon_wavenumbers, tot_wavenumbers

  @functools.cached_property
  def mask(self) -> np.ndarray:
    m, l = np.meshgrid(*self.modal_axes, indexing='ij')
    return abs(m) <= l

  def transform(self, x: torch.Tensor) -> torch.Tensor:
    wx = self.w * x
    fwx = torch.einsum('im,...ij->...mj', self.f, wx)
    return torch.einsum('mjl,...mj->...ml', self.p, fwx)

  def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
    px = torch.einsum('mjl,...ml->...mj', self.p, x)
    # note: explicit matrix multiplication is faster than an FFT at the
    # resolutions we use.
    return torch.einsum('im,...mj->...ij', self.f, px)

  def longitudinal_derivative(self, x: torch.Tensor) -> torch.Tensor:
    return fourier.real_basis_derivative(x, dim=-2)


def _unstack_m(x: torch.Tensor) -> torch.Tensor:
  """Splits the interleaved cos/sin dimension `2m` into `(2, m)`.

  Equivalent to jnp.reshape(x, (..., 2, M, L), order='F'):
  out[..., s, k, l] = x[..., 2 * k + s, l].
  """
  shape = x.shape[:-2] + (x.shape[-2] // 2, 2) + x.shape[-1:]
  return x.reshape(shape).transpose(-3, -2)


def _stack_m(x: torch.Tensor) -> torch.Tensor:
  """Inverse of `_unstack_m`."""
  shape = x.shape[:-3] + (x.shape[-3] * x.shape[-2],) + x.shape[-1:]
  return x.transpose(-3, -2).reshape(shape)


class FastSphericalHarmonics(SphericalHarmonics):
  """Spherical harmonics with cos/sin interleaved and a zero-imag `m = 0`.

  Modal shape is `(2 * longitude_wavenumbers, total_wavenumbers)` with a
  structural zero at m-index 1; the Legendre coefficient array is shared
  between the cos and sin components, halving its size. Named
  `RealSphericalHarmonicsWithZeroImag` in upstream dinosaur/NeuralGCM (used
  e.g. by the TL63 stochastic checkpoint).
  """

  def __init__(self, spec, *, device=None, dtype=torch.float32):
    super().__init__(spec, device=device, dtype=dtype)
    f = fourier.real_basis_with_zero_imag(
        wavenumbers=spec.longitude_wavenumbers, nodes=spec.longitude_nodes
    )
    p = associated_legendre.evaluate(
        n_m=spec.longitude_wavenumbers,
        n_l=spec.total_wavenumbers,
        x=spec.nodal_axes[1],
    )
    _buffer(self, 'f', f, dtype, device)
    _buffer(self, 'p', p, dtype, device)
    _buffer(self, 'w', spec.quadrature_weights, dtype, device)

  @property
  def modal_shape(self) -> tuple[int, int]:
    return (2 * self.spec.longitude_wavenumbers, self.spec.total_wavenumbers)

  @functools.cached_property
  def modal_axes(self) -> tuple[np.ndarray, np.ndarray]:
    m_pos = np.arange(1, self.spec.longitude_wavenumbers)
    m_pos_neg = np.stack([m_pos, -m_pos], axis=1).ravel()
    lon_wavenumbers = np.concatenate([[0, 0], m_pos_neg])
    tot_wavenumbers = np.arange(self.spec.total_wavenumbers)
    return lon_wavenumbers, tot_wavenumbers

  @functools.cached_property
  def mask(self) -> np.ndarray:
    m, l = np.meshgrid(*self.modal_axes, indexing='ij')
    i = np.arange(self.modal_shape[0])[:, np.newaxis]
    return (abs(m) <= l) & (i != 1)

  def transform(self, x: torch.Tensor) -> torch.Tensor:
    x = self.w * x
    x = torch.einsum('im,...ij->...mj', self.f, x)
    x = _unstack_m(x)
    x = torch.einsum('mjl,...smj->...sml', self.p, x)
    return _stack_m(x)

  def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
    x = _unstack_m(x)
    x = torch.einsum('mjl,...sml->...smj', self.p, x)
    x = _stack_m(x)
    return torch.einsum('im,...mj->...ij', self.f, x)

  def longitudinal_derivative(self, x: torch.Tensor) -> torch.Tensor:
    return fourier.real_basis_derivative_with_zero_imag(x, dim=-2)


# Upstream name for the layout used by e.g. the TL63 stochastic checkpoint.
RealSphericalHarmonicsWithZeroImag = FastSphericalHarmonics


class Grid(nn.Module):
  """Spectral transforms and differential operators for a `GridSpec`.

  Bundles a spherical-harmonics implementation with the standard spectral
  operators (Laplacian, latitudinal/longitudinal derivatives, vorticity /
  divergence ↔ velocity conversions). All methods take and return tensors on
  the module's device; pytree-valued inputs are mapped over their non-scalar
  tensor leaves.
  """

  def __init__(
      self,
      spec: GridSpec,
      *,
      impl: Callable[..., SphericalHarmonics] = RealSphericalHarmonics,
      device: torch.device | str | None = None,
      dtype: torch.dtype = torch.float32,
  ):
    super().__init__()
    self.spec = spec
    self.sh = impl(spec, device=device, dtype=dtype)

    _buffer(self, 'cos_lat', spec.cos_lat, dtype, device)
    _buffer(self, 'sec2_lat', spec.sec2_lat, dtype, device)

    eigs = spec.laplacian_eigenvalues
    _buffer(self, 'laplacian_eigenvalues', eigs, dtype, device)
    with np.errstate(divide='ignore', invalid='ignore'):
      inv_eigs = 1 / eigs
    inv_eigs[0] = 0
    _buffer(self, 'inverse_laplacian_eigenvalues', inv_eigs, dtype, device)

    # Coefficients of the sparse (bidiagonal in l) latitudinal derivative
    # operators; see Durran (2010), eq. (8.42)-(8.45).
    m, l = np.meshgrid(*self.sh.modal_axes, indexing='ij')
    mask = self.sh.mask
    a = np.sqrt(mask * (l**2 - m**2) / (4 * l**2 - 1))
    a[:, 0] = 0
    b = np.sqrt(mask * ((l + 1) ** 2 - m**2) / (4 * (l + 1) ** 2 - 1))
    b[:, -1] = 0
    _buffer(self, '_d_dlat_lm1', (l + 1) * a, dtype, device)
    _buffer(self, '_d_dlat_lp1', -l * b, dtype, device)
    _buffer(self, '_sec_lat_d_dlat_cos2_lm1', (l - 1) * a, dtype, device)
    _buffer(self, '_sec_lat_d_dlat_cos2_lp1', -(l + 2) * b, dtype, device)

    # Mask zeroing the highest total wavenumber (the common clip).
    clip_one = np.ones(self.sh.modal_shape[-1])
    clip_one[-1:] = 0
    _buffer(self, '_clip_mask_one', clip_one, dtype, device)

    _buffer(
        self,
        '_integration_weights',
        spec.quadrature_weights * spec.radius**2,
        dtype,
        device,
    )

  # -- metadata ------------------------------------------------------------

  @property
  def radius(self) -> float:
    return self.spec.radius

  @property
  def nodal_shape(self) -> tuple[int, int]:
    return self.spec.nodal_shape

  @property
  def modal_shape(self) -> tuple[int, int]:
    return self.sh.modal_shape

  @property
  def modal_axes(self) -> tuple[np.ndarray, np.ndarray]:
    return self.sh.modal_axes

  @property
  def mask(self) -> np.ndarray:
    return self.sh.mask

  @property
  def longitudes(self) -> np.ndarray:
    return self.spec.longitudes

  @property
  def latitudes(self) -> np.ndarray:
    return self.spec.latitudes

  # -- transforms ------------------------------------------------------------

  def to_nodal(self, x):
    """Maps `x` (tensor or pytree of tensors) from modal to nodal."""
    return pytree.map_fields(self.sh.inverse_transform, x)

  def to_modal(self, z):
    """Maps `z` (tensor or pytree of tensors) from nodal to modal."""
    return pytree.map_fields(self.sh.transform, z)

  # -- spectral operators ------------------------------------------------------

  def laplacian(self, x: torch.Tensor) -> torch.Tensor:
    """Computes `∇²(x)` in the spectral basis."""
    return x * self.laplacian_eigenvalues

  def inverse_laplacian(self, x: torch.Tensor) -> torch.Tensor:
    """Computes `(∇²)⁻¹(x)` in the spectral basis."""
    return x * self.inverse_laplacian_eigenvalues

  def clip_wavenumbers(self, x, n: int = 1):
    """Zeros out the highest `n` total wavenumbers."""
    if n <= 0:
      raise ValueError(f'`n` must be >= 0; got {n}.')
    if n == 1:
      mask = self._clip_mask_one
    else:
      l = self.sh.modal_shape[-1]
      arange = torch.arange(l, device=self._clip_mask_one.device)
      mask = (arange < l - n).to(self._clip_mask_one.dtype)
    return pytree.map_fields(lambda t: t * mask, x)

  def d_dlon(self, x: torch.Tensor) -> torch.Tensor:
    """Computes `∂x/∂λ` where λ denotes longitude."""
    return self.sh.longitudinal_derivative(x)

  def cos_lat_d_dlat(self, x: torch.Tensor) -> torch.Tensor:
    """Computes `cosθ ∂x/∂θ`, where θ denotes latitude.

    The result has a (clippable) numerical artifact in the highest
    total wavenumber.
    """
    x_lm1 = fourier.shift(self._d_dlat_lm1 * x, -1, dim=-1)
    x_lp1 = fourier.shift(self._d_dlat_lp1 * x, +1, dim=-1)
    return x_lm1 + x_lp1

  def sec_lat_d_dlat_cos2(self, x: torch.Tensor) -> torch.Tensor:
    """Computes `secθ ∂/∂θ(cos²θ x)`, where θ denotes latitude."""
    x_lm1 = fourier.shift(self._sec_lat_d_dlat_cos2_lm1 * x, -1, dim=-1)
    x_lp1 = fourier.shift(self._sec_lat_d_dlat_cos2_lp1 * x, +1, dim=-1)
    return x_lm1 + x_lp1

  def cos_lat_grad(
      self, x: torch.Tensor, clip: bool = True
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes `cosθ ∇(x)` where θ denotes latitude."""
    # clipping the last wavenumber removes the numerical artifact in d_dlat.
    raw = self.d_dlon(x) / self.radius, self.cos_lat_d_dlat(x) / self.radius
    if clip:
      return self.clip_wavenumbers(raw)
    return raw

  def k_cross(self, v: tuple) -> tuple:
    """Computes `k ✕ v`, where k is the normal unit vector."""
    return -v[1], v[0]

  def div_cos_lat(self, v: tuple, clip: bool = True) -> torch.Tensor:
    """Computes `∇ · (v cosθ)` where θ denotes latitude."""
    raw = (self.d_dlon(v[0]) + self.sec_lat_d_dlat_cos2(v[1])) / self.radius
    if clip:
      return self.clip_wavenumbers(raw)
    return raw

  def curl_cos_lat(self, v: tuple, clip: bool = True) -> torch.Tensor:
    """Computes `k · ∇ ✕ (v cosθ)` where θ denotes latitude."""
    raw = (self.d_dlon(v[1]) - self.sec_lat_d_dlat_cos2(v[0])) / self.radius
    if clip:
      return self.clip_wavenumbers(raw)
    return raw

  def integrate(self, z: torch.Tensor) -> torch.Tensor:
    """Approximates the integral of nodal values `z` over the sphere."""
    return torch.einsum('y,...xy->...', self._integration_weights, z)

  # -- velocity / vorticity conversions ---------------------------------------

  def cos_lat_vector(
      self,
      vorticity: torch.Tensor,
      divergence: torch.Tensor,
      clip: bool = True,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes `v cosθ` in the modal basis from modal vorticity/divergence."""
    stream_function = self.inverse_laplacian(vorticity)
    velocity_potential = self.inverse_laplacian(divergence)
    grad_potential = self.cos_lat_grad(velocity_potential, clip=clip)
    rot_stream = self.k_cross(self.cos_lat_grad(stream_function, clip=clip))
    return (
        grad_potential[0] + rot_stream[0],
        grad_potential[1] + rot_stream[1],
    )

  def uv_nodal_to_vor_div_modal(
      self, u_nodal: torch.Tensor, v_nodal: torch.Tensor, clip: bool = True
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Converts nodal `u, v` velocities to modal `vorticity, divergence`."""
    u_over_cos_lat = self.to_modal(u_nodal / self.cos_lat)
    v_over_cos_lat = self.to_modal(v_nodal / self.cos_lat)
    vorticity = self.curl_cos_lat((u_over_cos_lat, v_over_cos_lat), clip=clip)
    divergence = self.div_cos_lat((u_over_cos_lat, v_over_cos_lat), clip=clip)
    return vorticity, divergence

  def vor_div_to_uv_nodal(
      self,
      vorticity: torch.Tensor,
      divergence: torch.Tensor,
      clip: bool = True,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Converts modal `vorticity, divergence` to nodal `u, v` velocities."""
    u_cos_lat, v_cos_lat = self.cos_lat_vector(vorticity, divergence, clip=clip)
    u_nodal = self.to_nodal(u_cos_lat) / self.cos_lat
    v_nodal = self.to_nodal(v_cos_lat) / self.cos_lat
    return u_nodal, v_nodal


# In the spectral basis, a constant field of ones has this value in entry
# [0, 0]. This is a consequence of the way we normalize Legendre polynomials:
# 1 / (basis value of the constant mode) = sqrt(4 pi).
_CONSTANT_NORMALIZATION_FACTOR = 3.5449077


def add_constant(x: torch.Tensor, c) -> torch.Tensor:
  """Adds the constant `c` to the tensor `x` in the spectral basis."""
  x = x.clone()
  x[..., 0, 0] += _CONSTANT_NORMALIZATION_FACTOR * c
  return x
