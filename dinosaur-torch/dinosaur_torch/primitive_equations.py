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

"""The primitive equations written for a semi-implicit solver.

Sigma-coordinate path only: every published NeuralGCM checkpoint uses sigma
coordinates (the upstream hybrid-coordinate variant is not ported).

`PrimitiveEquations` is an `nn.Module`: the vertical weight matrices
(geopotential `G`, temperature-implicit `H`, Durran §8.6.5) are precomputed
as non-persistent buffers; the implicit-inverse blocks are built lazily per
step size. `State` is a dataclass registered as a torch pytree.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Mapping, Optional, Union

import numpy as np
import torch
from torch import nn

from dinosaur_torch import coordinate_systems
from dinosaur_torch import pytree
from dinosaur_torch import sigma_coordinates
from dinosaur_torch import spherical_harmonic
from dinosaur_torch import time_integration
from dinosaur_torch import units


# For consistency with commonly accepted notation, we use Greek letters
# within some of the functions below.
# pylint: disable=invalid-name


#  ===========================================================================
#  Data Structures
#  ===========================================================================


@pytree.state
class State:
  """Records the state of a system described by the primitive equations."""

  vorticity: torch.Tensor
  divergence: torch.Tensor
  temperature_variation: torch.Tensor
  log_surface_pressure: torch.Tensor
  tracers: Mapping[str, torch.Tensor] = dataclasses.field(
      default_factory=dict
  )
  sim_time: Optional[Union[float, torch.Tensor]] = None


@pytree.state
class DiagnosticState:
  """Stores nodal diagnostic values used to compute explicit tendencies.

  The expected shapes of the state are described in terms of # of layers
  `h`, # of longitude quadrature points `q` and # of latitude quadrature
  points `t`.

  Attributes:
    vorticity: nodal values of the vorticity field of shape [h, q, t].
    divergence: nodal values of the divergence field of shape [h, q, t].
    temperature_variation: nodal values of the T' field of shape [h, q, t].
    cos_lat_u: tuple of nodal values of cosθ * velocity_vector, each of shape
      [h, q, t].
    sigma_dot_explicit: nodal values of d𝜎/dt due to pressure gradient terms
      `u · ∇(log(ps))` of shape [h, q, t].
    sigma_dot_full: nodal values of d𝜎/dt due to all terms of shape
      [h, q, t].
    cos_lat_grad_log_sp: (2,) nodal values of cosθ · ∇(log(ps)) of shape
      [1, q, t].
    u_dot_grad_log_sp: nodal values of `u · ∇(log(ps))` of shape [h, q, t].
    tracers: mapping from tracer names to corresponding nodal values.
  """

  vorticity: torch.Tensor
  divergence: torch.Tensor
  temperature_variation: torch.Tensor
  cos_lat_u: tuple[torch.Tensor, torch.Tensor]
  sigma_dot_explicit: torch.Tensor
  sigma_dot_full: torch.Tensor
  cos_lat_grad_log_sp: tuple[torch.Tensor, torch.Tensor]
  u_dot_grad_log_sp: torch.Tensor
  tracers: Mapping[str, torch.Tensor]


class StateShapeError(Exception):
  """Exceptions for unexpected state shapes."""


def validate_state_shape(
    state: State, coords: coordinate_systems.CoordinateSystem
):
  """Validates that values in `state` have appropriate shapes."""
  modal_shape = tuple(coords.modal_shape)
  surface_modal_shape = tuple(coords.surface_modal_shape)
  if tuple(state.vorticity.shape) != modal_shape:
    raise StateShapeError(
        f'Expected vorticity shape {modal_shape}; '
        f'got shape {tuple(state.vorticity.shape)}.'
    )
  if tuple(state.divergence.shape) != modal_shape:
    raise StateShapeError(
        f'Expected divergence shape {modal_shape}; '
        f'got shape {tuple(state.divergence.shape)}.'
    )
  if tuple(state.temperature_variation.shape) != modal_shape:
    raise StateShapeError(
        f'Expected temperature_variation shape {modal_shape}; '
        f'got shape {tuple(state.temperature_variation.shape)}.'
    )
  if tuple(state.log_surface_pressure.shape) != surface_modal_shape:
    raise StateShapeError(
        f'Expected log_surface_pressure shape {surface_modal_shape}; '
        f'got shape {tuple(state.log_surface_pressure.shape)}.'
    )
  for tracer_name, array in state.tracers.items():
    if tuple(array.shape[-3:]) != modal_shape:
      raise StateShapeError(
          f'Expected tracer {tracer_name} shape {modal_shape}; '
          f'got shape {tuple(array.shape)}.'
      )


def _vertical_matvec(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
  return torch.einsum('gh,...hml->...gml', a, x)


def _vertical_matvec_per_wavenumber(
    a: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
  return torch.einsum('lgh,...hml->...gml', a, x)


def compute_diagnostic_state(
    state: State, coords: coordinate_systems.CoordinateSystem
) -> DiagnosticState:
  """Computes `DiagnosticState` in nodal basis based on the modal `state`."""
  grid = coords.horizontal
  to_nodal = grid.to_nodal

  nodal_vorticity = to_nodal(state.vorticity)
  nodal_divergence = to_nodal(state.divergence)
  nodal_temperature_variation = to_nodal(state.temperature_variation)
  tracers = to_nodal(state.tracers)
  u_cos_lat, v_cos_lat = grid.cos_lat_vector(
      state.vorticity, state.divergence, clip=False
  )
  nodal_cos_lat_u = (to_nodal(u_cos_lat), to_nodal(v_cos_lat))
  cos_lat_grad_log_sp = grid.cos_lat_grad(
      state.log_surface_pressure, clip=False
  )
  nodal_cos_lat_grad_log_sp = (
      to_nodal(cos_lat_grad_log_sp[0]),
      to_nodal(cos_lat_grad_log_sp[1]),
  )
  nodal_u_dot_grad_log_sp = (
      nodal_cos_lat_u[0] * nodal_cos_lat_grad_log_sp[0]
      + nodal_cos_lat_u[1] * nodal_cos_lat_grad_log_sp[1]
  ) * grid.sec2_lat

  vertical = coords.vertical
  f_explicit = vertical.cumulative_sigma_integral(nodal_u_dot_grad_log_sp)
  f_full = vertical.cumulative_sigma_integral(
      nodal_divergence + nodal_u_dot_grad_log_sp
  )
  # note: we only need velocities at the inner boundaries of coords.vertical.
  # level slices count from the end (leaves may carry a leading batch axis)
  sum_𝜎 = torch.cumsum(vertical.layer_thickness, 0)[:, None, None]
  surface = lambda f: f[..., -1:, :, :]
  inner = lambda f: f[..., :-1, :, :]
  sigma_dot_explicit = inner(sum_𝜎 * surface(f_explicit) - f_explicit)
  sigma_dot_full = inner(sum_𝜎 * surface(f_full) - f_full)
  return DiagnosticState(
      vorticity=nodal_vorticity,
      divergence=nodal_divergence,
      temperature_variation=nodal_temperature_variation,
      cos_lat_u=nodal_cos_lat_u,
      sigma_dot_explicit=sigma_dot_explicit,
      sigma_dot_full=sigma_dot_full,
      cos_lat_grad_log_sp=nodal_cos_lat_grad_log_sp,
      u_dot_grad_log_sp=nodal_u_dot_grad_log_sp,
      tracers=tracers,
  )


#  ===========================================================================
#  Vertical weight matrices (NumPy, construction time)
#  ===========================================================================


def get_sigma_ratios(
    coordinates: sigma_coordinates.SigmaCoordinates,
) -> np.ndarray:
  """Returns the log ratios of the sigma values for the given coordinates.

  These values are used as weights when computing geopotentials. In
  'Numerical Methods for Fluid Dynamics', Durran refers to these values as
  `𝜎[j]`.

  Returns:
    A vector 𝜶 with length `coordinates.layers` such that, for `n + 1`
    layers,
                 𝜶[n] = -log(𝜎[n])
                 𝜶[j] = log(𝜎[j + 1] / 𝜎[j]) / 2    for j < n
  """
  alpha = np.diff(np.log(coordinates.centers), append=0) / 2
  alpha[-1] = -np.log(coordinates.centers[-1])
  return alpha


def get_geopotential_weights(
    coordinates: sigma_coordinates.SigmaCoordinates,
    ideal_gas_constant: float,
) -> np.ndarray:
  """Returns a matrix of weights used to compute the geopotential.

  In 'Numerical Methods for Fluid Dynamics' §8.6.5, Durran refers to this
  matrix as `G`:

               𝜶[0]    𝜶[0] + 𝜶[1]    𝜶[1] + 𝜶[2]    𝜶[2] + 𝜶[3]    ᠁
    G / R  =   0       𝜶[1]           𝜶[1] + 𝜶[2]    𝜶[2] + 𝜶[3]    ᠁
               0       0              𝜶[2]           𝜶[2] + 𝜶[3]    ᠁
               ⋮       ⋮               ⋮              ⋮              ⋱

  where 𝜶 is the vector returned by `get_sigma_ratios`.
  """
  # Since this matrix is computed only once, we favor readability over
  # efficiency in its construction.
  alpha = get_sigma_ratios(coordinates)
  weights = np.zeros([coordinates.layers, coordinates.layers])
  for j in range(coordinates.layers):
    weights[j, j] = alpha[j]
    for k in range(j + 1, coordinates.layers):
      weights[j, k] = alpha[k] + alpha[k - 1]
  return ideal_gas_constant * weights


def get_temperature_implicit_weights(
    coordinates: sigma_coordinates.SigmaCoordinates,
    reference_temperature: np.ndarray,
    kappa: float,
) -> np.ndarray:
  """Returns weights used to compute implicit terms for the temperature.

  In 'Numerical Methods for Fluid Dynamics' §8.6.5, Durran refers to this
  matrix as `H`. Its entry in row `r` and column `s` is given by

    H[r, s] / Δ𝜎[s] = 𝜅T[r] · (P(r - s) 𝛼[r] + P(r - s - 1) 𝛼[r - 1]) / Δ𝜎[r]
                      - ̇K[r, s]
                      - K[r - 1, s]

  with

    K[r, s] = (T[r + 1] - T[r]) / (Δ𝜎[r + 1] + Δ𝜎[r])
              · (P(r - s) - sum(Δ𝜎[:r + 1]))
    K[r, s] = 0  if r < 0 or `r = coordinates.layers - 1`

  where `T` is the reference temperature and `P` is an indicator function
  that takes the value 0 on negative numbers and 1 on non-negative numbers.
  """
  if (
      reference_temperature.ndim != 1
      or reference_temperature.shape[-1] != coordinates.layers
  ):
    raise ValueError(
        '`reference_temp` must be a vector of length `coordinates.layers`; '
        f'got shape {reference_temperature.shape} and '
        f'{coordinates.layers} layers.'
    )

  # The function P in matrix form, where `p[r, s] = p(r - s)`
  p = np.tril(np.ones([coordinates.layers, coordinates.layers]))

  # Compute the first term in the sum above.
  alpha = get_sigma_ratios(coordinates)[..., np.newaxis]
  p_alpha = p * alpha
  p_alpha_shifted = np.roll(p_alpha, 1, axis=0)
  p_alpha_shifted[0] = 0
  h0 = (
      kappa
      * reference_temperature[..., np.newaxis]
      * (p_alpha + p_alpha_shifted)
      / coordinates.layer_thickness[..., np.newaxis]
  )

  # Constructing the values k[r, s].
  temp_diff = np.diff(reference_temperature)
  thickness_sum = (
      coordinates.layer_thickness[:-1] + coordinates.layer_thickness[1:]
  )
  # (T[r + 1] - T[r]) / (Δ𝜎[r + 1] + Δ𝜎[r])
  k0 = np.concatenate((temp_diff / thickness_sum, [0]), axis=0)[
      ..., np.newaxis
  ]

  thickness_cumulative = np.cumsum(coordinates.layer_thickness)[
      ..., np.newaxis
  ]
  # P(r - s) - sum(Δ𝜎[:r + 1])
  k1 = p - thickness_cumulative

  k = k0 * k1

  # `k_shifted[r, s] = k[r - 1, s]`, padded with zeros at `r = 0`.
  k_shifted = np.roll(k, 1, axis=0)
  k_shifted[0] = 0

  return (h0 - k - k_shifted) * coordinates.layer_thickness


def _get_implicit_term_matrix(
    eta: float,
    grid_spec: spherical_harmonic.GridSpec,
    coordinates: sigma_coordinates.SigmaCoordinates,
    reference_temperature: np.ndarray,
    kappa: float,
    ideal_gas_constant: float,
) -> np.ndarray:
  """Returns a matrix corresponding to `PrimitiveEquations.implicit_terms`.

  Shape is `[total_wavenumbers, 2 * layers + 1, 2 * layers + 1]`, acting on
  the stacked (divergence, temperature_variation, log_surface_pressure)
  column per total wavenumber.
  """
  eye = np.eye(coordinates.layers)[np.newaxis]
  lam = grid_spec.laplacian_eigenvalues
  g = get_geopotential_weights(coordinates, ideal_gas_constant)
  r = ideal_gas_constant
  h = get_temperature_implicit_weights(
      coordinates, reference_temperature, kappa
  )
  t = reference_temperature[:, np.newaxis]
  thickness = coordinates.layer_thickness[np.newaxis, np.newaxis, :]

  l = grid_spec.total_wavenumbers
  j = k = coordinates.layers

  row0 = np.concatenate(
      [
          np.broadcast_to(eye, [l, j, k]),
          eta * np.einsum('l,jk->ljk', lam, g),
          eta * r * np.einsum('l,jo->ljo', lam, t),
      ],
      axis=2,
  )
  row1 = np.concatenate(
      [
          eta * np.broadcast_to(h[np.newaxis], [l, j, k]),
          np.broadcast_to(eye, [l, j, k]),
          np.zeros([l, j, 1]),
      ],
      axis=2,
  )
  row2 = np.concatenate(
      [
          eta * np.broadcast_to(thickness, [l, 1, k]),
          np.zeros([l, 1, k]),
          np.ones([l, 1, 1]),
      ],
      axis=2,
  )
  return np.concatenate((row0, row1, row2), axis=1)


#  ===========================================================================
#  Helper functions
#  ===========================================================================


def div_sec_lat(
    m_component: torch.Tensor,
    n_component: torch.Tensor,
    grid: spherical_harmonic.Grid,
) -> torch.Tensor:
  """Computes div_sec_lat (aka H operator in Durran) in modal basis.

  Computes divergences of sec(θ) * (m, n) vector (equivalently operator H):

    H(M, N) = ((1 / cos²θ) * ∂M/∂λ + ∂N/∂(sinθ) / R)

  which captures some explicit tendencies in primitive equations. Note:
  this operator does not include the 1/a scaling factor.
  """
  m_modal = grid.to_modal(m_component * grid.sec2_lat)
  n_modal = grid.to_modal(n_component * grid.sec2_lat)
  return grid.div_cos_lat((m_modal, n_modal), clip=False)


def truncated_modal_orography(
    orography: torch.Tensor,
    coords: coordinate_systems.CoordinateSystem,
    wavenumbers_to_clip: int = 1,
) -> torch.Tensor:
  """Returns modal orography with `n` highest wavenumbers truncated."""
  grid = coords.horizontal
  expected_shape = grid.nodal_shape
  if tuple(orography.shape) != tuple(expected_shape):
    raise ValueError(f'Expected nodal orography with shape={expected_shape}')
  return grid.clip_wavenumbers(grid.to_modal(orography), n=wavenumbers_to_clip)


class Geopotential(nn.Module):
  """Computes nodal geopotential from nodal temperature (and moisture).

  Precomputes the Durran `G` weight matrix as a buffer. If
  `specific_humidity` is given, virtual temperature effects are included;
  cloud condensate (if given) is subtracted from the virtual temperature.
  """

  def __init__(
      self,
      coordinates: sigma_coordinates.SigmaCoordinates,
      gravity_acceleration: float,
      ideal_gas_constant: float,
      water_vapor_gas_constant: float | None = None,
      *,
      device: torch.device | str | None = None,
      dtype: torch.dtype = torch.float32,
  ):
    super().__init__()
    self.gravity_acceleration = gravity_acceleration
    self.ideal_gas_constant = ideal_gas_constant
    self.water_vapor_gas_constant = water_vapor_gas_constant
    weights = get_geopotential_weights(coordinates, ideal_gas_constant)
    self.register_buffer(
        'weights',
        torch.as_tensor(weights, dtype=dtype, device=device),
        persistent=False,
    )

  def forward(
      self,
      temperature: torch.Tensor,
      nodal_orography: torch.Tensor,
      specific_humidity: torch.Tensor | None = None,
      clouds: torch.Tensor | None = None,
  ) -> torch.Tensor:
    surface_geopotential = nodal_orography * self.gravity_acceleration
    if specific_humidity is not None:
      if self.water_vapor_gas_constant is None:
        raise ValueError(
            'Must provide `water_vapor_gas_constant` with '
            '`specific_humidity`.'
        )
      gas_const_ratio = (
          self.water_vapor_gas_constant / self.ideal_gas_constant
      )
      cloud_effect = 0.0 if clouds is None else clouds
      virtual_temp = temperature * (
          1 + (gas_const_ratio - 1) * specific_humidity - cloud_effect
      )
    else:
      virtual_temp = temperature
    return surface_geopotential + _vertical_matvec(self.weights, virtual_temp)


#  ===========================================================================
#  The `PrimitiveEquations` module
#  ===========================================================================


class PrimitiveEquations(nn.Module, time_integration.ImplicitExplicitODE):
  """Semi-implicit primitive equations on terrain-following sigma coordinates.

  Args:
    reference_temperature: array of shape [layers]. All temperature values
      are expressed as their difference from this value.
    orography: tensor of shape `coords.horizontal.modal_shape` describing
      the topography in modal representation.
    coords: horizontal and vertical discretization.
    physics_specs: object holding nondimensionalized physical constants.
    include_vertical_advection: whether to include tendencies from vertical
      advection.
    vertical_advection: 'centered', 'upwind', or a callable
      `f(levels, w, x)`.
    humidity_key: key for specific humidity in the tracers dict. If
      provided, moisture effects are included in the dynamics.
    cloud_keys: keys for cloud water species in the tracers dict. If
      provided, cloud effects are included in virtual temperature.
  """

  def __init__(
      self,
      reference_temperature: np.ndarray,
      orography: torch.Tensor,
      coords: coordinate_systems.CoordinateSystem,
      physics_specs: units.SimUnits,
      *,
      include_vertical_advection: bool = True,
      vertical_advection: Union[str, Callable] = 'centered',
      humidity_key: str | None = None,
      cloud_keys: tuple[str, ...] | None = None,
  ):
    super().__init__()
    if not np.allclose(
        coords.horizontal.radius, physics_specs.radius, rtol=1e-5
    ):
      raise ValueError(
          'inconsistent radius between coordinates and constants: '
          f'{coords.horizontal.radius=} != {physics_specs.radius=}'
      )
    if cloud_keys is not None and humidity_key is None:
      raise ValueError('cloud_keys requires humidity_key to be set.')

    self.coords = coords
    self.physics_specs = physics_specs
    self.include_vertical_advection = include_vertical_advection
    self.humidity_key = humidity_key
    self.cloud_keys = cloud_keys
    self.reference_temperature = np.asarray(reference_temperature)
    self._t_ref_is_variable = (
        np.unique(self.reference_temperature.ravel()).size > 1
    )
    if isinstance(vertical_advection, str):
      self._vertical_advection_fn = {
          'centered': type(coords.vertical).centered_vertical_advection,
          'upwind': type(coords.vertical).upwind_vertical_advection,
      }[vertical_advection]
    else:
      self._vertical_advection_fn = vertical_advection

    ref = coords.horizontal.cos_lat  # reference buffer for device/dtype
    device, dtype = ref.device, ref.dtype
    coordinates = coords.vertical.coordinates

    def buffer(name, array):
      self.register_buffer(
          name,
          torch.as_tensor(
              np.asarray(array, np.float64), dtype=dtype, device=device
          ),
          persistent=False,
      )

    if not isinstance(orography, torch.Tensor):
      orography = torch.as_tensor(
          np.ascontiguousarray(orography), dtype=dtype, device=device
      )
    self.register_buffer(
        'orography', orography.to(dtype=dtype, device=device),
        persistent=False,
    )

    # T_ref with spatial dimensions appended.
    buffer('T_ref', self.reference_temperature[:, np.newaxis, np.newaxis])
    # Coriolis parameter 2Ω sin(θ); constant along longitude.
    _, sin_lat = coords.horizontal.spec.nodal_axes
    buffer(
        'coriolis_parameter',
        2 * physics_specs.angular_velocity * sin_lat[np.newaxis, :],
    )
    buffer(
        'geopotential_weights',
        get_geopotential_weights(coordinates, physics_specs.R),
    )
    buffer(
        'neg_temperature_implicit_weights',
        -get_temperature_implicit_weights(
            coordinates, self.reference_temperature, physics_specs.kappa
        ),
    )
    buffer('_thickness_row', coordinates.layer_thickness[np.newaxis])
    buffer('_alpha_columns', get_sigma_ratios(coordinates)[:, None, None])
    buffer(
        '_thickness_columns', coordinates.layer_thickness[:, None, None]
    )

    # Inverse blocks of the implicit matrix, built lazily per step size.
    self._implicit_inverse_parts: dict[float, dict[Any, torch.Tensor]] = {}

  # -- moisture helpers --------------------------------------------------------

  def _get_tracer(self, state_or_aux: Any, key: str) -> torch.Tensor:
    if key not in state_or_aux.tracers:
      raise ValueError(
          f'`{key}` is not found in tracers: {state_or_aux.tracers.keys()}.'
      )
    return state_or_aux.tracers[key]

  def _get_specific_humidity(self, state_or_aux: Any) -> torch.Tensor:
    if self.humidity_key is None:
      raise ValueError('humidity_key is not set.')
    return self._get_tracer(state_or_aux, self.humidity_key)

  def _cloud_virtual_t_adjustment(self, aux_state: Any):
    """Adjustment to the virtual temperature due to clouds (negative)."""
    adjustment = 0.0
    if self.cloud_keys is not None:
      for key in self.cloud_keys:
        # clouds reduce the virtual temperature, hence the negative sign.
        adjustment = adjustment - self._get_tracer(aux_state, key)
    return adjustment

  def _virtual_temperature_adjustment(self, aux_state: Any):
    """Computes the factor (1 + 0.61q - q_cloud) for virtual temperature."""
    if self.humidity_key is None:
      return 1.0
    q = self._get_specific_humidity(aux_state)
    gas_const_ratio = self.physics_specs.R_vapor / self.physics_specs.R
    adjustment = 1 + (gas_const_ratio - 1) * q
    # _cloud_virtual_t_adjustment returns a negative value (-cloud_water).
    adjustment = adjustment + self._cloud_virtual_t_adjustment(aux_state)
    return adjustment

  # -- explicit tendencies -----------------------------------------------------

  def _vertical_tendency(
      self, w: torch.Tensor, x: torch.Tensor
  ) -> torch.Tensor:
    """Computes vertical nodal tendency of `x` due to vertical velocity."""
    return self._vertical_advection_fn(self.coords.vertical, w, x)

  def kinetic_energy_tendency(self, aux_state: DiagnosticState):
    """Explicit tendency of divergence due to the kinetic energy term."""
    grid = self.coords.horizontal
    nodal_cos_lat_u2 = torch.stack(list(aux_state.cos_lat_u)) ** 2
    kinetic = nodal_cos_lat_u2.sum(0) * grid.sec2_lat / 2
    return -grid.laplacian(grid.to_modal(kinetic))

  def orography_tendency(self) -> torch.Tensor:
    """Orography contribution to divergence tendency (geopotential)."""
    # this term broadcasts correctly as layers are leading indices.
    return -self.physics_specs.g * self.coords.horizontal.laplacian(
        self.orography
    )

  def horizontal_scalar_advection(
      self, scalar: torch.Tensor, aux_state: DiagnosticState
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Explicit tendency of `scalar` due to horizontal advection."""
    u, v = aux_state.cos_lat_u
    nodal_terms = scalar * aux_state.divergence
    modal_terms = -div_sec_lat(
        u * scalar, v * scalar, self.coords.horizontal
    )
    return nodal_terms, modal_terms

  def divergence_tendency_due_to_humidity(
      self, state: State, aux_state: DiagnosticState
  ) -> torch.Tensor:
    """Divergence tendencies from moist geopotential and pressure terms.

    The terms computed here correspond to the laplacian of the moist part
    of: (1) ∆(R (Tv - T) log(ps)) and (2) ∆(Φ(Tv) - Φ(T)).
    """
    grid = self.coords.horizontal
    physics_specs = self.physics_specs
    q = self._get_specific_humidity(aux_state)
    # contribution of (virtual - normal) temperature x laplacian(log ps).
    nodal_laplacian_lsp = grid.to_nodal(
        grid.laplacian(state.log_surface_pressure)
    )
    nodal_laplacian_correction_term = (
        q
        * nodal_laplacian_lsp
        * self.T_ref
        * (physics_specs.R_vapor - physics_specs.R)
    )
    # term differentiating the spatially dependent part of reference
    # virtual temperature.
    q_modal = self._get_specific_humidity(state)
    cos_lat_grad_q = grid.cos_lat_grad(q_modal, clip=False)
    nodal_cos_lat_grad_q = (
        grid.to_nodal(cos_lat_grad_q[0]),
        grid.to_nodal(cos_lat_grad_q[1]),
    )
    coefficient = self.T_ref * (physics_specs.R_vapor - physics_specs.R)
    nodal_dot_term = (
        coefficient
        * grid.sec2_lat
        * (
            nodal_cos_lat_grad_q[0] * aux_state.cos_lat_grad_log_sp[0]
            + nodal_cos_lat_grad_q[1] * aux_state.cos_lat_grad_log_sp[1]
        )
    )

    temperature = aux_state.temperature_variation + self.T_ref
    temperature_diff = (
        q * temperature * (physics_specs.R_vapor / physics_specs.R - 1)
    )
    geopotential_diff = _vertical_matvec(
        self.geopotential_weights, temperature_diff
    )

    return -grid.laplacian(grid.to_modal(geopotential_diff)) - grid.to_modal(
        nodal_dot_term + nodal_laplacian_correction_term
    )

  def vorticity_tendency_due_to_humidity(
      self, state: State, aux_state: DiagnosticState
  ) -> torch.Tensor:
    """Computes vorticity tendencies due to humidity."""
    grid = self.coords.horizontal
    physics_specs = self.physics_specs
    q_modal = self._get_specific_humidity(state)
    cos_lat_grad_q = grid.cos_lat_grad(q_modal, clip=False)
    nodal_cos_lat_grad_q = (
        grid.to_nodal(cos_lat_grad_q[0]),
        grid.to_nodal(cos_lat_grad_q[1]),
    )
    nodal_cos_lat_grad_log_sp = aux_state.cos_lat_grad_log_sp
    coefficient = self.T_ref * (physics_specs.R_vapor - physics_specs.R)
    nodal_curl_term = (
        coefficient
        * grid.sec2_lat
        * (
            nodal_cos_lat_grad_log_sp[0] * nodal_cos_lat_grad_q[1]
            - nodal_cos_lat_grad_log_sp[1] * nodal_cos_lat_grad_q[0]
        )
    )
    return grid.to_modal(nodal_curl_term)

  def _t_omega_over_sigma_sp(
      self,
      temperature_field: torch.Tensor,
      g_term: torch.Tensor,
      v_dot_grad_log_sp: torch.Tensor,
  ) -> torch.Tensor:
    """Computes nodal terms of the form `T * omega / p`.

    A helper for the temperature tendency terms of the form

      ∂T/∂t[n] ~ (T * ⍵/p)[n], where ⍵ = dp/dt

    using the scheme of 'Numerical Methods for Fluid Dynamics' §8.6.3,
    eq. 8.124, which approximates ⍵/p as:

      ⍵/p[n] = v·∇(ln(ps))[n] - (1 / Δ𝜎[n]) * (𝛼[n] sum(G[:n] Δ𝜎[:n]) +
                                               𝛼[n-1] sum(G[:n-1] Δ𝜎[:n-1]))
    """
    f = self.coords.vertical.cumulative_sigma_integral(g_term)
    alpha_f = self._alpha_columns * f
    # shift one level downward (level axis is -3; a leading batch axis may
    # be present)
    alpha_f_shifted = torch.cat(
        [
            torch.zeros_like(alpha_f[..., :1, :, :]),
            alpha_f[..., :-1, :, :],
        ],
        dim=-3,
    )
    g_part = (alpha_f + alpha_f_shifted) / self._thickness_columns
    return temperature_field * (v_dot_grad_log_sp - g_part)

  def curl_and_div_tendencies(
      self, aux_state: DiagnosticState
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Curl and divergence tendencies for vorticity ζ and divergence 𝛅.

    Computes the explicit tendencies (dζ_dt, d𝛅_dt) due to curl and
    divergence terms in the primitive equations:

      dζ_dt = -k · ∇ ✕ ((ζ + f)(k ✕ v) + d𝜎_dt ∂v/∂𝜎 + RT'∇(ln(p_s)))
      d𝛅_dt = - ∇ · ((ζ + f)(k ✕ v) + d𝜎_dt ∂v/∂𝜎 + RT'∇(ln(p_s)))
    """
    grid = self.coords.horizontal
    u, v = aux_state.cos_lat_u
    sec2_lat = grid.sec2_lat
    # note the cos_lat cancels out with sec2_lat and cos in derivative ops.
    total_vorticity = aux_state.vorticity + self.coriolis_parameter
    # note that u, v are switched to correspond to `k ✕ v = (-v, u)`.
    nodal_vorticity_u = -v * total_vorticity * sec2_lat
    nodal_vorticity_v = u * total_vorticity * sec2_lat
    # vertical and pressure gradient terms
    d𝜎_dt = aux_state.sigma_dot_full
    if self.include_vertical_advection:
      # vertical tendency equals `-1 * dot{sigma} * u`, hence the negation.
      sigma_dot_u = -self._vertical_tendency(d𝜎_dt, u)
      sigma_dot_v = -self._vertical_tendency(d𝜎_dt, v)
    else:
      sigma_dot_u = 0
      sigma_dot_v = 0

    adjustment = self._virtual_temperature_adjustment(aux_state)
    rt = self.physics_specs.R * aux_state.temperature_variation * adjustment

    grad_log_ps_u, grad_log_ps_v = aux_state.cos_lat_grad_log_sp
    vertical_term_u = (sigma_dot_u + rt * grad_log_ps_u) * sec2_lat
    vertical_term_v = (sigma_dot_v + rt * grad_log_ps_v) * sec2_lat
    combined_u = grid.to_modal(nodal_vorticity_u + vertical_term_u)
    combined_v = grid.to_modal(nodal_vorticity_v + vertical_term_v)
    dζ_dt = -grid.curl_cos_lat((combined_u, combined_v), clip=False)
    d𝛅_dt = -grid.div_cos_lat((combined_u, combined_v), clip=False)
    return (dζ_dt, d𝛅_dt)

  def nodal_temperature_vertical_tendency(
      self, aux_state: DiagnosticState
  ):
    """Computes explicit vertical tendency of the temperature."""
    # two types of terms of sigma_dot * ∂T/∂𝜎;
    # the second term is zero if T_ref does not depend on layer_id.
    if self.include_vertical_advection:
      tendency = self._vertical_tendency(
          aux_state.sigma_dot_full, aux_state.temperature_variation
      )
    else:
      tendency = 0
    if self._t_ref_is_variable:
      # T_ref has shape (layers, 1, 1) and broadcasts through the advection.
      tendency = tendency + self._vertical_tendency(
          aux_state.sigma_dot_explicit, self.T_ref
      )
    return tendency

  def nodal_temperature_adiabatic_tendency(
      self, aux_state: DiagnosticState
  ) -> torch.Tensor:
    """Explicit temperature tendency due to adiabatic processes."""
    g_explicit = aux_state.u_dot_grad_log_sp
    g_full = g_explicit + aux_state.divergence
    t_ref = self.T_ref
    mean_t_part = self._t_omega_over_sigma_sp(
        t_ref, g_explicit, aux_state.u_dot_grad_log_sp
    )
    if self.humidity_key is None:
      variation_t_part = self._t_omega_over_sigma_sp(
          aux_state.temperature_variation, g_full, aux_state.u_dot_grad_log_sp
      )
      return self.physics_specs.kappa * (mean_t_part + variation_t_part)
    else:
      gas_const_ratio = self.physics_specs.R_vapor / self.physics_specs.R
      heat_capacity_ratio = self.physics_specs.Cp_vapor / self.physics_specs.Cp
      q = self._get_specific_humidity(aux_state)
      # Here Tv refers to virtual temperature. The terms below capture
      # tendencies from full temperature variation and moist T_ref terms.
      variation_temperature_component = aux_state.temperature_variation * (
          (1 + (gas_const_ratio - 1) * q)
          / (1 + (heat_capacity_ratio - 1) * q)
      )
      humidity_reference_component = t_ref * (
          ((gas_const_ratio - heat_capacity_ratio) * q)
          / (1 + (heat_capacity_ratio - 1) * q)
      )
      variation_and_humidity_terms = (
          variation_temperature_component + humidity_reference_component
      )
      variation_and_Tv_part = self._t_omega_over_sigma_sp(
          variation_and_humidity_terms, g_full, aux_state.u_dot_grad_log_sp
      )
      return self.physics_specs.kappa * (mean_t_part + variation_and_Tv_part)

  def nodal_log_pressure_tendency(
      self, aux_state: DiagnosticState
  ) -> torch.Tensor:
    """Computes explicit tendency of the log_surface_pressure."""
    # computes -∑G[i] * ∆𝜎[i] where G[i] = u[i] · ∇(log(ps)).
    g = aux_state.u_dot_grad_log_sp
    return -self.coords.vertical.sigma_integral(g)

  def explicit_terms(self, state: State) -> State:
    """Computes explicit tendencies of the primitive equations."""
    grid = self.coords.horizontal
    aux_state = compute_diagnostic_state(state, self.coords)
    # tendencies that are computed in modal representation
    vorticity_dot, divergence_dot = self.curl_and_div_tendencies(aux_state)
    kinetic_energy_tendency = self.kinetic_energy_tendency(aux_state)
    orography_tendency = self.orography_tendency()

    if self.humidity_key is not None:
      vorticity_dot = vorticity_dot + self.vorticity_tendency_due_to_humidity(
          state, aux_state
      )
      divergence_dot = (
          divergence_dot
          + self.divergence_tendency_due_to_humidity(state, aux_state)
      )

    dT_dt_horizontal_nodal, dT_dt_horizontal_modal = (
        self.horizontal_scalar_advection(
            aux_state.temperature_variation, aux_state
        )
    )
    tracers_horizontal = {
        name: self.horizontal_scalar_advection(tracer, aux_state)
        for name, tracer in aux_state.tracers.items()
    }
    # tendencies in nodal domain
    dT_dt_vertical = self.nodal_temperature_vertical_tendency(aux_state)
    dT_dt_adiabatic = self.nodal_temperature_adiabatic_tendency(aux_state)
    log_sp_tendency = self.nodal_log_pressure_tendency(aux_state)
    if self.include_vertical_advection:
      sigma_dot_full = aux_state.sigma_dot_full
      vertical_tendency_fn = lambda x: self._vertical_tendency(
          sigma_dot_full, x
      )
    else:
      vertical_tendency_fn = lambda x: 0
    # combining tendencies
    to_modal = grid.to_modal
    tendency = State(
        vorticity=vorticity_dot,
        divergence=(
            divergence_dot + kinetic_energy_tendency + orography_tendency
        ),
        temperature_variation=(
            to_modal(dT_dt_horizontal_nodal + dT_dt_vertical + dT_dt_adiabatic)
            + dT_dt_horizontal_modal
        ),
        log_surface_pressure=to_modal(log_sp_tendency),
        tracers={
            name: to_modal(vertical_tendency_fn(aux_state.tracers[name])
                           + tracers_horizontal[name][0])
            + tracers_horizontal[name][1]
            for name in aux_state.tracers
        },
        sim_time=None if state.sim_time is None else 1.0,
    )
    # Note: clipping the final total wavenumber from the explicit tendencies
    # matches SPEEDY.
    return grid.clip_wavenumbers(tendency)

  # -- implicit tendencies -----------------------------------------------------

  def implicit_terms(self, state: State) -> State:
    """Returns the implicit terms of the primitive equations."""
    grid = self.coords.horizontal
    geopotential_diff = _vertical_matvec(
        self.geopotential_weights, state.temperature_variation
    )
    rt_log_p = (
        self.physics_specs.ideal_gas_constant
        * self.T_ref
        * state.log_surface_pressure
    )
    return State(
        vorticity=torch.zeros_like(state.vorticity),
        divergence=-grid.laplacian(geopotential_diff + rt_log_p),
        temperature_variation=_vertical_matvec(
            self.neg_temperature_implicit_weights, state.divergence
        ),
        log_surface_pressure=-_vertical_matvec(
            self._thickness_row, state.divergence
        ),
        tracers={
            name: torch.zeros_like(tracer)
            for name, tracer in state.tracers.items()
        },
        sim_time=None if state.sim_time is None else 0.0,
    )

  def _inverse_parts(
      self, step_size: float, device, dtype
  ) -> dict[Any, torch.Tensor]:
    """Inverse blocks of the implicit matrix for `step_size` (cached)."""
    key = float(step_size)
    parts = self._implicit_inverse_parts.get(key)
    if parts is None:
      matrix = _get_implicit_term_matrix(
          key,
          self.coords.horizontal.spec,
          self.coords.vertical.coordinates,
          self.reference_temperature,
          self.physics_specs.kappa,
          self.physics_specs.R,
      )
      assert matrix.dtype == np.float64
      inverse = np.linalg.inv(matrix)
      assert not np.isnan(inverse).any()
      layers = self.coords.vertical.layers
      blocks = {
          'div': slice(0, layers),
          'temp': slice(layers, 2 * layers),
          'logp': slice(2 * layers, 2 * layers + 1),
      }
      parts = {
          (out_name, in_name): torch.as_tensor(
              np.ascontiguousarray(inverse[:, out_slice, in_slice]),
              dtype=dtype,
              device=device,
          )
          for out_name, out_slice in blocks.items()
          for in_name, in_slice in blocks.items()
      }
      self._implicit_inverse_parts[key] = parts
    return parts

  def implicit_inverse(self, state: State, step_size: float) -> State:
    """Applies `(1 - step_size * implicit_terms)⁻¹` to `state`."""
    parts = self._inverse_parts(
        step_size, state.divergence.device, state.divergence.dtype
    )
    matvec = _vertical_matvec_per_wavenumber
    inverted_divergence = (
        matvec(parts['div', 'div'], state.divergence)
        + matvec(parts['div', 'temp'], state.temperature_variation)
        + matvec(parts['div', 'logp'], state.log_surface_pressure)
    )
    inverted_temperature_variation = (
        matvec(parts['temp', 'div'], state.divergence)
        + matvec(parts['temp', 'temp'], state.temperature_variation)
        + matvec(parts['temp', 'logp'], state.log_surface_pressure)
    )
    inverted_log_surface_pressure = (
        matvec(parts['logp', 'div'], state.divergence)
        + matvec(parts['logp', 'temp'], state.temperature_variation)
        + matvec(parts['logp', 'logp'], state.log_surface_pressure)
    )
    return State(
        vorticity=state.vorticity,
        divergence=inverted_divergence,
        temperature_variation=inverted_temperature_variation,
        log_surface_pressure=inverted_log_surface_pressure,
        tracers=state.tracers,
        sim_time=state.sim_time,
    )


def MoistPrimitiveEquations(*args, **kwargs) -> PrimitiveEquations:
  """Primitive equations with moisture effects from specific humidity."""
  return PrimitiveEquations(*args, humidity_key='specific_humidity', **kwargs)


def MoistPrimitiveEquationsWithCloudMoisture(
    *args, **kwargs
) -> PrimitiveEquations:
  """Moist primitive equations including cloud condensate effects."""
  return PrimitiveEquations(
      *args,
      humidity_key='specific_humidity',
      cloud_keys=(
          'specific_cloud_liquid_water_content',
          'specific_cloud_ice_water_content',
      ),
      **kwargs,
  )
