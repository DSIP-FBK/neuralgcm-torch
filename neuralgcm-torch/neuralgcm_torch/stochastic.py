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
"""Random fields with spatial and temporal correlations (SPPT-style).

Randomness uses plain integer seeds mixed (splitmix64) into per-draw
`torch.Generator`s, instead of the legacy key-splitting machinery. Sampled
realizations are therefore *statistically* equivalent to the original JAX implementation
(itself statistically equivalent to jax), not bitwise equal; the
deterministic AR(1) parameters (phi, sigma spectrum) match exactly.

Only the fields referenced by published checkpoints are ported:
`NoRandomField`, `ZerosRandomField`, `GaussianRandomField` and
`BatchGaussianRandomFieldModule` (which holds learnable correlation
parameters and is vectorized over the batch of fields).
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dinosaur_torch import pytree
from dinosaur_torch import scales
from dinosaur_torch import spherical_harmonic
from dinosaur_torch import units as units_lib

_SOFTPLUS_INVERSE_1 = 0.5413248546129181


@pytree.state
class RandomnessState:
  """Representation of random states on the sphere.

  Attributes:
    core: internal representation of the random state.
    nodal_value: random field values in the nodal representation.
    modal_value: random field values in the modal representation.
    prng_key: integer seed for the underlying RNG.
    prng_step: iteration counter mixed into the seed at each advance.
  """

  core: Optional[torch.Tensor] = None
  nodal_value: Optional[torch.Tensor] = None
  modal_value: Optional[torch.Tensor] = None
  prng_key: Optional[int] = None
  prng_step: Optional[int] = None


def _splitmix64(x: int) -> int:
  """Mixes an integer into a well-distributed 64-bit value."""
  x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
  z = x
  z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
  z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
  return z ^ (z >> 31)


def fold_in(key: int, data: int) -> int:
  """Derives a new seed from `key` and `data`."""
  return _splitmix64(_splitmix64(key) ^ _splitmix64(data))


def _generator(key: int, device) -> torch.Generator:
  gen = torch.Generator(device=device)
  gen.manual_seed(_splitmix64(key) % (2**63))
  return gen


def truncated_normal(
    key: int, lower: float, upper: float, shape, *, dtype, device
) -> torch.Tensor:
  """Samples a truncated standard normal via inverse-CDF."""
  gen = _generator(key, device)
  u = torch.rand(shape, generator=gen, dtype=torch.float64, device=device)
  cdf = torch.distributions.Normal(0.0, 1.0).cdf
  lo = cdf(torch.tensor(lower, dtype=torch.float64))
  hi = cdf(torch.tensor(upper, dtype=torch.float64))
  x = torch.special.ndtri(lo + u * (hi - lo))
  return x.clamp(lower, upper).to(dtype)


def nondimensionalize(x, physics_specs) -> float:
  if isinstance(x, (scales.Quantity, str)):
    return float(physics_specs.nondimensionalize(scales.Quantity(x)))
  return x


def maybe_nondimensionalize(x, physics_specs):
  if x is None or (isinstance(x, str) and x == 'None'):
    return None
  return nondimensionalize(x, physics_specs)


def make_positive_scalar(raw_parameter: torch.Tensor) -> torch.Tensor:
  """Positive [batch] scalar values, maps 0 --> 1 using a softplus(...)."""
  return F.softplus(raw_parameter + _SOFTPLUS_INVERSE_1)


def convert_param_to_positive_scalar(param, initial_value):
  """Converts a [batch] scalar parameter to a scalar value via softplus."""
  return initial_value * make_positive_scalar(param)


class NoRandomField(nn.Module):
  """A random field that returns `None` values."""

  HAIKU_NAME = 'no_random_field'

  def unconditional_sample(self, rng: int) -> RandomnessState:
    return RandomnessState(prng_key=fold_in(rng, 1), prng_step=0)

  def advance(self, state: RandomnessState) -> RandomnessState:
    return RandomnessState(
        prng_key=state.prng_key, prng_step=state.prng_step + 1
    )

  def to_nodal_values(self, core_state):
    return None

  def to_modal_values(self, core_state):
    return None


class ZerosRandomField(nn.Module):
  """A random field of deterministic zeros."""

  HAIKU_NAME = 'zeros_random_field'

  def __init__(self, grid: spherical_harmonic.Grid):
    super().__init__()
    self.grid = grid

  def _zeros_state(self, key, step) -> RandomnessState:
    ref = self.grid.cos_lat
    zeros = lambda shape: torch.zeros(
        shape, dtype=ref.dtype, device=ref.device
    )
    return RandomnessState(
        core=zeros(self.grid.modal_shape),
        nodal_value=zeros(self.grid.nodal_shape),
        modal_value=zeros(self.grid.modal_shape),
        prng_key=key,
        prng_step=step,
    )

  def unconditional_sample(self, rng: int) -> RandomnessState:
    return self._zeros_state(fold_in(rng, 1), 0)

  def advance(self, state: RandomnessState) -> RandomnessState:
    return self._zeros_state(state.prng_key, state.prng_step + 1)

  def to_nodal_values(self, core_state):
    return torch.zeros_like(core_state) if core_state is None else (
        self.grid.to_nodal(core_state))

  def to_modal_values(self, core_state):
    return core_state


class _GaussianRandomFieldBase(nn.Module):
  """Shared machinery for (batched) Gaussian random fields.

  Subclasses provide `_correlation_times()`, `_correlation_lengths()` and
  `_variances()` returning tensors with a leading batch dim (size 1 for a
  single field).
  """

  def __init__(self, grid: spherical_harmonic.Grid, dt: float,
               clip: float = 6.0):
    super().__init__()
    self.grid = grid
    self.dt = dt
    self.clip = clip
    ref = grid.cos_lat
    mask = torch.as_tensor(np.asarray(grid.mask), device=ref.device)
    self.register_buffer('mask', mask, persistent=False)
    self.register_buffer(
        '_total_wavenumbers',
        torch.as_tensor(
            np.asarray(grid.modal_axes[1], np.float64),
            dtype=ref.dtype, device=ref.device,
        ),
        persistent=False,
    )
    self.register_buffer(
        '_n_lon_wavenumbers',
        torch.as_tensor(
            np.asarray(grid.mask.sum(axis=0), np.float64),
            dtype=ref.dtype, device=ref.device,
        ),
        persistent=False,
    )

  # -- parameter accessors (overridden by subclasses) -----------------------

  def _correlation_times(self) -> torch.Tensor:
    raise NotImplementedError

  def _correlation_lengths(self) -> torch.Tensor:
    raise NotImplementedError

  def _variances(self) -> torch.Tensor:
    raise NotImplementedError

  # -- AR(1) parameters ------------------------------------------------------

  def _phi(self) -> torch.Tensor:
    """One-step correlation, shape (fields, 1, 1)."""
    tau = self._correlation_times()
    return torch.exp(-self.dt / tau)[:, None, None]

  def _one_minus_phi2(self) -> torch.Tensor:
    tau = self._correlation_times()
    return -torch.expm1(-2 * self.dt / tau)[:, None, None]

  def _sigma_array(self) -> torch.Tensor:
    """Per-wavenumber std devs sigma_n, shape (fields, 1, L); see Palmer."""
    radius = self.grid.radius
    kt = ((self._correlation_lengths() / radius) ** 2 / 2)[:, None]
    n = self._total_wavenumbers
    sigmas_unnormed = torch.exp(-0.5 * kt * n * (n + 1))  # (fields, L)
    sum_unnormed_vars = torch.sum(
        self._n_lon_wavenumbers * sigmas_unnormed**2, dim=-1, keepdim=True
    )
    surf_area = 4 * np.pi * radius**2
    integrated_variance = (self._variances() * surf_area)[:, None]
    normalization = torch.sqrt(
        integrated_variance
        * self._one_minus_phi2()[:, 0, 0, None]
        / sum_unnormed_vars
    )
    return (normalization * sigmas_unnormed / radius)[:, None, :]

  # -- sampling ---------------------------------------------------------------

  def _num_fields(self) -> int:
    return self._variances().shape[0]

  def _eta(self, key: int) -> torch.Tensor:
    """Masked truncated-normal noise, shape (fields, m, L)."""
    ref = self.grid.cos_lat
    shape = (self._num_fields(),) + tuple(self.grid.modal_shape)
    eta = truncated_normal(
        key, -self.clip, self.clip, shape, dtype=ref.dtype, device=ref.device
    )
    return torch.where(self.mask, eta, torch.zeros_like(eta))

  def _squeeze(self, x: torch.Tensor) -> torch.Tensor:
    """Drops the fields dim for single-field subclasses."""
    return x

  def unconditional_sample(self, rng: int) -> RandomnessState:
    """Returns a randomly initialized state for the autoregressive process."""
    sample_key = fold_in(rng, 0)
    next_key = fold_in(rng, 1)
    core = self._squeeze(
        self._one_minus_phi2() ** (-0.5) * self._sigma_array() * self._eta(
            sample_key)
    )
    return RandomnessState(
        core=core,
        nodal_value=self.to_nodal_values(core),
        modal_value=self.to_modal_values(core),
        prng_key=next_key,
        prng_step=0,
    )

  def advance(self, state: RandomnessState) -> RandomnessState:
    """Updates the core state of the random field."""
    if state.core is None:
      raise ValueError('Got state.core=None when a value is expected.')
    if isinstance(state.prng_key, (tuple, list)):
      return self._advance_batched(state)
    step_key = fold_in(state.prng_key, int(state.prng_step))
    next_core = self._squeeze(
        self._phi()
    ) * state.core + self._squeeze(self._sigma_array() * self._eta(step_key))
    return RandomnessState(
        core=next_core,
        nodal_value=self.to_nodal_values(next_core),
        modal_value=self.to_modal_values(next_core),
        prng_key=state.prng_key,
        prng_step=state.prng_step + 1,
    )

  def _advance_batched(self, state: RandomnessState) -> RandomnessState:
    """Advances a member-batched state (see `ensembles.stack_states`).

    The core carries an explicit (member, fields, ...) layout (no
    `_squeeze`), `prng_key` is one key per member, and each member draws
    exactly the noise its sequential advance would draw.
    """
    step = int(state.prng_step)
    eta = torch.stack(
        [self._eta(fold_in(key, step)) for key in state.prng_key]
    )
    next_core = self._phi() * state.core + self._sigma_array() * eta
    return RandomnessState(
        core=next_core,
        nodal_value=self.grid.to_nodal(next_core),
        modal_value=next_core,
        prng_key=tuple(state.prng_key),
        prng_step=state.prng_step + 1,
    )

  def to_modal_values(self, core_state):
    return core_state

  def to_nodal_values(self, core_state):
    return self.grid.to_nodal(core_state)


class GaussianRandomField(_GaussianRandomFieldBase):
  """A single Gaussian random field with fixed parameters.

  See Appendix 8 of Palmer et al. (2009): the field follows an AR(1)
  recursion `U(t + dt) = phi U(t) + sigma_n eta` in spectral space, with the
  average pointwise variance over the sphere equal to `variance`.
  """

  HAIKU_NAME = 'gaussian_random_field'

  def __init__(
      self,
      grid: spherical_harmonic.Grid,
      dt: float,
      correlation_time: Union[float, str, scales.Quantity],
      correlation_length: Union[float, str, scales.Quantity],
      variance: Union[float, str, scales.Quantity],
      clip: float = 6.0,
      *,
      physics_specs: Optional[units_lib.SimUnits] = None,
  ):
    super().__init__(grid, dt, clip)
    ref = grid.cos_lat
    as_tensor = lambda x: torch.as_tensor(
        nondimensionalize(x, physics_specs),
        dtype=ref.dtype, device=ref.device,
    ).reshape(1)
    self.register_buffer(
        '_correlation_time', as_tensor(correlation_time), persistent=False
    )
    self.register_buffer(
        '_correlation_length', as_tensor(correlation_length),
        persistent=False,
    )
    self.register_buffer('_variance', as_tensor(variance), persistent=False)

  def _correlation_times(self):
    return self._correlation_time

  def _correlation_lengths(self):
    return self._correlation_length

  def _variances(self):
    return self._variance

  def _squeeze(self, x: torch.Tensor) -> torch.Tensor:
    return x.squeeze(0)


class BatchGaussianRandomFieldModule(_GaussianRandomFieldBase):
  """Batch of independent Gaussian random fields with learned correlations.

  State arrays carry a leading batch dim indexing the fields. The
  correlation times/lengths are learnable (softplus-positive around the
  initial values); variances are fixed.
  """

  HAIKU_NAME = 'batch_gaussian_random_field_module'

  def __init__(
      self,
      grid: spherical_harmonic.Grid,
      dt: float,
      initial_correlation_times: Sequence,
      initial_correlation_lengths: Sequence,
      variances: Sequence,
      field_subset: Optional[Sequence[int]] = None,
      n_fixed_fields: Optional[int] = None,
      clip: float = 6.0,
      *,
      physics_specs: Optional[units_lib.SimUnits] = None,
  ):
    super().__init__(grid, dt, clip)
    lengths = [
        len(initial_correlation_times),
        len(initial_correlation_lengths),
        len(variances),
    ]
    if len(set(lengths)) != 1:
      raise ValueError(f'Argument lengths differed: {lengths=}')
    n_fixed_fields = n_fixed_fields or 0

    if field_subset is not None:
      if not field_subset:
        raise ValueError(
            '`field_subset` must be `None` or non-empty, got '
            f'{field_subset=}'
        )
      get = lambda seq: [seq[i] for i in field_subset]
      initial_correlation_times = get(initial_correlation_times)
      initial_correlation_lengths = get(initial_correlation_lengths)
      variances = get(variances)

    ref = grid.cos_lat
    as_tensor = lambda seq: torch.as_tensor(
        [nondimensionalize(x, physics_specs) for x in seq],
        dtype=ref.dtype, device=ref.device,
    )
    self.n_fields = len(variances)
    self.n_fixed_fields = n_fixed_fields
    self.register_buffer(
        '_variances_fixed', as_tensor(variances), persistent=False
    )
    self.register_buffer(
        '_initial_correlation_times', as_tensor(initial_correlation_times),
        persistent=False,
    )
    self.register_buffer(
        '_initial_correlation_lengths',
        as_tensor(initial_correlation_lengths),
        persistent=False,
    )
    n_learned = self.n_fields - n_fixed_fields
    self.correlation_times_raw = nn.Parameter(
        torch.zeros(n_learned, dtype=ref.dtype, device=ref.device)
    )
    self.correlation_lengths_raw = nn.Parameter(
        torch.zeros(n_learned, dtype=ref.dtype, device=ref.device)
    )

  def _padded(self, raw: torch.Tensor) -> torch.Tensor:
    if self.n_fixed_fields:
      raw = torch.cat([raw, torch.zeros_like(raw[:1]).expand(
          self.n_fixed_fields)])
    return raw

  def _correlation_times(self):
    return convert_param_to_positive_scalar(
        self._padded(self.correlation_times_raw),
        self._initial_correlation_times,
    )

  def _correlation_lengths(self):
    return convert_param_to_positive_scalar(
        self._padded(self.correlation_lengths_raw),
        self._initial_correlation_lengths,
    )

  def _variances(self):
    return self._variances_fixed

  def import_haiku(self, params: dict, prefix: str) -> None:
    bundle = params[prefix]
    with torch.no_grad():
      self.correlation_times_raw.copy_(bundle['correlation_times_raw'])
      self.correlation_lengths_raw.copy_(bundle['correlation_lengths_raw'])


class GaussianRandomFieldModule(_GaussianRandomFieldBase):
  """A single Gaussian random field with learned correlation scales.

  The per-field unit of `DictOfGaussianRandomFieldModules`: correlation
  time and length are learnable (softplus-positive around the initial
  values, like the batch module); the variance is fixed.
  """

  HAIKU_NAME = 'gaussian_random_field_module'

  def __init__(
      self,
      grid: spherical_harmonic.Grid,
      dt: float,
      initial_correlation_time: Union[float, str, scales.Quantity],
      initial_correlation_length: Union[float, str, scales.Quantity],
      variance: Union[float, str, scales.Quantity],
      clip: float = 6.0,
      *,
      physics_specs: Optional[units_lib.SimUnits] = None,
  ):
    super().__init__(grid, dt, clip)
    ref = grid.cos_lat
    as_tensor = lambda x: torch.as_tensor(
        nondimensionalize(x, physics_specs),
        dtype=ref.dtype, device=ref.device,
    ).reshape(1)
    self.register_buffer(
        '_initial_correlation_time', as_tensor(initial_correlation_time),
        persistent=False,
    )
    self.register_buffer(
        '_initial_correlation_length',
        as_tensor(initial_correlation_length),
        persistent=False,
    )
    self.register_buffer('_variance', as_tensor(variance), persistent=False)
    self.correlation_time_raw = nn.Parameter(
        torch.zeros((), dtype=ref.dtype, device=ref.device)
    )
    self.correlation_length_raw = nn.Parameter(
        torch.zeros((), dtype=ref.dtype, device=ref.device)
    )

  def _correlation_times(self):
    return convert_param_to_positive_scalar(
        self.correlation_time_raw.reshape(1), self._initial_correlation_time
    )

  def _correlation_lengths(self):
    return convert_param_to_positive_scalar(
        self.correlation_length_raw.reshape(1),
        self._initial_correlation_length,
    )

  def _variances(self):
    return self._variance

  def _squeeze(self, x: torch.Tensor) -> torch.Tensor:
    return x.squeeze(0)

  def import_haiku(self, params: dict, prefix: str) -> None:
    bundle = params[prefix]
    with torch.no_grad():
      self.correlation_time_raw.copy_(
          bundle['correlation_time_raw'].reshape(())
      )
      self.correlation_length_raw.copy_(
          bundle['correlation_length_raw'].reshape(())
      )


class DictOfGaussianRandomFieldModules(nn.Module):
  """Dictionary of independent learned Gaussian random fields.

  The `RandomnessState` leaves (`core`, `nodal_value`, `modal_value`) are
  dicts keyed by field name. Draws use per-field subkeys derived from the
  state's integer key (statistically, not bitwise, equivalent to the
  legacy key-splitting).
  """

  HAIKU_NAME = 'dict_of_gaussian_random_field_modules'

  def __init__(self, fields: dict):
    super().__init__()
    self.fields = nn.ModuleDict(fields)

  def unconditional_sample(self, rng: int) -> RandomnessState:
    core, nodal, modal = {}, {}, {}
    for i, (name, field) in enumerate(self.fields.items()):
      sample = field.unconditional_sample(fold_in(rng, i))
      core[name] = sample.core
      nodal[name] = sample.nodal_value
      modal[name] = sample.modal_value
    return RandomnessState(
        core=core,
        nodal_value=nodal,
        modal_value=modal,
        prng_key=fold_in(rng, len(self.fields)),
        prng_step=0,
    )

  def advance(self, state: RandomnessState) -> RandomnessState:
    batched = isinstance(state.prng_key, (tuple, list))
    if batched:
      # one sub-key chain per member, matching the sequential draws
      step_keys = tuple(
          fold_in(key, int(state.prng_step)) for key in state.prng_key
      )
    else:
      step_key = fold_in(state.prng_key, int(state.prng_step))
    core, nodal, modal = {}, {}, {}
    for i, (name, field) in enumerate(self.fields.items()):
      advanced = field.advance(
          RandomnessState(
              core=state.core[name],
              prng_key=tuple(fold_in(k, i) for k in step_keys)
              if batched
              else fold_in(step_key, i),
              prng_step=0,
          )
      )
      core[name] = advanced.core
      nodal[name] = advanced.nodal_value
      modal[name] = advanced.modal_value
    return RandomnessState(
        core=core,
        nodal_value=nodal,
        modal_value=modal,
        prng_key=tuple(state.prng_key) if batched else state.prng_key,
        prng_step=state.prng_step + 1,
    )

  def import_haiku(self, params: dict, prefix: str) -> None:
    # each field is a named child module in the legacy __init__.
    for name, field in self.fields.items():
      field.import_haiku(params, f'{prefix}/~/{name}')
