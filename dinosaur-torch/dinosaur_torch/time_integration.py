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

"""Implicit-explicit time integration routines for ODEs.

States are torch pytrees (tensors, dicts, registered dataclasses). Time
steppers combine stage tendencies leafwise with a single fused
`linear_combination` per stage; trajectories are plain Python loops (wrap
the step in `torch.compile` / CUDA-graph capture for performance).
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Sequence, Union

import numpy as np
import torch

from dinosaur_torch import filtering
from dinosaur_torch import pytree
from dinosaur_torch import spherical_harmonic


PyTreeState = Any
TimeStepFn = Callable[[PyTreeState], PyTreeState]

# For consistency with commonly accepted notation, we use Greek letters
# within some of the functions below.
# pylint: disable=invalid-name,non-ascii-name


def linear_combination(trees: Sequence, coeffs: Sequence[float]):
  """Computes `sum_i coeffs[i] * trees[i]` leafwise.

  `None` leaves (matched across all trees) pass through as `None`.
  """

  def combine(*leaves):
    if leaves[0] is None:
      return None
    return sum(c * leaf for c, leaf in zip(coeffs, leaves))

  return pytree.tree_map(combine, *trees)


class ExplicitODE:
  """Describes a set of ODEs with only explicit terms."""

  def explicit_terms(self, state: PyTreeState) -> PyTreeState:
    """Evaluates explicit terms in the ODE."""
    raise NotImplementedError

  @classmethod
  def from_functions(cls, explicit_terms) -> ExplicitODE:
    """Constructs an `ExplicitODE` instance with given methods."""
    explicit_ode = cls()
    explicit_ode.explicit_terms = explicit_terms
    return explicit_ode


class ImplicitExplicitODE:
  """Describes a set of ODEs with implicit & explicit terms.

  The structure of the equation is assumed to be:

    ∂x/∂t = explicit_terms(x) + implicit_terms(x)

  `explicit_terms(x)` includes terms that should use explicit time-stepping
  and `implicit_terms(x)` includes terms that should be modeled implicitly.

  Typically the explicit terms are non-linear and the implicit terms are
  linear. This simplifies solves but isn't strictly necessary.
  """

  def explicit_terms(self, state: PyTreeState) -> PyTreeState:
    """Evaluates explicit terms in the ODE."""
    raise NotImplementedError

  def implicit_terms(self, state: PyTreeState) -> PyTreeState:
    """Evaluates implicit terms in the ODE."""
    raise NotImplementedError

  def implicit_inverse(
      self, state: PyTreeState, step_size: float
  ) -> PyTreeState:
    """Applies `(1 - step_size * implicit_terms)⁻¹` to `state`."""
    raise NotImplementedError

  @classmethod
  def from_functions(
      cls, explicit_terms, implicit_terms, implicit_inverse
  ) -> ImplicitExplicitODE:
    """Constructs an `ImplicitExplicitODE` instance with given methods."""
    ode = cls()
    ode.explicit_terms = explicit_terms
    ode.implicit_terms = implicit_terms
    ode.implicit_inverse = implicit_inverse
    return ode


@dataclasses.dataclass
class TimeReversedImExODE(ImplicitExplicitODE):
  """An ImplicitExplicitODE reversed in time.

  The reversed ODE follows the equation:

    ∂x/∂t = -explicit_terms(x) - implicit_terms(x)
  """

  forward_eq: ImplicitExplicitODE

  def explicit_terms(self, state: PyTreeState) -> PyTreeState:
    forward_term = self.forward_eq.explicit_terms(state)
    return pytree.tree_map(lambda x: -x, forward_term)

  def implicit_terms(self, state: PyTreeState) -> PyTreeState:
    forward_term = self.forward_eq.implicit_terms(state)
    return pytree.tree_map(lambda x: -x, forward_term)

  def implicit_inverse(
      self, state: PyTreeState, step_size: float
  ) -> PyTreeState:
    return self.forward_eq.implicit_inverse(state, -step_size)


def compose_equations(
    equations: Sequence[Union[ImplicitExplicitODE, ExplicitODE]],
) -> ImplicitExplicitODE:
  """Combines `equations` with exactly one ImplicitExplicitODE instance."""
  implicit_explicit_eqs = [
      eq for eq in equations if isinstance(eq, ImplicitExplicitODE)
  ]
  if len(implicit_explicit_eqs) != 1:
    raise ValueError(
        'compose_equations supports at most 1 ImplicitExplicitODE '
        f'got {len(implicit_explicit_eqs)}'
    )
  (implicit_explicit_equation,) = implicit_explicit_eqs

  def explicit_fn(x: PyTreeState) -> PyTreeState:
    explicit_tendencies = [eq.explicit_terms(x) for eq in equations]
    return pytree.tree_map(
        lambda *args: sum(x for x in args if x is not None),
        *explicit_tendencies,
    )

  return ImplicitExplicitODE.from_functions(
      explicit_fn,
      implicit_explicit_equation.implicit_terms,
      implicit_explicit_equation.implicit_inverse,
  )


def backward_forward_euler(
    equation: ImplicitExplicitODE, time_step: float
) -> TimeStepFn:
  """Time stepping via forward and backward Euler methods.

  This method is first order accurate.
  """
  dt = time_step
  F = equation.explicit_terms
  G_inv = equation.implicit_inverse

  def step_fn(u0):
    g = linear_combination([u0, F(u0)], [1, dt])
    return G_inv(g, dt)

  return step_fn


def crank_nicolson_rk2(
    equation: ImplicitExplicitODE, time_step: float
) -> TimeStepFn:
  """Time stepping via Crank-Nicolson and 2nd order Runge-Kutta (Heun).

  This method is second order accurate.

  Reference:
    Chandler, G. J. & Kerswell, R. R. Invariant recurrent solutions embedded
    in a turbulent two-dimensional Kolmogorov flow. J. Fluid Mech. 722,
    554–595 (2013). https://doi.org/10.1017/jfm.2013.122 (Section 3)
  """
  dt = time_step
  F = equation.explicit_terms
  G = equation.implicit_terms
  G_inv = equation.implicit_inverse

  def step_fn(u0):
    g = linear_combination([u0, G(u0)], [1, 0.5 * dt])
    h1 = F(u0)
    u1 = G_inv(linear_combination([g, h1], [1, dt]), 0.5 * dt)
    u2 = G_inv(
        linear_combination([g, F(u1), h1], [1, 0.5 * dt, 0.5 * dt]), 0.5 * dt
    )
    return u2

  return step_fn


def low_storage_runge_kutta_crank_nicolson(
    alphas: Sequence[float],
    betas: Sequence[float],
    gammas: Sequence[float],
    equation: ImplicitExplicitODE,
    time_step: float,
) -> TimeStepFn:
  """Time stepping via "low-storage" Runge-Kutta and Crank-Nicolson steps.

  These schemes are second order accurate for the implicit terms, but
  potentially higher order accurate for the explicit terms. This seems to be
  a favorable tradeoff when the explicit terms dominate, e.g., for modeling
  turbulent fluids.

  Per Canuto: "[these methods] have been widely used for the
  time-discretization in applications of spectral methods."

  Reference:
    Canuto, C., Yousuff Hussaini, M., Quarteroni, A. & Zang, T. A.
    Spectral Methods: Evolution to Complex Geometries and Applications to
    Fluid Dynamics. (Springer Berlin Heidelberg, 2007).
    https://doi.org/10.1007/978-3-540-30728-0 (Appendix D.3)
  """
  α = alphas
  β = betas
  γ = gammas
  dt = time_step
  F = equation.explicit_terms
  G = equation.implicit_terms
  G_inv = equation.implicit_inverse

  if len(alphas) - 1 != len(betas) != len(gammas):
    raise ValueError('number of RK coefficients does not match')

  def step_fn(u):
    h = None
    for k in range(len(β)):
      if h is None:
        h = F(u)
      else:
        h = linear_combination([F(u), h], [1, β[k]])
      µ = 0.5 * dt * (α[k + 1] - α[k])
      u = G_inv(
          linear_combination([u, h, G(u)], [1, γ[k] * dt, µ]), µ
      )
    return u

  return step_fn


def crank_nicolson_rk3(
    equation: ImplicitExplicitODE, time_step: float
) -> TimeStepFn:
  """Time stepping via Crank-Nicolson and RK3 ('Williamson')."""
  return low_storage_runge_kutta_crank_nicolson(
      alphas=[0, 1 / 3, 3 / 4, 1],
      betas=[0, -5 / 9, -153 / 128],
      gammas=[1 / 3, 15 / 16, 8 / 15],
      equation=equation,
      time_step=time_step,
  )


def crank_nicolson_rk4(
    equation: ImplicitExplicitODE, time_step: float
) -> TimeStepFn:
  """Time stepping via Crank-Nicolson and RK4 ('Carpenter-Kennedy')."""
  # pylint: disable=line-too-long
  return low_storage_runge_kutta_crank_nicolson(
      alphas=[0, 0.1496590219993, 0.3704009573644, 0.6222557631345, 0.9582821306748, 1],
      betas=[0, -0.4178904745, -1.192151694643, -1.697784692471, -1.514183444257],
      gammas=[0.1496590219993, 0.3792103129999, 0.8229550293869, 0.6994504559488, 0.1530572479681],
      equation=equation,
      time_step=time_step,
  )


@dataclasses.dataclass
class ImExButcherTableau:
  """Butcher Tableau for implicit-explicit Runge-Kutta methods."""

  a_ex: Sequence[Sequence[float]]
  a_im: Sequence[Sequence[float]]
  b_ex: Sequence[float]
  b_im: Sequence[float]

  def __post_init__(self):
    if (
        len({
            len(self.a_ex) + 1,
            len(self.a_im) + 1,
            len(self.b_ex),
            len(self.b_im),
        })
        > 1
    ):
      raise ValueError('inconsistent Butcher tableau')


def imex_runge_kutta(
    tableau: ImExButcherTableau,
    equation: ImplicitExplicitODE,
    time_step: float,
) -> TimeStepFn:
  """Time stepping with Implicit-Explicit Runge-Kutta."""
  dt = time_step
  F = equation.explicit_terms
  G = equation.implicit_terms
  G_inv = equation.implicit_inverse

  a_ex = tableau.a_ex
  a_im = tableau.a_im
  b_ex = tableau.b_ex
  b_im = tableau.b_im

  num_steps = len(b_ex)

  def step_fn(y0):
    f = [None] * num_steps
    g = [None] * num_steps

    f[0] = F(y0)
    g[0] = G(y0)

    for i in range(1, num_steps):
      trees, coeffs = [y0], [1.0]
      for j in range(i):
        if a_ex[i - 1][j]:
          trees.append(f[j])
          coeffs.append(dt * a_ex[i - 1][j])
        if a_im[i - 1][j]:
          trees.append(g[j])
          coeffs.append(dt * a_im[i - 1][j])
      Y_star = linear_combination(trees, coeffs)
      Y = G_inv(Y_star, dt * a_im[i - 1][i])
      if any(a_ex[j][i] for j in range(i, num_steps - 1)) or b_ex[i]:
        f[i] = F(Y)
      if any(a_im[j][i] for j in range(i, num_steps - 1)) or b_im[i]:
        g[i] = G(Y)

    trees, coeffs = [y0], [1.0]
    for j in range(num_steps):
      if b_ex[j]:
        trees.append(f[j])
        coeffs.append(dt * b_ex[j])
      if b_im[j]:
        trees.append(g[j])
        coeffs.append(dt * b_im[j])
    return linear_combination(trees, coeffs)

  return step_fn


def imex_rk_sil3(
    equation: ImplicitExplicitODE, time_step: float
) -> TimeStepFn:
  """Time stepping with the SIL3 implicit-explicit RK scheme.

  This method is second-order accurate for the implicit terms and third-order
  accurate for the explicit terms.

  Reference:
    Whitaker, J. S. & Kar, S. K. Implicit-Explicit Runge-Kutta Methods for
    Fast-Slow Wave Problems. Monthly Weather Review vol. 141 3426-3434 (2013)
    http://dx.doi.org/10.1175/mwr-d-13-00132.1
  """
  return imex_runge_kutta(
      tableau=ImExButcherTableau(
          a_ex=[[1 / 3], [1 / 6, 1 / 2], [1 / 2, -1 / 2, 1]],
          a_im=[[1 / 6, 1 / 6], [1 / 3, 0, 1 / 3], [3 / 8, 0, 3 / 8, 1 / 4]],
          b_ex=[1 / 2, -1 / 2, 1, 0],
          b_im=[3 / 8, 0, 3 / 8, 1 / 4],
      ),
      equation=equation,
      time_step=time_step,
  )


#  ===========================================================================
#  Time integration filters, for use with step_with_filters.
#  ===========================================================================


def runge_kutta_step_filter(state_filter):
  """Convert a state filter into a Runge-Kutta time integration filter."""

  def _filter(u: PyTreeState, u_next: PyTreeState) -> PyTreeState:
    del u  # unused
    return state_filter(u_next)

  return _filter


def exponential_step_filter(
    grid: spherical_harmonic.Grid,
    dt: float,
    tau: float = 0.010938,
    order: int = 18,
    cutoff: float = 0,
):
  """Returns an exponential step filter.

  This filter simulates dampening on modes according to:

    (∂u_k / ∂t) ≈ -(u_k / 𝜏) * ((k - cutoff) / (1 - cutoff)) ** (2 * order)

  For more details see `filtering.exponential_filter`.

  Args:
    grid: the `spherical_harmonic.Grid` to use for the computation.
    dt: size of the time step to be used for each filter application.
    tau: timescale over which modes are reduced by the corresponding
      exponential factors determined by the wavenumbers, `order` and
      `cutoff`. Default value represents attenuation of `16` for a time step
      of 20 minutes.
    order: controls the polynomial order of the exponential filter.
    cutoff: a hard threshold with which to start attenuation.

  Returns:
    A function that accepts (u, u_next) and returns a filtered u_next.
  """
  filter_fn = filtering.exponential_filter(grid, dt / tau, order, cutoff)
  return runge_kutta_step_filter(filter_fn)


def horizontal_diffusion_step_filter(
    grid: spherical_harmonic.Grid, dt: float, tau: float, order: int = 1
):
  """Returns a horizontal diffusion step filter.

  This filter simulates dampening on modes according to:

    (∂u_k / ∂t) ≈ -(u_k / 𝜏) * (((k * (k + 1)) / (L * (L + 1))) ** order)

  Where L is the maximum total wavenumber. For more details see
  `filtering.horizontal_diffusion_filter`.
  """
  eigenvalues = grid.spec.laplacian_eigenvalues
  scale = dt / (tau * abs(eigenvalues[-1]) ** order)
  filter_fn = filtering.horizontal_diffusion_filter(grid, scale, order)
  return runge_kutta_step_filter(filter_fn)


#  ===========================================================================
#  Utility functions for deriving trajectories and steps.
#  ===========================================================================


def step_with_filters(
    step_fn: TimeStepFn, filters: Sequence[Callable]
) -> TimeStepFn:
  """Returns a step function with `filters` applied to outputs in order."""

  def _step_fn(u: PyTreeState) -> PyTreeState:
    u_next = step_fn(u)
    for filter_fn in filters:
      u_next = filter_fn(u, u_next)
    return u_next

  return _step_fn


def repeated(fn: TimeStepFn, steps: int) -> TimeStepFn:
  """Returns a version of fn() that is repeatedly applied `steps` times."""
  if steps == 1:
    return fn

  def f_repeated(x: PyTreeState) -> PyTreeState:
    for _ in range(steps):
      x = fn(x)
    return x

  return f_repeated


def trajectory_from_step(
    step_fn: TimeStepFn,
    outer_steps: int,
    inner_steps: int,
    *,
    start_with_input: bool = False,
    post_process_fn: Callable = lambda x: x,
) -> Callable[[PyTreeState], tuple[PyTreeState, Any]]:
  """Returns a function that accumulates repeated applications of `step_fn`.

  Computes a trajectory by repeatedly calling `step_fn()`
  `outer_steps * inner_steps` times; output frames are stacked along a new
  leading dimension.

  Args:
    step_fn: function that takes a state and returns state after one step.
    outer_steps: number of steps to save in the generated trajectory.
    inner_steps: number of repeated calls to step_fn() between saved steps.
    start_with_input: if True, output the trajectory at steps [0, ...,
      steps-1] instead of steps [1, ..., steps].
    post_process_fn: function to apply to trajectory outputs.

  Returns:
    A function that takes an initial state and returns a tuple consisting of:
      (1) the final frame of the trajectory.
      (2) trajectory of length `outer_steps` representing time evolution.
  """
  if inner_steps != 1:
    step_fn = repeated(step_fn, inner_steps)

  def multistep(x: PyTreeState) -> tuple[PyTreeState, Any]:
    frames = []
    for _ in range(outer_steps):
      x_next = step_fn(x)
      frame = x if start_with_input else x_next
      frames.append(post_process_fn(frame))
      x = x_next
    def stack(*leaves):
      if leaves[0] is None:
        return None
      if isinstance(leaves[0], torch.Tensor):
        return torch.stack(leaves)
      return np.asarray(leaves)

    stacked = pytree.tree_map(stack, *frames)
    return x, stacked

  return multistep


#  ===========================================================================
#  Utilities for digital filter initialization.
#  ===========================================================================


def accumulate_repeated(
    step_fn: TimeStepFn, weights: np.ndarray, state: PyTreeState
) -> PyTreeState:
  """Accumulate the weighted average of repeatedly applying a function."""
  averaged = pytree.tree_map(torch.zeros_like, state)
  for weight in weights:
    state = step_fn(state)
    averaged = linear_combination([averaged, state], [1, float(weight)])
  return averaged


def _dfi_lanczos_weights(
    time_span: float, cutoff_period: float, dt: float
) -> np.ndarray:
  """Calculate Lanczos weights for digital filter initialization."""
  N = round(time_span / (2 * dt))
  n = np.arange(1, N + 1)
  w = np.sinc(n / (N + 1)) * np.sinc(n * time_span / (cutoff_period * N))
  return w


def digital_filter_initialization(
    equation: ImplicitExplicitODE,
    ode_solver: Callable[[ImplicitExplicitODE, float], TimeStepFn],
    filters: Sequence[Callable],
    time_span: float,
    cutoff_period: float,
    dt: float,
) -> TimeStepFn:
  """Create a function to perform digital filter initialization.

  Args:
    equation: equation to solve for forward dynamics. This equation must be
      reversible (i.e., it should only include dynamics).
    ode_solver: ODE solver to use for time-stepping.
    filters: sequence of filters to apply after each ODE step forward or
      backwards.
    time_span: the ODE is solved over the time interval
      [-time_span/2, time_span/2]. Typically 6 hours.
    cutoff_period: cutoff period for the Lanczos filter. Typically matches
      time_span.
    dt: time step size.

  Returns:
    Function that can be applied to an initial state to filter it.

  Reference:
    Lynch, P. & Huang, X.-Y. Initialization of the HIRLAM Model Using a
    Digital Filter. Mon. Weather Rev. 120, 1019–1034 (1992)
    https://doi.org/10.1175/1520-0493(1992)120<1019:IOTHMU>2.0.CO;2
  """

  def f(state: PyTreeState) -> PyTreeState:
    forward_step = step_with_filters(ode_solver(equation, dt), filters)
    backward_step = step_with_filters(
        ode_solver(TimeReversedImExODE(equation), dt), filters
    )
    # for times [1, ..., N] and [-1, ..., -N]
    weights = _dfi_lanczos_weights(time_span, cutoff_period, dt)
    init_weight = 1.0  # for time=0
    total_weight = init_weight + 2 * weights.sum()
    # normalize
    init_weight /= total_weight
    weights /= total_weight
    # add up the weighted contributions.
    init_term = pytree.tree_map(lambda x: x * init_weight, state)
    forward_term = accumulate_repeated(forward_step, weights, state)
    backward_term = accumulate_repeated(backward_step, weights, state)
    return linear_combination(
        [init_term, forward_term, backward_term], [1, 1, 1]
    )

  return f


def maybe_fix_sim_time_roundoff(state: PyTreeState, dt: float) -> PyTreeState:
  """Returns `state` with sim_time rounded to an integer value of `dt`."""
  if getattr(state, 'sim_time', None) is not None:
    state = dataclasses.replace(
        state, sim_time=dt * torch.round(state.sim_time / dt)
    )
  return state
