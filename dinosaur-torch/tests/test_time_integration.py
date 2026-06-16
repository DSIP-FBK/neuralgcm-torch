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

"""Tests for time_integration (ported from the original Dinosaur, pytest style)."""

import numpy as np
import pytest
import torch

from dinosaur_torch import pytree
from dinosaur_torch import time_integration


def _to_numpy(tree):
  return pytree.tree_map(
      lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x, tree
  )


def _stack_over_time(closed_form, x0, time):
  outs = [_to_numpy(closed_form(x0, t)) for t in time]
  return pytree.tree_map(lambda *xs: np.stack(xs), *outs)


def harmonic_oscillator(x0, t):
  x0 = np.asarray(x0)
  theta = np.arctan(x0[0] / x0[1])
  r = np.linalg.norm(x0, 2, axis=0)
  return r * np.stack([np.sin(t + theta), np.cos(t + theta)])


def _zeros_like(x):
  return torch.zeros_like(x)


def _stack_pair(a, b):
  return torch.stack([a, b])


class CustomODE(time_integration.ImplicitExplicitODE):

  def __init__(self, explicit_terms, implicit_terms, implicit_inverse):
    self.explicit_terms = explicit_terms
    self.implicit_terms = implicit_terms
    self.implicit_inverse = implicit_inverse


class CustomExplicitODE(time_integration.ExplicitODE):

  def __init__(self, explicit_terms):
    self.explicit_terms = explicit_terms


ALL_TEST_PROBLEMS = [
    # x(t) = np.ones(10)
    pytest.param(
        dict(
            explicit_terms=lambda x: 0 * x,
            implicit_terms=lambda x: 0 * x,
            implicit_inverse=lambda x, eta: x,
            dt=1e-2,
            inner_steps=10,
            outer_steps=5,
            initial_state=np.ones(10),
            closed_form=lambda x0, t: x0,
            tolerances=[1e-12] * 5,
        ),
        id='zero_derivative',
    ),
    # x(t) = 5 * t * np.ones(3)
    pytest.param(
        dict(
            explicit_terms=lambda x: 5 * torch.ones_like(x),
            implicit_terms=lambda x: 0 * x,
            implicit_inverse=lambda x, eta: x,
            dt=1e-2,
            inner_steps=10,
            outer_steps=5,
            initial_state=np.ones(3),
            closed_form=lambda x0, t: np.asarray(x0) + 5 * t,
            tolerances=[1e-12] * 5,
        ),
        id='constant_derivative',
    ),
    # x(t) = np.arange(3) * np.exp(t), explicit terms only.
    pytest.param(
        dict(
            explicit_terms=lambda x: x,
            implicit_terms=lambda x: 0 * x,
            implicit_inverse=lambda x, eta: x,
            dt=1e-2,
            inner_steps=20,
            outer_steps=5,
            initial_state=np.arange(3.0),
            closed_form=lambda x0, t: np.arange(3) * np.exp(t),
            tolerances=[5e-2, 1e-4, 1e-6, 1e-9, 1e-6],
        ),
        id='linear_derivative_explicit',
    ),
    # x(t) = np.arange(3) * np.exp(t), implicit terms only.
    pytest.param(
        dict(
            explicit_terms=lambda x: 0 * x,
            implicit_terms=lambda x: x,
            implicit_inverse=lambda x, eta: x / (1 - eta),
            dt=1e-2,
            inner_steps=20,
            outer_steps=5,
            initial_state=np.arange(3.0),
            closed_form=lambda x0, t: np.arange(3) * np.exp(t),
            tolerances=[5e-2, 5e-5, 1e-5, 1e-5, 3e-5],
        ),
        id='linear_derivative_implicit',
    ),
    # x(t) = np.arange(3) * np.exp(t), split implicit/explicit.
    pytest.param(
        dict(
            explicit_terms=lambda x: x / 2,
            implicit_terms=lambda x: x / 2,
            implicit_inverse=lambda x, eta: x / (1 - eta / 2),
            dt=1e-2,
            inner_steps=20,
            outer_steps=5,
            initial_state=np.arange(3) * np.exp(0),
            closed_form=lambda x0, t: np.arange(3.0) * np.exp(t),
            tolerances=[1e-4, 2e-5, 2e-6, 1e-6, 2e-5],
        ),
        id='linear_derivative_semi_implicit',
    ),
    pytest.param(
        dict(
            explicit_terms=lambda x: _stack_pair(x[1], -x[0]),
            implicit_terms=_zeros_like,
            implicit_inverse=lambda x, eta: x,
            dt=1e-2,
            inner_steps=20,
            outer_steps=5,
            initial_state=np.ones(2),
            closed_form=harmonic_oscillator,
            tolerances=[1e-2, 3e-5, 6e-8, 5e-11, 6e-8],
        ),
        id='harmonic_oscillator_explicit',
    ),
    pytest.param(
        dict(
            explicit_terms=_zeros_like,
            implicit_terms=lambda x: _stack_pair(x[1], -x[0]),
            implicit_inverse=lambda x, eta: _stack_pair(
                x[0] + eta * x[1], x[1] - eta * x[0]
            )
            / (1 + eta**2),
            dt=1e-2,
            inner_steps=20,
            outer_steps=5,
            initial_state=np.ones(2),
            closed_form=harmonic_oscillator,
            tolerances=[1e-2, 2e-5, 2e-6, 1e-6, 6e-6],
        ),
        id='harmonic_oscillator_implicit',
    ),
]


ALL_TIME_STEPPERS = [
    time_integration.backward_forward_euler,
    time_integration.crank_nicolson_rk2,
    time_integration.crank_nicolson_rk3,
    time_integration.crank_nicolson_rk4,
    time_integration.imex_rk_sil3,
]


@pytest.mark.parametrize('problem', ALL_TEST_PROBLEMS)
def test_implicit_inverse(problem, device):
  """`implicit_inverse` solves (y - eta * G(y)) = x for each test case."""
  eta = 0.3
  initial_state = torch.as_tensor(
      problem['initial_state'], dtype=torch.float64, device=device
  )
  solved = problem['implicit_inverse'](initial_state, eta)
  reconstructed = solved - eta * problem['implicit_terms'](solved)
  np.testing.assert_allclose(
      reconstructed.cpu().numpy(), initial_state.cpu().numpy(), rtol=1e-7
  )


@pytest.mark.parametrize('problem', ALL_TEST_PROBLEMS)
def test_integration(problem, device):
  """Time integration is accurate for a range of test cases."""
  dt = problem['dt']
  inner_steps = problem['inner_steps']
  outer_steps = problem['outer_steps']
  time = dt * inner_steps * (1 + np.arange(outer_steps))
  expected = _stack_over_time(problem['closed_form'],
                              problem['initial_state'], time)

  for atol, time_stepper in zip(problem['tolerances'], ALL_TIME_STEPPERS):
    equation = CustomODE(
        problem['explicit_terms'],
        problem['implicit_terms'],
        problem['implicit_inverse'],
    )
    step = time_stepper(equation, dt)
    input_state = torch.as_tensor(
        problem['initial_state'], dtype=torch.float64, device=device
    )
    trajectory_fn = time_integration.trajectory_from_step(
        step, outer_steps, inner_steps
    )
    _, actual = trajectory_fn(input_state)
    np.testing.assert_allclose(
        expected,
        actual.cpu().numpy(),
        atol=atol,
        rtol=0,
        err_msg=time_stepper.__name__,
    )


@pytest.mark.parametrize('time_stepper', ALL_TIME_STEPPERS)
def test_pytree_state(time_stepper, device):
  equation = CustomODE(
      explicit_terms=lambda x: pytree.tree_map(_zeros_like, x),
      implicit_terms=lambda x: pytree.tree_map(_zeros_like, x),
      implicit_inverse=lambda x, eta: x,
  )
  one = torch.ones((), dtype=torch.float64, device=device)
  u0 = {'x': one, 'y': one}
  u1 = time_stepper(equation, 1.0)(u0)
  assert set(u1) == {'x', 'y'}
  for v in u1.values():
    assert float(v) == 1.0


@pytest.mark.parametrize('use_pytree', [False, True], ids=['array', 'pytree'])
def test_multiple_equations(use_pytree, device):
  tolerances = [1e-4, 2e-5, 2e-6, 1e-6, 2e-5]
  dt = 1e-2
  inner_steps = 20
  outer_steps = 5

  if not use_pytree:
    initial_state = np.arange(3) * np.exp(0)
    closed_form = lambda x0, t: np.arange(3.0) * np.exp(t)
    equation_a = CustomODE(
        explicit_terms=lambda x: 3 * x / 8,
        implicit_terms=lambda x: x / 2,
        implicit_inverse=lambda x, eta: x / (1 - eta / 2),
    )
    equation_b = CustomExplicitODE(explicit_terms=lambda x: x / 8)
    input_state = torch.as_tensor(
        initial_state, dtype=torch.float64, device=device
    )
    select = lambda tree: tree
  else:
    initial_state = {'s': np.arange(3) * np.exp(0)}
    closed_form = lambda x0, t: {'s': np.arange(3.0) * np.exp(t)}
    equation_a = CustomODE(
        explicit_terms=lambda x: {'s': x['s'] / 8},
        implicit_terms=lambda x: {'s': x['s'] / 2},
        implicit_inverse=lambda x, eta: {'s': x['s'] / (1 - eta / 2)},
    )
    equation_b = CustomExplicitODE(
        explicit_terms=lambda x: {'s': 3 * x['s'] / 8}
    )
    input_state = {
        's': torch.as_tensor(
            initial_state['s'], dtype=torch.float64, device=device
        )
    }
    select = lambda tree: tree['s']

  equation = time_integration.compose_equations([equation_a, equation_b])
  time = dt * inner_steps * (1 + np.arange(outer_steps))
  expected = _stack_over_time(closed_form, initial_state, time)

  for atol, time_stepper in zip(tolerances, ALL_TIME_STEPPERS):
    step = time_stepper(equation, dt)
    trajectory_fn = time_integration.trajectory_from_step(
        step, outer_steps, inner_steps
    )
    _, actual = trajectory_fn(input_state)
    np.testing.assert_allclose(
        select(expected),
        select(_to_numpy(actual)),
        atol=atol,
        err_msg=time_stepper.__name__,
    )


def test_accumulate_repeated(device):
  result = time_integration.accumulate_repeated(
      lambda x: 2 * x,
      np.arange(4),
      torch.ones((), dtype=torch.float64, device=device),
  )
  assert float(result) == 0 * 2 + 1 * 4 + 2 * 8 + 3 * 16


def test_dfi_lanczos_weights():
  weights = time_integration._dfi_lanczos_weights(10, 10, 0.1)
  assert weights.size == 50
  assert (np.diff(weights) < 0).all()
  np.testing.assert_almost_equal(weights[0], 1.0, decimal=2)
  np.testing.assert_almost_equal(weights[-1], 0.0, decimal=2)


def test_digital_filter_initialization(device):
  def explicit_terms(x):
    growth = torch.full_like(x[:1], 0.1)
    return torch.cat([growth, x[2:3], -x[1:2]])

  eq = time_integration.ImplicitExplicitODE.from_functions(
      # x[0] is linear growth; x[1], x[2] are a simple harmonic oscillator
      explicit_terms=explicit_terms,
      implicit_terms=_zeros_like,
      implicit_inverse=lambda x, eta: x,
  )
  dfi = time_integration.digital_filter_initialization(
      equation=eq,
      ode_solver=time_integration.imex_rk_sil3,
      filters=[],
      time_span=20.0,
      cutoff_period=20.0,
      dt=0.01,
  )
  result = dfi(
      torch.as_tensor([1.0, 1.0, 1.0], dtype=torch.float64, device=device)
  )
  expected = np.array([1.0, 0.0, 0.0])  # oscillating terms are filtered
  np.testing.assert_allclose(expected, result.cpu().numpy(), atol=1e-3)
