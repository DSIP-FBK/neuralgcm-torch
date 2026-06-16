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

"""Real Fourier basis evaluation, and derivative coefficients.

Basis construction returns float64 NumPy arrays (setup time); the derivative
operators act on tensors (run time).
"""
from __future__ import annotations

import numpy as np
import scipy.linalg
import torch


def shift(x: torch.Tensor, offset: int, dim: int) -> torch.Tensor:
  """Like `torch.roll`, but zero-padding instead of wrapping around."""
  n = x.shape[dim]
  if abs(offset) >= n:
    return torch.zeros_like(x)
  zeros = torch.zeros_like(x.narrow(dim, 0, abs(offset)))
  if offset > 0:
    return torch.cat([zeros, x.narrow(dim, 0, n - offset)], dim=dim)
  else:
    return torch.cat([x.narrow(dim, -offset, n + offset), zeros], dim=dim)


def real_basis(wavenumbers: int, nodes: int) -> np.ndarray:
  """Returns the real-valued Fourier basis.

  Args:
    wavenumbers: number of wavenumbers.
    nodes: number of equally spaced nodes in the range [0, 2π). Must satisfy
      wavenumbers >= nodes.

  Returns:
    The nodes ✕ (2 * wavenumbers - 1) matrix F, such that

      F[i, 0] = 1 / √2π
      F[i, 2j - 1] = cos(j xᵢ) / √π    1 ≤ j < wavenumbers
      F[i, 2j]     = sin(j xᵢ) / √π    1 ≤ j < wavenumbers

    where x is a vector of m equally-spaced nodes so xᵢ = i * 2π / nodes.
    i.e., the columns of F are the real Fourier basis functions sampled at x.

    The normalization of the basis functions is chosen such that they have
    unit L²([0, 2π]) norm.
  """
  if nodes < wavenumbers:
    raise ValueError(
        '`real_basis` requires nodes >= wavenumbers; '
        f'got m = {nodes} and n = {wavenumbers}.'
    )

  dft = scipy.linalg.dft(nodes)[:, :wavenumbers] / np.sqrt(np.pi)

  cos = np.real(dft[:, 1:])
  sin = -np.imag(dft[:, 1:])

  f = np.empty(shape=[nodes, 2 * wavenumbers - 1], dtype=np.float64)
  f[:, 0] = 1 / np.sqrt(2 * np.pi)
  f[:, 1::2] = cos
  f[:, 2::2] = sin
  return f


def real_basis_derivative(u: torch.Tensor, /, dim: int = -1) -> torch.Tensor:
  """Calculate the derivative of a signal in the real Fourier basis.

  Args:
    u: signal to differentiate, in the real Fourier basis.
    dim: the (negative) dimension along which the basis is laid out.

  Returns:
    The derivative of `u` along `dim`. In particular, if
    `u_x = real_basis_derivative(u)`:

      u_x[..., 0]      = 0
      u_x[..., 2j - 1] =  j * u[..., 2j]      1 ≤ j < wavenumbers
      u_x[..., 2j]     = -j * u[..., 2j - 1]  1 ≤ j < wavenumbers
  """
  if u.shape[dim] % 2 != 1:
    raise ValueError(f'{u.shape=} along {dim=} is not odd')
  if dim >= 0:
    raise ValueError('dim must be negative')
  i = torch.arange(u.shape[dim], device=u.device).reshape(
      (-1,) + (1,) * (-1 - dim)
  )
  j = (i + 1) // 2
  u_down = shift(u, -1, dim)
  u_up = shift(u, +1, dim)
  return j * torch.where((i % 2).bool(), u_down, -u_up)


def real_basis_with_zero_imag(wavenumbers: int, nodes: int) -> np.ndarray:
  """Real basis with a zero imaginary part."""
  if nodes < wavenumbers:
    raise ValueError(
        '`real_basis` requires nodes >= wavenumbers; '
        f'got m = {nodes} and n = {wavenumbers}.'
    )

  dft = scipy.linalg.dft(nodes)[:, :wavenumbers] / np.sqrt(np.pi)

  cos = np.real(dft[:, 1:])
  sin = -np.imag(dft[:, 1:])

  f = np.empty(shape=[nodes, 2 * wavenumbers], dtype=np.float64)
  f[:, 0] = 1 / np.sqrt(2 * np.pi)
  f[:, 1] = 0
  f[:, 2::2] = cos
  f[:, 3::2] = sin
  return f


def real_basis_derivative_with_zero_imag(
    u: torch.Tensor, /, dim: int = -1
) -> torch.Tensor:
  """Calculate the derivative along a real basis with zero imaginary part."""
  if u.shape[dim] % 2:
    raise ValueError(f'{u.shape=} along {dim=} is not even')
  if dim >= 0:
    raise ValueError('dim must be negative')
  i = torch.arange(u.shape[dim], device=u.device).reshape(
      (-1,) + (1,) * (-1 - dim)
  )
  j = i // 2
  u_down = shift(u, -1, dim)
  u_up = shift(u, +1, dim)
  return j * torch.where(((i + 1) % 2).bool(), u_down, -u_up)


def quadrature_nodes(nodes: int) -> tuple[np.ndarray, np.ndarray]:
  """Returns nodes and weights for the trapezoidal rule.

  Args:
    nodes: the number of nodes and weights to return.

  Returns:
    A pair (nodes, weights); values and weights to use for quadrature, with
    dtype `np.float64`.
  """
  xs = np.linspace(0, 2 * np.pi, nodes, endpoint=False)
  weights = 2 * np.pi / nodes
  return xs, weights
