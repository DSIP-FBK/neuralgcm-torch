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

"""Tests for fourier (ported from the original Dinosaur, pytest style)."""

import numpy as np
import pytest
import torch

from dinosaur_torch import fourier


@pytest.mark.parametrize('wavenumbers,nodes', [(4, 7), (11, 11), (32, 63)])
def test_basis(wavenumbers, nodes):
  f = fourier.real_basis(wavenumbers, nodes)
  node_pts = np.linspace(0, 2 * np.pi, nodes, endpoint=False)
  np.testing.assert_allclose(f[:, 0], 1 / np.sqrt(2 * np.pi))
  for j in range(1, wavenumbers):
    np.testing.assert_allclose(
        f[:, 2 * j - 1], np.cos(j * node_pts) / np.sqrt(np.pi), atol=1e-12
    )
    np.testing.assert_allclose(
        f[:, 2 * j], np.sin(j * node_pts) / np.sqrt(np.pi), atol=1e-12
    )


@pytest.mark.parametrize('wavenumbers,seed', [(4, 0), (11, 0), (32, 0)])
def test_derivatives(wavenumbers, seed):
  f = np.random.RandomState(seed).normal(size=[2 * wavenumbers - 1])
  f_t = torch.as_tensor(f, dtype=torch.float64)
  f_x = fourier.real_basis_derivative(f_t).numpy()
  np.testing.assert_allclose(f_x[0], 0)
  for j in range(1, wavenumbers):
    np.testing.assert_allclose(f_x[2 * j - 1], j * f[2 * j], atol=1e-12)
    np.testing.assert_allclose(f_x[2 * j], -j * f[2 * j - 1], atol=1e-12)


@pytest.mark.parametrize('wavenumbers', [4, 16, 256])
def test_normalized(wavenumbers):
  nodes = 2 * wavenumbers - 1
  f = fourier.real_basis(wavenumbers, nodes)
  _, w = fourier.quadrature_nodes(nodes)
  eye = np.eye(2 * wavenumbers - 1)
  np.testing.assert_allclose((f.T * w).dot(f), eye, atol=1e-12)
