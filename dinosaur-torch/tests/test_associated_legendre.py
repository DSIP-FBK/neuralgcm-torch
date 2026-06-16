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

"""Tests for associated_legendre (ported from the original Dinosaur, pytest style)."""

import numpy as np
import pytest
import scipy.special as sps

from dinosaur_torch import associated_legendre


@pytest.mark.parametrize('n', [4, 11, 32])
def test_orthonormality(n):
  z, w = associated_legendre.gauss_legendre_nodes(n)
  p = associated_legendre.evaluate(n, n, z)
  eye = np.eye(n, dtype=np.float64)
  inner_products = np.einsum('mil,mik,i->mlk', p, p, w)
  for m in range(n):
    np.testing.assert_allclose(eye, inner_products[m], atol=1e-8)
    eye[m, m] = 0


@pytest.mark.parametrize('m,l', [(4, 4), (3, 8), (12, 20)])
def test_against_scipy(m, l):
  x, _ = associated_legendre.gauss_legendre_nodes(l + 1)
  p_lm = associated_legendre.evaluate(m + 1, l + 1, x)[-1, :, -1]
  q_lm = sps.lpmv(m, l, x)
  ratio = q_lm[0] / p_lm[0]
  for p_j, q_j in zip(p_lm, q_lm):
    np.testing.assert_almost_equal(q_j / ratio, p_j)
