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

"""Tests for filtering (ported from the original Dinosaur, pytest style)."""

import numpy as np
import pytest
import torch

from dinosaur_torch import filtering
from dinosaur_torch import spherical_harmonic


def _to_numpy(x):
  if isinstance(x, torch.Tensor):
    return x.detach().cpu().numpy()
  return np.asarray(x)


def test_make_filter_fn():
  # filter should only affect 3-D tensors whose shape broadcasts with (2, 1, 3)
  f = filtering._make_filter_fn(torch.zeros(2, 1, 3))
  x = {
      'a': 1.0,
      'b': torch.ones(1, 2, 3),
      'c': torch.ones(2, 2, 3),
  }
  y = f(x)
  assert y['a'] == 1.0  # scalar passes through
  np.testing.assert_array_equal(_to_numpy(y['b']), _to_numpy(x['b']))
  np.testing.assert_array_equal(_to_numpy(y['c']), np.zeros((2, 2, 3)))


def test_exponential_filter(device):
  grid = spherical_harmonic.Grid(spherical_harmonic.GridSpec.TL127(),
                                 device=device)
  inputs = torch.ones(grid.modal_shape, device=device)
  scaling = _to_numpy(filtering.exponential_filter(grid, attenuation=16)(inputs))
  assert abs(scaling[0, 0] - 1.0) < 1e-9
  assert abs(scaling[0, -1] - 1.125e-7) < 1e-9


@pytest.mark.parametrize('order', [1, 2, 3])
def test_horizontal_diffusion_filter(order, device):
  grid = spherical_harmonic.Grid(spherical_harmonic.GridSpec.TL127(),
                                 device=device)
  inputs = torch.ones((3,) + grid.modal_shape, device=device)
  timescale = torch.tensor([1.0, 2.0, 3.0], device=device)[:, None, None]
  eigenvalues = grid.laplacian_eigenvalues
  scale = 0.1 / (timescale * abs(eigenvalues[-1]) ** order)
  scaling = _to_numpy(
      filtering.horizontal_diffusion_filter(grid, scale, order)(inputs)
  )
  assert abs(scaling[0, 0, 0] - 1.0) < 1e-9
  assert abs(scaling[1, 0, 0] - 1.0) < 1e-9
  assert abs(scaling[2, 0, 0] - 1.0) < 1e-9
  assert abs(scaling[0, 0, -1] - np.exp(-0.1)) < 1e-6
  assert abs(scaling[1, 0, -1] - np.exp(-0.1 / 2)) < 1e-6
  assert abs(scaling[2, 0, -1] - np.exp(-0.1 / 3)) < 1e-6


@pytest.mark.parametrize('filter_fn,filter_kwargs', [
    (filtering.exponential_filter,
     {'attenuation': torch.tensor([1.0, 2.0]).reshape(2, 1, 1, 1)}),
    (filtering.horizontal_diffusion_filter,
     {'scale': torch.tensor([1.0, 2.0]).reshape(2, 1, 1, 1)}),
])
def test_time_filter_variation(filter_fn, filter_kwargs, device):
  """Filter args given as arrays produce the same result as independent calls."""
  filter_kwargs = {k: v.to(device) for k, v in filter_kwargs.items()}
  (time_slices,) = {v.shape[0] for v in filter_kwargs.values()}
  grid = spherical_harmonic.Grid(spherical_harmonic.GridSpec.TL63(),
                                 device=device)
  inputs = torch.ones((time_slices, 3) + grid.modal_shape, device=device)
  out = filter_fn(grid, **filter_kwargs)(inputs)
  for i in range(time_slices):
    kwargs_i = {k: v[i].item() for k, v in filter_kwargs.items()}
    expected_i = _to_numpy(filter_fn(grid, **kwargs_i)(inputs[i]))
    np.testing.assert_allclose(expected_i, _to_numpy(out[i]), rtol=1e-6)
