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

"""Tests for scales (ported from the original Dinosaur, pytest style)."""

import numpy as np
import pytest
import torch

from dinosaur_torch import scales

units = scales.units


@pytest.mark.parametrize('get_x,scalez', [
    (lambda: 24 * units.m / units.s,
     (10 * units.m, 1 * units.hour)),
    (lambda: 9.8 * units.m / units.s ** 2,
     (123 * units.mile, 1 * units.year, 32 * units.degK)),
    (lambda: np.arange(20) * units.J,
     (6 * units.mile, np.pi * units.week, 32 * units.kilogram)),
    (lambda: torch.arange(77) * units.J / units.m ** 2,
     (11 * units.angstrom, np.pi * units.fortnight, 1 * units.UK_ton)),
])
def test_round_trip(get_x, scalez):
  x = get_x()
  scale = scales.Scale(*scalez)
  y = scale.nondimensionalize(x)
  assert not isinstance(y, scales.Quantity)
  z = scale.dimensionalize(y, x.units)
  np.testing.assert_allclose(
      np.asarray(x.magnitude), np.asarray(z.magnitude), rtol=1e-6
  )
  assert x.units == z.units


@pytest.mark.parametrize('get_x,scalez', [
    (lambda: 24 * units.m / units.s, (1 * units.hour,)),
    (lambda: 9.8 * units.m / units.s ** 2,
     (123 * units.mile, 32 * units.degK)),
    (lambda: np.arange(20) * units.J,
     (6 * units.mile, 32 * units.kilogram)),
    (lambda: torch.arange(77) * units.J / units.m ** 2, (1 * units.UK_ton,)),
])
def test_unspecified_scale(get_x, scalez):
  x = get_x()
  scale = scales.Scale(*scalez)
  with pytest.raises(ValueError, match='No scale has been set'):
    scale.nondimensionalize(x)


@pytest.mark.parametrize('scalez', [
    (1 * units.J,),
    (123 * units.mile, 1 / units.year, 32 * units.degK),
    (6 * units.mile, np.pi * units.week * units.newton, 32 * units.kilogram),
    (11 * units.KPH, np.pi * units.fortnight, 1 * units.UK_ton),
])
def test_illegal_compound_dimension(scalez):
  with pytest.raises(ValueError, match='All scales must describe a single dimension'):
    scales.Scale(*scalez)


@pytest.mark.parametrize('scalez', [
    (1 * units.m, 10 * units.parsec),
    (123 * units.mile, 1 * units.year, 32 * units.year),
    (6 * units.mile, np.pi * units.lb, 32 * units.kilogram),
    (11 * units.hour, np.pi * units.fortnight, 1 * units.UK_ton),
])
def test_duplicate_scale(scalez):
  with pytest.raises(ValueError, match='Got duplicate scales for dimension'):
    scales.Scale(*scalez)


@pytest.mark.parametrize('scale,quantity', [
    (scales.Scale(1 * units.m), 1 * units.m),
    (scales.Scale(17 * units.m), 11 * units.mm),
    (scales.Scale(33 * units.m, 11 * units.year),
     np.pi * units.m ** 2 / units.s ** 2),
    (scales.Scale(123 * units.kg, 345 * units.m, 456 * units.year),
     5 * units.pascal),
])
def test_round_trip_non_standard(scale, quantity):
  nondim = scale.nondimensionalize(quantity)
  reconstructed = scale.dimensionalize(nondim, quantity.units)
  assert quantity.units == reconstructed.units
  np.testing.assert_allclose(quantity.m, reconstructed.m)
