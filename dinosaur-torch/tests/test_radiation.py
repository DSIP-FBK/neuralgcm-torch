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

"""Tests for radiation (ported from the original Dinosaur, pytest style)."""

import datetime

import numpy as np
import pytest
import torch

from dinosaur_torch import radiation
from dinosaur_torch import scales
from dinosaur_torch import spherical_harmonic
from dinosaur_torch import units as units_lib

TWOPI = 2 * np.pi
units = scales.units


def _to_numpy(x):
  if isinstance(x, torch.Tensor):
    return x.detach().cpu().numpy()
  return np.asarray(x)


def _get_expected_value_modulo_2pi(expected, actual):
  actual = _to_numpy(actual)
  expected = np.fmod(expected, TWOPI)
  expected = np.where(expected < actual - np.pi, expected + TWOPI, expected)
  expected = np.where(expected > actual + np.pi, expected - TWOPI, expected)
  return expected


# ---------------------------------------------------------------------------
# datetime helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('when,expected', [
    (np.datetime64('1980-01-01T00:00:00.000000000'),
     datetime.datetime(1980, 1, 1, 0, 0)),
    (np.datetime64('2001-02-03T04:05'),
     datetime.datetime(2001, 2, 3, 4, 5)),
    (np.datetime64('1982-05-04'),
     datetime.datetime(1982, 5, 4, 0, 0)),
])
def test_datetime64_to_datetime(when, expected):
  actual = radiation.datetime64_to_datetime(when)
  assert actual == expected


@pytest.mark.parametrize('when,expected_days', [
    (datetime.datetime(1984, 1, 1), 366),
    (datetime.datetime(1985, 1, 1), 365),
    (datetime.datetime(2020, 1, 1), 366),
    (datetime.datetime(2001, 1, 1), 365),
])
def test_days_in_year(when, expected_days):
  assert radiation.days_in_year(when) == expected_days


@pytest.mark.parametrize('when,exp_orbital_phase,exp_synodic_phase', [
    # 1980 is a leap year
    (datetime.datetime(1980, 1, 1, 0, 0), 0, 0),
    (datetime.datetime(1980, 1, 1, 12, 0),
     0.5 * TWOPI / 366, np.pi),
    (datetime.datetime(1980, 1, 1, 23, 59),
     (1439 / 1440) * TWOPI / 366, 1439 * TWOPI / 1440),
    (datetime.datetime(1980, 5, 1, 0, 0),
     (31 + 29 + 31 + 30) * TWOPI / 366, 0),
    (datetime.datetime(1980, 12, 31, 23, 59),
     TWOPI - (1 / 1440) * TWOPI / 366, 1439 * TWOPI / 1440),
    # 1981 is not a leap year
    (datetime.datetime(1981, 1, 1, 0, 0), 0, 0),
    (datetime.datetime(1981, 1, 1, 12, 0),
     0.5 * TWOPI / 365, np.pi),
    (datetime.datetime(1981, 1, 1, 23, 59),
     (1439 / 1440) * TWOPI / 365, 1439 * TWOPI / 1440),
    (datetime.datetime(1981, 5, 1, 0, 0),
     (31 + 28 + 31 + 30) * TWOPI / 365, 0),
    (datetime.datetime(1981, 12, 31, 23, 59),
     TWOPI - (1 / 1440) * TWOPI / 365, 1439 * TWOPI / 1440),
    # distant future
    (datetime.datetime(2022, 5, 1, 0, 0),
     (31 + 28 + 31 + 30) * TWOPI / 365, 0),
    (datetime.datetime(2045, 12, 31, 23, 59),
     TWOPI - (1 / 1440) * TWOPI / 365, 1439 * TWOPI / 1440),
])
def test_datetime_to_orbital_time(when, exp_orbital_phase, exp_synodic_phase):
  actual = radiation.datetime_to_orbital_time(when)
  assert abs(actual.orbital_phase - exp_orbital_phase) < 1e-12
  assert abs(actual.synodic_phase - exp_synodic_phase) < 1e-12


def test_get_direct_solar_irradiance_no_units():
  flux = radiation.get_direct_solar_irradiance(
      orbital_phase=torch.tensor(
          np.linspace(0, TWOPI, 4, endpoint=False), dtype=torch.float64
      ),
      mean_irradiance=1.2,
      variation=0.3,
      perihelion=0,
  )
  np.testing.assert_allclose(_to_numpy(flux), [1.5, 1.2, 0.9, 1.2], atol=1e-7)


# Days of the year when equation of time is nearly zero
@pytest.mark.parametrize('when', [
    datetime.datetime(2000, 4, 16),
    datetime.datetime(2000, 6, 14),
    datetime.datetime(2000, 8, 31),
    datetime.datetime(2000, 12, 25),
])
def test_equation_of_time(when):
  ot = radiation.datetime_to_orbital_time(when)
  delta_phase = radiation.equation_of_time(
      torch.tensor(ot.orbital_phase, dtype=torch.float64)
  )
  np.testing.assert_allclose(_to_numpy(delta_phase), 0, atol=0.005)


# ---------------------------------------------------------------------------
# SolarRadiation module
# ---------------------------------------------------------------------------

_SPEC = spherical_harmonic.GridSpec.T85()
_PHYSICS = units_lib.SimUnits.from_si()
_REF_DT = radiation.WB_REFERENCE_DATETIME


def test_radiation_shapes():
  sr = radiation.SolarRadiation(_SPEC, _PHYSICS, _REF_DT)
  tisr = sr.radiation_flux(time=0)
  assert tuple(tisr.shape) == _SPEC.nodal_shape
  sha = sr.solar_hour_angle(time=0)
  assert tuple(sha.shape) == _SPEC.nodal_shape


def test_radiation_flux_nondimensionalization():
  sr = radiation.SolarRadiation(_SPEC, _PHYSICS, _REF_DT)
  perihelion_datetime = datetime.datetime(1979, 1, 3)
  time = sr.datetime_to_time(perihelion_datetime)
  actual_max = float(np.max(_to_numpy(sr.radiation_flux(time))))
  expected_max = float(_PHYSICS.nondimensionalize(
      radiation.TOTAL_SOLAR_IRRADIANCE + radiation.SOLAR_IRRADIANCE_VARIATION
  ))
  np.testing.assert_allclose(actual_max, expected_max, rtol=5e-5)


@pytest.mark.parametrize('reference_datetime,days_until_aphelion', [
    (radiation.WB_REFERENCE_DATETIME, 3 + radiation.DAYS_PER_YEAR / 2),
    (radiation.WB_REFERENCE_DATETIME, 3 - radiation.DAYS_PER_YEAR / 2),
    (datetime.datetime(1979, 7, 4), 0),
    (datetime.datetime(1980, 7, 3), 0),
    (datetime.datetime(2022, 5, 4), 61),
])
def test_reference_datetime(reference_datetime, days_until_aphelion):
  sr = radiation.SolarRadiation(_SPEC, _PHYSICS, reference_datetime)
  time = _PHYSICS.nondimensionalize(days_until_aphelion * units.day)
  actual_max = float(np.max(_to_numpy(sr.radiation_flux(time))))
  expected_max = float(_PHYSICS.nondimensionalize(
      radiation.TOTAL_SOLAR_IRRADIANCE - radiation.SOLAR_IRRADIANCE_VARIATION
  ))
  np.testing.assert_allclose(actual_max, expected_max, rtol=5e-5)


def test_normalized_radiation_flux():
  sr = radiation.SolarRadiation.normalized(_SPEC, _PHYSICS, _REF_DT)
  perihelion_datetime = datetime.datetime(1979, 1, 3)
  time = sr.datetime_to_time(perihelion_datetime)
  tisr = sr.radiation_flux(time=time)
  np.testing.assert_allclose(float(np.max(_to_numpy(tisr))), 1.0, rtol=5e-5)


@pytest.mark.parametrize('when,expected_time_days', [
    (datetime.datetime(1979, 1, 1, 0, 0), 0.),
    (datetime.datetime(1979, 1, 1, 0, 1), 1 / (60 * 24)),
    (datetime.datetime(1979, 1, 1, 12, 0), 0.5),
    (datetime.datetime(1979, 1, 3, 0, 0), 2.),
    (datetime.datetime(1980, 1, 1, 0, 0), 365.),
    (datetime.datetime(1981, 1, 1, 0, 0), 731.),
    (datetime.datetime(2020, 1, 1, 0, 0), 14975.),
    (datetime.datetime(2022, 5, 4, 4, 20), 15829.180555555555),
])
def test_datetime_to_time(when, expected_time_days):
  sr = radiation.SolarRadiation(_SPEC, _PHYSICS, _REF_DT)
  actual_time = sr.datetime_to_time(when)
  expected_time = _PHYSICS.nondimensionalize(expected_time_days * units.day)
  assert abs(actual_time - expected_time) < 1e-9


@pytest.mark.parametrize('when,expected_orbital_phase,expected_synodic_phase', [
    (radiation.WB_REFERENCE_DATETIME, 0, 0),
    (radiation.WB_REFERENCE_DATETIME + datetime.timedelta(days=365.25),
     TWOPI, 365.25 * TWOPI),
    (radiation.WB_REFERENCE_DATETIME + datetime.timedelta(days=-365.25),
     -TWOPI, -365.25 * TWOPI),
    (datetime.datetime(2019, 1, 1, 0, 0), 40 * TWOPI, 40 * 365.25 * TWOPI),
])
def test_time_to_orbital_time(when, expected_orbital_phase,
                              expected_synodic_phase):
  sr = radiation.SolarRadiation(_SPEC, _PHYSICS, _REF_DT)
  time = sr.datetime_to_time(when)
  actual = sr.time_to_orbital_time(time)

  exp_op = _get_expected_value_modulo_2pi(expected_orbital_phase,
                                          actual.orbital_phase)
  assert abs(actual.orbital_phase - exp_op) < 1e-9

  exp_sp = _get_expected_value_modulo_2pi(expected_synodic_phase,
                                          actual.synodic_phase)
  assert abs(actual.synodic_phase - exp_sp) < 1e-9


def test_solar_hour_angle():
  sr = radiation.SolarRadiation(_SPEC, _PHYSICS, _REF_DT)
  time = sr.datetime_to_time(
      datetime.datetime(1979, 4, 15, 0, 0)  # equation of time ≈ 0
  )
  actual = sr.solar_hour_angle(time)
  lon, _ = _SPEC.nodal_mesh
  expected = _get_expected_value_modulo_2pi(lon - np.pi, actual)
  np.testing.assert_allclose(_to_numpy(actual), expected, atol=1e-4)


def test_nondim_minutes_day_year_constants():
  days_per_year = _PHYSICS.nondimensionalize(units.year / units.day)
  np.testing.assert_almost_equal(days_per_year, radiation.DAYS_PER_YEAR)
  minutes_per_day = _PHYSICS.nondimensionalize(units.day / units.minute)
  np.testing.assert_almost_equal(minutes_per_day, radiation.MINUTES_PER_DAY)
