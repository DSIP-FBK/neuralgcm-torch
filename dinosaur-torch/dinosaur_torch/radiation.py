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

"""Top of atmosphere incident solar radiation.

This code was inspired by pysolar, however that library is intended as a
surface-of-Earth solar radiation model that includes atmospheric scattering
effects; the constants used to compute solar irradiance differ. Time is
represented in radians, which greatly reduces conversions by factors of pi.
"""

from __future__ import annotations

import dataclasses
import datetime
from typing import Union

import numpy as np
import torch
from torch import nn

from dinosaur_torch import scales
from dinosaur_torch import spherical_harmonic
from dinosaur_torch import units as units_lib

units = scales.units

Numeric = Union[float, torch.Tensor]

DAYS_PER_YEAR = 365.25
MINUTES_PER_DAY = 1440
SECONDS_PER_DAY = 86400
# TSI: Energy input to the top of the Earth's atmosphere
# https://www.ncei.noaa.gov/products/climate-data-records/total-solar-irradiance
TOTAL_SOLAR_IRRADIANCE = 1361 * units.W / units.meter**2
# Seasonal variation in apparent solar irradiance due to Earth-Sun distance
SOLAR_IRRADIANCE_VARIATION = 47 * units.W / units.meter**2  # .5 * 6.9% * TSI
# Approximate perihelion, when Earth is closest to the sun (Jan 3rd)
PERIHELION = 3 * 2 * np.pi / DAYS_PER_YEAR  # radians
# Approximate equinox (March 20 on non-leap year), when dihedral is zero
SPRING_EQUINOX = 79 * 2 * np.pi / DAYS_PER_YEAR  # radians
# Angle between Earth's rotational axis and its orbital axis
EARTH_AXIS_INCLINATION = 23.45 * np.pi / 180  # radians

# Reference datetime for WeatherBench
WB_REFERENCE_DATETIME = datetime.datetime(1979, 1, 1, 0, 0)


@dataclasses.dataclass(frozen=True)
class OrbitalTime:
  """Nondimensional time based on orbital dynamics.

  Attributes:
    orbital_phase: phase of the Earth's orbit around the Sun in radians. The
      values 0, 2pi correspond to January 1st, midnight UTC.
    synodic_phase: phase of the Earth's rotation around its axis in radians,
      relative to the Sun. The values 0, 2pi correspond to midnight UTC.
  """

  orbital_phase: Numeric
  synodic_phase: Numeric


def datetime64_to_datetime(when: np.datetime64) -> datetime.datetime:
  """Returns datetime corresponding to the provided numpy datetime64."""
  ts = (when - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
  return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).replace(tzinfo=None)


def days_in_year(when: datetime.datetime) -> int:
  """Returns the number of days in the year of the provided datetime."""
  return (
      datetime.datetime(year=when.year, month=12, day=31).timetuple().tm_yday
  )


def datetime_to_orbital_time(when: datetime.datetime) -> OrbitalTime:
  """Returns the OrbitalTime associated with the provided datetime."""
  days_this_year = days_in_year(when)
  full_days = when.timetuple().tm_yday - 1
  fraction_of_day = (60 * when.hour + when.minute) / MINUTES_PER_DAY
  fraction_of_year = (full_days + fraction_of_day) / days_this_year
  return OrbitalTime(
      orbital_phase=2 * np.pi * fraction_of_year,
      synodic_phase=2 * np.pi * fraction_of_day,
  )


def datetime_to_time(
    when: datetime.datetime | np.datetime64,
    physics_specs: units_lib.SimUnits,
    reference_datetime: datetime.datetime | np.datetime64,
) -> float:
  """Returns nondimensional time corresponding to the specified datetime.

  Args:
    when: datetime for which to compute nondimensional time.
    physics_specs: object holding physical constants and definition of
      custom units to use for initialization of the state.
    reference_datetime: datetime corresponding to nondimensionalized time=0.
  """
  if isinstance(when, np.datetime64):
    when = datetime64_to_datetime(when)
  if isinstance(reference_datetime, np.datetime64):
    reference_datetime = datetime64_to_datetime(reference_datetime)
  difference = when - reference_datetime
  days = difference.days + difference.seconds / SECONDS_PER_DAY
  return physics_specs.nondimensionalize(days * units.day)


def get_direct_solar_irradiance(
    orbital_phase: torch.Tensor,
    mean_irradiance: Numeric,
    variation: Numeric,
    perihelion: float = PERIHELION,
) -> torch.Tensor:
  """Returns solar radiation flux incident on the top of the atmosphere.

  Formula includes 6.9% seasonal variation due to Earth's elliptical orbit,
  but neglects the 0.1% variation of the 11-year solar cycle (Schwabe
  cycle).
  https://earth.gsfc.nasa.gov/climate/research/solar-radiation
  https://en.wikipedia.org/wiki/Solar_constant
  """
  return mean_irradiance + variation * torch.cos(orbital_phase - perihelion)


def get_declination(orbital_phase: torch.Tensor) -> torch.Tensor:
  """Angle between the Earth-Sun line and the Earth equatorial plane."""
  # https://en.wikipedia.org/wiki/Declination
  return EARTH_AXIS_INCLINATION * torch.sin(orbital_phase - SPRING_EQUINOX)


def equation_of_time(orbital_phase: torch.Tensor) -> torch.Tensor:
  """Returns the value to add to mean solar time to get actual solar time."""
  # https://en.wikipedia.org/wiki/Equation_of_time
  b = orbital_phase - SPRING_EQUINOX
  added_minutes = (
      9.87 * torch.sin(2 * b) - 7.53 * torch.cos(b) - 1.5 * torch.sin(b)
  )
  # Output normalized as a correction to synodic_phase
  return 2 * np.pi * added_minutes / MINUTES_PER_DAY


def get_hour_angle(
    orbital_phase: torch.Tensor,
    synodic_phase: torch.Tensor,
    longitude: torch.Tensor,
) -> torch.Tensor:
  """Angular displacement of the sun east or west of the local meridian."""
  # https://en.wikipedia.org/wiki/Hour_angle
  eot = equation_of_time(orbital_phase)
  solar_time = synodic_phase + eot + longitude
  return solar_time - np.pi


def get_solar_sin_altitude(
    orbital_phase: torch.Tensor,
    synodic_phase: torch.Tensor,
    longitude: torch.Tensor,
    latitude: torch.Tensor,
) -> torch.Tensor:
  """Returns sine of the solar altitude angle."""
  # https://en.wikipedia.org/wiki/Solar_zenith_angle
  declination = get_declination(orbital_phase)
  hour_angle = get_hour_angle(orbital_phase, synodic_phase, longitude)
  first_term = (
      torch.cos(latitude) * torch.cos(declination) * torch.cos(hour_angle)
  )
  second_term = torch.sin(latitude) * torch.sin(declination)
  return first_term + second_term


def get_radiation_flux(
    orbital_time: OrbitalTime,
    longitude: torch.Tensor,
    latitude: torch.Tensor,
    mean_irradiance: Numeric,
    variation: Numeric,
) -> torch.Tensor:
  """Returns TOA incident radiation flux."""
  sin_altitude = get_solar_sin_altitude(
      orbital_phase=orbital_time.orbital_phase,
      synodic_phase=orbital_time.synodic_phase,
      longitude=longitude,
      latitude=latitude,
  )
  is_daytime = sin_altitude > 0
  flux = get_direct_solar_irradiance(
      orbital_phase=orbital_time.orbital_phase,
      mean_irradiance=mean_irradiance,
      variation=variation,
  )
  return flux * is_daytime * sin_altitude


class SolarRadiation(nn.Module):
  """Top of atmosphere incident solar radiation (TISR).

  Holds the nodal lon/lat mesh as buffers; `radiation_flux(time)` maps a
  nondimensional time (float or 0-d tensor) to the nondimensionalized flux
  on the grid.
  """

  def __init__(
      self,
      grid_spec: spherical_harmonic.GridSpec,
      physics_specs: units_lib.SimUnits,
      reference_datetime: datetime.datetime | np.datetime64,
      *,
      device: torch.device | str | None = None,
      dtype: torch.dtype = torch.float32,
      normalized: bool = False,
  ):
    """Initialize SolarRadiation.

    Args:
      grid_spec: horizontal grid description.
      physics_specs: object holding physical constants and definition of
        custom units to use for initialization of the state.
      reference_datetime: datetime corresponding to nondimensional time = 0.
      device: device for the lon/lat buffers.
      dtype: dtype for the lon/lat buffers.
      normalized: if True, fluxes are normalized to the range [0, 1].
    """
    super().__init__()
    if isinstance(reference_datetime, np.datetime64):
      reference_datetime = datetime64_to_datetime(reference_datetime)
    self.reference_datetime = reference_datetime
    self.reference_orbital_time = datetime_to_orbital_time(reference_datetime)
    self.physics_specs = physics_specs

    lon, sin_lat = grid_spec.nodal_mesh
    self.register_buffer(
        'lon', torch.as_tensor(lon, dtype=dtype, device=device),
        persistent=False,
    )
    self.register_buffer(
        'lat',
        torch.as_tensor(np.arcsin(sin_lat), dtype=dtype, device=device),
        persistent=False,
    )

    self.orbital_rate = OrbitalTime(
        orbital_phase=float(
            physics_specs.nondimensionalize(2 * np.pi / units.year)
        ),
        synodic_phase=float(
            physics_specs.nondimensionalize(2 * np.pi / units.day)
        ),
    )
    self.total_solar_irradiance = float(
        physics_specs.nondimensionalize(TOTAL_SOLAR_IRRADIANCE)
    )
    self.solar_irradiance_variation = float(
        physics_specs.nondimensionalize(SOLAR_IRRADIANCE_VARIATION)
    )
    if normalized:
      scale = self.total_solar_irradiance + self.solar_irradiance_variation
      self.total_solar_irradiance /= scale
      self.solar_irradiance_variation /= scale

  @classmethod
  def normalized(cls, *args, **kwargs) -> SolarRadiation:
    """Initialize SolarRadiation for normalized solar radiation."""
    return cls(*args, normalized=True, **kwargs)

  def _as_tensor(self, time: Numeric) -> torch.Tensor:
    if not isinstance(time, torch.Tensor):
      time = torch.as_tensor(
          time, dtype=self.lon.dtype, device=self.lon.device
      )
    return time

  def datetime_to_time(self, when: datetime.datetime | np.datetime64):
    """Returns nondimensional time corresponding to the specified datetime."""
    return datetime_to_time(when, self.physics_specs, self.reference_datetime)

  def time_to_orbital_time(self, time: Numeric) -> OrbitalTime:
    """Returns the OrbitalTime corresponding to the given nondim time.

    The phase arithmetic stays in the input's type (python float64 for
    floats, tensor dtype for tensors) so that wrapping large phases does
    not lose precision unnecessarily.
    """

    def phase(reference, rate):
      p = reference + rate * time
      # Reduce the magnitude of the result to avoid loss of precision
      # downstream (torch.fmod is not very precise on float32).
      return p - p // (2 * np.pi) * (2 * np.pi)

    return OrbitalTime(
        orbital_phase=phase(
            self.reference_orbital_time.orbital_phase,
            self.orbital_rate.orbital_phase,
        ),
        synodic_phase=phase(
            self.reference_orbital_time.synodic_phase,
            self.orbital_rate.synodic_phase,
        ),
    )

  def solar_hour_angle(self, time: Numeric) -> torch.Tensor:
    """Returns solar hour angle in radians."""
    now = self.time_to_orbital_time(time)
    return get_hour_angle(
        orbital_phase=self._as_tensor(now.orbital_phase),
        synodic_phase=self._as_tensor(now.synodic_phase),
        longitude=self.lon,
    )

  def radiation_flux(self, time: Numeric) -> torch.Tensor:
    """Returns non-dimensionalized TOA incident solar radiation flux."""
    now = self.time_to_orbital_time(time)
    return get_radiation_flux(
        OrbitalTime(
            orbital_phase=self._as_tensor(now.orbital_phase),
            synodic_phase=self._as_tensor(now.synodic_phase),
        ),
        self.lon,
        self.lat,
        mean_irradiance=self.total_solar_irradiance,
        variation=self.solar_irradiance_variation,
    )

  forward = radiation_flux
