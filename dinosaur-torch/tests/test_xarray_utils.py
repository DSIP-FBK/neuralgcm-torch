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

"""Tests for xarray dataset preparation utilities."""

import numpy as np
import pandas as pd
import pytest
import xarray

from dinosaur_torch import horizontal_interpolation
from dinosaur_torch import spherical_harmonic
from dinosaur_torch import xarray_utils


def era5_style_dataset(n_lat=33, n_lon=64, n_time=4):
  lat = np.linspace(-90, 90, n_lat)
  lon = np.arange(n_lon) * (360 / n_lon)
  time = pd.date_range('2020-01-01', periods=n_time, freq='6h')
  rs = np.random.RandomState(0)
  return xarray.Dataset(
      {
          'temperature': (
              ('time', 'latitude', 'longitude'),
              250 + 30 * rs.rand(n_time, n_lat, n_lon),
          ),
          'sea_surface_temperature': (
              ('time', 'latitude', 'longitude'),
              280 + 10 * rs.rand(n_time, n_lat, n_lon),
          ),
      },
      coords={'time': time, 'latitude': lat, 'longitude': lon},
  )


def test_grid_spec_from_dataset():
  ds = era5_style_dataset()
  spec = xarray_utils.grid_spec_from_dataset(ds)
  assert spec.nodal_shape == (64, 33)
  assert spec.latitude_spacing == 'equiangular_with_poles'
  np.testing.assert_allclose(
      np.rad2deg(spec.longitudes), ds.longitude.data, atol=1e-10
  )
  np.testing.assert_allclose(
      np.rad2deg(spec.latitudes), ds.latitude.data, atol=1e-10
  )


def test_selective_temporal_shift():
  ds = era5_style_dataset()
  shifted = xarray_utils.selective_temporal_shift(
      ds, variables=['sea_surface_temperature'], time_shift='6 hours'
  )
  assert shifted.sizes['time'] == ds.sizes['time'] - 1
  np.testing.assert_array_equal(
      shifted.temperature.data, ds.temperature.data[1:]
  )
  np.testing.assert_array_equal(
      shifted.sea_surface_temperature.data,
      ds.sea_surface_temperature.data[:-1],
  )

  with pytest.raises(ValueError, match='evenly'):
    xarray_utils.selective_temporal_shift(
        ds, variables=['sea_surface_temperature'], time_shift='9 hours'
    )


def test_fill_nan_with_nearest():
  ds = era5_style_dataset()
  sst = ds.sea_surface_temperature.copy(deep=True)
  sst[:, 10:12, 20:22] = np.nan
  filled = xarray_utils.fill_nan_with_nearest(sst)
  assert not filled.isnull().any()
  # untouched values are preserved
  np.testing.assert_array_equal(
      filled.data[:, :5, :5], sst.data[:, :5, :5]
  )
  # filled values come from neighboring cells (same value range)
  assert filled.data[:, 10:12, 20:22].min() >= sst.min()
  assert filled.data[:, 10:12, 20:22].max() <= sst.max()


def test_regrid_horizontal_conservative(device):
  ds = era5_style_dataset()
  source = xarray_utils.grid_spec_from_dataset(ds)
  target = spherical_harmonic.GridSpec.TL31()
  regridder = horizontal_interpolation.ConservativeRegridder(
      source, target, device=device
  )
  out = xarray_utils.regrid_horizontal(ds, regridder)
  assert out.temperature.shape == (
      ds.sizes['time'],
  ) + target.nodal_shape[::-1][::-1]  # (time, lon, lat) -> sizes by dims
  assert out.sizes['longitude'] == target.nodal_shape[0]
  assert out.sizes['latitude'] == target.nodal_shape[1]
  assert np.isfinite(out.temperature.data).all()
  # conservative regridding approximately preserves the global mean
  weights = np.cos(np.deg2rad(ds.latitude))
  source_mean = ds.temperature.weighted(weights).mean().item()
  target_weights = np.cos(np.deg2rad(out.latitude))
  target_mean = out.temperature.weighted(target_weights).mean().item()
  assert abs(source_mean - target_mean) < 0.5


def test_regrid_horizontal_descending_latitude(device):
  ds = era5_style_dataset()
  flipped = ds.isel(latitude=slice(None, None, -1))
  source = xarray_utils.grid_spec_from_dataset(ds)
  target = spherical_harmonic.GridSpec.TL31()
  regridder = horizontal_interpolation.ConservativeRegridder(
      source, target, device=device
  )
  out = xarray_utils.regrid_horizontal(ds, regridder)
  out_flipped = xarray_utils.regrid_horizontal(flipped, regridder)
  np.testing.assert_array_equal(
      out.temperature.data, out_flipped.temperature.data
  )
