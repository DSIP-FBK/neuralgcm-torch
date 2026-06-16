# Copyright 2024 Google LLC
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
"""Utilities for preparing xarray datasets for model consumption.

These helpers operate on datasets with `latitude` / `longitude` coordinates
in degrees (ERA5 conventions) and convert to/from the `(longitude,
latitude)` trailing-axes layout used by this package.
"""

from __future__ import annotations

from typing import Sequence, TypeVar, Union

import numpy as np
import pandas as pd
import torch
import xarray

from dinosaur_torch import horizontal_interpolation
from dinosaur_torch import spherical_harmonic

DatasetOrDataArray = TypeVar(
    'DatasetOrDataArray', xarray.Dataset, xarray.DataArray
)


def infer_longitude_offset(
    lon: Union[np.ndarray, xarray.DataArray],
) -> float:
  """Infers the longitude offset in radians from longitudes in degrees."""
  lon = np.asarray(lon)
  if lon.max() < 2 * np.pi:
    raise ValueError(f'Expected longitude values in degrees, got {lon=}')
  return float(lon[0] * np.pi / 180)


def infer_latitude_spacing(
    lat: Union[np.ndarray, xarray.DataArray],
) -> str:
  """Infers the latitude spacing name from latitude values in degrees."""
  lat = np.asarray(lat)
  if np.allclose(np.diff(lat), lat[1] - lat[0]):
    if np.isclose(lat.max(), 90.0):
      return 'equiangular_with_poles'
    return 'equiangular'
  return 'gauss'


def grid_spec_from_dataset(
    dataset: Union[xarray.Dataset, xarray.DataArray],
) -> spherical_harmonic.GridSpec:
  """Builds a nodal-only `GridSpec` matching a dataset's lat/lon coords.

  The resulting spec carries no spectral truncation (wavenumber counts are
  set to a linear truncation of the latitude count) and is intended for use
  with the regridders in `horizontal_interpolation`, which only read the
  nodal coordinates.
  """
  lon = dataset['longitude'].data
  lat = dataset['latitude'].data
  return spherical_harmonic.GridSpec(
      longitude_wavenumbers=lat.size,
      total_wavenumbers=lat.size + 1,
      longitude_nodes=lon.size,
      latitude_nodes=lat.size,
      latitude_spacing=infer_latitude_spacing(lat),
      longitude_offset=infer_longitude_offset(lon),
  )


def ensure_ascending_latitude(data: DatasetOrDataArray) -> DatasetOrDataArray:
  """Returns `data` with ascending latitude, reversing if needed."""
  latitude = data.coords['latitude']
  if (latitude.diff('latitude') > 0).all():
    return data
  elif (latitude.diff('latitude') < 0).all():
    return data.isel(latitude=slice(None, None, -1))
  else:
    raise ValueError(f'non-monotonic latitude: {latitude.data}')


def selective_temporal_shift(
    dataset: xarray.Dataset,
    variables: Sequence[str] = (),
    time_shift: Union[str, np.timedelta64, pd.Timedelta] = '0 hour',
    time_name: str = 'time',
) -> xarray.Dataset:
  """Shifts `variables` in time and truncates the head/tail accordingly.

  A positive `time_shift` produces a dataset where, at each time, the
  values of `variables` come from an earlier time of the original dataset
  (e.g. forcings observed `time_shift` before the forecast init).
  """
  time_shift = pd.Timedelta(time_shift)
  time_spacing = dataset[time_name][1] - dataset[time_name][0]

  shift, remainder = divmod(time_shift, time_spacing)
  shift = int(shift)
  if shift == 0 or not variables:
    return dataset
  if remainder:
    raise ValueError(f'Does not divide evenly, got {remainder=}')

  ds = dataset.copy()
  if shift > 0:
    ds = ds.isel({time_name: slice(shift, None)})
    for var in variables:
      ds[var] = dataset.variables[var].isel({time_name: slice(None, -shift)})
  else:
    ds = ds.isel({time_name: slice(None, shift)})
    for var in variables:
      ds[var] = dataset.variables[var].isel({time_name: slice(-shift, None)})
  return ds


def fill_nan_with_nearest(data: DatasetOrDataArray) -> DatasetOrDataArray:
  """Replaces NaN values with the nearest horizontal non-NaN values.

  The NaN mask must be identical across all non-spatial dimensions (e.g.
  a fixed land mask for sea surface temperature).
  """
  from sklearn import neighbors  # deferred: sklearn import is slow

  def fill_nan_for_array(array: xarray.DataArray) -> xarray.DataArray:
    if 'latitude' not in array.dims and 'longitude' not in array.dims:
      return array

    if array.chunks:
      raise ValueError(
          f'Expected data to be loaded in memory, got chunks={array.chunks}.'
          ' Consider calling .compute() first.'
      )

    extra_dims = list(set(array.dims) - {'latitude', 'longitude'})
    isnan_mask = array.isnull().any(extra_dims)
    allnan_mask = array.isnull().all(extra_dims)

    if not isnan_mask.any():
      return array
    if allnan_mask.all():
      raise ValueError('all values are NaN')
    if not isnan_mask.equals(allnan_mask):
      raise ValueError('NaN mask is not fixed across non-spatial dimensions')

    lat, lon = xarray.broadcast(array.latitude, array.longitude)
    lat = lat.transpose(*isnan_mask.dims)
    lon = lon.transpose(*isnan_mask.dims)

    index_coords = np.deg2rad(
        np.stack(
            [lat.data[~isnan_mask.data], lon.data[~isnan_mask.data]], axis=-1
        )
    )
    query_coords = np.deg2rad(
        np.stack(
            [lat.data[isnan_mask.data], lon.data[isnan_mask.data]], axis=-1
        )
    )

    tree = neighbors.BallTree(index_coords, metric='haversine')
    indices = tree.query(query_coords, return_distance=False).squeeze(axis=-1)

    source_lats = xarray.DataArray(
        lat.data[~isnan_mask.data][indices], dims=['query']
    )
    source_lons = xarray.DataArray(
        lon.data[~isnan_mask.data][indices], dims=['query']
    )
    target_lats = xarray.DataArray(lat.data[isnan_mask.data], dims=['query'])
    target_lons = xarray.DataArray(lon.data[isnan_mask.data], dims=['query'])

    array = array.copy(deep=True)
    array.loc[{'latitude': target_lats, 'longitude': target_lons}] = (
        array.loc[{'latitude': source_lats, 'longitude': source_lons}]
    )
    return array

  if 'latitude' not in data.dims or 'longitude' not in data.dims:
    raise ValueError(f'did not find latitude and longitude dimensions: {data}')

  if isinstance(data, xarray.DataArray):
    return fill_nan_for_array(data)
  elif isinstance(data, xarray.Dataset):
    return data.map(fill_nan_for_array)
  else:
    raise TypeError(f'data must be a DataArray or Dataset: {data}')


def regrid_horizontal(
    data: DatasetOrDataArray,
    regridder: horizontal_interpolation.Regridder,
    latlon_tolerance: float = 1e-3,
) -> DatasetOrDataArray:
  """Horizontally regrids a dataset with one of this package's regridders.

  Args:
    data: source data with `latitude` / `longitude` coordinates in degrees
      matching `regridder.source_grid`.
    regridder: regridder to apply.
    latlon_tolerance: maximum absolute coordinate difference, in degrees.

  Returns:
    Regridded data with new `latitude` and `longitude` coordinates.
  """
  data = ensure_ascending_latitude(data)

  old_lon = np.rad2deg(regridder.source_grid.longitudes)
  old_lat = np.rad2deg(regridder.source_grid.latitudes)

  if abs(old_lon - data.longitude.data).max() > latlon_tolerance:
    raise ValueError(
        'inconsistent longitude between data and source grid:'
        f' {data.longitude.data} vs {old_lon}'
    )
  if abs(old_lat - data.latitude.data).max() > latlon_tolerance:
    raise ValueError(
        'inconsistent latitude between data and source grid:'
        f' {data.latitude.data} vs {old_lat}'
    )

  device = next(iter(regridder.buffers())).device

  def regrid_to_numpy(x):
    field = torch.as_tensor(
        np.ascontiguousarray(x), dtype=torch.float32, device=device
    )
    return regridder(field).cpu().numpy()

  data = xarray.apply_ufunc(
      regrid_to_numpy,
      data,
      input_core_dims=[['longitude', 'latitude']],
      output_core_dims=[['longitude', 'latitude']],
      exclude_dims={'longitude', 'latitude'},
      vectorize=True,  # loop over time & level, for lower memory usage
  )
  data.coords['longitude'] = np.rad2deg(regridder.target_grid.longitudes)
  data.coords['latitude'] = np.rad2deg(regridder.target_grid.latitudes)

  return data
