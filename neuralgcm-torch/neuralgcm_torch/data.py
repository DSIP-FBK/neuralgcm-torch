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
"""Data pipeline: training examples from ERA5-style xarray datasets.

`TrajectoryDataset` windows a time series into rollout-training examples:

  dataset = data.TrajectoryDataset(era5, model, outer_steps=2)
  inputs, forcings, targets = dataset[0]

Each example holds the model inputs at the initial time, the forcings
covering the rollout window, and the next `outer_steps` frames as targets
(all dicts of CPU tensors in SI units, ready for
`training.rollout_loss`). The source dataset must already be on the
model's data grid (see `dinosaur_torch.xarray_utils` for regridding)
with a time spacing that is a multiple of the model timestep.

NeuralGCM models operate on single examples (no batch dimension); use
`torch.utils.data.DataLoader(dataset, batch_size=None, ...)` and
accumulate gradients across examples instead of batching.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import xarray

from neuralgcm_torch import api


Example = tuple[dict, dict, dict]


class TrajectoryDataset(torch.utils.data.Dataset):
  """Rollout-training windows over an ERA5-style dataset.

  Args:
    dataset: in-memory dataset on the model's data grid, with the model's
      input and forcing variables, evenly spaced `time`.
    model: the `PressureLevelModel` whose conversion rules (variables,
      coordinates, units, reference time) define the example tensors.
    outer_steps: number of target frames per example.
    stride: offset (in dataset time indices) between successive examples.
    device: device for the example tensors; keep the default (CPU) when
      loading with DataLoader workers.
  """

  def __init__(
      self,
      dataset: xarray.Dataset,
      model: api.PressureLevelModel,
      outer_steps: int,
      stride: int = 1,
      device: Any = 'cpu',
  ):
    if dataset.chunks:
      raise ValueError(
          'dataset must be loaded in memory; call .compute() first'
      )
    self.dataset = dataset
    self.model = model
    self.outer_steps = outer_steps
    self.stride = stride
    self.device = device

    times = dataset['time'].data
    spacing = times[1] - times[0]
    if not (np.diff(times) == spacing).all():
      raise ValueError('dataset times must be evenly spaced')
    ratio = spacing / model.timestep
    if abs(ratio - round(ratio)) > 1e-6 or round(ratio) < 1:
      raise ValueError(
          f'dataset time spacing {spacing} is not a positive multiple of '
          f'the model timestep {model.timestep}'
      )
    self._num_examples = (
        dataset.sizes['time'] - outer_steps - 1
    ) // stride + 1
    if self._num_examples <= 0:
      raise ValueError(
          f'dataset too short: {dataset.sizes["time"]} times for '
          f'{outer_steps} target steps'
      )

  def __len__(self) -> int:
    return self._num_examples

  def __getitem__(self, index: int) -> Example:
    if not 0 <= index < self._num_examples:
      raise IndexError(index)
    start = index * self.stride
    window = self.dataset.isel(
        time=slice(start, start + self.outer_steps + 1)
    )
    inputs = self.model.inputs_from_xarray(
        window.isel(time=[0]), device=self.device
    )
    forcings = self.model.forcings_from_xarray(window, device=self.device)
    targets = self.model.inputs_from_xarray(
        window.isel(time=slice(1, None)), device=self.device
    )
    return inputs, forcings, targets
