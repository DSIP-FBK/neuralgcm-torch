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

"""Fine-tune a converted NeuralGCM checkpoint on real ERA5 data.

Demonstrates the full training loop on the 2.8 deg deterministic model:
short-rollout windows sampled across a month of ARCO-ERA5, the spectral
rollout loss, and a held-out before/after forecast evaluation.

Run from the repository root (fetch needs network; training needs GPU):

  uv run --no-sync python neuralgcm-torch/tools/finetune_era5.py \
      neuralgcm-torch/notebooks/checkpoints/deterministic_2_8_deg.pt \
      --fetch-only          # download + regrid the training windows once
  uv run --no-sync python neuralgcm-torch/tools/finetune_era5.py \
      neuralgcm-torch/notebooks/checkpoints/deterministic_2_8_deg.pt \
      --steps 200

Multi-GPU: launch with torchrun (one process per GPU); examples are
sharded across ranks with `distributed.example_sampler`.

The raw 0.25 deg fetch is ~1 GB per hourly snapshot, so windows are
downloaded one at a time, regridded to the model's data grid
immediately, and cached on disk (~15 MB total) for re-runs.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import xarray

from dinosaur_torch import horizontal_interpolation
from dinosaur_torch import xarray_utils
from neuralgcm_torch import api
from neuralgcm_torch import data as data_lib
from neuralgcm_torch import distributed
from neuralgcm_torch import training

ERA5_PATH = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'


def open_era5(model):
  full_era5 = xarray.open_zarr(
      ERA5_PATH, chunks=None, storage_options=dict(token='anon')
  )
  return full_era5[model.input_variables + model.forcing_variables].pipe(
      xarray_utils.selective_temporal_shift,
      variables=model.forcing_variables,
      time_shift='24 hours',
  )


def regrid_to_model(ds, model, device):
  era5_grid = xarray_utils.grid_spec_from_dataset(ds)
  regridder = horizontal_interpolation.ConservativeRegridder(
      era5_grid, model.data_grid, skipna=True, device=device
  )
  out = xarray_utils.regrid_horizontal(ds, regridder)
  return xarray_utils.fill_nan_with_nearest(out)


def fetch_windows(model, args, device):
  """Downloads + regrids training windows and the eval slice (cached)."""
  os.makedirs(args.cache_dir, exist_ok=True)
  train_path = os.path.join(args.cache_dir, 'train_windows.nc')
  eval_path = os.path.join(args.cache_dir, 'eval.nc')
  if os.path.exists(train_path) and os.path.exists(eval_path):
    return train_path, eval_path

  era5 = open_era5(model)
  frames = args.outer_steps + 1
  starts = np.datetime64(args.train_start) + np.arange(args.windows) * (
      np.timedelta64(args.window_stride_hours, 'h')
  )
  # one cache file per window, so an interrupted fetch resumes
  windows = []
  for i, start in enumerate(starts):
    part_path = os.path.join(args.cache_dir, f'window_{i:03d}.nc')
    if not os.path.exists(part_path):
      times = start + np.arange(frames) * np.timedelta64(1, 'h')
      t = time.perf_counter()
      raw = era5.sel(time=times).compute()
      regrid_to_model(raw, model, device).to_netcdf(part_path)
      print(
          f'window {i + 1}/{len(starts)} @ {start}'
          f' ({time.perf_counter() - t:.1f}s)',
          flush=True,
      )
    windows.append(xarray.load_dataset(part_path))
  xarray.concat(windows, dim='time').to_netcdf(train_path)

  if not os.path.exists(eval_path):
    eval_times = slice(args.eval_start, args.eval_end, 24)
    raw = era5.sel(time=eval_times).compute()
    regrid_to_model(raw, model, device).to_netcdf(eval_path)
  return train_path, eval_path


def window_datasets(train_path, model, outer_steps):
  """One evenly spaced TrajectoryDataset per fetched window."""
  ds = xarray.load_dataset(train_path)
  frames = outer_steps + 1
  parts = [
      data_lib.TrajectoryDataset(
          ds.isel(time=slice(i, i + frames)), model, outer_steps=outer_steps
      )
      for i in range(0, ds.sizes['time'], frames)
  ]
  return torch.utils.data.ConcatDataset(parts)


@torch.no_grad()
def evaluate(model, eval_path, lead_days=3):
  """Day-`lead_days` T850/Z500 RMSE from the first eval init."""
  ds = xarray.load_dataset(eval_path)
  inputs = model.inputs_from_xarray(ds.isel(time=0))
  forcings = model.forcings_from_xarray(ds.isel(time=[0]))
  state = model.encode(inputs, forcings, rng=0)
  state, outputs = model.unroll(
      state, forcings, steps=lead_days, timedelta='24 hours'
  )
  predictions = model.data_to_xarray(
      outputs, times=ds.time.values[1 : lead_days + 1]
  )
  target = ds.isel(time=lead_days)
  prediction = predictions.isel(time=lead_days - 1)
  weights = np.cos(np.deg2rad(ds.latitude))
  weights = weights / weights.mean()

  def rmse(name, level):
    err = (
        prediction[name].sel(level=level) - target[name].sel(level=level)
    ) ** 2
    return float(np.sqrt((err * weights).mean()))

  return {
      f't850_rmse_day{lead_days}': rmse('temperature', 850),
      f'z500_rmse_day{lead_days}': rmse('geopotential', 500) / 9.80665,
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('checkpoint')
  parser.add_argument('--device', default=None)
  parser.add_argument('--cache-dir', default='/tmp/ngcm_finetune')
  parser.add_argument('--train-start', default='2020-01-01T00')
  parser.add_argument('--windows', type=int, default=40)
  parser.add_argument('--window-stride-hours', type=int, default=18)
  parser.add_argument('--eval-start', default='2020-02-02T00')
  parser.add_argument('--eval-end', default='2020-02-06T00')
  parser.add_argument('--outer-steps', type=int, default=1)
  parser.add_argument('--steps', type=int, default=200)
  parser.add_argument('--lr', type=float, default=3e-6)
  parser.add_argument('--loss', choices=['spectral', 'nodal'],
                      default='spectral')
  parser.add_argument('--wavenumber-cutoff', type=int, default=None)
  parser.add_argument('--fetch-only', action='store_true')
  parser.add_argument('--out', default='/tmp/ngcm_finetune/result')
  args = parser.parse_args()

  rank, world_size = distributed.init()
  device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
  model = api.PressureLevelModel.from_checkpoint(args.checkpoint, device)

  train_path, eval_path = fetch_windows(model, args, device)
  if args.fetch_only:
    print(f'cached: {train_path}, {eval_path}')
    return

  dataset = window_datasets(train_path, model, args.outer_steps)
  print(f'{len(dataset)} training examples')

  if args.loss == 'spectral':
    def loss_fn(model, inputs, forcings, targets, rng, scales):
      return training.spectral_rollout_loss(
          model, inputs, forcings, targets, rng, scales,
          wavenumber_cutoff=args.wavenumber_cutoff,
      )
  else:
    loss_fn = None  # train_step's default nodal rollout_loss

  before = evaluate(model, eval_path) if rank == 0 else {}

  module = model.model
  optimizer = torch.optim.AdamW(module.parameters(), lr=args.lr)
  if world_size > 1:
    ddp_loss = distributed.wrap(model)
    sampler = distributed.example_sampler(dataset)
  else:
    ddp_loss, sampler = None, torch.utils.data.RandomSampler(
        dataset, replacement=False,
        generator=torch.Generator().manual_seed(0),
    )

  losses = []
  start = time.perf_counter()
  step = 0
  while step < args.steps:
    if hasattr(sampler, 'set_epoch'):
      sampler.set_epoch(step)
    for index in iter(sampler):
      if step >= args.steps:
        break
      example = dataset[index]
      if ddp_loss is not None:
        loss = distributed.train_step(ddp_loss, optimizer, example, rng=step)
      else:
        loss = training.train_step(
            model, optimizer, example, rng=step, loss_fn=loss_fn
        )
      losses.append(loss)
      step += 1
      if step % 10 == 0 and rank == 0:
        recent = float(np.mean(losses[-10:]))
        print(f'step {step:4d} loss {recent:.4f} '
              f'({(time.perf_counter() - start) / step:.2f}s/step)',
              flush=True)

  if rank == 0:
    after = evaluate(model, eval_path)
    os.makedirs(args.out, exist_ok=True)
    torch.save(module.state_dict(), os.path.join(args.out, 'finetuned.pt'))
    result = {
        'args': {k: str(v) for k, v in vars(args).items()},
        'losses': losses,
        'loss_first10': float(np.mean(losses[:10])),
        'loss_last10': float(np.mean(losses[-10:])),
        'eval_before': before,
        'eval_after': after,
    }
    with open(os.path.join(args.out, 'metrics.json'), 'w') as f:
      json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'losses'},
                     indent=2))


if __name__ == '__main__':
  main()
