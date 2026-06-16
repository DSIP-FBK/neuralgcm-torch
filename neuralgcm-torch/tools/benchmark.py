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

"""Benchmark eager vs torch.compile inference for a converted checkpoint.

Usage (from the repository root):

  uv run --no-sync python neuralgcm-torch/tools/benchmark.py \
      /tmp/converted_2_8_deg.pt --steps 50

Builds the model, encodes a synthetic near-balanced atmospheric state and
times `advance` steps in eager mode and after `PressureLevelModel.compile`.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from neuralgcm_torch import api

GRAVITY = 9.80665


def _smooth_noise(rs, shape):
  """Horizontally smooth random fields (smooth synthetic benchmark inputs)."""
  import scipy.ndimage

  *batch, lon, lat = shape
  coarse = rs.randn(*batch, 8, 4)
  zoom = [1] * len(batch) + [lon / 8, lat / 4]
  return scipy.ndimage.zoom(coarse, zoom, order=3, mode='grid-wrap')


def synthetic_state(model: api.PressureLevelModel, seed=1):
  """Encodes a synthetic near-balanced state (ISA profile, weak winds)."""
  rs = np.random.RandomState(seed)
  levels = model.data_levels.astype(np.float64)
  lon, lat = model.longitudes.size, model.latitudes.size
  shape = (1, len(levels), lon, lat)
  heights_m = 44330.77 * (1 - (levels / 1013.25) ** 0.1903)
  isa_t = np.maximum(288.15 - 0.0065 * heights_m, 216.65)
  humidity = 1e-3 * np.exp(-heights_m / 2000.0)

  def tensor(x):
    return torch.as_tensor(
        np.ascontiguousarray(x), dtype=model.dtype, device=model.device
    )

  inputs = {
      'u_component_of_wind': tensor(3 * _smooth_noise(rs, shape)),
      'v_component_of_wind': tensor(3 * _smooth_noise(rs, shape)),
      'temperature': tensor(
          isa_t[None, :, None, None] + _smooth_noise(rs, shape)
      ),
      'geopotential': tensor(
          GRAVITY
          * (heights_m[None, :, None, None] + 5 * _smooth_noise(rs, shape))
      ),
      'specific_humidity': tensor(
          humidity[None, :, None, None] * (1 + 0.3 * _smooth_noise(rs, shape))
      ),
      'specific_cloud_ice_water_content': tensor(
          1e-6 * np.abs(_smooth_noise(rs, shape))
      ),
      'specific_cloud_liquid_water_content': tensor(
          1e-6 * np.abs(_smooth_noise(rs, shape))
      ),
      'sim_time': tensor(np.zeros(1)),
  }
  forcings = {
      'sim_time': tensor(np.zeros(1)),
      'sea_ice_cover': tensor(
          np.clip(rs.rand(1, lon, lat) * 1.5 - 0.5, 0, 1)
      ),
      'sea_surface_temperature': tensor(273.0 + 30 * rs.rand(1, lon, lat)),
  }
  return inputs, forcings


def time_advance(model: api.PressureLevelModel, state, forcings, steps):
  """Times `steps` advance calls; returns (seconds_per_step, final_state)."""
  if model.device.type == 'cuda':
    torch.cuda.synchronize()
  start = time.perf_counter()
  for _ in range(steps):
    state = model.advance(state, forcings)
  if model.device.type == 'cuda':
    torch.cuda.synchronize()
  elapsed = time.perf_counter() - start
  return elapsed / steps, state


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('checkpoint', help='path to a converted checkpoint')
  parser.add_argument('--device', default=None)
  parser.add_argument('--steps', type=int, default=50)
  parser.add_argument('--warmup', type=int, default=5)
  parser.add_argument('--skip-compile', action='store_true')
  parser.add_argument(
      '--cudagraphs',
      action='store_true',
      help='capture the compiled advance submodules as CUDA graphs',
  )
  parser.add_argument(
      '--max-autotune',
      action='store_true',
      help='compile with inductor max-autotune (autotuned GEMM/conv '
      'kernels); passed via inductor options so it composes with the '
      'comprehensive_padding workaround and --cudagraphs',
  )
  parser.add_argument(
      '--json',
      default=None,
      help='append a JSON line with the results to this file',
  )
  args = parser.parse_args()

  device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
  model = api.PressureLevelModel.from_checkpoint(
      args.checkpoint, device=device
  )
  dt_seconds = model.timestep / np.timedelta64(1, 's')
  print(f'device={device} timestep={dt_seconds:.0f}s')

  inputs, forcings = synthetic_state(model)
  state = model.encode(inputs, forcings, rng=0)

  time_advance(model, state, forcings, args.warmup)
  eager_dt, eager_state = time_advance(model, state, forcings, args.steps)
  sim_days_per_min = 60 / eager_dt * dt_seconds / 86400
  print(
      f'eager:    {eager_dt * 1e3:8.2f} ms/step '
      f'({sim_days_per_min:8.1f} simulated days/minute)'
  )

  mode = 'eager'
  record = {
      'checkpoint': os.path.basename(args.checkpoint),
      'device': device,
      'steps': args.steps,
      'eager_ms': eager_dt * 1e3,
  }

  if args.skip_compile:
    if args.json:
      with open(args.json, 'a') as f:
        f.write(json.dumps(record | {'mode': mode}) + '\n')
    return

  mode = 'compile'
  if args.max_autotune:
    mode = 'max-autotune'
  if args.cudagraphs:
    mode += '+cudagraphs'
  options = None
  if args.max_autotune:
    options = dict(torch._inductor.list_mode_options('max-autotune-no-cudagraphs'))

  t = time.perf_counter()
  model.compile(state, forcings, options=options, cudagraphs=args.cudagraphs)
  _, _ = time_advance(model, state, forcings, 1)  # first compiled call
  compile_s = time.perf_counter() - t
  print(f'compile:  {compile_s:8.2f} s (one-time, {mode})')

  time_advance(model, state, forcings, args.warmup)
  compiled_dt, compiled_state = time_advance(
      model, state, forcings, args.steps
  )
  sim_days_per_min = 60 / compiled_dt * dt_seconds / 86400
  print(
      f'compiled: {compiled_dt * 1e3:8.2f} ms/step '
      f'({sim_days_per_min:8.1f} simulated days/minute) '
      f'[{eager_dt / compiled_dt:.2f}x]'
  )

  # numerics: same trajectory endpoint as eager
  rel_diffs = {}
  for name in ('vorticity', 'temperature_variation'):
    eager_v = getattr(eager_state.state, name)
    compiled_v = getattr(compiled_state.state, name)
    scale = eager_v.abs().max()
    rel = ((eager_v - compiled_v).abs().max() / scale).item()
    rel_diffs[name] = rel
    print(f'max rel diff after {args.steps} steps, {name}: {rel:.2e}')

  if args.json:
    record |= {
        'mode': mode,
        'compiled_ms': compiled_dt * 1e3,
        'compile_s': compile_s,
        'speedup': eager_dt / compiled_dt,
        'sim_days_per_min': sim_days_per_min,
        'max_rel_diff': rel_diffs,
    }
    with open(args.json, 'a') as f:
      f.write(json.dumps(record) + '\n')


if __name__ == '__main__':
  main()
