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
"""Rollout training with torch.optim.

The model (encoder, physics parameterization, corrector/dycore, decoder)
is differentiable end-to-end, so training is a plain PyTorch loop over
`data.TrajectoryDataset` examples:

  model = api.PressureLevelModel.from_checkpoint(path, device='cuda')
  dataset = data.TrajectoryDataset(era5, model, outer_steps=2)
  optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-5)
  for inputs, forcings, targets in loader:
    optimizer.zero_grad()
    loss = training.rollout_loss(model, inputs, forcings, targets, rng=seed)
    loss.backward()
    optimizer.step()

The loss here is a latitude-weighted MSE on the decoded pressure-level
outputs, normalized per variable; upstream NeuralGCM trains with more
elaborate spectral objectives, but this is a reasonable default for
fine-tuning. Memory scales linearly with the number of advance steps kept
on the autodiff tape, so keep `outer_steps` small (1-3 outer frames).

Note: the dynamical core uses a grad-free snapshot of the (possibly
learned) orography taken when the checkpoint was imported; orography
correction parameters receive gradients only through the encoder/decoder
paths.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from neuralgcm_torch import api
from neuralgcm_torch import model_states

# Per-variable normalization scales (squared denominators for the MSE), in
# SI units, of the order of typical analysis errors / short-range forecast
# differences. Variables not listed fall back to the targets' standard
# deviation.
DEFAULT_LOSS_SCALES: Dict[str, float] = {
    'geopotential': 100.0,  # m^2/s^2
    'temperature': 1.0,  # K
    'u_component_of_wind': 2.0,  # m/s
    'v_component_of_wind': 2.0,  # m/s
    'specific_humidity': 1e-3,  # kg/kg
    'specific_cloud_ice_water_content': 1e-4,  # kg/kg
    'specific_cloud_liquid_water_content': 1e-4,  # kg/kg
}


def _to_device(tree, device, dtype=None):
  if isinstance(tree, dict):
    return {k: _to_device(v, device, dtype) for k, v in tree.items()}
  return tree.to(device=device, dtype=dtype)


def rollout_predictions(
    model: api.PressureLevelModel,
    inputs: dict,
    forcings: dict,
    target_sim_times: torch.Tensor,
    rng: Optional[int] = None,
) -> tuple[list[dict], model_states.ModelState]:
  """Encodes `inputs` and decodes a prediction at each target time.

  Unlike `PressureLevelModel.unroll`, gradients flow through the rollout.

  Args:
    model: the API model (its `model` module holds the parameters).
    inputs: model inputs at the initial time (`data.TrajectoryDataset`
      format, any device).
    forcings: forcings covering the rollout window.
    target_sim_times: 1-d tensor of nondimensional output times, each a
      multiple of the model timestep after the initial time.
    rng: optional integer seed for stochastic models.

  Returns:
    Tuple of (list of decoded output dicts in SI units, final state).
  """
  module = model.model
  inputs = _to_device(inputs, model.device, model.dtype)
  forcings = _to_device(forcings, model.device, model.dtype)

  forcing = module.forcing_fn(forcings, inputs['sim_time'][-1])
  state = module.encode(model._wb_inputs(inputs), forcing, rng)  # pylint: disable=protected-access

  dt = float(
      model.to_nondim_units(
          model.timestep / np.timedelta64(1, 's'), 's'
      )
  )
  outputs = []
  current = float(inputs['sim_time'][-1])
  for target_time in target_sim_times:
    steps = round((float(target_time) - current) / dt)
    if steps < 1:
      raise ValueError(
          f'target time {float(target_time)} is not after the current '
          f'time {current} (dt={dt})'
      )
    for _ in range(steps):
      forcing = module.forcing_fn(forcings, state.state.sim_time)
      state = module.advance(state, forcing)
    current += steps * dt
    forcing = module.forcing_fn(forcings, state.state.sim_time)
    outputs.append(model._from_wb_outputs(module.decode(state, forcing)))  # pylint: disable=protected-access
  return outputs, state


def rollout_loss(
    model: api.PressureLevelModel,
    inputs: dict,
    forcings: dict,
    targets: dict,
    rng: Optional[int] = None,
    loss_scales: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
  """Latitude-weighted, per-variable-normalized MSE over a rollout.

  Args:
    model: the API model.
    inputs: initial-time inputs (`data.TrajectoryDataset` format).
    forcings: forcings covering the rollout window.
    targets: target frames with a leading time axis and a `sim_time`
      vector defining the output times.
    rng: optional integer seed for stochastic models.
    loss_scales: per-variable normalization scales (SI units); defaults to
      `DEFAULT_LOSS_SCALES`, falling back to the targets' std for unknown
      variables.

  Returns:
    Scalar loss tensor (gradients flow to the model parameters).
  """
  if loss_scales is None:
    loss_scales = DEFAULT_LOSS_SCALES
  targets = _to_device(targets, model.device, model.dtype)
  predictions, _ = rollout_predictions(
      model, inputs, forcings, targets['sim_time'], rng
  )

  lat_weights = torch.as_tensor(
      np.cos(np.deg2rad(model.latitudes)),
      dtype=model.dtype,
      device=model.device,
  )
  lat_weights = lat_weights / lat_weights.mean()

  total = torch.zeros((), dtype=model.dtype, device=model.device)
  count = 0
  for step, prediction in enumerate(predictions):
    for name, predicted in prediction.items():
      if name == 'sim_time' or name not in targets:
        continue
      target = targets[name][step]
      scale = loss_scales.get(name)
      if scale is None:
        scale = max(float(target.std()), 1e-30)
      error = (predicted - target) / scale
      total = total + (error.square() * lat_weights).mean()
      count += 1
  return total / count


def spectral_rollout_loss(
    model: api.PressureLevelModel,
    inputs: dict,
    forcings: dict,
    targets: dict,
    rng: Optional[int] = None,
    loss_scales: Optional[Dict[str, float]] = None,
    wavenumber_cutoff: Optional[int] = None,
) -> torch.Tensor:
  """Per-variable-normalized MSE accumulated in spherical-harmonic space.

  The normalized error of each decoded variable is transformed to modal
  coefficients on the data grid and its squared coefficients are summed,
  which by Parseval equals an exactly area-weighted mean-square error on
  the sphere (up to the basis normalization constant) — the spectral
  form of the objectives NeuralGCM trains with upstream.

  Args:
    model: the API model.
    inputs: initial-time inputs (`data.TrajectoryDataset` format).
    forcings: forcings covering the rollout window.
    targets: target frames with a leading time axis and `sim_time`.
    rng: optional integer seed for stochastic models.
    loss_scales: per-variable normalization scales (SI units).
    wavenumber_cutoff: if set, only total wavenumbers below the cutoff
      enter the loss — fine-tune the resolvable scales without fitting
      the spectral tail.

  Returns:
    Scalar loss tensor.
  """
  if loss_scales is None:
    loss_scales = DEFAULT_LOSS_SCALES
  targets = _to_device(targets, model.device, model.dtype)
  predictions, _ = rollout_predictions(
      model, inputs, forcings, targets['sim_time'], rng
  )
  grid = model.model.decoder.output_coords.horizontal

  total = torch.zeros((), dtype=model.dtype, device=model.device)
  count = 0
  for step, prediction in enumerate(predictions):
    for name, predicted in prediction.items():
      if name == 'sim_time' or name not in targets:
        continue
      target = targets[name][step]
      scale = loss_scales.get(name)
      if scale is None:
        scale = max(float(target.std()), 1e-30)
      modal = grid.to_modal((predicted - target) / scale)
      if wavenumber_cutoff is not None:
        modal = modal[..., :wavenumber_cutoff]
      total = total + modal.square().sum(dim=(-2, -1)).mean()
      count += 1
  return total / count


def train_step(
    model: api.PressureLevelModel,
    optimizer: torch.optim.Optimizer,
    example,
    rng: Optional[int] = None,
    loss_scales: Optional[Dict[str, float]] = None,
    loss_fn=None,
) -> float:
  """One optimizer update on a single (inputs, forcings, targets) example.

  `loss_fn` defaults to `rollout_loss`; pass `spectral_rollout_loss` (or
  a `functools.partial` of it) for the spectral objective.
  """
  if loss_fn is None:
    loss_fn = rollout_loss
  inputs, forcings, targets = example
  optimizer.zero_grad()
  loss = loss_fn(model, inputs, forcings, targets, rng, loss_scales)
  loss.backward()
  optimizer.step()
  return float(loss.detach())
