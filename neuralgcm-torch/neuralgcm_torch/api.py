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
"""Inference API for models that predict dense data on pressure levels.

`PressureLevelModel` wraps a `StochasticModularStepModel` (built from a
converted checkpoint by `model_builder.from_checkpoint`) with xarray
input/output handling, mirroring the upstream NeuralGCM API:

  model = PressureLevelModel.from_checkpoint('converted.pt', device='cuda')
  inputs = model.inputs_from_xarray(era5_slice)
  forcings = model.forcings_from_xarray(era5_slice)
  state = model.encode(inputs, forcings, rng=42)
  state, outputs = model.unroll(state, forcings, steps=24)
  predictions = model.data_to_xarray(outputs, times=times)

Datasets use ERA5/WeatherBench conventions: `latitude` / `longitude` in
degrees (latitude ascending), `level` in hPa, full variable names
(`temperature`, `u_component_of_wind`, ...) with dimensions
`[time, level, latitude, longitude]` in any order. Inputs and outputs of
the model methods are dicts of torch tensors in SI units with trailing
`(longitude, latitude)` axes; a leading time axis is always present in
`inputs` / `forcings` dicts (length one for a single snapshot).
"""

from __future__ import annotations

import datetime
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import xarray

from dinosaur_torch import spherical_harmonic
from dinosaur_torch import scales
from neuralgcm_torch import checkpoint as checkpoint_lib
from neuralgcm_torch import model_builder
from neuralgcm_torch import model_states
from neuralgcm_torch import models

TimedeltaLike = Union[str, np.timedelta64, pd.Timedelta, datetime.timedelta]
Numeric = Union[float, np.ndarray, torch.Tensor]

_ABBREVIATED_NAMES = {
    'u_component_of_wind': 'u',
    'v_component_of_wind': 'v',
    'geopotential': 'z',
    'temperature': 't',
}
_FULL_NAMES = {v: k for k, v in _ABBREVIATED_NAMES.items()}


class _CloneOutputs(torch.nn.Module):
  """Clones a compiled module's outputs out of the CUDA-graph memory pool.

  CUDA-graph replays overwrite the graph's output buffers in place. The
  advanced model state outlives the next replay (it is the next step's
  input, and for models with memory also the step after that), so outputs
  must be materialized into ordinary allocator memory right after each
  replay.
  """

  def __init__(self, inner: torch.nn.Module):
    super().__init__()
    self.inner = inner

  def forward(self, *args, **kwargs):
    outputs = self.inner(*args, **kwargs)
    return torch.utils._pytree.tree_map(
        lambda x: x.clone() if isinstance(x, torch.Tensor) else x, outputs
    )


def _calculate_sub_steps(
    timestep: np.timedelta64, duration: TimedeltaLike
) -> int:
  """Number of internal time-steps that make up `duration`."""
  duration = pd.Timedelta(duration)
  time_step_ratio = duration / pd.Timedelta(timestep)
  if abs(time_step_ratio - round(time_step_ratio)) > 1e-6:
    raise ValueError(
        f'non-integral time-step ratio: {duration=} is not a multiple of '
        f'the internal model timestep {timestep}'
    )
  return round(time_step_ratio)


class PressureLevelModel:
  """Inference-only API for NeuralGCM models on ERA5 pressure-level data."""

  def __init__(
      self,
      model: models.StochasticModularStepModel,
      config: dict,
      device: Any = None,
      dtype: torch.dtype = torch.float32,
  ):
    self.model = model
    self.config = config
    self.device = torch.device(device) if device is not None else (
        torch.device('cpu')
    )
    self.dtype = dtype
    self._physics = checkpoint_lib.sim_units_from_config(config)
    self._data_grid_spec = checkpoint_lib.grid_spec_from_config(
        config['data_grid'] or config['model_grid']
    )
    self._ref_datetime = np.datetime64(config['reference_datetime'])

  @classmethod
  def from_checkpoint(
      cls,
      checkpoint: Union[str, dict],
      device: Any = None,
      dtype: torch.dtype = torch.float32,
  ) -> PressureLevelModel:
    """Builds the model from a converted checkpoint (dict or file path)."""
    if not isinstance(checkpoint, dict):
      checkpoint = checkpoint_lib.load(checkpoint)
    model = model_builder.from_checkpoint(
        checkpoint, device=device, dtype=dtype
    )
    return cls(model, checkpoint['config'], device=device, dtype=dtype)

  #  =========================================================================
  #  Static model properties.
  #  =========================================================================

  @property
  def input_variables(self) -> list[str]:
    """Variable names required in `inputs` datasets."""
    return list(self.config['input_variables'])

  @property
  def forcing_variables(self) -> list[str]:
    """Variable names required in `forcings` datasets."""
    return list(self.config['forcing_variables'])

  @property
  def tracer_variables(self) -> list[str]:
    """The subset of `input_variables` treated as tracers."""
    return list(self.config['tracer_variables'])

  @property
  def timestep(self) -> np.timedelta64:
    """Spacing between internal model timesteps."""
    return np.timedelta64(
        int(round(self.config['timestep_seconds'])), 's'
    )

  @property
  def data_grid(self) -> spherical_harmonic.GridSpec:
    """Horizontal grid spec of input/output data."""
    return self._data_grid_spec

  @property
  def data_levels(self) -> np.ndarray:
    """Pressure levels of input/output data, in hPa."""
    return np.asarray(self.config['data_pressure_levels'])

  @property
  def longitudes(self) -> np.ndarray:
    """Longitudes of input/output data, in degrees."""
    return np.rad2deg(self._data_grid_spec.longitudes)

  @property
  def latitudes(self) -> np.ndarray:
    """Latitudes of input/output data, in degrees."""
    return np.rad2deg(self._data_grid_spec.latitudes)

  #  =========================================================================
  #  Units and time conversions.
  #  =========================================================================

  def to_nondim_units(self, value: Numeric, units: str) -> Numeric:
    """Scales a value to the model's internal non-dimensional units."""
    return self._physics.nondimensionalize(value * scales.parse_units(units))

  def from_nondim_units(self, value: Numeric, units: str) -> Numeric:
    """Scales a value from the model's internal non-dimensional units."""
    return self._physics.dimensionalize(
        value, scales.parse_units(units).units
    ).magnitude

  def datetime64_to_sim_time(self, datetime64: np.ndarray) -> np.ndarray:
    """Converts datetime64 values to nondimensional `sim_time`."""
    hours = (datetime64 - self._ref_datetime) / np.timedelta64(1, 'h')
    return self._physics.nondimensionalize(hours * scales.units.hour)

  def sim_time_to_datetime64(self, sim_time: np.ndarray) -> np.ndarray:
    """Converts nondimensional `sim_time` values to datetime64."""
    minutes = self._physics.dimensionalize(
        np.asarray(sim_time, np.float64), scales.units.minute
    ).magnitude
    return self._ref_datetime + np.array(
        np.round(minutes).astype(np.int64), 'timedelta64[m]'
    )

  #  =========================================================================
  #  xarray conversions.
  #  =========================================================================

  def _check_coords(self, dataset: xarray.Dataset, levels: bool):
    lon = dataset['longitude'].data
    lat = dataset['latitude'].data
    if lon.shape != self.longitudes.shape or (
        np.abs(lon - self.longitudes).max() > 1e-3
    ):
      raise ValueError(
          f'longitude coordinate mismatch: {lon} vs {self.longitudes}'
      )
    if lat.shape != self.latitudes.shape or (
        np.abs(lat - self.latitudes).max() > 1e-3
    ):
      raise ValueError(
          f'latitude coordinate mismatch: {lat} vs {self.latitudes}'
      )
    if levels:
      lvl = dataset['level'].data
      if lvl.shape != self.data_levels.shape or (
          np.abs(lvl - self.data_levels).max() > 1e-3
      ):
        raise ValueError(
            f'pressure level mismatch: {lvl} vs {self.data_levels}'
        )

  def _sim_time(self, dataset: xarray.Dataset) -> torch.Tensor:
    times = np.atleast_1d(dataset['time'].data)
    if np.issubdtype(times.dtype, np.floating):
      sim_time = times  # already nondimensional
    else:
      sim_time = self.datetime64_to_sim_time(times)
    return torch.as_tensor(
        sim_time, dtype=self.dtype, device=self.device
    )

  def _data_from_xarray(
      self,
      dataset: xarray.Dataset,
      variables: list[str],
      with_level: bool,
      device: Any = None,
  ) -> dict[str, torch.Tensor]:
    for k in variables:
      if k not in dataset:
        raise ValueError(f'expected variable {k} not found in dataset')
    self._check_coords(dataset, levels=with_level)
    if 'time' not in dataset.dims:
      dataset = dataset.expand_dims('time')
    dims = ('time', 'level', 'longitude', 'latitude') if with_level else (
        'time', 'longitude', 'latitude'
    )
    device = self.device if device is None else device
    data = {
        k: torch.as_tensor(
            np.ascontiguousarray(dataset[k].transpose(*dims).data),
            dtype=self.dtype,
            device=device,
        )
        for k in variables
    }
    data['sim_time'] = self._sim_time(dataset).to(device)
    return data

  def inputs_from_xarray(
      self, dataset: xarray.Dataset, device: Any = None
  ) -> dict[str, torch.Tensor]:
    """Extracts input tensors (with a leading time axis) from a dataset."""
    return self._data_from_xarray(
        dataset, self.input_variables, with_level=True, device=device
    )

  def forcings_from_xarray(
      self, dataset: xarray.Dataset, device: Any = None
  ) -> dict[str, torch.Tensor]:
    """Extracts forcing tensors (with a leading time axis) from a dataset."""
    return self._data_from_xarray(
        dataset, self.forcing_variables, with_level=False, device=device
    )

  def data_from_xarray(
      self, dataset: xarray.Dataset
  ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Extracts inputs and forcings from a dataset."""
    return (
        self.inputs_from_xarray(dataset),
        self.forcings_from_xarray(dataset),
    )

  def data_to_xarray(
      self,
      data: dict[str, Any],
      times: Optional[np.ndarray] = None,
      members: Optional[np.ndarray] = None,
  ) -> xarray.Dataset:
    """Converts model predictions to an `xarray.Dataset`.

    Args:
      data: dict of arrays shaped `([time,] level, longitude, latitude)`
        for level variables or `([time,] longitude, latitude)` for surface
        variables. A `sim_time` entry, if present, is ignored.
      times: coordinate array of length `time`, or `None` if the arrays
        have no leading time dimension.
      members: coordinate array for the ensemble member axis of batched
        outputs (which follows the time axis; see `encode_ensemble`), or
        `None` for unbatched data.

    Returns:
      Dataset with `latitude` / `longitude` in degrees and `level` in hPa.
    """
    coords = {
        'longitude': self.longitudes,
        'latitude': self.latitudes,
        'level': self.data_levels,
    }
    if times is not None:
      coords['time'] = np.asarray(times)
    if members is not None:
      coords['member'] = np.asarray(members)
    n_levels = self.data_levels.size
    data_vars = {}
    for k, v in data.items():
      if k == 'sim_time':
        continue
      if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
      v = np.asarray(v)
      ndim = (
          v.ndim
          + (1 if times is None else 0)
          - (0 if members is None else 1)
      )
      if ndim == 4:
        if v.shape[-3] == 1 and n_levels != 1:
          # singleton vertical axis on surface diagnostics (e.g. the
          # precipitation outputs); named `surface` as in the published
          # NeuralGCM datasets.
          dims = ('surface', 'longitude', 'latitude')
        else:
          dims = ('level', 'longitude', 'latitude')
      elif ndim == 3:
        dims = ('longitude', 'latitude')
      else:
        raise ValueError(f'unsupported array shape for {k}: {v.shape}')
      if members is not None:
        dims = ('member',) + dims
      if times is not None:
        dims = ('time',) + dims
      data_vars[k] = (dims, v)
    return xarray.Dataset(data_vars, coords=coords)

  #  =========================================================================
  #  Model methods.
  #  =========================================================================

  def _wb_inputs(self, inputs: dict) -> dict:
    """Renames inputs to the WeatherBench field names with nested tracers."""
    inputs = dict(inputs)
    return {
        'u': inputs['u_component_of_wind'],
        'v': inputs['v_component_of_wind'],
        't': inputs['temperature'],
        'z': inputs['geopotential'],
        'sim_time': inputs['sim_time'],
        'tracers': {k: inputs[k] for k in self.tracer_variables},
    }

  def _from_wb_outputs(self, outputs: dict) -> dict:
    """Flattens tracers and diagnostics, restores the full variable names."""
    outputs = dict(outputs)
    outputs.update(outputs.pop('tracers', {}))
    outputs.update(outputs.pop('diagnostics', None) or {})
    return {_FULL_NAMES.get(k, k): v for k, v in outputs.items()}

  def _forcing_at(self, forcings: Optional[dict], sim_time) -> dict:
    if forcings is None:
      return None
    if isinstance(sim_time, torch.Tensor) and sim_time.ndim == 1:
      sim_time = sim_time[-1]
    return self.model.forcing_fn(forcings, sim_time)

  @torch.no_grad()
  def encode(
      self,
      inputs: dict,
      forcings: Optional[dict] = None,
      rng: Optional[int] = None,
  ) -> model_states.ModelState:
    """Encodes pressure-level inputs and forcings to a model state.

    Args:
      inputs: dict from `inputs_from_xarray`; the last time slice is
        encoded.
      forcings: dict from `forcings_from_xarray`.
      rng: optional integer seed for the stochastic state (required for
        reproducible stochastic models, ignored by deterministic ones).

    Returns:
      Model state on sigma levels (modal representation).
    """
    forcing = self._forcing_at(forcings, inputs['sim_time'])
    return self.model.encode(self._wb_inputs(inputs), forcing, rng)

  @torch.no_grad()
  def encode_ensemble(
      self,
      inputs: dict,
      forcings: Optional[dict] = None,
      rngs: Sequence[int] = (),
  ) -> model_states.ModelState:
    """Encodes one member per seed and stacks them along a member axis.

    The result is a member-batched state (see `ensembles`): `advance` and
    `unroll` run all members through one batched model call, and decoded
    outputs gain a member axis after time (pass `members=` to
    `data_to_xarray`). Each member evolves exactly as it would when
    encoded individually with its seed (up to float reassociation in the
    batched kernels).
    """
    from neuralgcm_torch import ensembles

    return ensembles.stack_states(
        [self.encode(inputs, forcings, rng=rng) for rng in rngs]
    )

  @torch.no_grad()
  def advance(
      self,
      state: model_states.ModelState,
      forcings: Optional[dict] = None,
  ) -> model_states.ModelState:
    """Advances the model state by one internal timestep."""
    forcing = self._forcing_at(forcings, state.state.sim_time)
    return self.model.advance(state, forcing)

  @torch.no_grad()
  def decode(
      self,
      state: model_states.ModelState,
      forcings: Optional[dict] = None,
  ) -> dict:
    """Decodes a model state to pressure-level outputs (SI units)."""
    forcing = self._forcing_at(forcings, state.state.sim_time)
    return self._from_wb_outputs(self.model.decode(state, forcing))

  @torch.no_grad()
  def compile(
      self,
      state: model_states.ModelState,
      forcings: Optional[dict] = None,
      options: Optional[dict] = None,
      cudagraphs: bool = False,
      **compile_kwargs,
  ) -> None:
    """Compiles the advance step with `torch.compile` (in place).

    The corrector (dynamical core substeps) and the physics
    parameterization (neural network) carry essentially all of the compute
    of an `advance` and take only tensor inputs; the stochastic-field
    update stays eager (it draws from a `torch.Generator`).

    One eager `advance` is run first to materialize the lazily-built
    implicit-solve matrices, so they become constants for the compiler
    instead of being traced into the graph.

    Args:
      state: a representative model state (e.g. the result of `encode`);
        it is not modified.
      forcings: forcing data matching `state`.
      options: extra inductor options, merged over the defaults below.
      cudagraphs: also capture each compiled submodule as a CUDA graph
        (inductor cudagraph trees), removing the per-kernel launch
        overhead from replays. Requires a CUDA device.
      **compile_kwargs: forwarded to `torch.compile`.
    """
    # `comprehensive_padding` pads the strides of intermediate buffers,
    # which corrupts the spectral transforms in this graph (the padded
    # bytes are uninitialized and an extern FFT kernel consumes them as
    # data; observed with torch 2.12 + cu13). Padding buys nothing here,
    # so turn it off rather than depend on allocator luck.
    options = {'comprehensive_padding': False} | (options or {})
    if cudagraphs:
      options = {'triton.cudagraphs': True} | options
    forcing = self._forcing_at(forcings, state.state.sim_time)
    self.model.advance(state, forcing)  # warm lazy buffers
    step = self.model.advance_module
    corrector = torch.compile(
        step.corrector, options=options, **compile_kwargs
    )
    parameterization = torch.compile(
        step.physics_parameterization, options=options, **compile_kwargs
    )
    if cudagraphs:
      corrector = _CloneOutputs(corrector)
      parameterization = _CloneOutputs(parameterization)
    step.corrector = corrector
    step.physics_parameterization = parameterization

  @torch.no_grad()
  def unroll(
      self,
      state: model_states.ModelState,
      forcings: Optional[dict] = None,
      *,
      steps: int,
      timedelta: Optional[TimedeltaLike] = None,
      start_with_input: bool = False,
  ) -> tuple[model_states.ModelState, dict]:
    """Unrolls predictions over many time-steps.

    Usage:

      advanced_state, outputs = model.unroll(state, forcings, steps=N)

    where `outputs` is a dict of decoded variables with a leading time
    axis of size `N`.

    Args:
      state: initial model state.
      forcings: forcing data covering the unroll period (with a leading
        time axis); the forcing nearest in time is used at every step.
      steps: number of output frames.
      timedelta: spacing between output frames (must be a multiple of
        `timestep`); defaults to `timestep`.
      start_with_input: if True, outputs are at `[0, ..., (steps-1)*dt]`
        relative to the initial time rather than `[dt, ..., steps*dt]`.

    Returns:
      Tuple of (advanced state, outputs).
    """
    if timedelta is None:
      inner_steps = 1
    else:
      inner_steps = _calculate_sub_steps(self.timestep, timedelta)

    def post_process(x: model_states.ModelState) -> dict:
      return self.decode(x, forcings)

    return self.model.trajectory(
        state,
        outer_steps=steps,
        inner_steps=inner_steps,
        forcing_data=forcings,
        start_with_input=start_with_input,
        post_process_fn=post_process,
    )
