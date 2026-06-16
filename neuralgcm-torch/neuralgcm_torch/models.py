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
"""The model class composing encoder/advance/decoder/forcing modules.

Only `StochasticModularStepModel` is ported; it is the model class used by
all published checkpoint configs. Use
`neuralgcm_torch.model_builder.from_checkpoint` to construct one from
a converted checkpoint.
"""

from __future__ import annotations

import zlib
from typing import Any, Optional

from torch import nn

from dinosaur_torch import time_integration
from neuralgcm_torch import model_states
from neuralgcm_torch import stochastic

ModelState = model_states.ModelState

_FINALIZE_SALT = zlib.crc32(b'finalize_state')  # arbitrary uint32 value


class StochasticModularStepModel(nn.Module):
  """Dynamical model with modular components and stochasticity.

  Components operate on `ModelState`s; the advance module owns the random
  field, which is (re)initialized by `encode` via `finalize_state`.

  Randomness uses integer seeds (see `neuralgcm_torch.stochastic`):
  pass `rng` to `encode` for reproducible stochastic models; draws are
  statistically (not bitwise) equivalent to the original JAX implementation.
  """

  HAIKU_NAME = 'stochastic_modular_step_model'

  def __init__(
      self,
      encoder: nn.Module,
      decoder: nn.Module,
      advance_module: nn.Module,
      forcing_module: nn.Module,
  ):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.advance_module = advance_module
    self.forcing_module = forcing_module

  def forcing_fn(self, forcing_data: dict, sim_time) -> dict:
    """Returns the forcing at `sim_time`, possibly using `forcing_data`."""
    return self.forcing_module(forcing_data, sim_time)

  def encode(
      self, x: dict, forcing=None, rng: Optional[int] = None
  ) -> ModelState:
    """Encodes inputs (with a leading time axis) to a model state."""
    model_state = self.encoder(x, forcing, rng)
    finalize_rng = None if rng is None else stochastic.fold_in(
        rng, _FINALIZE_SALT
    )
    return self.advance_module.finalize_state(
        model_state, forcing, finalize_rng
    )

  def advance(self, x: ModelState, forcing=None) -> ModelState:
    """Returns the model state advanced by one time step."""
    return self.advance_module(x, forcing)

  def decode(self, x: ModelState, forcing=None) -> dict:
    """Decodes a model state to the data representation."""
    return self.decoder(x, forcing)

  def trajectory(
      self,
      x: ModelState,
      outer_steps: int,
      inner_steps: int = 1,
      *,
      forcing_data: Optional[dict] = None,
      start_with_input: bool = False,
      post_process_fn=lambda x: x,
  ) -> tuple[ModelState, Any]:
    """Returns the final state and a trajectory of `outer_steps` frames."""

    def step_fn(x: ModelState) -> ModelState:
      sim_time = getattr(x.state, 'sim_time', None)
      forcing = self.forcing_fn(forcing_data, sim_time)
      return self.advance(x, forcing)

    return time_integration.trajectory_from_step(
        step_fn,
        outer_steps,
        inner_steps,
        start_with_input=start_with_input,
        post_process_fn=post_process_fn,
    )(x)

  def import_haiku(
      self, params: dict, prefix: str = HAIKU_NAME
  ) -> None:
    # all children are created in the legacy __init__ (hence the '~').
    for child in (
        self.advance_module,
        self.encoder,
        self.decoder,
        self.forcing_module,
    ):
      if hasattr(child, 'import_haiku'):
        child.import_haiku(params, f'{prefix}/~/{child.HAIKU_NAME}')
