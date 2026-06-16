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
"""Modules that parameterize composed time-steppers.

Only `StochasticPhysicsParameterizationStep` is ported; it is the step used
by all published checkpoint configs. Diagnostics and perturbation modules
are not ported: published checkpoints use the no-op defaults, so the
`diagnostics` field of the model state is always an empty dict.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

from torch import nn

from neuralgcm_torch import model_states

ModelState = model_states.ModelState


class StochasticPhysicsParameterizationStep(nn.Module):
  """Advances a `ModelState` with stochastic physics tendencies + dycore.

  Each of the `num_substeps` substeps computes physics tendencies from the
  current state (and randomness), advances the state with the corrector
  (which was built with the substep `dt`), and advances the random field.

  Args:
    corrector: corrector module built with `dt / num_substeps`.
    physics_parameterization: parameterization module built with
      `dt / num_substeps`.
    randomness_module: random field providing `unconditional_sample` and
      `advance`.
    num_substeps: number of substeps per call.
    diagnostics_module: optional module called as `module(x, pp_tendency,
      forcing)` after each substep (and at encoding time); its output dict
      is stored on `ModelState.diagnostics`. Published checkpoints use
      none; see `neuralgcm_torch.diagnostics`.
  """

  HAIKU_NAME = 'stochastic_physics_parameterization_step'

  def __init__(
      self,
      corrector: nn.Module,
      physics_parameterization: nn.Module,
      randomness_module: nn.Module,
      num_substeps: int = 1,
      diagnostics_module: Optional[nn.Module] = None,
  ):
    super().__init__()
    self.corrector = corrector
    self.physics_parameterization = physics_parameterization
    self.randomness_module = randomness_module
    self.num_substeps = num_substeps
    self.diagnostics_module = diagnostics_module

  def finalize_state(
      self, x: ModelState, forcing=None, rng: Optional[int] = None
  ) -> ModelState:
    """Populates the randomness (and diagnostics) of an encoded state."""
    randomness = self.randomness_module.unconditional_sample(
        rng if rng is not None else 0
    )
    x = dataclasses.replace(x, randomness=randomness, diagnostics={})
    if self.diagnostics_module is not None:
      pp_tendency = self.physics_parameterization(
          x.state, x.memory, x.diagnostics, x.randomness.nodal_value, forcing
      )
      diagnostics = self.diagnostics_module(x, pp_tendency, forcing)
      x = dataclasses.replace(x, diagnostics=diagnostics)
    return x

  def forward(self, state: ModelState, forcing=None) -> ModelState:
    x = state
    for _ in range(self.num_substeps):
      pp_tendency = self.physics_parameterization(
          x.state, x.memory, x.diagnostics, x.randomness.nodal_value, forcing
      )
      next_state = self.corrector(x.state, pp_tendency, forcing)
      next_randomness = self.randomness_module.advance(x.randomness)
      next_memory = x.state if x.memory is not None else None
      next_diagnostics = (
          {} if self.diagnostics_module is None
          else self.diagnostics_module(x, pp_tendency, forcing)
      )
      x = ModelState(
          state=next_state,
          memory=next_memory,
          diagnostics=next_diagnostics,
          randomness=next_randomness,
      )
    return x

  def import_haiku(self, params: dict, prefix: str) -> None:
    # all children are created in the legacy __init__ (hence the '~').
    for child in (
        self.corrector,
        self.physics_parameterization,
        self.randomness_module,
        self.diagnostics_module,
    ):
      if child is not None and hasattr(child, 'import_haiku'):
        child.import_haiku(params, f'{prefix}/~/{child.HAIKU_NAME}')
