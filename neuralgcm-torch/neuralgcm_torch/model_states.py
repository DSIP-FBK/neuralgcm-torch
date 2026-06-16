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
"""State containers shared by encoders, steps and the model API."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional, Union

import torch

from dinosaur_torch import pytree
from neuralgcm_torch import stochastic


@pytree.state
class WeatherbenchState:
  """A WeatherBench state described using velocity components."""

  u: torch.Tensor
  v: torch.Tensor
  t: torch.Tensor
  z: torch.Tensor
  sim_time: Optional[Union[float, torch.Tensor]] = None
  tracers: Dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
  diagnostics: Dict[str, torch.Tensor] = dataclasses.field(
      default_factory=dict
  )


@pytree.state
class ModelState:
  """Model state decomposed into deterministic and stochastic components.

  Attributes:
    state: prognostic variables describing the state of the atmosphere.
    memory: optional model fields/predictions providing past time context.
    diagnostics: optional diagnostic values computed in the model space.
    randomness: an optional random field used to stochastically perturb the
      advance step of the model.
  """

  state: Any
  memory: Any = None
  diagnostics: Dict[str, torch.Tensor] = dataclasses.field(
      default_factory=dict
  )
  randomness: stochastic.RandomnessState = dataclasses.field(
      default_factory=stochastic.RandomnessState
  )
