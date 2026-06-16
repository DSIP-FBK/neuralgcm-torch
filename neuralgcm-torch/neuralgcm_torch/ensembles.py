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
"""Member-batched ensembles: many members through one model call.

Ensemble members differ only in their stochastic state, so instead of
looping members through `advance`, the member states are stacked along a
new leading batch axis and advanced together — the deterministic compute
(dycore + physics network) then runs as batched tensor ops, which is much
faster than a member loop for the launch-bound model sizes.

Conventions for a batched `ModelState` (M members):

- prognostic / memory / diagnostic tensors gain a leading member axis:
  `(M, level, m, l)` modal, `(M, level, lon, lat)` nodal;
- `sim_time` stays a shared scalar (members advance in lockstep);
- `RandomnessState.core/nodal_value/modal_value` carry an explicit
  `(M, fields, ...)` layout (the single-field `_squeeze` convention is
  reinstated on extraction), and `prng_key` becomes one key per member,
  so each member draws exactly the noise its sequential advance would.

Build a batched state with `api.PressureLevelModel.encode_ensemble` (or
`stack_states` over individually encoded members); `advance`, `unroll`
and `decode` then work unchanged, with outputs gaining a member axis.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import torch

from neuralgcm_torch import model_states
from neuralgcm_torch import stochastic


def _map_state_leaves(fn, trees, key=None):
  """Maps `fn(key, *leaves)` over aligned non-randomness pytree leaves."""
  t0 = trees[0]
  if isinstance(t0, stochastic.RandomnessState):
    raise AssertionError('randomness handled separately')
  if dataclasses.is_dataclass(t0) and not isinstance(t0, type):
    return type(t0)(**{
        f.name: _map_state_leaves(
            fn, [getattr(t, f.name) for t in trees], f.name
        )
        for f in dataclasses.fields(t0)
    })
  if isinstance(t0, dict):
    return {
        k: _map_state_leaves(fn, [t[k] for t in trees], k) for k in t0
    }
  return fn(key, *trees)


def _stack_leaf(key, *leaves):
  if leaves[0] is None:
    if any(leaf is not None for leaf in leaves):
      raise ValueError(f'inconsistent None leaf {key!r} across members')
    return None
  if key == 'sim_time':
    # members advance in lockstep; keep the shared scalar
    for leaf in leaves[1:]:
      if not torch.equal(torch.as_tensor(leaf), torch.as_tensor(leaves[0])):
        raise ValueError('ensemble members must share sim_time')
    return leaves[0]
  return torch.stack(leaves)


def _stack_randomness(
    states: Sequence[stochastic.RandomnessState],
) -> stochastic.RandomnessState:
  steps = {s.prng_step for s in states}
  if len(steps) != 1:
    raise ValueError(f'members at different prng steps: {steps}')

  def stack_value(values):
    if values[0] is None:
      return None
    if isinstance(values[0], dict):
      return {k: stack_value([v[k] for v in values]) for k in values[0]}
    stacked = torch.stack(values)
    # reinstate the fields axis dropped by single-field `_squeeze`
    return stacked.unsqueeze(-3) if stacked.ndim == 3 else stacked

  return stochastic.RandomnessState(
      core=stack_value([s.core for s in states]),
      nodal_value=stack_value([s.nodal_value for s in states]),
      modal_value=stack_value([s.modal_value for s in states]),
      prng_key=tuple(s.prng_key for s in states),
      prng_step=states[0].prng_step,
  )


def stack_states(
    states: Sequence[model_states.ModelState],
) -> model_states.ModelState:
  """Stacks individually encoded member states into one batched state."""
  if len(states) < 2:
    raise ValueError('an ensemble needs at least 2 members')
  batched = _map_state_leaves(
      _stack_leaf,
      [dataclasses.replace(s, randomness=None) for s in states],
  )
  return dataclasses.replace(
      batched, randomness=_stack_randomness([s.randomness for s in states])
  )


def num_members(state: model_states.ModelState) -> int:
  """Number of members in a batched state."""
  if isinstance(state.randomness.prng_key, (tuple, list)):
    return len(state.randomness.prng_key)
  raise ValueError('not a member-batched state')


def _member_randomness(
    state: stochastic.RandomnessState, index: int
) -> stochastic.RandomnessState:
  def pick(value):
    if value is None:
      return None
    if isinstance(value, dict):
      return {k: pick(v) for k, v in value.items()}
    member = value[index]
    # single-field modules store unbatched values `_squeeze`d; multi-field
    # modules keep the fields axis (assumed > 1 there)
    return member.squeeze(-3) if member.shape[-3] == 1 else member

  return stochastic.RandomnessState(
      core=pick(state.core),
      nodal_value=pick(state.nodal_value),
      modal_value=pick(state.modal_value),
      prng_key=state.prng_key[index],
      prng_step=state.prng_step,
  )


def member_state(
    state: model_states.ModelState, index: int
) -> model_states.ModelState:
  """Extracts one member as a regular (unbatched) model state."""

  def pick(key, leaf):
    if leaf is None or key == 'sim_time':
      return leaf
    return leaf[index]

  picked = _map_state_leaves(
      pick, [dataclasses.replace(state, randomness=None)]
  )
  return dataclasses.replace(
      picked, randomness=_member_randomness(state.randomness, index)
  )
