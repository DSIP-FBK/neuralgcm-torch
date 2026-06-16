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
"""Modules that map dictionaries of fields through neural-network towers.

Field dictionaries are packed into a single feature tensor, passed through
a tower, and unpacked into a dictionary of specified output shapes.

IMPORTANT: packing follows **jax pytree semantics** — dictionary keys are
visited in sorted order (recursively) — because the published checkpoints
were trained with features packed that way. Plain torch pytrees preserve
insertion order instead, so the helpers here sort explicitly.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
from torch import nn


def _sorted_items(tree: dict):
  """Yields (key_path, leaf) depth-first with sorted keys per dict level."""
  for k in sorted(tree):
    v = tree[k]
    if isinstance(v, dict):
      yield from _sorted_items(v)
    else:
      yield k, v


def _align_batch_dims(leaves: Sequence[torch.Tensor]) -> list[torch.Tensor]:
  """Broadcasts leaves to a common leading (batch) shape.

  Leaves share a trailing core rank (the minimum ndim present); any extra
  leading dimensions are batch dimensions (e.g. ensemble members) and are
  broadcast across all leaves, so member-batched state features combine
  with member-shared static/forcing features.
  """
  tail = min(leaf.ndim for leaf in leaves)
  if all(leaf.ndim == tail for leaf in leaves):
    return list(leaves)
  batch = torch.broadcast_shapes(
      *[leaf.shape[: leaf.ndim - tail] for leaf in leaves]
  )
  return [leaf.expand(batch + leaf.shape[leaf.ndim - tail:]) for leaf in leaves]


def pack_dict(inputs: dict, dim: int = -3) -> torch.Tensor:
  """Concatenates leaves along `dim` in sorted-key order.

  `dim` counts from the end; leaves may carry extra leading batch
  dimensions, which are broadcast to a common shape.
  """
  leaves = _align_batch_dims([leaf for _, leaf in _sorted_items(inputs)])
  return torch.cat(leaves, dim=dim)


def stack_dict(inputs: dict, dim: int = 0) -> torch.Tensor:
  """Stacks leaves along a new `dim` in sorted-key order (see pack_dict)."""
  leaves = _align_batch_dims([leaf for _, leaf in _sorted_items(inputs)])
  return torch.stack(leaves, dim=dim)


def _build_like(tree: dict, values: dict) -> dict:
  if isinstance(tree, dict):
    return {k: _build_like(tree[k], values) for k in tree}
  raise AssertionError  # pragma: no cover


def _ordered_items(tree: dict):
  """Yields (key, leaf) depth-first in INSERTION order per dict level."""
  for k, v in tree.items():
    if isinstance(v, dict):
      yield from _ordered_items(v)
    else:
      yield k, v


def unpack_to_dict(
    array: torch.Tensor, output_shapes: dict, dim: int = -3
) -> dict:
  """Splits `array` along `dim` into a dict shaped like `output_shapes`.

  `output_shapes` maps names to full leaf shapes; the split sizes are taken
  from each shape's `dim` entry, in **insertion order**. The caller must
  order `output_shapes` to match the legacy pytree flatten order: sorted
  keys for plain dicts, dataclass field order for State-shaped outputs
  (vorticity, divergence, temperature_variation, log_surface_pressure,
  tracers).
  """
  items = list(_ordered_items(output_shapes))
  sizes = [shape[dim] for _, shape in items]
  pieces = torch.split(array, sizes, dim=dim)
  by_key = {k: piece for (k, _), piece in zip(items, pieces)}

  def rebuild(shapes: dict) -> dict:
    return {
        k: rebuild(v) if isinstance(v, dict) else by_key[k]
        for k, v in shapes.items()
    }

  return rebuild(output_shapes)


def unstack_to_dict(
    array: torch.Tensor, output_shapes: dict, dim: int = 0
) -> dict:
  """Unstacks `array` along `dim` into a dict shaped like `output_shapes`.

  Slices are assigned in insertion order (see `unpack_to_dict`).
  """
  items = list(_ordered_items(output_shapes))
  pieces = torch.unbind(array, dim=dim)
  if len(pieces) != len(items):
    raise ValueError(
        f'cannot unstack {len(pieces)} slices into {len(items)} outputs'
    )
  by_key = {k: piece for (k, _), piece in zip(items, pieces)}

  def rebuild(shapes: dict) -> dict:
    return {
        k: rebuild(v) if isinstance(v, dict) else by_key[k]
        for k, v in shapes.items()
    }

  return rebuild(output_shapes)


class NodalMapping(nn.Module):
  """Maps a dict of nodal features to a dict of specified structure.

  Packs the inputs into a single (n, lon, lat) tensor along the feature
  axis (-3), applies the tower, and unpacks the (m, lon, lat) result into
  `output_shapes`.
  """

  HAIKU_NAME = 'nodal_mapping'

  def __init__(self, tower: nn.Module, output_shapes: Dict[str, Tuple]):
    super().__init__()
    self.tower = tower
    self.output_shapes = output_shapes
    self.feature_axis = -3

  def forward(self, inputs: dict) -> dict:
    array = pack_dict(inputs, self.feature_axis)
    # ndim 4 carries a leading batch (ensemble member) dimension
    if array.ndim not in (3, 4):
      raise ValueError(f'Expected input array with ndim=3, got {array.shape=}')
    outputs = self.tower(array)
    if outputs.ndim != array.ndim:
      raise ValueError(f'Expected outputs with ndim=3, got {outputs.shape=}')
    return unpack_to_dict(outputs, self.output_shapes, self.feature_axis)

  def import_haiku(self, params: dict, prefix: str) -> None:
    self.tower.import_haiku(params, f'{prefix}/~/{self.tower.HAIKU_NAME}')


class NodalVolumeMapping(nn.Module):
  """Maps a dict of nodal volume features to a dict of given structure.

  Stacks the inputs into a (channel, level, lon, lat) tensor, applies the
  tower, and unstacks the (n, level, lon, lat) result into `output_shapes`.
  All input leaves must share the same shape.
  """

  HAIKU_NAME = 'nodal_volume_mapping'

  def __init__(self, tower: nn.Module, output_shapes: Dict[str, Tuple]):
    super().__init__()
    self.tower = tower
    self.output_shapes = output_shapes
    # counted from the end so a leading batch (member) dimension may be
    # present; equals dim 0 for unbatched (level, lon, lat) leaves.
    self.feature_axis = -4

  def forward(self, inputs: dict) -> dict:
    array = stack_dict(inputs, self.feature_axis)
    if array.ndim not in (4, 5):
      raise ValueError(f'Expected input array with ndim=4, got {array.shape=}')
    outputs = self.tower(array)
    if outputs.ndim != array.ndim:
      raise ValueError(f'Expected outputs with ndim=4, got {outputs.shape=}')
    return unstack_to_dict(outputs, self.output_shapes, self.feature_axis)

  def import_haiku(self, params: dict, prefix: str) -> None:
    self.tower.import_haiku(params, f'{prefix}/~/{self.tower.HAIKU_NAME}')
