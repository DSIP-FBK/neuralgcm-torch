# Copyright 2023 Google LLC
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

"""Tiny helpers over `torch.utils._pytree` for dataclasses-of-tensors.

States in this package are plain dataclasses registered as torch pytrees, so
they compose natively with `torch.compile`, `torch.func` and CUDA graphs.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, TypeVar

import torch
import torch.utils._pytree as torch_pytree

T = TypeVar('T')

tree_map = torch_pytree.tree_map
tree_flatten = torch_pytree.tree_flatten
tree_unflatten = torch_pytree.tree_unflatten
tree_leaves = torch_pytree.tree_leaves


def state(cls: type[T]) -> type[T]:
  """Class decorator: dataclass registered as a torch pytree node.

  All fields are pytree children, in declaration order.
  """
  cls = dataclasses.dataclass(cls)
  torch_pytree.register_dataclass(cls)
  return cls


def map_fields(fn: Callable[..., Any], tree: Any, *rests: Any) -> Any:
  """`tree_map` applied only to non-scalar tensor leaves.

  Tensor leaves with `ndim > 0` are mapped through `fn`; all other leaves
  (python scalars, 0-d tensors such as `sim_time`, strings, None) pass
  through unchanged.
  """

  def g(x, *xs):
    if isinstance(x, torch.Tensor) and x.ndim > 0:
      return fn(x, *xs)
    return x

  return tree_map(g, tree, *rests)
