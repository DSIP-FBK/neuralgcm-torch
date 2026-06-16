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

"""Shared fixtures: tests take an explicit `device` (cuda when available).

Unlike the original JAX implementation there is no global default device; modules are
constructed with `device=...` like any other PyTorch code. Set
DINOSAUR_TORCH_TEST_DEVICE=cpu (or cuda) to override.
"""
import os

import pytest
import torch


@pytest.fixture(scope='session')
def device() -> torch.device:
  name = os.environ.get('DINOSAUR_TORCH_TEST_DEVICE')
  if name is None:
    name = 'cuda' if torch.cuda.is_available() else 'cpu'
  return torch.device(name)
