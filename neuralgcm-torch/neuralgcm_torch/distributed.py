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
"""Multi-GPU data-parallel fine-tuning with DistributedDataParallel.

NeuralGCM models take single examples (no batch axis), so data
parallelism is the natural multi-GPU strategy: every rank holds a full
replica, runs different examples, and DDP averages gradients during
backward. Spatial sharding is not warranted at these sizes (0.19M-31M
parameters; a replica plus its rollout activations fits one device).

DDP only hooks gradient synchronization into graphs produced by the
wrapped module's ``forward``, while NeuralGCM training drives the model
through ``encode``/``advance``/``decode``. ``wrap`` therefore wraps the
whole rollout-loss computation as the forward pass, which is the
supported pattern for models trained through method calls.

Usage, one process per GPU (``torchrun --nproc_per_node=N train.py``):

  rank, world_size = distributed.init()
  model = api.PressureLevelModel.from_checkpoint(path, device=f'cuda:{rank}')
  ddp_loss = distributed.wrap(model)
  optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-5)

  dataset = data.TrajectoryDataset(era5, model, outer_steps=2)
  sampler = distributed.example_sampler(dataset)
  loader = torch.utils.data.DataLoader(
      dataset, batch_size=None, sampler=sampler
  )
  for epoch in range(epochs):
    sampler.set_epoch(epoch)
    for i, example in enumerate(loader):
      loss = distributed.train_step(ddp_loss, optimizer, example, rng=i)
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.distributed as dist

from neuralgcm_torch import api
from neuralgcm_torch import training


def init() -> tuple[int, int]:
  """Initializes the default process group from torchrun's environment.

  Returns (rank, world_size); (0, 1) when not launched under torchrun,
  so training scripts work unchanged in single-process runs.
  """
  if 'RANK' not in os.environ or not dist.is_available():
    return 0, 1
  backend = 'nccl' if torch.cuda.is_available() else 'gloo'
  dist.init_process_group(backend)
  if torch.cuda.is_available():
    local_rank = int(os.environ.get('LOCAL_RANK', dist.get_rank()))
    torch.cuda.set_device(local_rank % torch.cuda.device_count())
  return dist.get_rank(), dist.get_world_size()


class RolloutLossModule(torch.nn.Module):
  """`training.rollout_loss` as a forward pass, for DDP to hook into."""

  def __init__(
      self,
      model: api.PressureLevelModel,
      loss_scales: Optional[Dict[str, float]] = None,
  ):
    super().__init__()
    self.api_model = model
    self.module = model.model  # registers the parameters with DDP
    self.loss_scales = loss_scales

  def forward(self, inputs, forcings, targets, rng=None):
    return training.rollout_loss(
        self.api_model, inputs, forcings, targets, rng, self.loss_scales
    )


def wrap(
    model: api.PressureLevelModel,
    loss_scales: Optional[Dict[str, float]] = None,
    find_unused_parameters: bool = True,
    **ddp_kwargs,
) -> torch.nn.parallel.DistributedDataParallel:
  """Wraps the rollout loss of `model` for distributed training.

  `find_unused_parameters` defaults to True: which parameters join a
  given loss graph depends on the checkpoint's wiring (e.g. orography
  correction parameters only reach the loss through the encoder/decoder
  paths, and diagnostic heads only when their outputs are decoded), and
  a parameter missing from one rank's graph would otherwise hang the
  all-reduce. The bookkeeping cost is negligible at NeuralGCM sizes.
  """
  device_ids = None
  if next(model.model.parameters()).device.type == 'cuda':
    device_ids = [torch.cuda.current_device()]
  return torch.nn.parallel.DistributedDataParallel(
      RolloutLossModule(model, loss_scales),
      device_ids=device_ids,
      find_unused_parameters=find_unused_parameters,
      **ddp_kwargs,
  )


def example_sampler(
    dataset: torch.utils.data.Dataset, shuffle: bool = True, **kwargs
) -> torch.utils.data.DistributedSampler:
  """A DistributedSampler over `data.TrajectoryDataset` examples."""
  return torch.utils.data.DistributedSampler(
      dataset, shuffle=shuffle, **kwargs
  )


def train_step(
    ddp_loss: torch.nn.parallel.DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    example,
    rng: Optional[int] = None,
) -> float:
  """One synchronized optimizer update on this rank's example."""
  inputs, forcings, targets = example
  optimizer.zero_grad()
  loss = ddp_loss(inputs, forcings, targets, rng)
  loss.backward()
  optimizer.step()
  return float(loss.detach())
