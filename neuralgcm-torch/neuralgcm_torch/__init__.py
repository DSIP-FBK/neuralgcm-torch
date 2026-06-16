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

"""Idiomatic-PyTorch rewrite of NeuralGCM (legacy API scope)."""

import neuralgcm_torch.api
import neuralgcm_torch.checkpoint
import neuralgcm_torch.correctors
import neuralgcm_torch.data
import neuralgcm_torch.decoders
import neuralgcm_torch.diagnostics
import neuralgcm_torch.embeddings
import neuralgcm_torch.encoders
import neuralgcm_torch.features
import neuralgcm_torch.filters
import neuralgcm_torch.forcings
import neuralgcm_torch.layers
import neuralgcm_torch.mappings
import neuralgcm_torch.model_builder
import neuralgcm_torch.model_states
import neuralgcm_torch.models
import neuralgcm_torch.orographies
import neuralgcm_torch.parameterizations
import neuralgcm_torch.pretrained
import neuralgcm_torch.steps
import neuralgcm_torch.stochastic
import neuralgcm_torch.towers
import neuralgcm_torch.training
import neuralgcm_torch.transforms

from neuralgcm_torch.api import PressureLevelModel

__version__ = '0.1.0'
