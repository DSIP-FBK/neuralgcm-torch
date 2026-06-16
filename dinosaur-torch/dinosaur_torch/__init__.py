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

"""Idiomatic-PyTorch rewrite of the Dinosaur spectral dynamical core."""

import dinosaur_torch.associated_legendre
import dinosaur_torch.coordinate_systems
import dinosaur_torch.filtering
import dinosaur_torch.fourier
import dinosaur_torch.horizontal_interpolation
import dinosaur_torch.primitive_equations
import dinosaur_torch.radiation
import dinosaur_torch.vertical_interpolation
import dinosaur_torch.pytree
import dinosaur_torch.scales
import dinosaur_torch.sigma_coordinates
import dinosaur_torch.spherical_harmonic
import dinosaur_torch.time_integration
import dinosaur_torch.units
import dinosaur_torch.xarray_utils

__version__ = '0.1.0'
