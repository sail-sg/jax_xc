# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test functionals."""

import jax.numpy as jnp
import pkgutil
from importlib import import_module

import jax_xc

names = [name for _, name, _ in pkgutil.iter_modules(jax_xc.__path__)]
for name in names:
  if name in [
    "gga_x_fd_lb94",
    "gga_x_fd_revlb94",
    "gga_x_gg99",
    "gga_x_kgg99",
    "hyb_gga_xc_case21",
    "hyb_mgga_xc_b94_hyb",
    "hyb_mgga_xc_br3p86",
    "lda_x_1d_exponential",
    "lda_x_1d_soft",
    "mgga_c_b94",
    "mgga_x_b00",
    "mgga_x_bj06",
    "mgga_x_br89",
    "mgga_x_br89_1",
    "mgga_x_mbr",
    "mgga_x_mbrxc_bg",
    "mgga_x_mbrxh_bg",
    "mgga_x_mggac",
    "mgga_x_rpp09",
    "mgga_x_tb09",
  ]:
    continue
  try:
    mod = import_module(f"jax_xc.{name}")
    fn = getattr(mod, name)
    if name.startswith("mgga") or name.startswith("hyb_mgga"):
      res = fn(lambda x: jnp.sum(x, axis=-1), lambda x: x, False)
    else:
      res = fn(lambda x: jnp.sum(x, axis=-1), False)
    print(name, res(jnp.array([1., 2., 3.])))
  except Exception as e:
    print(name, e)
