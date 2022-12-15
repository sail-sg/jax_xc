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

import jax
import ctypes
import jax.numpy as jnp
from jax.config import config
from absl.testing import absltest, parameterized
from absl import logging
import numpy as np
import time

from jax_xc import libxc as pylibxc
from jax_xc.libxc import libxc
from jax_xc import utils, impl, functionals

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

THRESHOLD = 2e-10

# NOT-IMPLEMENTED due to jax's lack of support
SKIP_LIST = [
  "gga_x_fd_lb94",  # Becke-Roussel not having an closed-form expression
  "gga_x_fd_revlb94",  # Becke-Roussel not having an closed-form expression
  "gga_x_gg99",  # Becke-Roussel not having an closed-form expression
  "gga_x_kgg99",  # Becke-Roussel not having an closed-form expression
  "hyb_gga_xc_case21",  # Becke-Roussel not having an closed-form expression
  "hyb_mgga_xc_b94_hyb",  # Becke-Roussel not having an closed-form expression
  "hyb_mgga_xc_br3p86",  # Becke-Roussel not having an closed-form expression
  "lda_x_1d_exponential",  # Integral
  "lda_x_1d_soft",  # Integral
  "mgga_c_b94",  # Becke-Roussel not having an closed-form expression
  "mgga_x_b00",  # Becke-Roussel not having an closed-form expression
  "mgga_x_bj06",  # Becke-Roussel not having an closed-form expression
  "mgga_x_br89",  # Becke-Roussel not having an closed-form expression
  "mgga_x_br89_1",  # Becke-Roussel not having an closed-form expression
  "mgga_x_mbr",  # Becke-Roussel not having an closed-form expression
  "mgga_x_mbrxc_bg",  # Becke-Roussel not having an closed-form expression
  "mgga_x_mbrxh_bg",  # Becke-Roussel not having an closed-form expression
  "mgga_x_mggac",  # Becke-Roussel not having an closed-form expression
  "mgga_x_rpp09",  # Becke-Roussel not having an closed-form expression
  "mgga_x_tb09",  # Becke-Roussel not having an closed-form expression
  "gga_x_wpbeh",  # TODO: jit too long & error for E1_scaled
  "gga_c_ft97",  # TODO: jit too long & error for E1_scaled
  "lda_xc_tih",  # vxc functional, error from libxc
  'gga_c_pbe_jrgx',  # vxc functional, error from libxc
  "gga_x_lb",  # vxc functional, error from libxc
]


class _TestGetParams(parameterized.TestCase):

  @parameterized.parameters(
    *(pylibxc.util.xc_available_functional_names()),
  )
  def test_get_params_unpol(self, name):

    def inp_to_impl_args(inp, fn_type):
      if fn_type == "lda":
        return (inp["rho"],)
      elif fn_type == "gga":
        return (inp["rho"], inp["sigma"])
      elif fn_type == "mgga":
        return (inp["rho"], inp["sigma"], inp["lapl"], inp["tau"])

    def jit_impl_fn(impl, fn_type, p):
      if fn_type == "lda":
        return jax.jit(jax.vmap(lambda x0: impl(x0, p.params, p), in_axes=(0,)))
      elif fn_type == "gga":
        return jax.jit(
          jax.vmap(lambda x0, x1: impl(x0, x1, p.params, p), in_axes=(0, 0))
        )
      elif fn_type == "mgga":
        return jax.jit(
          jax.vmap(
            lambda x0, x1, x2, x3: impl(x0, x1, x2, x3, p.params, p),
            in_axes=(0, 0, 0, 0)
          )
        )

    func = pylibxc.LibXCFunctional(name, 1)
    x = ctypes.cast(func.xc_func, ctypes.c_void_p)
    p = libxc.get_p(x.value)
    name = p['name']
    if name in SKIP_LIST:
      logging.info(f"Skipping {name} due to existing in SKIP_LIST")
      return
    p = utils.dict_to_namedtuple(p, "P")
    if not hasattr(impl, p.name):
      logging.info(f"Skipping {p.name} due to no maple code implementation")
      return
    fn_type = utils.functional_name_to_type(name)

    batch = 100
    seed = 0
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 4)
    inp = {}
    inp["rho"] = jax.random.uniform(
      keys[0], (batch,), dtype=jnp.float64, minval=1e-10, maxval=1e2
    )
    inp["sigma"] = jax.random.uniform(
      keys[1], (batch,), dtype=jnp.float64, minval=1e-10, maxval=1e2
    )
    inp["lapl"] = jax.random.uniform(
      keys[2], (batch,), dtype=jnp.float64, minval=1e-10, maxval=1e2
    )
    inp["tau"] = jax.random.uniform(
      keys[3], (batch,), dtype=jnp.float64, minval=1e-10, maxval=1e2
    )
    start_time = time.time()
    res1 = func.compute(inp)
    end_time = time.time()
    logging.info(f"pylibxc {name} took {end_time - start_time} seconds")
    res1_zk = res1["zk"].squeeze()

    impl_fn = getattr(impl, p.name).unpol
    impl_fn = jit_impl_fn(impl_fn, fn_type, p)

    start_time = time.time()
    res2_zk = impl_fn(*inp_to_impl_args(inp, fn_type))
    end_time = time.time()
    logging.info(
      f"jax_xc {name} took {end_time - start_time} compilation seconds"
    )
    res2_zk = impl_fn(*inp_to_impl_args(inp, fn_type))
    snd_end_time = time.time()
    logging.info(
      f"jax_xc {name} took {snd_end_time - end_time} execution seconds"
    )
    # absolute(res2_zk - res1_zk) <= (atol + rtol * absolute(res1_zk)
    np.testing.assert_allclose(res2_zk, res1_zk, rtol=THRESHOLD, atol=THRESHOLD)

  @parameterized.parameters(
    *pylibxc.util.xc_available_functional_numbers(),
  )
  def test_get_hyb_params(self, name):
    func = pylibxc.LibXCFunctional(name, 1)
    x = ctypes.cast(func.xc_func, ctypes.c_void_p)
    p = libxc.get_p(x.value)
    name = p['name']
    if name in SKIP_LIST:
      logging.info(f"Skipping {name} due to existing in SKIP_LIST")
      return
    if 'func_aux' not in p:
      logging.info(f"Skipping {name} due to not hybrid functional")
      return
    impl_fn = getattr(functionals, name)
    functional = impl_fn(lambda x: 1, False)
    alpha = functional.cam_alpha
    beta = functional.cam_beta
    omega = functional.cam_omega
    nlc_b = functional.nlc_b
    nlc_C = functional.nlc_C
    self.assertTrue(alpha is not None)
    self.assertTrue(beta is not None)
    self.assertTrue(omega is not None)
    self.assertTrue(nlc_b is not None)
    self.assertTrue(nlc_C is not None)


if __name__ == "__main__":
  absltest.main()
