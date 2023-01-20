# Copyright 2022 Garena Online Private Limited
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

import jax
import ctypes
import jax.numpy as jnp
from jax.config import config
from absl.testing import absltest, parameterized
from absl import logging
import numpy as np
from jax.tree_util import Partial
import pandas as pd

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


class _TestImpl(parameterized.TestCase):

  def test_impl(self):

    def get_impl_fn_and_inputs(inputs, impl, fn_type, p, polarized):
      rho0, rho1, sigma0, sigma1, sigma2, lapl0, lapl1, tau0, tau1 = inputs
      impl_fn = jax.vmap(Partial(impl, params=p.params, p=p))

      if polarized:
        libxc_input_args = {
          "rho": jnp.stack([rho0, rho1], -1),
          "sigma": jnp.stack([sigma0, sigma1, sigma2], -1),
          "lapl": jnp.stack([lapl0, lapl1], -1),
          "tau": jnp.stack([tau0, tau1], -1),
        }
        if fn_type == "lda":
          fn_input_args = (rho0, rho1)

        elif fn_type == "gga":
          fn_input_args = (rho0, rho1, sigma0, sigma1, sigma2)

        elif fn_type == "mgga":
          fn_input_args = (
            rho0, rho1, sigma0, sigma1, sigma2, lapl0, lapl1, tau0, tau1
          )
      else:
        rho = rho0 + rho1
        sigma = sigma0 + sigma2 + 2 * sigma1
        lapl = lapl0 + lapl1
        tau = tau0 + tau1
        libxc_input_args = dict(rho=rho, sigma=sigma, lapl=lapl, tau=tau)
        if fn_type == "lda":
          fn_input_args = (rho,)
        elif fn_type == "gga":
          fn_input_args = (rho, sigma)
        elif fn_type == "mgga":
          fn_input_args = (rho, sigma, lapl, tau)
      return impl_fn, fn_input_args, libxc_input_args

    for polarized in [False, True]:

      func = pylibxc.LibXCFunctional(name, int(polarized) + 1)
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
      # 9 represent the polarized cases:
      # r0, r1, s0, s1, s2, l0, l1, t0, t1
      inputs = jax.random.uniform(
        keys[0], (9, batch), dtype=jnp.float64, minval=1e-5, maxval=1e2
      )
      # sigma should have: s0 + s2 - 2 * s1 >= 0
      # update slice of inputs[2 : 5, :] to 1 if
      # inputs[2] + inputs[4] - 2 * inputs[3] < 0
      inputs = inputs.at[2:5, :].set(
        jnp.where(inputs[2] + inputs[4] - 2 * inputs[3] < 0, 1, inputs[2:5])
      )

      impl_module = getattr(impl, p.name)
      impl_fn = impl_module.pol if polarized else impl_module.unpol
      impl_fn, input_args, libxc_input_args = get_impl_fn_and_inputs(
        inputs, impl_fn, fn_type, p, polarized
      )
      impl_fn = jax.jit(impl_fn)
      res2_zk = impl_fn(*input_args)

      res1 = func.compute(libxc_input_args)
      res1_zk = res1["zk"].squeeze()

      # absolute(res2_zk - res1_zk) <= (atol + rtol * absolute(res1_zk)
      np.testing.assert_allclose(
        res2_zk, res1_zk, rtol=THRESHOLD, atol=THRESHOLD
      )

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
