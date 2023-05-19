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

from jax_xc import libxc as pylibxc
from jax_xc.libxc import libxc
from jax_xc import utils, impl, functionals

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

THRESHOLD = 2e-10

# NOT-IMPLEMENTED due to jax's lack of support
SKIP_LIST = [
  "gga_x_fd_lb94",  # Becke-Roussel not having an closed-form expression
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
  "mgga_x_2d_prp10",  # vxc functional, error from libxc
]

names = pylibxc.util.xc_available_functional_names()
impl_names = []
hybrid_names = []

for n in names:
  func = pylibxc.LibXCFunctional(n, 1)
  x = ctypes.cast(func.xc_func, ctypes.c_void_p)
  p = libxc.get_p(x.value)
  p = utils.dict_to_namedtuple(p, "P")
  if (
    p.maple_name not in SKIP_LIST and p.maple_name != "" and
    p.maple_name != "DEORBITALIZE"
  ):
    impl_names.append(p.name)
  if p.maple_name == "":
    hybrid_names.append(p.name)
  assert n == p.name


class _TestImpl(parameterized.TestCase):

  @parameterized.parameters(*impl_names)
  def test_unpol(self, name):
    self._test_impl(name, False)

  @parameterized.parameters(*impl_names)
  def test_pol(self, name):
    self._test_impl(name, True)

  def _test_impl(self, name, polarized):
    batch = 100
    # r0, r1, s0, s1, s2, l0, l1, t0, t1
    inputs = jax.random.uniform(
      jax.random.PRNGKey(10),
      (9, batch),
      dtype=jnp.float64,
      minval=1e-5,
      maxval=1e2,
    )
    inputs = inputs.at[2:5, :].set(
      jnp.where(inputs[2] + inputs[4] - 2 * inputs[3] < 0, 1, inputs[2:5])
    )
    rho0, rho1, sigma0, sigma1, sigma2, lapl0, lapl1, tau0, tau1 = inputs
    func = pylibxc.LibXCFunctional(name, int(polarized) + 1)
    x = ctypes.cast(func.xc_func, ctypes.c_void_p)
    p = libxc.get_p(x.value)
    p = utils.dict_to_namedtuple(p, "P")
    logging.info(
      "Testing %s, implemented by maple file %s", p.name, p.maple_name
    )
    if polarized:
      r, s, l, t = (
        (rho0, rho1), (sigma0, sigma1, sigma2), (lapl0, lapl1), (tau0, tau1)
      )
      libxc_input_args = {
        "rho": jnp.stack([rho0, rho1], -1),
        "sigma": jnp.stack([sigma0, sigma1, sigma2], -1),
        "lapl": jnp.stack([lapl0, lapl1], -1),
        "tau": jnp.stack([tau0, tau1], -1),
      }
    else:
      r, s, l, t = (
        rho0 + rho1, sigma0 + sigma2 + 2 * sigma1, lapl0 + lapl1, tau0 + tau1
      )
      libxc_input_args = {
        "rho": r,
        "sigma": s,
        "lapl": l,
        "tau": t,
      }

    module = getattr(impl, p.maple_name)
    fn = module.pol if polarized else module.unpol
    res2_zk = jax.jit(lambda *args: fn(p, *args))(r, s, l, t)
    res1 = func.compute(libxc_input_args)
    res1_zk = res1["zk"].squeeze()
    # absolute(res2_zk - res1_zk) <= (atol + rtol * absolute(res1_zk)
    np.testing.assert_allclose(res2_zk, res1_zk, rtol=THRESHOLD, atol=THRESHOLD)

  @parameterized.parameters(*hybrid_names)
  def test_get_hyb_params(self, name):
    func = pylibxc.LibXCFunctional(name, 1)
    x = ctypes.cast(func.xc_func, ctypes.c_void_p)
    p = libxc.get_p(x.value)
    p = utils.dict_to_namedtuple(p, "P")
    impl_fn = getattr(functionals, name)
    functional = impl_fn(False)
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
