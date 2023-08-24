# Copyright 2022 Garena Online Private Limited
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

import jax
import jax.numpy as jnp
from jax.config import config
from absl.testing import absltest, parameterized
from absl import logging
import numpy as np

import jax_xc
from jax_xc.utils import get_p
from jax_xc import libxc as pylibxc
from functools import partial
from jaxtyping import Array, Float64

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
lda = []
gga = []
mgga = []

for n in names:
  p = get_p(n, 1)
  assert n == p.name
  if (
    p.maple_name not in SKIP_LIST and p.maple_name != "" and
    p.maple_name != "DEORBITALIZE"
  ):
    if p.name.startswith("mgga") or p.name.startswith("hyb_mgga"):
      mgga.append(p.name)
    elif p.name.startswith("gga") or p.name.startswith("hyb_gga"):
      gga.append(p.name)
    elif p.name.startswith("lda") or p.name.startswith("hyb_lda"):
      lda.append(p.name)


def sigma(rho, r):
  jac = jax.jacfwd(rho)(r)
  if jac.ndim == 2:
    return jnp.stack(
      [
        jnp.dot(jac[0], jac[0]),
        jnp.dot(jac[0], jac[1]),
        jnp.dot(jac[1], jac[1]),
      ]
    )
  else:
    return jnp.dot(jac, jac)


def lapl(rho, r):
  hess = jax.hessian(rho)(r)
  return jnp.diagonal(hess, axis1=-2, axis2=-1).sum(axis=-1)


def tau(rho, mo, r, deorbitalize=None):
  mo_jac = jax.jacfwd(mo)(r)
  if deorbitalize is None:
    tau = jnp.sum(jnp.real(jnp.conj(mo_jac) * mo_jac), axis=[-1, -2]) / 2
  else:
    tau = rho(r) * deorbitalize
  return tau


def rho1(r: Float64[Array, "3"]) -> Float64[Array, ""]:
  return jnp.prod(jax.scipy.stats.norm.pdf(r, loc=0, scale=1))


def rho2(r: Float64[Array, "3"]) -> Float64[Array, ""]:
  return jnp.prod(jax.scipy.stats.cauchy.pdf(r, loc=0, scale=1))


def rho3(r: Float64[Array, "3"]) -> Float64[Array, "2"]:
  return jnp.stack([rho1(r), rho2(r)], axis=0)


def mo1(r: Float64[Array, "3"]) -> Float64[Array, "8"]:
  # create 8 orbitals
  r = r[None, :] * jnp.arange(8)[:, None]
  return jnp.sum(jnp.sin(r), axis=-1) + jnp.sum(jnp.cos(r), axis=-1) * 1.j


def mo2(r: Float64[Array, "3"]) -> Float64[Array, "8"]:
  r = r[None, :] * jnp.arange(8)[:, None] * 2
  return jnp.sum(jnp.sin(r), axis=-1) + jnp.sum(jnp.cos(r), axis=-1) * 1.j


def mo3(r: Float64[Array, "3"]) -> Float64[Array, "2 8"]:
  return jnp.stack([mo1(r), mo2(r)], axis=0)


class _TestAgainstLibxc(parameterized.TestCase):

  @parameterized.parameters(*lda, *gga, *mgga)
  def test_unpol(self, name):
    self._test_impl(name, 0, rho1, mo1)

  @parameterized.parameters(*lda, *gga, *mgga)
  def test_pol(self, name):
    self._test_impl(name, 1, rho3, mo3)

  def _test_impl(self, name, polarized, rho, mo):
    batch = 100
    r = jax.random.uniform(
      jax.random.PRNGKey(42),
      (batch, 3),
      dtype=jnp.float64,
      minval=-3,
      maxval=3,
    )
    rho_r = jax.vmap(rho)(r)
    sigma_r = jax.vmap(partial(sigma, rho))(r)
    lapl_r = jax.vmap(partial(lapl, rho))(r)
    tau_r = jax.vmap(partial(tau, rho, mo))(r)

    # libxc
    func = pylibxc.LibXCFunctional(name, int(polarized) + 1)
    logging.info(
      "Testing %s, implemented by maple file %s", p.name, p.maple_name
    )
    res1 = func.compute(
      {
        "rho": rho_r,
        "sigma": sigma_r,
        "lapl": lapl_r,
        "tau": tau_r,
      },
      do_vxc=False
    )
    res1_zk = res1["zk"].squeeze()

    # jax_xc experimental
    epsilon_xc = getattr(jax_xc.experimental, name)(polarized)
    energy_density = epsilon_xc(rho, mo)
    res2_zk = jax.vmap(energy_density)(r)

    # jax_xc
    epsilon_xc = getattr(jax_xc, name)(polarized)
    energy_density = lambda r: epsilon_xc(rho, r, mo)
    res3_zk = jax.vmap(energy_density)(r)

    # absolute(res2_zk - res1_zk) <= (atol + rtol * absolute(res1_zk)
    np.testing.assert_allclose(res2_zk, res1_zk, rtol=THRESHOLD, atol=THRESHOLD)
    np.testing.assert_allclose(res3_zk, res1_zk, rtol=THRESHOLD, atol=THRESHOLD)


if __name__ == "__main__":
  absltest.main()
