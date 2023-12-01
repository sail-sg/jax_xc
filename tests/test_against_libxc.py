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
from jaxtyping import Array, Float64, Complex128
from autofd import function

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

THRESHOLD = {
  "mgga_x_br89_explicit": 1e-9,
  "gga_c_op_pw91": 1e-14,
  "lda_x_rel": 1e-13,
  "mgga_x_pjs18": 1e-11,
  "mgga_x_m08": 1e-12,
  "mgga_c_m08": 1e-12,
  "mgga_x_edmgga": 1e-12,
  "mgga_x_ft98": 1e-12,
  "mgga_x_m061": 1e-13,
  "hyb_mgga_x_pjs18": 1e-11,
  "gga_k_meyer": 1e-12,
  "mgga_x_sa_tpss": 1e-13,
  "gga_x_beefvdw": 1e-10,
  "gga_x_pbepow": 1e-10,
  "mgga_c_bc95": 1e-12,
  "mgga_x_m06l": 1e-12,
}

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
sensitive = []

for n in names:
  p = get_p(n, 1)
  assert n == p.name
  if (
    p.maple_name not in SKIP_LIST and p.maple_name != "" and
    p.maple_name != "DEORBITALIZE"
  ):
    if p.maple_name in THRESHOLD:
      sensitive.append((p.name, p.maple_name))
    elif p.name.startswith("mgga") or p.name.startswith("hyb_mgga"):
      mgga.append((p.name, p.maple_name))
    elif p.name.startswith("gga") or p.name.startswith("hyb_gga"):
      gga.append((p.name, p.maple_name))
    elif p.name.startswith("lda") or p.name.startswith("hyb_lda"):
      lda.append((p.name, p.maple_name))


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


@function
def rho1(r: Float64[Array, "3"]) -> Float64[Array, ""]:
  return jnp.prod(jax.scipy.stats.norm.pdf(r, loc=0, scale=1))


@function
def rho2(r: Float64[Array, "3"]) -> Float64[Array, ""]:
  return jnp.prod(jax.scipy.stats.cauchy.pdf(r, loc=0, scale=1))


@function
def rho3(r: Float64[Array, "3"]) -> Float64[Array, "2"]:
  return jnp.stack([rho1(r), rho2(r)], axis=0)


@function
def mo1(r: Float64[Array, "3"]) -> Complex128[Array, "8"]:
  # create 8 orbitals
  r = r[None, :] * jnp.arange(8)[:, None]
  return jnp.sum(jnp.sin(r), axis=-1) + jnp.sum(jnp.cos(r), axis=-1) * 1.j


@function
def mo2(r: Float64[Array, "3"]) -> Complex128[Array, "8"]:
  r = r[None, :] * jnp.arange(8)[:, None] * 2
  return jnp.sum(jnp.sin(r), axis=-1) + jnp.sum(jnp.cos(r), axis=-1) * 1.j


@function
def mo3(r: Float64[Array, "3"]) -> Complex128[Array, "2 8"]:
  return jnp.stack([mo1(r), mo2(r)], axis=0)


class _TestAgainstLibxc(parameterized.TestCase):

  @parameterized.parameters(*lda)
  def test_lda(self, name, maple_name):
    self._test_impl(name, maple_name, 0, rho1)
    self._test_impl(name, maple_name, 1, rho3)

  @parameterized.parameters(*gga)
  def test_gga(self, name, maple_name):
    self._test_impl(name, maple_name, 0, rho1)
    self._test_impl(name, maple_name, 1, rho3)

  @parameterized.parameters(*mgga)
  def test_mgga(self, name, maple_name):
    self._test_impl(name, maple_name, 0, rho1, mo1)
    self._test_impl(name, maple_name, 1, rho3, mo3)

  @parameterized.parameters(*sensitive)
  def test_sensitive(self, name, maple_name):
    self._test_impl(name, maple_name, 0, rho1, mo1)
    self._test_impl(name, maple_name, 1, rho3, mo3)

  def _test_impl(self, name, maple_name, polarized, rho, mo=None):
    threshold = THRESHOLD.get(maple_name, 1e-14)
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
    if mo is not None:
      tau_r = jax.vmap(partial(tau, rho, mo))(r)
    else:
      tau_r = None

    # libxc
    func = pylibxc.LibXCFunctional(name, int(polarized) + 1)
    logging.info("Testing %s, implemented by maple file %s", name, maple_name)
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

    # jax_xc
    epsilon_xc = getattr(jax_xc, name)(polarized)
    energy_density = lambda r: epsilon_xc(rho, r, mo)
    res2_zk = jax.jit(jax.vmap(energy_density))(r)

    # absolute(res2_zk - res1_zk) <= (atol + rtol * absolute(res1_zk)
    np.testing.assert_allclose(res2_zk, res1_zk, rtol=threshold, atol=threshold)

    # jax_xc experimental
    try:
      from autofd import function
      rho = function(rho)
      args = (rho, mo) if mo is not None else (rho,)
      epsilon_xc = getattr(jax_xc.experimental, name)(*args)
      res3_zk = jax.jit(jax.vmap(epsilon_xc))(r)
      np.testing.assert_allclose(
        res3_zk, res1_zk, rtol=threshold, atol=threshold
      )
    except ImportError:
      logging.info("Skipping experimental test because autofd is not found")


if __name__ == "__main__":
  absltest.main()
