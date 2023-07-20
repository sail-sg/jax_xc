#!/usr/bin/env python3
# Copyright 2022 Garena Online Private Limited
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

import jax
import jax.numpy as jnp
import jax_xc
from absl.testing import absltest, parameterized


class _TestGrad(parameterized.TestCase):

  @parameterized.parameters(
    (jax_xc.lda_x,),
    (jax_xc.gga_x_pbe,),
  )
  def test_not_nan(self, xc_fn_fac):
    """Test that the gradient of the functional is not nan."""
    xc_fn = xc_fn_fac(polarized=False)
    r = jnp.array([[10., 10., 10.], [9, 9, 9]])

    def rho(r, s=1):
      return jnp.prod(jax.scipy.stats.norm.pdf(r, loc=0, scale=s))

    def grad(r):
      return jax.grad(xc_fn, argnums=1)(rho, r)

    def param_grad(s):
      return jax.grad(lambda s: xc_fn(lambda r: rho(r, s), r), argnums=1)(s)

    v = jax.vmap(lambda r: xc_fn(rho, r))(r)
    self.assertFalse(jnp.isnan(v).any())
    g = jax.vmap(grad)(r)
    self.assertFalse(jnp.isnan(g).any())
    pg = param_grad(1.5)
    self.assertFalse(jnp.isnan(pg).any())

  # def test_grad_is_not_nan(self):
  #   """Test that the gradient of the functional is not nan."""
  #   xc_fn = jax_xc.lda_x(polarized=False)
  #   r = jnp.array([[10., 10., 10.], [9, 9, 9]])

  #   def rho(r):
  #     return jnp.prod(jax.scipy.stats.norm.pdf(r, loc=0, scale=1))

  #   def grad(r):
  #     return jax.grad(xc_fn, argnums=1)(rho, r)

  #   g = jax.vmap(grad)(r)
  #   self.assertFalse(jnp.isnan(g).any())


if __name__ == "__main__":
  absltest.main()
