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

  def test_grad_is_not_nan(self):
    """Test that the gradient of the functional is not nan."""
    lda = jax_xc.lda_x(polarized=False)
    r = jnp.array([[10., 10., 10.], [9, 9, 9]])

    def rho(r):
      return jnp.prod(jax.scipy.stats.norm.pdf(r, loc=0, scale=1))

    def grad(r):
      return jax.grad(lda, argnums=1)(rho, r)

    g = jax.vmap(grad)(r)
    self.assertFalse(jnp.isnan(g).any())


if __name__ == "__main__":
  absltest.main()
