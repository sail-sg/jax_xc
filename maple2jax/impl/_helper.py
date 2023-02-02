# Copyright 2022 Garena Online Private Limited
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

import jax
import jax.numpy as jnp
import tensorflow_probability as tfp


def Heaviside(x):
  return jnp.where(x >= 0., 1., 0.)


def xc_E1_scaled(x):
  """
  TODO: This function should not be called due to slow performance.
  https://github.com/google/jax/issues/13543
  """
  return jax.scipy.special.exp1(x) * jnp.exp(x)


def lambertw(x):
  return tfp.substrates.jax.math.lambertw(x)


def xc_erfcx(x):
  return jnp.exp(x**2) * jax.scipy.special.erfc(x)
