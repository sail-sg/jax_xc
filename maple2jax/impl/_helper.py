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
import jax.numpy as jnp
import tensorflow_probability as tfp


def my_piecewise3(c, x1, x2):
  """
  #define my_piecewise3(c, x1, x2) ((c) ? (x1) : (x2))
  """
  return jnp.where(c, x1, x2)


def my_piecewise5(c1, x1, c2, x2, x3):
  """
  #define my_piecewise5(c1, x1, c2, x2, x3) ((c1) ? (x1) : ((c2) ? (x2) : (x3)))
  """
  return jnp.where(c1, x1, jnp.where(c2, x2, x3))


def Heaviside(x):
  return jnp.where(x >= 0., 1., 0.)


def xc_E1_scaled(x):
  return jax.scipy.special.exp1(x) * jnp.exp(x)


def lambertw(x):
  return tfp.substrates.jax.math.lambertw(x)


def xc_erfcx(x):
  return jnp.exp(x**2) * jax.scipy.special.erfc(x)
