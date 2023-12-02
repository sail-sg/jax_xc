# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils for the impl of the functionals."""
import jax
from jax import lax
import jax.numpy as jnp
import tensorflow_probability as tfp
from typing import Callable, Optional, NamedTuple


def Heaviside(x):
  return jnp.heaviside(x, 1.)


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


def lax_cond(a, b, c):
  if isinstance(b, int):
    b = float(b)
  if isinstance(c, int):
    c = float(c)
  return lax.cond(a, lambda _: b, lambda _: c, None)


def rho_to_arguments(
  p: NamedTuple,
  rho: Callable,
  r: jnp.ndarray,
  mo: Optional[Callable] = None,
  deorbitalize: Optional[float] = None,
):
  """Converts a density function and coordinate to the arguments
  needed by the impl of the functionals.
  Aka, dens0, dens1, s0, s1, s2, l0, l1, tau0, tau1.
  """
  if p.type not in ["lda", "gga", "mgga"]:
    raise ValueError(
      f"functional_type must be one of 'lda', 'gga', 'mgga', got {p.type}. "
      "This is an internal error of jax_xc, please report it."
    )

  if p.type == "mgga":
    if mo is None:
      raise ValueError(
        "Molecular orbital function are required for mgga functionals"
      )

  # compute density
  density = rho(r)
  if density.shape not in ((), (2,)):
    raise RuntimeError(
      "The density function rho passed to the functional "
      "needs to return either a scalar (unpolarized), "
      f"or a shape (2,) array (polarized). Got shape {density.shape}."
    )
  polarized = (density.shape == (2,))

  if (p.nspin == 1 and polarized) or (p.nspin == 2 and not polarized):
    raise ValueError(
      f"The functional is initialized with nspin={p.nspin}, "
      f"while the density function returns array of shape {density.shape}."
    )

  if p.type == "lda":
    return (density,)

  # compute s
  jac, hvp = jax.linearize(jax.jacrev(rho), r)
  if polarized:
    s0 = jnp.dot(jac[0], jac[0])
    s1 = jnp.dot(jac[0], jac[1])
    s2 = jnp.dot(jac[1], jac[1])
    s = (s0, s1, s2)
  else:
    s = jnp.dot(jac, jac)

  if p.type == "gga":
    return (density, s)

  # compute l
  # normally, r is a 3d vector for a coordinate in real space.
  eye = jnp.eye(r.shape[-1])
  ll = sum([hvp(eye[i])[..., i] for i in range(r.shape[-1])])

  # compute tau
  mo_jac = jax.jacfwd(mo)(r)
  if polarized and mo_jac.shape != (2, mo_jac.shape[1], r.shape[-1]):
    raise ValueError(
      "Since this functional is initialized to be polarized."
      "mo must return an array of shape (2, N), where 2 stands for two spins, "
      "and N is the number of molecular orbitals."
    )
  elif not polarized and mo_jac.shape != (mo_jac.shape[0], r.shape[-1]):
    raise ValueError(
      "Since this functional is initialized to be unpolarized."
      "mo must return an array of shape (N,), where N stands for the number of "
      "molecular orbitals."
    )
  tau = jnp.sum(jnp.real(jnp.conj(mo_jac) * mo_jac), axis=[-1, -2]) / 2
  if deorbitalize is not None:
    tau = density * deorbitalize
  return (density, s, ll, tau)
