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


def energy_functional(p, impl, deorbitalize=None):
  from autofd.operators import compose, nabla
  from autofd.general_array import (
    with_spec,
    Spec,
    SpecTree,
  )

  # filter 0 density
  def _impl(r, *args):
    dens = r if p.nspin == 1 else r.sum()
    ret = lax.cond(
      (dens < p.dens_threshold), lambda *_: 0., lambda *_: impl(p, r, *args),
      None
    )
    return ret

  # define the energy functional, that takes a rho function
  # and an optional mo function.
  def epsilon_xc(rho: Callable, mo: Optional[Callable] = None):
    r"""epsilon_xc is the xc energy density functional.
    The exchange correlation energy is defined as
    .. raw:: latex

      E_{xc} = \int \rho(r) \epsilon_{xc}[\rho](r) dr

    Therefore the way to use this functional is to

    .. code-block:: python

      energy_density = epsilon_xc(rho)

      def Exc(rho):
        return integrate(compose(mul, energy_density, rho))

      Vxc = jax.grad(Exc)(rho)

    `compose` and `integrate` are operators imported from autofd.

    Args:
      rho: the density function `f[3] -> f[2]` if polarized,
        `f[3] -> f[]` otherwise.
      mo: the molecular orbital function `f[3] -> f[2, nmo]` if polarized,
        `f[3] -> f[nmo]` otherwise.

    Returns:
      The energy density function, `f[3] -> f[2]` if polarized,
      `f[3] -> f[]` otherwise.
    """
    o_spec = SpecTree.from_ret(rho)
    i_spec = SpecTree.from_args(rho)
    dtype = o_spec.dtype
    scalar = Spec((), dtype)
    # Check for any errors
    if o_spec.shape not in ((), (2,)):
      raise RuntimeError(
        "The density function rho passed to the functional "
        "needs to return either a scalar (unpolarized), "
        f"or a shape (2,) array (polarized). Got shape {o_spec.shape}."
      )
    polarized = (o_spec.shape == (2,))
    if (p.nspin == 1 and polarized) or (p.nspin == 2 and not polarized):
      raise ValueError(
        f"The functional is initialized with nspin={p.nspin}, "
        f"while the density function returns array of shape {o_spec.shape}."
      )

    @with_spec((o_spec,), scalar)
    def lda_energy(r):
      return _impl(r)

    if p.type == "lda":
      return compose(lda_energy, rho)

    # 1st order derivative
    nabla_rho = nabla(rho, method=jax.jacrev)

    @with_spec(
      (nabla_rho.ret_spec,),
      Spec((3,) if polarized else (), dtype),
    )
    def compute_s(jac):
      if polarized:
        return jnp.stack(
          [
            jnp.dot(jac[0], jac[0]),
            jnp.dot(jac[0], jac[1]),
            jnp.dot(jac[1], jac[1]),
          ]
        )
      else:
        return jnp.dot(jac, jac)

    @with_spec((o_spec, compute_s.ret_spec), scalar)
    def gga_energy(r, s):
      return _impl(r, s)

    # compute the functional
    if p.type == "gga":
      return compose(
        gga_energy, rho, compose(compute_s, nabla_rho), share_inputs=True
      )

    # 2nd order derivative
    hess_rho = nabla(nabla_rho, method=jax.jacfwd)

    @with_spec(
      (hess_rho.ret_spec,),
      Spec((2,) if polarized else (), dtype),
    )
    def compute_l(hess):
      return jnp.diagonal(hess, axis1=-2, axis2=-1).sum(axis=-1)

    # Now deal with the terms related to mo
    if mo is None:
      raise ValueError(
        "Molecular orbital function are required for mgga functionals."
      )
    mo_o_spec = SpecTree.from_ret(mo)
    mo_i_spec = SpecTree.from_args(mo)
    if mo_i_spec != i_spec:
      raise ValueError("mo must take the same argument as rho.")
    if mo_o_spec.shape != (*(2,) * polarized, mo_o_spec.shape[-1]):
      raise ValueError(
        "mo must return (2, N) if polarized, or (N,) if not. "
        f"Got {mo_o_spec.shape} while polarized={polarized}."
      )
    nabla_mo = nabla(mo, method=jax.jacfwd)

    @with_spec((nabla_mo.ret_spec,), o_spec)
    def compute_tau(mo_jac):
      tau = jnp.sum(jnp.real(jnp.conj(mo_jac) * mo_jac), axis=[-1, -2]) / 2
      return tau

    if deorbitalize is None:
      tau_fn = compose(compute_tau, nabla_mo)
    else:
      tau_fn = rho * deorbitalize(rho, mo)

    # compute the functional
    @with_spec((o_spec, compute_s.ret_spec, compute_l.ret_spec, o_spec), scalar)
    def mgga_energy(r, s, l, tau):
      return _impl(r, s, l, tau)

    return compose(
      mgga_energy,
      rho,
      compose(compute_s, nabla_rho),
      compose(compute_l, hess_rho),
      tau_fn,
      share_inputs=True,
    )

  return epsilon_xc


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
