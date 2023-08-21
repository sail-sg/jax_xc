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
from jaxtyping import Array


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
  import autofd.operators as o
  from autofd.general_array import (
    SpecTree,
    return_annotation,
    _dtype_to_jaxtyping,
  )

  # filter 0 density
  def _impl(r, s=None, l=None, tau=None):
    dens = r if p.nspin == 1 else r.sum()
    ret = lax.cond(
      (dens < p.dens_threshold), lambda *_: 0.,
      lambda *_: impl(p, r, s, l, tau), None
    )
    return ret

  # define the energy functional, that takes a rho function
  # and an optional mo function.
  def epsilon_xc(rho: Callable, mo: Optional[Callable] = None):
    if p.type == "mgga":
      if mo is None:
        raise ValueError(
          "Molecular orbital function are required for mgga functionals."
        )

    o_spec = SpecTree.from_ret(rho)
    i_spec = SpecTree.from_args(rho)[0]
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

    # compute arguments relating to density
    T = _dtype_to_jaxtyping[o_spec.dtype.name]

    # 0th order
    if p.type == "lda":

      def _energy(r: return_annotation(rho)) -> return_annotation(rho):
        return _impl(r)

      return o.compose(_energy, rho)

    # 1st order
    nabla_rho = o.nabla(rho)

    def compute_s(
      jac: return_annotation(nabla_rho),
    ) -> T[Array, ("3" if polarized else "")]:
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

    s_fn = o.compose(compute_s, nabla_rho)

    # compute the functional
    if p.type == "gga":

      def _energy(r: return_annotation(rho),
                  s: return_annotation(s_fn)) -> return_annotation(rho):
        return _impl(r, s)

      return o.compose(_energy, rho, s_fn, share_inputs=True)

    # 2nd order
    hess_rho = o.nabla(nabla_rho)

    def compute_l(
      hess: return_annotation(hess_rho),
    ) -> T[Array, ("2" if polarized else "")]:
      return jnp.diagonal(hess, axis1=-2, axis2=-1).sum(axis=-1)

    l_fn = o.compose(compute_l, hess_rho)

    # Now deal with mo
    mo_o_spec = SpecTree.from_ret(mo)
    mo_i_spec = SpecTree.from_args(mo)[0]
    if mo_i_spec != i_spec:
      raise ValueError("mo must take the same argument as rho.")
    if mo_o_spec.shape != (*(2,) * polarized, mo_o_spec.shape[1]):
      raise ValueError(
        "mo must return (2, N) if polarized, or (N,) if not. "
        f"Got {mo_o_spec.shape} while polarized={polarized}."
      )
    nabla_mo = o.nabla(mo)

    def compute_tau(
      mo_jac: return_annotation(nabla_mo),
    ) -> return_annotation(rho):
      tau = jnp.sum(mo_jac**2, axis=[-1, -2]) / 2
      return tau

    def compute_tau_deorbitalize(
      density: return_annotation(rho),
      deo: return_annotation(rho),
    ) -> return_annotation(rho):
      return density * deo

    if deorbitalize is None:
      tau_fn = o.compose(compute_tau, nabla_mo)
    else:
      tau_fn = o.compose(
        compute_tau_deorbitalize, rho, deorbitalize(rho, mo), share_inputs=True
      )

    # compute the functional
    def _energy(
      r: return_annotation(rho), s: return_annotation(s_fn),
      l: return_annotation(l_fn), tau: return_annotation(tau_fn)
    ) -> return_annotation(rho):
      return _impl(r, s, l, tau)

    return o.compose(_energy, rho, s_fn, l_fn, tau_fn, share_inputs=True)

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
  mo_jac = jax.jacobian(mo)(r)
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
  tau = jnp.sum(mo_jac**2, axis=[-1, -2]) / 2
  if deorbitalize is not None:
    tau = density * deorbitalize
  return (density, s, ll, tau)
