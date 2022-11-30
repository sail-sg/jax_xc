"""Utilities for the functional implementations."""

import jax
import jax.numpy as jnp
from jax import lax
from . import libxc as pylibxc
from .libxc import libxc
import ctypes
from collections import namedtuple


def get_p(name, polarized, *ext_params):
  func = pylibxc.LibXCFunctional(name, int(polarized) + 1)
  if ext_params:
    func.set_ext_params(ext_params)
  x = ctypes.cast(func.xc_func, ctypes.c_void_p)
  p = libxc.get_p(x.value)
  params = p["params"]
  # change lambda to lambda_
  if "lambda" in p:
    p["lambda_"] = p.pop("lambda")
  if "lambda" in params:
    params["lambda_"] = params.pop("lambda")

  params = namedtuple("Params", params.keys())(*params.values())
  p["params"] = params
  p = namedtuple("P", p.keys())(*p.values())

  # TODO: recursively change lambda to lambda_
  return p


def rho_to_arguments(rho, r, polarized, functional_type, mo=None):
  """Converts a density function and coordinate to the arguments
  needed by the impl of the functionals.
  Aka, dens0, dens1, s0, s1, s2, l0, l1, tau0, tau1.
  """
  if functional_type not in ["lda", "gga", "mgga"]:
    raise ValueError("functional_type must be one of 'lda', 'gga', 'mgga'")

  if functional_type == "mgga":
    if mo is None:
      raise ValueError(
        "molecular orbital function are required for mgga functionals"
      )

  ret = []
  # compute density
  dens = rho(r)
  if polarized:
    if dens.shape != (2,):
      raise ValueError(
        "rho must return a array of shape (2,) when polarized=True"
      )
    ret.append(dens[0])
    ret.append(dens[1])
  else:
    if dens.shape != ():
      raise ValueError("rho must return a scalar when polarized=False")
    ret.append(dens)

  if functional_type == "lda":
    return ret

  # shape checking only needs to be done once for the output of rho
  # then we can assume the jac & hess are of the correct shape.

  # compute s
  jac, hvp = jax.linearize(jax.jacrev(rho), r)
  if polarized:
    s0 = jnp.dot(jac[0], jac[0])
    s1 = jnp.dot(jac[0], jac[1])
    s2 = jnp.dot(jac[1], jac[1])
    ret.extend([s0, s1, s2])
  else:
    s = jnp.dot(jac, jac)
    ret.append(s)

  if functional_type == "gga":
    return ret

  # compute l
  eye = jnp.eye(r.shape[0])
  ll = lax.fori_loop(0, 3, lambda i, val: val + hvp(eye[i])[..., i], 0.0)
  if polarized:
    ret.extend([ll[0], ll[1]])
  else:
    ret.append(ll)

  # compute tau
  mo_jac = jax.jacobian(mo)(r)
  tau = jnp.sum(mo_jac**2, axis=[-1, -2]) / 2

  if polarized:
    if mo_jac.shape != (2, mo_jac.shape[1], r.shape[0]):
      raise ValueError(
        "mo must return a array of shape (2, N) when polarized=True"
      )
    ret.extend([tau[0], tau[1]])
  else:
    if mo_jac.shape != (mo_jac.shape[0], r.shape[0]):
      raise ValueError(
        "mo must return a array of shape (N,) when polarized=False"
      )
    ret.append(tau)

  return ret
