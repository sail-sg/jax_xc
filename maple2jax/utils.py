"""Utilities for the functional implementations."""

import jax
import jax.numpy as jnp
from jax import lax
from . import impl
from . import libxc as pylibxc
from .libxc import libxc
import ctypes
from collections import namedtuple
from typing import NamedTuple


def fn_name_to_type(fn_name: str) -> str:
  """Converts a functional name to the type of functional it is.

  Parameters:
  ----------
  fn_name : str
      The name of the functional

  Returns:
  -------
  str
      The type of functional, one of "lda", "gga", "mgga"
  """
  if fn_name.startswith("lda") or fn_name.startswith("hyb_lda"):
    return "lda"
  elif fn_name.startswith("gga") or fn_name.startswith("hyb_gga"):
    return "gga"
  elif fn_name.startswith("mgga") or fn_name.startswith("hyb_mgga"):
    return "mgga"
  else:
    raise ValueError("Unknown functional type")


def dict_to_namedtuple(d: dict, name: str):
  """Recursively convert a dict to a namedtuple

  Parameters:
  ----------
  d : dict
      A dictionary obtained from `get_p`
  name : str
      The name of the namedtuple

  Notes:
  ------
  If the dict contains a key "lambda", it will be renamed to "lambda_".
  If the value is a dict, it will be converted to a namedtuple,
  based on the key name. If the value is a list, it will remain a list
  but with elements converted to namedtuples.
  """

  if "lambda" in d:
    d["lambda_"] = d.pop("lambda")

  for k, v in d.items():
    if isinstance(v, dict):
      d[k] = dict_to_namedtuple(v, k)
    elif isinstance(v, (list, tuple)):
      d[k] = [dict_to_namedtuple(i, k) for i in v]

  return namedtuple(name, d.keys())(*d.values())


def get_p(name, polarized, *ext_params):
  func = pylibxc.LibXCFunctional(name, int(polarized) + 1)
  if ext_params:
    func.set_ext_params(ext_params)
  x = ctypes.cast(func.xc_func, ctypes.c_void_p)
  p = libxc.get_p(x.value)

  p = dict_to_namedtuple(p, "P")
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


def _single_fn_call(p: NamedTuple, rho, r, polarized: bool, type: str, mo=None):
  impl_fn = getattr(impl, p.maple_name)
  func = impl_fn.pol if polarized else impl_fn.unpol

  return func(
    *rho_to_arguments(rho, r, polarized, type, mo), params=p.params, p=p
  )


def recursive_fn_call(
  p: NamedTuple, rho, r, polarized: bool, type: str, mo=None
):
  # If it's hybrid functional
  if p.maple_name == "":
    return sum(
      [
        recursive_fn_call(
          fn_aux_p, rho, r, polarized, fn_name_to_type(fn_aux_p.name), mo
        ) * mix_coef for fn_aux_p, mix_coef in zip(p.func_aux, p.mix_coef)
      ]
    )
  # If it's deorbitalized functional
  if p.maple_name == "DEORBITALIZE":
    res = _single_fn_call(
      p.func_aux[1], rho, r, polarized, fn_name_to_type(p.func_aux[1].name), mo
    )
    fn_aux_p = p.func_aux[0]
    impl_fn = getattr(impl, fn_aux_p.maple_name)
    fn_aux = impl_fn.pol if polarized else impl_fn.unpol

    args = rho_to_arguments(
      rho, r, polarized, fn_name_to_type(fn_aux_p.name), mo
    )

    # Modify tau of args
    if polarized:
      args[-1] = args[1] * res
      args[-2] = args[0] * res
    else:
      args[-1] = args[0] * res

    return fn_aux(*args, params=fn_aux_p.params, p=fn_aux_p)

  # If it's a normal functional
  return _single_fn_call(p, rho, r, polarized, type, mo)
