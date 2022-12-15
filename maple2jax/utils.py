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
"""Utilities for the functional implementations."""

import jax
import jax.numpy as jnp
from jax import lax
from . import impl
from . import libxc as pylibxc
from .libxc import libxc
import ctypes
from collections import namedtuple
from typing import NamedTuple, Callable, Optional


class HybridFunctional(Callable):
  cam_alpha: float
  cam_beta: float
  cam_omega: float
  nlc_b: float
  nlc_C: float


def add_hyb_params(p: NamedTuple) -> Callable[[Callable], HybridFunctional]:
  """A decorator that adds the hybrid params to the function.

  Parameters:
  ----------
  p : NamedTuple
      The namedtuple containing the hybrid parameters

  Returns:
  -------
  Callable
      The decorated function
  """

  def decorator(func: Callable) -> HybridFunctional:
    # fraction of full Hartree-Fock exchange, used both for
    # usual hybrids as well as range-separated ones
    func.cam_alpha = p.cam_alpha
    # fraction of short-range only(!) exchange in
    # range-separated hybrids
    func.cam_beta = p.cam_beta
    # the range separation constant
    func.cam_omega = p.cam_omega
    # Non-local correlation, b parameter
    func.nlc_b = p.nlc_b
    # Non-local correlation, C parameter
    func.nlc_C = p.nlc_C
    return func

  return decorator


def functional_name_to_type(fnl_name: str) -> str:
  """Converts a functional name to the type of functional it is.

  Parameters:
  ----------
  fnl_name : str
      The name of the functional

  Returns:
  -------
  str
      The type of functional, one of "lda", "gga", "mgga"
  """
  if fnl_name.startswith("lda") or fnl_name.startswith("hyb_lda"):
    return "lda"
  elif fnl_name.startswith("gga") or fnl_name.startswith("hyb_gga"):
    return "gga"
  elif fnl_name.startswith("mgga") or fnl_name.startswith("hyb_mgga"):
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


def call_functional(
  p: NamedTuple,
  rho: Callable,
  r: jnp.ndarray,
  polarized: bool,
  mo: Optional[Callable] = None
):

  def _invoke_single_functional(
    p: NamedTuple,
    rho: Callable,
    r: jnp.ndarray,
    polarized: bool,
    mo: Optional[Callable] = None
  ):
    # raise error if p.maple_name is not in impl
    if not hasattr(impl, p.maple_name):
      raise ValueError(f"Functional {p.maple_name} not implemented")
    impl_fn = getattr(impl, p.maple_name)
    func = impl_fn.pol if polarized else impl_fn.unpol
    return func(
      *rho_to_arguments(rho, r, polarized, functional_name_to_type(p.name), mo),
      params=p.params,
      p=p
    )

  # If it's hybrid functional
  if p.maple_name == "":
    return sum(
      [
        call_functional(fn_aux_p, rho, r, polarized, mo) * mix_coef
        for fn_aux_p, mix_coef in zip(p.func_aux, p.mix_coef)
      ]
    )
  # If it's deorbitalized functional
  if p.maple_name == "DEORBITALIZE":
    if len(p.func_aux) != 2:
      raise ValueError("DEORBITALIZE must have two auxiliary functionals")
    for fn_aux in p.func_aux:
      if functional_name_to_type(fn_aux.name) != "mgga":
        raise ValueError("deorbitalized functional must be mgga functional")

    res = _invoke_single_functional(p.func_aux[1], rho, r, polarized, mo)
    fn_aux_p = p.func_aux[0]
    # raise error if fn_aux_p.maple_name is not in impl
    if not hasattr(impl, fn_aux_p.maple_name):
      raise ValueError(f"Functional {fn_aux_p.maple_name} not implemented")
    impl_fn = getattr(impl, fn_aux_p.maple_name)
    fn_aux = impl_fn.pol if polarized else impl_fn.unpol

    args = rho_to_arguments(rho, r, polarized, "mgga", mo)

    # Modify tau of args
    if polarized:
      args[-1] = args[1] * res
      args[-2] = args[0] * res
    else:
      args[-1] = args[0] * res

    return fn_aux(*args, params=fn_aux_p.params, p=fn_aux_p)

  # If it's normal functional
  return _invoke_single_functional(p, rho, r, polarized, mo)
