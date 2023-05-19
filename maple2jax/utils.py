# Copyright 2022 Garena Online Private Limited
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
"""Utilities for the functional implementations."""

from . import libxc as pylibxc
from .libxc import libxc
import ctypes
from collections import namedtuple


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
