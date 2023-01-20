#!/usr/bin/env python3
# Copyright 2022 Garena Online Private Limited
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

from jax_xc import libxc as pylibxc
from jax_xc.libxc import libxc
from absl.testing import absltest, parameterized
from absl import logging
import numpy as np
import ctypes


class _TestGetParams(parameterized.TestCase):

  @parameterized.parameters(
    *pylibxc.util.xc_available_functional_numbers(),
  )
  def test_get_params_unpol(self, name):
    func = pylibxc.LibXCFunctional(name, 1)
    logging.info("Testing %s", func._xc_func_name)
    x = ctypes.cast(func.xc_func, ctypes.c_void_p)
    out = libxc.get_params(x.value)
    self.assertTrue(isinstance(out, dict))
    self.assertTrue(all(map(lambda x: isinstance(x, str), out.keys())))
    self.assertTrue(all(map(lambda x: isinstance(x, np.ndarray), out.values())))

  @parameterized.parameters(
    *pylibxc.util.xc_available_functional_numbers(),
  )
  def test_get_p_unpol(self, name):
    func = pylibxc.LibXCFunctional(name, 1)
    logging.info("Testing %s", func._xc_func_name)
    x = ctypes.cast(func.xc_func, ctypes.c_void_p)
    p = libxc.get_p(x.value)
    logging.info(p)


if __name__ == "__main__":
  absltest.main()
