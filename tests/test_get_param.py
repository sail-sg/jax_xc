#!/usr/bin/env python3
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
