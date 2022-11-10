#!/usr/bin/env python3

import pylibxc
from jax_xc import libxc_params
import ctypes

func=pylibxc.LibXCFunctional("GGA_C_AM05", 1)
x=ctypes.cast(func.xc_func, ctypes.c_void_p)
y=ctypes.cast(func.xc_func_info, ctypes.c_void_p)
out = libxc_params.gga_c_am05_params_to_numpy(x.value)
print(out)

