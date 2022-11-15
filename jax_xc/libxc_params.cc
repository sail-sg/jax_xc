#include <map>
#include <string>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "register.h"

std::map<std::string, py::array> get_params(uint64_t xc_func) {
  xc_func_type *func = reinterpret_cast<xc_func_type *>(xc_func);
  return registry[func->info->init](func);
}

PYBIND11_MODULE(libxc_params, m) {
  m.doc() = "Utility to extract libxc params."; // optional module docstring
  m.def("get_params", &get_params);
}
