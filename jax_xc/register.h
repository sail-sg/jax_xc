#ifndef REGISTER_H_
#define REGISTER_H_

#include <array>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "visit_struct.hpp"
#include "xc.h"

namespace py = pybind11;

typedef void (*init_fn)(xc_func_type *);
typedef std::map<std::string, py::array> (*get_param_fn)(xc_func_type *);

// defined in register.c
extern std::map<init_fn, get_param_fn> registry;
static bool Register(init_fn f1, get_param_fn f2) {
  registry[f1] = f2;
  return true;
}

template <typename T> decltype(auto) ToNumpy(const T &a) {
  return py::array(std::array<int, 0>({}), &a);
}

template <typename T, size_t N> decltype(auto) ToNumpy(const T (&a)[N]) {
  return py::array(std::array<int, 1>({N}), a);
}

template <typename T, size_t N, size_t M>
decltype(auto) ToNumpy(const T (&a)[N][M]) {
  return py::array(std::array<int, 2>({N, M}), a);
}

#define REGISTER_PARAMS(STRUCT, ...)                                           \
  VISITABLE_STRUCT(STRUCT, __VA_ARGS__);                                       \
  static auto STRUCT##_to_numpy(xc_func_type *func) {                          \
    std::map<std::string, py::array> ret;                                      \
    visit_struct::for_each(*reinterpret_cast<STRUCT *>(func->params),          \
                           [&](const char *name, const auto &value) {          \
                             ret[name] = ToNumpy(value);                       \
                           });                                                 \
    return ret;                                                                \
  }
#define REGISTER_INIT(INIT, STRUCT)                                            \
  static bool INIT##_registered = Register(INIT, STRUCT##_to_numpy);

#endif // REGISTER_H_
