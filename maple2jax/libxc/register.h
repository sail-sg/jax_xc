/*
 * Copyright 2022 Garena Online Private Limited
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef MAPLE2JAX_LIBXC_REGISTER_H_
#define MAPLE2JAX_LIBXC_REGISTER_H_

#include <array>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "visit_struct.hpp"  // NOLINT
#include "xc.h"              // NOLINT

namespace py = pybind11;

typedef void (*init_fn)(xc_func_type*);
typedef std::map<std::string, py::array> (*get_param_fn)(xc_func_type*);

// defined in register.c
extern std::map<init_fn, get_param_fn> registry;
extern std::map<void*, std::string> work_to_maple_name;

static bool Register(init_fn f1, get_param_fn f2) {
  registry[f1] = f2;
  return true;
}

static bool RegisterMaple(void* work, const std::string& maple_name) {
  work_to_maple_name[work] = maple_name;
  return true;
}

template <typename T>
decltype(auto) ToNumpy(const T& a) {
  return py::array(std::array<int, 0>({}), &a);
}

template <typename T, size_t N>
decltype(auto) ToNumpy(const T (&a)[N]) {
  return py::array(
      std::array<int, 1>({N}),
      reinterpret_cast<const typename std::remove_pointer<T>::type*>(a));
}

template <typename T, size_t N, size_t M>
decltype(auto) ToNumpy(const T (&a)[N][M]) {
  return py::array(
      std::array<int, 2>({N, M}),
      reinterpret_cast<const typename std::remove_pointer<T>::type*>(a));
}

#define REGISTER_PARAMS(STRUCT, ...)                                  \
  VISITABLE_STRUCT(STRUCT, __VA_ARGS__);                              \
  static auto STRUCT##_to_numpy(xc_func_type* func) {                 \
    std::map<std::string, py::array> ret;                             \
    visit_struct::for_each(*reinterpret_cast<STRUCT*>(func->params),  \
                           [&](const char* name, const auto& value) { \
                             ret[name] = ToNumpy(value);              \
                           });                                        \
    return ret;                                                       \
  }
#define REGISTER_INIT(INIT, STRUCT) \
  static bool INIT##_registered = Register(INIT, STRUCT##_to_numpy);
// define RGISTER_WORK
#define REGISTER_MAPLE(WORK, MAPLENAME)          \
  static bool WORK##_registered = RegisterMaple( \
      const_cast<void*>(reinterpret_cast<const void*>(&WORK)), MAPLENAME);

#endif  // MAPLE2JAX_LIBXC_REGISTER_H_
