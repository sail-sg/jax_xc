// Copyright 2022 Garena Online Private Limited
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <array>
#include <iostream>
#include <map>
#include <string>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "register.h"  // NOLINT
#include "util.h"      // NOLINT

REGISTER_MAPLE(xc_deorbitalize_func, std::string("DEORBITALIZE"));

std::map<std::string, py::array> get_params(uint64_t xc_func) {
  xc_func_type* func = reinterpret_cast<xc_func_type*>(xc_func);
  if (registry.count(func->info->init) == 0) {
    return {};
  }
  return registry[func->info->init](func);
}

/**
 * This function will return the name of the original maple file,
 * given the xc_func pointer.
 */
std::string get_maple_name(uint64_t xc_func) {
  xc_func_type* func = reinterpret_cast<xc_func_type*>(xc_func);
  void* key = NULL;
  if (func->info->lda != NULL) {
    key = const_cast<void*>(reinterpret_cast<const void*>(func->info->lda));
  } else if (func->info->gga != NULL) {
    key = const_cast<void*>(reinterpret_cast<const void*>(func->info->gga));
  } else if (func->info->mgga != NULL) {
    key = const_cast<void*>(reinterpret_cast<const void*>(func->info->mgga));
  }

  // special cases
  // TODO: check if libxc fixed this.
  std::string name(xc_functional_get_name(func->info->number));
  if (name == "mgga_x_2d_prhg07_prp10") {
    return "mgga_x_2d_prp10";
  }
  if (name == "hyb_mgga_xc_b98") {
    return "mgga_xc_b98";
  }

  if (func->func_aux != NULL) {
    return "";
  }

  if (key != NULL) {
    if (work_to_maple_name.count(key) == 0) {
      std::string error_msg_name = func->info->name;
      std::string error_msg_number = std::to_string(func->info->number);
      throw std::runtime_error(error_msg_name + " " + error_msg_number);
    } else {
      return work_to_maple_name.at(key);
    }
  }
  // if neither hybrid nor in lda/gga/mgga
  throw std::runtime_error(
      "Functional is neither hybrid nor any of lda/gga/mgga");
}

std::string name_to_type(const std::string& name) {
  if (name.rfind("lda", 0) == 0 || name.rfind("hyb_lda", 0) == 0) {
    return "lda";
  } else if (name.rfind("gga", 0) == 0 || name.rfind("hyb_gga", 0) == 0) {
    return "gga";
  } else if (name.rfind("mgga", 0) == 0 || name.rfind("hyb_mgga", 0) == 0) {
    return "mgga";
  } else {
    throw std::runtime_error("Unknown functional type");
  }
}

py::dict get_p(uint64_t xc_func) {
  xc_func_type* func = reinterpret_cast<xc_func_type*>(xc_func);
  py::dict ret;
  std::string name(xc_functional_get_name(func->info->number));
  ret["name"] = py::str(name);
  ret["cam_omega"] = func->cam_omega;
  ret["cam_alpha"] = func->cam_alpha;
  ret["cam_beta"] = func->cam_beta;
  ret["nlc_b"] = func->nlc_b;
  ret["nlc_C"] = func->nlc_C;
  ret["dens_threshold"] = func->dens_threshold;
  ret["zeta_threshold"] = func->zeta_threshold;
  ret["sigma_threshold"] = func->sigma_threshold;
  ret["tau_threshold"] = func->tau_threshold;
  ret["zeta_threshold"] = func->zeta_threshold;
  ret["sigma_threshold"] = func->sigma_threshold;
  ret["params"] = get_params(xc_func);
  ret["maple_name"] = get_maple_name(xc_func);
  ret["type"] = name_to_type(name);
  ret["nspin"] = func->nspin;
  /*
  if (func->hyb_number_terms > 0) {
    auto hyb_type = py::array(std::array<int, 1>({
          func->hyb_number_terms}), func->hyb_type);
    auto hyb_coeff = py::array(std::array<int, 1>({
          func->hyb_number_terms}), func->hyb_coeff);
    auto hyb_omega = py::array(std::array<int, 1>({
          func->hyb_number_terms}), func->hyb_omega);
    ret["hyb_type"] = hyb_type;
    ret["hyb_coeff"] = hyb_coeff;
    ret["hyb_omega"] = hyb_omega;
  }
  */
  if (func->n_func_aux > 0) {
    py::list l;
    for (int i = 0; i < func->n_func_aux; ++i) {
      l.append(get_p(reinterpret_cast<uint64_t>(func->func_aux[i])));
    }
    ret["func_aux"] = l;
    ret["mix_coef"] =
        py::array(std::array<int, 1>({func->n_func_aux}), func->mix_coef);
  }
  return ret;
}

PYBIND11_MODULE(libxc, m) {
  m.doc() = "Utility to extract libxc params.";  // optional module docstring
  m.def("get_params", &get_params);
  m.def("get_p", &get_p);
}
