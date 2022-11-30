#include <array>
#include <iostream>
#include <map>
#include <string>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "register.h"

std::map<std::string, py::array> get_params(uint64_t xc_func) {
  xc_func_type* func = reinterpret_cast<xc_func_type*>(xc_func);
  if (registry.count(func->info->init) == 0) {
    return {};
  }
  return registry[func->info->init](func);
}

std::string get_maple_name(uint64_t xc_func) {
  xc_func_type* func = reinterpret_cast<xc_func_type*>(xc_func);
  // if func->info->lda is not NULL
  if (func->info->lda != NULL) {
    return work_to_maple_name[const_cast<void*>(
        reinterpret_cast<const void*>(func->info->lda))];
  }
  // if func->info->gga is not NULL
  if (func->info->gga != NULL) {
    return work_to_maple_name[const_cast<void*>(
        reinterpret_cast<const void*>(func->info->gga))];
  }
  // if func->info->mgga is not NULL
  if (func->info->mgga != NULL) {
    return work_to_maple_name[const_cast<void*>(
        reinterpret_cast<const void*>(func->info->mgga))];
  }
  // if hybrid functional, return empty string
  if (func->func_aux != NULL) {
    return "";
  }
  // if neither hybrid nor in lda/gga/mgga
  throw std::runtime_error(
      "Functional is neither hybrid nor any of lda/gga/mgga");
}

py::dict get_p(uint64_t xc_func) {
  xc_func_type* func = reinterpret_cast<xc_func_type*>(xc_func);
  py::dict ret;
  ret["name"] = py::str(xc_functional_get_name(func->info->number));
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
