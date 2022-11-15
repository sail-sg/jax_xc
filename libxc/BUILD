load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_library")

exports_files([
    "wrap.py",
    "wrap.cc.jinja",
])

py_library(
    name = "jax_xc",
    srcs = ["__init__.py"],
    data = [":libxc_params.so"],
    visibility = ["//visibility:public"],
)
