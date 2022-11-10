load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_library")

pybind_extension(
    name = "libxc_params",
    srcs = [
        "libxc_params.cc",
    ],
    copts = ["-std=c++14"],
    deps = [
        "@libxc//:xc_inc",
        "@visit_struct",
    ],
)

py_library(
    name = "jax_xc",
    srcs = ["__init__.py"],
    data = [":libxc_params.so"],
    visibility = ["//visibility:public"],
)
