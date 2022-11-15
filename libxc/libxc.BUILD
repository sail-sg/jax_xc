load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_library")

py_binary(
    name = "wrap",
    srcs = ["wrap.py"],
    visibility = ["//visibility:public"],
)

genrule(
    name = "gen_xc_version",
    srcs = ["xc_version.h.in"],
    outs = ["xc_version.h"],
    cmd = "cat $< | sed s/@VERSION@/6.0.0/g | sed s/@XC_MAJOR_VERSION@/6/g | sed s/@XC_MINOR_VERSION@/0/g | sed s/@XC_MICRO_VERSION@/0/g > $@",
)

genrule(
    name = "gen_config",
    srcs = [],
    outs = ["config.h"],
    cmd = """echo '#define PACKAGE_VERSION "6.0.0"\n#include <stdio.h>' > $@""",
)

cc_library(
    name = "xc_inc",
    hdrs = [
        "xc_version.h",
        "config.h",
    ] + glob([
        "src/maple2c/**/*.c",
        "src/*.c",
        "src/*.h",
        "*.h",
    ]),
    visibility = ["//visibility:public"],
)
