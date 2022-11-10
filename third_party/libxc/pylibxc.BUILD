load("@rules_python//python:defs.bzl", "py_library")

genrule(
    name = "copy_libxc",
    srcs = ["@libxc//:libxc.so"],
    outs = ["libxc.so"],
    cmd = "cp $< $@",
)

py_library(
    name = "pylibxc",
    srcs = glob(
        ["*.py"],
    ),
    data = [":libxc.so"],
    visibility = ["//visibility:public"],
)
