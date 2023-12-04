load("@python_abi//:abi.bzl", "abi_tag", "python_tag")
load("@rules_python//python:packaging.bzl", "py_wheel")
load("@bazel_skylib//lib:selects.bzl", "selects")

selects.config_setting_group(
    name = "macos_arm64",
    match_all = [
        "@platforms//os:macos",
        "@platforms//cpu:arm64",
    ],
)

selects.config_setting_group(
    name = "macos_x86_64",
    match_all = [
        "@platforms//os:macos",
        "@platforms//cpu:x86_64",
    ],
)

selects.config_setting_group(
    name = "linux_x86_64",
    match_all = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
)

py_wheel(
    name = "jax_xc_wheel",
    abi = abi_tag(),
    author = "Kunhao Zheng, Min Lin",
    author_email = "zhengkh@sea.com, linmin@sea.com",
    classifiers = [
        "Topic :: Scientific/Engineering :: Chemistry",
        "Programming Language :: Python :: 3",
    ],
    description_file = "@jax_xc//:README.rst",
    distribution = "jax_xc",
    platform = select({
        ":macos_arm64": "macosx_11_0_arm64",
        ":macos_x86_64": "macosx_11_0_x86_64",
        ":linux_x86_64": "manylinux_2_17_x86_64",
        "//conditions:default": "manylinux_2_17_x86_64",
    }),
    python_requires = ">=3.9",
    python_tag = python_tag(),
    requires = [
        "jax",
        "numpy",
        "tensorflow-probability",
        "autofd",
    ],
    version = "0.0.9",
    deps = [
        "@maple2jax//jax_xc",
        "@maple2jax//jax_xc:experimental",
        "@maple2jax//jax_xc:functionals",
        "@maple2jax//jax_xc:utils",
        "@maple2jax//jax_xc/impl",
        "@maple2jax//jax_xc/libxc",
        "@maple2jax//jax_xc/libxc:libxc.so",
    ],
)
