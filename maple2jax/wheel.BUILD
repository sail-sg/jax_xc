load("@python_abi//:abi.bzl", "abi_tag", "python_tag")
load("@rules_python//python:packaging.bzl", "py_wheel")

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
    platform = "manylinux_2_17_x86_64",
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
