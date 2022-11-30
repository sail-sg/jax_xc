load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
    name = "tmp",
    srcs = ["tmp.py"],
    deps = [
        "@libxc//pylibxc",
    ],
)

py_binary(
    name = "test_jax_xc",
    testonly = 1,
    srcs = ["test_jax_xc.py"],
    deps = [
        "@libxc//pylibxc",
        "@maple2jax//jax_xc",
    ],
)
