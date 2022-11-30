load("@rules_python//python:packaging.bzl", "py_package", "py_wheel")

py_wheel(
    name = "jax_xc_wheel",
    distribution = "jax_xc",
    python_tag = "py3",
    version = "0.0.1",
    deps = [
        "@maple2jax//jax_xc",
        "@maple2jax//jax_xc:functionals",
        "@maple2jax//jax_xc:utils",
        "@maple2jax//jax_xc/impl",
        "@maple2jax//jax_xc/libxc",
        "@maple2jax//jax_xc/libxc:libxc.so",
    ],
)
