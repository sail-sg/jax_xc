load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

rules_python_version = "740825b7f74930c62f44af95c9a4c1bd428d2c53"  # Latest @ 2021-06-23

http_archive(
    name = "rules_python",
    sha256 = "3474c5815da4cb003ff22811a36a11894927eda1c2e64bf2dac63e914bfdf30f",
    strip_prefix = "rules_python-{}".format(rules_python_version),
    url = "https://github.com/bazelbuild/rules_python/archive/{}.zip".format(rules_python_version),
)

http_archive(
    name = "visit_struct",
    build_file = "//third_party/visit_struct:visit_struct.BUILD",
    strip_prefix = "visit_struct-1.0",
    urls = [
        "https://github.com/garbageslam/visit_struct/archive/refs/tags/v1.0.tar.gz",
    ],
)

http_archive(
    name = "pybind11_bazel",
    sha256 = "fec6281e4109115c5157ca720b8fe20c8f655f773172290b03f57353c11869c2",
    strip_prefix = "pybind11_bazel-72cbbf1fbc830e487e3012862b7b720001b70672",
    urls = [
        "https://github.com/pybind/pybind11_bazel/archive/72cbbf1fbc830e487e3012862b7b720001b70672.zip",
    ],
)

# We still require the pybind library.
http_archive(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    strip_prefix = "pybind11-2.10.1",
    urls = ["https://github.com/pybind/pybind11/archive/v2.10.1.tar.gz"],
)

load("@pybind11_bazel//:python_configure.bzl", "python_configure")

python_configure(
    name = "local_config_python",
    python_version = "3",
)

load("//jax_xc:libxc.bzl", "libxc_repo")

libxc_repo(name = "libxc")
