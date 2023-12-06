# Copyright 2022 Garena Online Private Limited
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "test_get_param",
    srcs = ["test_get_param.py"],
    deps = [
        "@jax_xc_repo//jax_xc",
    ],
)

py_test(
    name = "test_against_libxc",
    timeout = "eternal",
    srcs = ["test_against_libxc.py"],
    shard_count = 8,
    deps = [
        "@jax_xc_repo//jax_xc",
    ],
)

py_test(
    name = "test_grad",
    srcs = ["test_grad.py"],
    deps = [
        "@jax_xc_repo//jax_xc",
    ],
)
