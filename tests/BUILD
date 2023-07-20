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
        "@maple2jax//jax_xc",
    ],
)

py_test(
    name = "test_impl",
    srcs = ["test_impl.py"],
    deps = [
        "@maple2jax//jax_xc",
    ],
)

py_test(
    name = "test_grad",
    srcs = ["test_grad.py"],
    deps = [
        "@maple2jax//jax_xc",
    ],
)
