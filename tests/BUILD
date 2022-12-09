# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
