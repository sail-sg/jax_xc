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

"""Repository rule to automatically generate targets from libxc."""

def _impl(rctx):
    rctx.report_progress("Download and extract libxc-6.0.0.tar.gz")
    rctx.download_and_extract(
        url = "https://gitlab.com/libxc/libxc/-/archive/6.0.0/libxc-6.0.0.tar.gz",
        stripPrefix = "libxc-6.0.0",
        sha256 = "0c774e8e195dd92800b9adf3df5f5721e29acfe9af4b191a9937c7de4f9aa9f6",
    )
    rctx.file("WORKSPACE", "")

    # setup build rules for libxc.so
    rctx.delete("src/xc-info.c")
    rctx.delete("src/xc-sanity.c")
    rctx.delete("src/xc-threshold.c")
    rctx.delete("src/test.c")
    rctx.delete("src/genwiki.c")
    rctx.patch(Label("//maple2jax/libxc:so_naming.patch"))
    rctx.execute(["mkdir", "jax_xc"])
    rctx.execute(["mv", "pylibxc", "jax_xc/libxc"])
    rctx.execute(["mv", "src", "jax_xc/libxc/"])
    rctx.execute(["mv", "xc_version.h.in", "jax_xc/libxc/"])
    rctx.symlink(Label("//maple2jax/libxc:wrap.py"), "jax_xc/libxc/wrap.py")
    rctx.symlink(Label("//maple2jax/libxc:wrap.cc.jinja"), "jax_xc/libxc/wrap.cc.jinja")
    rctx.symlink(Label("//maple2jax/libxc:register.cc"), "jax_xc/libxc/src_cc/register.cc")
    rctx.symlink(Label("//maple2jax/libxc:register.h"), "jax_xc/libxc/src_cc/register.h")
    rctx.symlink(Label("//maple2jax/libxc:libxc.cc"), "jax_xc/libxc/src_cc/libxc.cc")
    rctx.symlink(Label("//maple2jax/libxc:gen_build.py"), "jax_xc/libxc/gen_build.py")
    rctx.symlink(Label("//maple2jax/libxc:build.jinja"), "jax_xc/libxc/build.jinja")

    ret = rctx.execute([
        "python3",
        "jax_xc/libxc/gen_build.py",
        "--src",
        "jax_xc/libxc/src",
        "--template",
        "jax_xc/libxc/build.jinja",
        "--build",
        "jax_xc/libxc/BUILD",
    ])
    if ret.return_code != 0:
        fail("Failed to generate BUILD file for libxc.so, " + ret.stderr)

    # setup build rules for impl
    rctx.execute(["mkdir", "-p", "jax_xc/impl"])
    rctx.execute(["mv", "maple", "jax_xc/impl"])
    rctx.symlink(Label("//maple2jax/impl:__init__.py"), "jax_xc/impl/__init__.py")
    rctx.symlink(Label("//maple2jax/impl:gen_build.py"), "jax_xc/impl/gen_build.py")
    rctx.symlink(Label("//maple2jax/impl:gen_maple.py"), "jax_xc/impl/gen_maple.py")
    rctx.symlink(Label("//maple2jax/impl:maple_template.jinja"), "jax_xc/impl/maple_template.jinja")
    rctx.symlink(Label("//maple2jax/impl:gen_py.py"), "jax_xc/impl/gen_py.py")
    rctx.symlink(Label("//maple2jax/impl:python_template.jinja"), "jax_xc/impl/python_template.jinja")
    rctx.symlink(Label("//maple2jax/impl:_helper.py"), "jax_xc/impl/_helper.py")
    rctx.symlink(Label("//maple2jax/impl:build.jinja"), "jax_xc/impl/build.jinja")

    ret = rctx.execute([
        "python3",
        "jax_xc/impl/gen_build.py",
        "--maple",
        "jax_xc/impl/maple",
        "--build",
        "jax_xc/impl/BUILD",
        "--template",
        "jax_xc/impl/build.jinja",
    ])
    if ret.return_code != 0:
        fail("Failed to generate BUILD file for impl, " + ret.stderr)

    # setup build rules for jax_xc
    rctx.symlink(Label("//maple2jax:build.jinja"), "jax_xc/BUILD")
    rctx.symlink(Label("//maple2jax:__init__.py"), "jax_xc/__init__.py")
    rctx.symlink(Label("//maple2jax:gen_py.py"), "jax_xc/gen_py.py")
    rctx.symlink(Label("//maple2jax:utils.py"), "jax_xc/utils.py")
    rctx.symlink(Label("//maple2jax:python_template.jinja"), "jax_xc/python_template.jinja")
    rctx.symlink(Label("//maple2jax:wheel.BUILD"), "BUILD")

maple2jax_repo = repository_rule(_impl, environ = ["GITHUB_ACTIONS"])
