# Copyright 2022 Garena Online Private Limited
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.

"""Repository rule to automatically generate targets from libxc."""

def _impl(rctx):
    rctx.path(Label("//gen_repo/libxc:wrap.py"))
    rctx.path(Label("//gen_repo/libxc:wrap.cc.jinja"))
    rctx.path(Label("//gen_repo/libxc:register.cc"))
    rctx.path(Label("//gen_repo/libxc:register.h"))
    rctx.path(Label("//gen_repo/libxc:libxc.cc"))
    rctx.path(Label("//gen_repo/libxc:gen_build.py"))
    rctx.path(Label("//gen_repo/libxc:build.jinja"))
    rctx.path(Label("//gen_repo/impl:__init__.py"))
    rctx.path(Label("//gen_repo/impl:gen_build.py"))
    rctx.path(Label("//gen_repo/impl:gen_maple.py"))
    rctx.path(Label("//gen_repo/impl:maple_template.jinja"))
    rctx.path(Label("//gen_repo/impl:gen_py.py"))
    rctx.path(Label("//gen_repo/impl:utils.py"))
    rctx.path(Label("//gen_repo/impl:python_template.jinja"))
    rctx.path(Label("//gen_repo/impl:build.jinja"))
    rctx.path(Label("//gen_repo:build.jinja"))
    rctx.path(Label("//gen_repo:__init__.py"))
    rctx.path(Label("//gen_repo:gen_py.py"))
    rctx.path(Label("//gen_repo:utils.py"))
    rctx.path(Label("//gen_repo:python_template.jinja"))
    rctx.path(Label("//gen_repo:experimental.jinja"))
    rctx.path(Label("//gen_repo:wheel.BUILD"))

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
    rctx.execute(["mkdir", "jax_xc"])
    rctx.execute(["mv", "pylibxc", "jax_xc/libxc"])
    rctx.execute(["mv", "src", "jax_xc/libxc/"])
    rctx.execute(["mv", "xc_version.h.in", "jax_xc/libxc/"])
    rctx.symlink(Label("//gen_repo/libxc:wrap.py"), "jax_xc/libxc/wrap.py")
    rctx.symlink(Label("//gen_repo/libxc:wrap.cc.jinja"), "jax_xc/libxc/wrap.cc.jinja")
    rctx.symlink(Label("//gen_repo/libxc:register.cc"), "jax_xc/libxc/src_cc/register.cc")
    rctx.symlink(Label("//gen_repo/libxc:register.h"), "jax_xc/libxc/src_cc/register.h")
    rctx.symlink(Label("//gen_repo/libxc:libxc.cc"), "jax_xc/libxc/src_cc/libxc.cc")
    rctx.symlink(Label("//gen_repo/libxc:gen_build.py"), "jax_xc/libxc/gen_build.py")
    rctx.symlink(Label("//gen_repo/libxc:build.jinja"), "jax_xc/libxc/build.jinja")
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
    rctx.symlink(Label("//gen_repo/impl:__init__.py"), "jax_xc/impl/__init__.py")
    rctx.symlink(Label("//gen_repo/impl:gen_build.py"), "jax_xc/impl/gen_build.py")
    rctx.symlink(Label("//gen_repo/impl:gen_maple.py"), "jax_xc/impl/gen_maple.py")
    rctx.symlink(Label("//gen_repo/impl:maple_template.jinja"), "jax_xc/impl/maple_template.jinja")
    rctx.symlink(Label("//gen_repo/impl:gen_py.py"), "jax_xc/impl/gen_py.py")
    rctx.symlink(Label("//gen_repo/impl:utils.py"), "jax_xc/impl/utils.py")
    rctx.symlink(Label("//gen_repo/impl:python_template.jinja"), "jax_xc/impl/python_template.jinja")
    rctx.symlink(Label("//gen_repo/impl:build.jinja"), "jax_xc/impl/build.jinja")
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
    rctx.symlink(Label("//gen_repo:build.jinja"), "jax_xc/BUILD")
    rctx.symlink(Label("//gen_repo:__init__.py"), "jax_xc/__init__.py")
    rctx.symlink(Label("//gen_repo:gen_py.py"), "jax_xc/gen_py.py")
    rctx.symlink(Label("//gen_repo:utils.py"), "jax_xc/utils.py")
    rctx.symlink(Label("//gen_repo:python_template.jinja"), "jax_xc/python_template.jinja")
    rctx.symlink(Label("//gen_repo:experimental.jinja"), "jax_xc/experimental.jinja")
    rctx.symlink(Label("//gen_repo:wheel.BUILD"), "BUILD")

gen_repo = repository_rule(_impl, environ = ["GITHUB_ACTIONS"])
