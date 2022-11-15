def _impl(rctx):
    rctx.report_progress("Download and extract libxc-6.0.0.tar.gz")
    rctx.download_and_extract(
        url = "https://gitlab.com/libxc/libxc/-/archive/6.0.0/libxc-6.0.0.tar.gz",
        stripPrefix = "libxc-6.0.0",
        sha256 = "0c774e8e195dd92800b9adf3df5f5721e29acfe9af4b191a9937c7de4f9aa9f6",
    )
    rctx.file("WORKSPACE", "")
    rctx.symlink(Label("//jax_xc:wrap.py"), "wrap.py")
    rctx.symlink(Label("//jax_xc:wrap.cc.jinja"), "wrap.cc.jinja")
    rctx.symlink(Label("//jax_xc:register.cc"), "src_cc/register.cc")
    rctx.symlink(Label("//jax_xc:register.h"), "src_cc/register.h")
    rctx.symlink(Label("//jax_xc:libxc_params.cc"), "src_cc/libxc_params.cc")
    rctx.symlink(Label("//jax_xc:libxc.BUILD"), "BUILD")
    ret = rctx.execute(["ls", "-1", "src"])
    files = ret.stdout.split("\n")
    for f in files:
        if f.endswith(".c"):
            rctx.report_progress("Processing " + f)
            ret = rctx.execute(["python3", "wrap.py", "--path", "src/" + f])
            if ret.return_code != 0:
                fail("Failed to wrap " + f + ": " + ret.stderr)
            if len(ret.stdout) > 0:
                rctx.file("src_cc/" + f[:-2] + ".cc", ret.stdout)
                content = rctx.read("src/" + f)
                rctx.file("src_c/" + f, content)
                rctx.delete("src/" + f)

libxc_repo = repository_rule(_impl)
