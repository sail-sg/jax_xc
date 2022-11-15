
def _genrule(src, tgt):
    return """
genrule(
    name = "gen_{}",
    srcs = ["src/{}"],
    outs = ["src_cc/{}"],
    cmd = "$(execpath :wrap) --path $< --out $@ --template $(execpath :wrap.cc.jinja)",
    tools = [":wrap", ":wrap.cc.jinja"],
)
""".format(tgt, src, tgt)

def _impl(rctx):
    rctx.report_progress("Download and extract libxc-6.0.0.tar.gz")
    rctx.download_and_extract(
        url = "https://gitlab.com/libxc/libxc/-/archive/6.0.0/libxc-6.0.0.tar.gz",
        stripPrefix = "libxc-6.0.0",
        sha256 = "0c774e8e195dd92800b9adf3df5f5721e29acfe9af4b191a9937c7de4f9aa9f6",
    )
    rctx.file("WORKSPACE", "")
    rctx.symlink(Label("//libxc:pylibxc.BUILD"), "pylibxc/BUILD")
    rctx.delete("src/xc-info.c")
    rctx.delete("src/xc-sanity.c")
    rctx.delete("src/xc-threshold.c")
    rctx.delete("src/test.c")
    rctx.delete("src/genwiki.c")
    rctx.symlink(Label("//libxc:wrap.py"), "wrap.py")
    rctx.symlink(Label("//libxc:wrap.cc.jinja"), "wrap.cc.jinja")
    rctx.symlink(Label("//libxc:register.cc"), "src_cc/register.cc")
    rctx.symlink(Label("//libxc:register.h"), "src_cc/register.h")
    rctx.symlink(Label("//libxc:libxc.cc"), "src_cc/libxc.cc")
    
    build_content = rctx.read(Label("//libxc:libxc.BUILD"))
    cc = []
    for f in rctx.path("src/").readdir():
        if f.basename.endswith(".c"):
            if (f.basename.startswith("funcs_") or
                f.basename.startswith("work_")):
                continue
            build_content += _genrule(f.basename, f.basename + "c")
            cc += ["\"src_cc/" + f.basename + "c\""]
    pybind = """
cc_binary(
    name = "libxc.so",
    copts = ["-std=c++14", "-fexceptions"],
    features = [
        "-use_header_modules",  # Required for pybind11.
        "-parse_headers",
    ],
    linkshared = 1,
    includes = [
        ".",
        "src",
    ],
    local_defines = [
        "XC_DONT_COMPILE_VXC",
        "XC_DONT_COMPILE_FXC",
        "XC_DONT_COMPILE_KXC",
        "XC_DONT_COMPILE_LXC",
    ],
    deps = [
        ":xc_inc",
        "@visit_struct",
        "@pybind11",
        "@local_config_python//:python_headers",
    ],
    srcs = [
        "src_cc/register.h",
        "src_cc/register.cc",
        "src_cc/libxc.cc",
        {}
    ],
    visibility = ["//visibility:public"],
)
""".format(",\n        ".join(cc))
    build_content += pybind
    rctx.file("BUILD", build_content)

libxc_repo = repository_rule(_impl)
