load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

genrule(
    name = "gen_xc_version",
    srcs = ["xc_version.h.in"],
    outs = ["xc_version.h"],
    cmd = "cat $< | sed s/@VERSION@/6.0.0/g | sed s/@XC_MAJOR_VERSION@/6/g | sed s/@XC_MINOR_VERSION@/0/g | sed s/@XC_MICRO_VERSION@/0/g > $@",
)

genrule(
    name = "gen_config",
    srcs = [],
    outs = ["config.h"],
    cmd = """echo '#define PACKAGE_VERSION "6.0.0"\n#include <stdio.h>' > $@""",
)

cc_library(
    name = "xc_inc",
    hdrs = [
        "src/funcs_gga.c",
        "src/funcs_hyb_gga.c",
        "src/funcs_hyb_lda.c",
        "src/funcs_hyb_mgga.c",
        "src/funcs_key.c",
        "src/funcs_lda.c",
        "src/funcs_mgga.c",
        "src/work_gga.c",
        "src/work_gga_inc.c",
        "src/work_lda.c",
        "src/work_lda_inc.c",
        "src/work_mgga.c",
        "src/work_mgga_inc.c",
        "xc_version.h",
        "config.h",
    ] + glob([
        "src_c/*.c",
        "src/maple2c/**/*.c",
        "src/*.h",
        "*.h",
    ]),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "xc_c",
    srcs = glob(
        [
            "src/*.c",
        ],
        exclude = [
            "src/funcs_gga.c",
            "src/funcs_hyb_gga.c",
            "src/funcs_hyb_lda.c",
            "src/funcs_hyb_mgga.c",
            "src/funcs_key.c",
            "src/funcs_lda.c",
            "src/funcs_mgga.c",
            "src/work_gga.c",
            "src/work_gga_inc.c",
            "src/work_lda.c",
            "src/work_lda_inc.c",
            "src/work_mgga.c",
            "src/work_mgga_inc.c",
            "src/test.c",
            "src/genwiki.c",
            "src/xc-info.c",
            "src/xc-sanity.c",
            "src/xc-threshold.c",
        ],
    ),
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
    deps = [":xc_inc"],
)

cc_library(
    name = "xc",
    srcs = glob(
        [
            "src_cc/*.cc",
            "src_cc/*.h",
            ":xc_c",
        ],
        exclude = [
            "src_cc/libxc_params.cc",
        ],
    ),
    copts = ["-std=c++14"],
    includes = [
        ".",
        "src",
        "src_c",
    ],
    linkopts = ["-Wl,-rpath,'$$ORIGIN'"],
    # linkshared = True,
    local_defines = [
        "XC_DONT_COMPILE_VXC",
        "XC_DONT_COMPILE_FXC",
        "XC_DONT_COMPILE_KXC",
        "XC_DONT_COMPILE_LXC",
    ],
    deps = [
        ":xc_c",
        "@pybind11",
        "@visit_struct",
    ],
)

cc_import(
    name = "xc_shared",
    hdrs = [
        "src_cc/register.h",
        "xc_version.h",
        "config.h",
    ] + glob(["src/*.h"]),
    shared_library = ":libxc.so",
)

# pybind_extension(
#     name = "libxc_params",
#     srcs = ["src_cc/libxc_params.cc"],
#     copts = ["-std=c++14"],
#     includes = [
#         ".",
#         "src",
#     ],
#     linkstatic = False,
#     deps = [
#         ":xc",
#         "@visit_struct",
#     ],
# )

pybind_extension(
    name = "libxc_params",
    srcs = [
        ":xc_c",
    ] + glob([
        "src_cc/*.h",
        "src_cc/*.cc",
    ]),
    copts = ["-std=c++14"],
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
    ],
)
