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
    hdrs = glob([
        "src/maple2c/**/*.c",
        "src/work_*.c",
        "src/funcs_*.c",
    ]) + [
        "src/xc.h",
        "xc_version.h",
        "config.h",
        "src/xc_funcs.h",
        "src/xc_funcs_removed.h",
    ],
    includes = [
        ".",
        "src",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "libxc.so",
    srcs = glob(
        [
            "src/*.c",
            "src/*.h",
        ],
        exclude = [
            "src/work_*.c",
            "src/funcs_*.c",
            "src/test.c",
            "src/xc-*.c",
            "src/genwiki.c",
        ],
    ),
    copts = ["-Wno-unused"],
    includes = [
        ".",
        "src",
    ],
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = [":xc_inc"],
)
