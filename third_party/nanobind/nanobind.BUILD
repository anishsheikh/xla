licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "nanobind",
    srcs = glob([
        "src/*.cpp",
    ]),
    copts = ["-fexceptions"],
    defines = ["NB_SHARED=1"],
    includes = ["include"],
    textual_hdrs = glob(
        [
            "include/**/*.h",
            "src/*.h",
        ],
    ),
    deps = [
        "@robin_map",
        "@xla//third_party/python_runtime:headers",
    ],
)
