load("@tsl//tsl:tsl.default.bzl", "filegroup")
load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "c_api_headers",
    srcs = ["c_api.h"],
)

filegroup(
    name = "api_base_headers",
    srcs = ["api_base.h"],
)

cc_library(
    name = "api",
    hdrs = ["api.h"],
    deps = [
        ":api_base",
        ":c_api",
    ],
)

cc_library(
    name = "api_base",
    hdrs = [":api_base_headers"],
    visibility = ["//visibility:private"],
    deps = [":c_api"],
)

cc_library(
    name = "c_api",
    hdrs = ["c_api.h"],
)

cc_library(
    name = "c_internal_api",
    hdrs = ["c_internal_api.h"],
    deps = [":c_api"],
)
