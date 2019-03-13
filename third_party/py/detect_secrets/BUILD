# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "detect_secrets",
    srcs = ["//third_party/py:empty.py"],
    deps = [
<<<<<<< HEAD:third_party/py/detect_secrets/BUILD
        requirement("detect-secrets"),
        requirement("PyYAML"),
=======
        requirement("dash"),
        requirement("plotly"),
        requirement("Click"),
        "//third_party/py/flask",
>>>>>>> b6c05513a... Add implicit dependency.:third_party/py/dash/BUILD
    ],
)
