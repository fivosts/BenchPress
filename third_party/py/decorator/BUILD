# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

py_library(
    name = "decorator",
    srcs = ["//third_party/py:empty.py"],
    deps = [
<<<<<<< HEAD:third_party/py/decorator/BUILD
        requirement("decorator"),
=======
        requirement("GitPython"),
        requirement("gitdb2"),
>>>>>>> 3f95f099d... Add implicit gitdb dep.:third_party/py/git/BUILD
    ],
)
