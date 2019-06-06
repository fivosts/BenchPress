# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

py_library(
    name = "click",
    srcs = ["//third_party/py:empty.py"],
    deps = [
<<<<<<< HEAD:third_party/py/click/BUILD
        requirement("click"),
=======
        requirement("PyGithub"),
        requirement("jwt"),
>>>>>>> 723c493e3... Add implicit dependency.:third_party/py/github/BUILD
    ],
)
