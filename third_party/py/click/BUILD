# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

py_library(
<<<<<<< HEAD:third_party/py/click/BUILD
    name = "click",
=======
    name = "github",
>>>>>>> 10d0bba21... Remove the empty python library files.:third_party/py/github/BUILD
    srcs = ["//third_party/py:empty.py"],
    deps = [
<<<<<<< HEAD:third_party/py/click/BUILD
        requirement("click"),
=======
        requirement("PyGithub"),
<<<<<<< HEAD:third_party/py/click/BUILD
        requirement("jwt"),
>>>>>>> 723c493e3... Add implicit dependency.:third_party/py/github/BUILD
=======
        requirement("jwt"),  # Implicit dependency.
        requirement("cryptography"),  # Implicit dependency.
>>>>>>> fc3608193... Add missing implicit dependency.:third_party/py/github/BUILD
    ],
)
