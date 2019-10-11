# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache v2

py_library(
    name = "fasteners",
    srcs = ["//third_party/py:empty.py"],
    deps = [
<<<<<<< HEAD:third_party/py/fasteners/BUILD
        requirement("fasteners"),
        requirement("monotonic"),  # Implicit dependency.
=======
        requirement("fs"),
        requirement("fs.sshfs"),
        requirement("paramiko"),
        requirement("appdirs"),
        requirement("property-cached"),
>>>>>>> a3d34805d... Add implicit deps to fs package.:third_party/py/fs/BUILD
    ],
)
