# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

py_library(
    name = "requests",
<<<<<<< HEAD
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("requests"),
        requirement("chardet"),  # Implicit dependency.
        requirement("certifi"),  # Implicit dependency.
        requirement("idna"),  # Implicit dependency.
        "//third_party/py/urllib3",  # Implicit dependency.
=======
    srcs = ["requests.py"],
    deps = [
        requirement("requests"),
        requirement("idna"),
>>>>>>> 9ef956645... Add explicit requests dependency.
    ],
)
