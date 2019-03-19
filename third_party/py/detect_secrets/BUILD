# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

# WARNING: This module assumes that the 'git' binary can be being found in the
# system PATH. This may not be true of all environments, e.g. docker containers.
# Run //third_party/py/git/tests:git_test to test if you're not sure if this
# holds.
py_library(
    name = "detect_secrets",
    srcs = ["detect_secrets.py"],
    deps = [
        requirement("detect-secrets"),
        requirement("PyYAML"),
    ],
)
