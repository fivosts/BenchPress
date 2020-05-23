# Top level package of the phd repo.

load("@bazel_gazelle//:def.bzl", "gazelle")
load("@build_stack_rules_proto//python:python_grpc_library.bzl", "python_grpc_library")

config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
    visibility = ["//visibility:public"],
)

# Golang.
# Gazelle directive:
# gazelle:prefix github.com/ChrisCummins/phd
# gazelle:proto disable

gazelle(name = "gazelle")
