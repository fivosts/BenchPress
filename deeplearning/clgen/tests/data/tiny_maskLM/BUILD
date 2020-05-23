# This package contains the data for a tiny CLgen instance using BERT maskLM corpus processing.

filegroup(
    name = "tiny_maskLM",
    srcs = [
        "config.pbtxt",
        "corpus.tar.bz2",
    ],
    visibility = ["//deeplearning/clgen:__subpackages__"],
)

py_test(
    name = "config_test",
    srcs = ["config_test.py"],
    data = [":tiny_maskLM"],
    deps = [
        "//deeplearning/clgen",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8/py:bazelutil",
        "//deeplearning/clgen:pbutil",
        "//labm8/py:test",
    ],
)
