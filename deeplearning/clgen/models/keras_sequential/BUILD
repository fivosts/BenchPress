py_library(
    name = "keras_sequential",
    srcs = ["keras_sequential.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/clgen/models:backends",
        "//deeplearning/clgen/models:builders",
        "//deeplearning/clgen/models:data_generators",
        "//deeplearning/clgen:samplers",
        "//deeplearning/clgen:telemetry",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8/py:crypto",
        "//labm8/py:humanize",
        "//labm8/py:labdate",
        "//labm8/py:logutil",
        "//labm8/py:pbutil",
        "//third_party/py/keras",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "keras_sequential_test",
    size = "enormous",
    srcs = ["test_keras_sequential"],
    deps = [
        "//deeplearning/clgen/models/keras_sequential:keras_sequential",
        "//deeplearning/clgen/models:models",
        "//deeplearning/clgen:sample_observers",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//deeplearning/clgen/tests:fixtures",
        "//labm8/py:app",
        "//labm8/py:crypto",
        "//labm8/py:fs",
        "//labm8/py:test",
    ],
)
