py_library(
    name = "tf_sequential",
    srcs = [
        "helper.py",
        "tf_sequential.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/clgen/models:backends",
        "//deeplearning/clgen/models:builders",
        "//deeplearning/clgen/models:data_generators",
        "//deeplearning/clgen:sample_observers",
        "//deeplearning/clgen:samplers",
        "//deeplearning/clgen:telemetry",
        "//deeplearning/clgen:tf",
        "//deeplearning/clgen/dashboard:dashboard_db",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8/py:app",
        "//labm8/py:crypto",
        "//labm8/py:gpu_scheduler",
        "//labm8/py:humanize",
        "//labm8/py:labdate",
        "//labm8/py:logutil",
        "//deeplearning/clgen:pbutil",
        "//third_party/py/numpy",
        "//third_party/py/progressbar",
    ],
)

py_test(
    name = "tf_sequential_test",
    timeout = "long",
    srcs = ["test_tf_sequential.py"],
    deps = [
        "//deeplearning/clgen/models:models",
        "//deeplearning/clgen/models/tf_sequential:tf_sequential",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//deeplearning/clgen/tests:fixtures",
        "//labm8/py:app",
        "//labm8/py:crypto",
        "//labm8/py:fs",
        "//labm8/py:test",
    ],
)
