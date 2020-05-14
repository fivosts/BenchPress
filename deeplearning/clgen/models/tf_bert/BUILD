py_library(
    name = "tf_bert",
    srcs = [
        "tf_bert.py",
        "model.py",
        "optimizer.py"
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/clgen/models:backends",
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
        "//labm8/py:pbutil",
        "//third_party/py/numpy",
        "//third_party/py/progressbar",
    ],
)
