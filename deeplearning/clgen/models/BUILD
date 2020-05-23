# This package implements the machine learning models for CLgen.
#
# Copyright (c) 2016-2020 Chris Cummins.
#
# clgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clgen.  If not, see <https://www.gnu.org/licenses/>.

py_library(
    name = "backends",
    srcs = ["backends.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/clgen:samplers",
        "//deeplearning/clgen/corpuses:atomizers",
        "//deeplearning/clgen/dashboard",
        "//deeplearning/clgen/dashboard:dashboard_db",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8/py:cache",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "builders",
    srcs = ["builders.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//deeplearning/clgen:pbutil",
        "//third_party/py/keras",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "builders_test",
    srcs = ["test_builders.py"],
    deps = [
        ":builders",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//deeplearning/clgen/tests:fixtures",
        "//labm8/py:test",
    ],
)

py_library(
    name = "data_generators",
    srcs = ["data_generators.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8/py:humanize",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "data_generators_test",
    srcs = ["test_data_generators.py"],
    deps = [
        ":data_generators",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//deeplearning/clgen/tests:fixtures",
        "//labm8/py:test",
    ],
)

py_library(
    name = "models",
    srcs = ["models.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":builders",
        "//deeplearning/clgen/models/keras_sequential:keras_sequential",
        "//deeplearning/clgen/models/tf_sequential:tf_sequential",
        "//deeplearning/clgen/models/tf_bert:tf_bert",
        "//deeplearning/clgen:cache",
        "//deeplearning/clgen:sample_observers",
        "//deeplearning/clgen:samplers",
        "//deeplearning/clgen:telemetry",
        "//deeplearning/clgen/corpuses",
        "//deeplearning/clgen/dashboard:dashboard_db",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8/py:crypto",
        "//labm8/py:humanize",
        "//labm8/py:lockfile",
        "//deeplearning/clgen:pbutil",
        "//labm8/py:system",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "pretrained",
    srcs = ["pretrained.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":builders",
        "//deeplearning/clgen/models/keras_sequential:keras_sequential",
        "//deeplearning/clgen/models/tf_sequential:tf_sequential",
        "//deeplearning/clgen/models/tf_bert:tf_bert",
        "//deeplearning/clgen:cache",
        "//deeplearning/clgen:sample_observers",
        "//deeplearning/clgen:samplers",
        "//deeplearning/clgen:telemetry",
        "//deeplearning/clgen/corpuses:atomizers",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8/py:cache",
        "//labm8/py:humanize",
        "//labm8/py:labdate",
        "//deeplearning/clgen:pbutil",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "models_test",
    srcs = ["test_models.py"],
    deps = [
        ":models",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//deeplearning/clgen/tests:fixtures",
        "//labm8/py:crypto",
        "//labm8/py:fs",
        "//labm8/py:test",
    ],
)
