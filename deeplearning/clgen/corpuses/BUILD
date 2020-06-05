# This package implements corpuses for training CLgen.
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
    name = "atomizers",
    srcs = ["atomizers.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/py:labdate",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "test_atomizers",
    srcs = ["test_atomizers.py"],
    data = [":token_lists"],
    deps = [
        ":atomizers",
        "//deeplearning/clgen/tests:fixtures",
        "//labm8/py:bazelutil",
        "//labm8/py:test",
    ],
)

py_library(
    name = "corpuses",
    srcs = ["corpuses.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":atomizers",
        ":github",
        ":encoded",
        ":preprocessed",
        "//deeplearning/clgen:cache",
        "//deeplearning/clgen/dashboard:dashboard_db",
        "//deeplearning/clgen/preprocessors",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8/py:bazelutil",
        "//labm8/py:crypto",
        "//labm8/py:hashcache",
        "//labm8/py:humanize",
        "//labm8/py:lockfile",
        "//deeplearning/clgen:pbutil",
        "//labm8/py:prof",
        "//third_party/py/checksumdir",
        "//third_party/py/numpy",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "test_corpuses",
    srcs = ["test_corpuses.py"],
    deps = [
        ":corpuses",
        ":encoded",
        ":preprocessed",
        "//deeplearning/clgen/preprocessors:common",
        "//deeplearning/clgen/preprocessors:cxx",
        "//deeplearning/clgen/preprocessors:opencl",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//deeplearning/clgen/tests:fixtures",
        "//labm8/py:fs",
        "//labm8/py:test",
    ],
)

py_library(
    name = "encoded",
    srcs = ["encoded.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":atomizers",
        ":preprocessed",
        "//deeplearning/clgen/preprocessors",
        "//labm8/py:humanize",
        "//labm8/py:sqlutil",
        "//third_party/py/numpy",
        "//third_party/py/progressbar",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "test_encoded",
    srcs = ["test_encoded.py"],
    deps = [
        ":atomizers",
        ":encoded",
        ":preprocessed",
        "//deeplearning/clgen/tests:fixtures",
        "//labm8/py:test",
    ],
)

py_library(
    name = "preprocessed",
    srcs = ["preprocessed.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":github",
        "//deeplearning/clgen/preprocessors",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8/py:crypto",
        "//labm8/py:fs",
        "//labm8/py:sqlutil",
        "//third_party/py/progressbar",
        "//third_party/py/sqlalchemy",
    ],
)

py_library(
    name = "github",
    srcs = ["github.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/py:fs",
    ],
)

filegroup(
    name = "token_lists",
    srcs = ["token_lists.json"],
    visibility = ["//visibility:public"],
)
