# Core Python libraries.
#
<<<<<<< HEAD:labm8/py/BUILD
# Copyright 2014-2020 Chris Cummins <chrisc.101@gmail.com>.
=======
# Copyright 2014-2019 Chris Cummins <chrisc.101@gmail.com>.
>>>>>>> bb562b8d7... Refresh labm8 for new deps.:labm8/BUILD
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
<<<<<<< HEAD:labm8/py/BUILD
=======

load("//tools/bzl:exports.bzl", "exports_pip", "exports_repo")

licenses(["notice"])  # Apache 2.0

exports_files([
    "LICENSE",
    "README.md",
    "CNAME",
])
>>>>>>> bb562b8d7... Refresh labm8 for new deps.:labm8/BUILD

exports_repo(
    name = "export",
    extra_files = [
        "labm8/labm8.jpg",
    ],
    github_repo = "labm8",
    move_file_mapping = {
        "labm8/README.md": "README.md",
        "labm8/LICENSE": "LICENSE",
        "labm8/travis.yml": ".travis.yml",
    },
    targets = ["//labm8:all"],
)

exports_pip(
    name = "deploy",
    package_name = "labm8",
    classifiers = [
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description = "Utility libraries for doing science",
    keywords = [
        "utility",
        "library",
        "bazel",
        "protobuf",
    ],
    license = "Apache License, Version 2.0",
    long_description_file = "//labm8:README.md",
    package_root = "//labm8",
)

py_library(
    name = "app",
    srcs = ["app.py"],
    data = ["//labm8/py/internal:workspace_status"],
    visibility = ["//visibility:public"],
    deps = [
<<<<<<< HEAD:labm8/py/BUILD
<<<<<<< HEAD:labm8/py/BUILD
        ":shell",
        "//labm8/py/internal:flags_parsers",
        "//labm8/py/internal:labm8_logging",
=======
=======
        ":shell",
>>>>>>> 13ac44cd6... Add missing shell module.:labm8/BUILD
        "//:build_info",
        "//labm8/internal:flags_parsers",
        "//labm8/internal:logging",
>>>>>>> 996b7e72c... Move config and build_info to top level package.:labm8/BUILD
        "//third_party/py/absl",
    ],
)

py_library(
    name = "app_test_flags",
    srcs = ["app_test_flags.py"],
    deps = [
        ":app",
        ":sqlutil",
    ],
)

py_test(
    name = "app_test",
    srcs = ["app_test.py"],
    deps = [
        ":app",
        ":app_test_flags",
        ":test",
    ],
)

py_library(
    name = "archive",
    srcs = ["archive.py"],
<<<<<<< HEAD:labm8/py/BUILD
    visibility = ["//visibility:public"],
=======
    deps = [
        "//third_party/py/absl",
    ],
>>>>>>> bd678d383... Auto BUILD format.:labm8/BUILD
)

py_test(
    name = "archive_test",
    srcs = ["archive_test.py"],
    deps = [
        ":app",
        ":archive",
<<<<<<< HEAD:labm8/py/BUILD
<<<<<<< HEAD:labm8/py/BUILD
        ":test",
=======
        "//third_party/py/absl",
        "//third_party/py/pytest",
>>>>>>> bd678d383... Auto BUILD format.:labm8/BUILD
=======
        "//labm8:test",
        "//third_party/py/absl",
>>>>>>> ad2622a47... Port tests to //labm8:test.:labm8/BUILD
    ],
)

py_library(
    name = "bazelutil",
    srcs = ["bazelutil.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":archive",
<<<<<<< HEAD:labm8/py/BUILD
<<<<<<< HEAD:labm8/py/BUILD
<<<<<<< HEAD:labm8/py/BUILD
        ":fs",
=======
        "//config:getconfig",
>>>>>>> 0e24e1d99... Add a labm8.bazelutil.DataArchive class.:labm8/BUILD
=======
>>>>>>> 996b7e72c... Move config and build_info to top level package.:labm8/BUILD
=======
        ":fs",
>>>>>>> 7851b2a76... Add extra dependencies on //labm8:fs.:labm8/BUILD
    ],
)

py_test(
    name = "bazelutil_test",
    size = "small",
    srcs = ["bazelutil_test.py"],
<<<<<<< HEAD:labm8/py/BUILD
    data = ["//labm8/py/test_data:hello_world_file"],
=======
    data = ["//labm8/test_data:hello_world_file"],
>>>>>>> bb562b8d7... Refresh labm8 for new deps.:labm8/BUILD
    deps = [
        ":app",
        ":bazelutil",
        ":fs",
<<<<<<< HEAD:labm8/py/BUILD
        ":test",
=======
        "//labm8:test",
        "//third_party/py/absl",
>>>>>>> ad2622a47... Port tests to //labm8:test.:labm8/BUILD
    ],
)

py_library(
    name = "cache",
    srcs = ["cache.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":crypto",
        ":fs",
        ":io",
    ],
)

py_test(
    name = "cache_test",
    size = "small",
    srcs = ["cache_test.py"],
<<<<<<< HEAD:labm8/py/BUILD
    data = ["//labm8/py/test_data:jsoncache_file"],
=======
    data = ["//labm8/test_data:jsoncache_file"],
>>>>>>> bb562b8d7... Refresh labm8 for new deps.:labm8/BUILD
    deps = [
        ":app",
        ":cache",
        ":fs",
<<<<<<< HEAD:labm8/py/BUILD
        ":test",
=======
        "//labm8:test",
        "//third_party/py/absl",
>>>>>>> ad2622a47... Port tests to //labm8:test.:labm8/BUILD
    ],
)

py_library(
    name = "crypto",
    srcs = ["crypto.py"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "crypto_test",
    size = "small",
    srcs = ["crypto_test.py"],
    data = [
<<<<<<< HEAD:labm8/py/BUILD
        "//labm8/py/test_data:empty_file_file",
        "//labm8/py/test_data:hello_world_file",
=======
        "//labm8/test_data:empty_file_file",
        "//labm8/test_data:hello_world_file",
>>>>>>> bb562b8d7... Refresh labm8 for new deps.:labm8/BUILD
    ],
    deps = [
        ":app",
        ":crypto",
<<<<<<< HEAD:labm8/py/BUILD
        ":test",
=======
        "//labm8:test",
        "//third_party/py/absl",
>>>>>>> ad2622a47... Port tests to //labm8:test.:labm8/BUILD
    ],
)

py_library(
    name = "graph",
    srcs = ["graph.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":app",
        ":fmt",
    ],
)

py_library(
    name = "hashcache",
    srcs = ["hashcache.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":app",
        ":crypto",
        ":fs",
        ":sqlutil",
        "//labm8/py:humanize",
        "//third_party/py/checksumdir",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "hashcache_test",
    size = "small",
    srcs = ["hashcache_test.py"],
    deps = [
        ":app",
        ":hashcache",
<<<<<<< HEAD:labm8/py/BUILD
        ":test",
    ],
)

py_library(
    name = "humanize",
    srcs = ["humanize.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/humanize",
    ],
)

py_test(
    name = "humanize_test",
    size = "small",
    srcs = ["humanize_test.py"],
    deps = [
        ":app",
        ":humanize",
        ":labdate",
        ":test",
=======
        "//labm8:test",
        "//third_party/py/absl",
>>>>>>> ad2622a47... Port tests to //labm8:test.:labm8/BUILD
    ],
)

py_library(
    name = "decorators",
    srcs = ["decorators.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":app",
        ":humanize",
    ],
)

py_test(
    name = "decorators_test",
    srcs = ["decorators_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":app",
        ":decorators",
<<<<<<< HEAD:labm8/py/BUILD
        ":test",
        "//third_party/py/pytest",
=======
        "//labm8:test",
        "//third_party/py/absl",
>>>>>>> ad2622a47... Port tests to //labm8:test.:labm8/BUILD
    ],
)

py_library(
    name = "dockerutil",
    srcs = ["dockerutil.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":app",
        ":bazelutil",
        ":labtypes",
    ],
)

py_test(
    name = "dockerutil_test",
    srcs = ["dockerutil_test.py"],
    data = [
        "//labm8/py/test_data:basic_app.tar",
    ],
    shard_count = 3,
    visibility = ["//visibility:public"],
    deps = [
        ":dockerutil",
        ":system",
        ":test",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "dockerutil",
    srcs = ["dockerutil.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":app",
        ":bazelutil",
        ":labtypes",
    ],
)

py_test(
    name = "dockerutil_test",
    srcs = ["dockerutil_test.py"],
    data = [
        "//labm8/test_data:basic_app.tar",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":dockerutil",
        ":system",
        ":test",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "fs",
    srcs = ["fs.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":humanize",
        ":labtypes",
        "//third_party/py/send2trash",
    ],
)

py_test(
    name = "fs_test",
    size = "small",
    srcs = ["fs_test.py"],
    data = [
<<<<<<< HEAD:labm8/py/BUILD
        "//labm8/py/test_data:data1_file",
        "//labm8/py/test_data:empty_file_file",
        "//labm8/py/test_data:hello_world_file",
        "//labm8/py/test_data:testdir_files",
=======
        "//labm8/test_data:data1_file",
        "//labm8/test_data:empty_file_file",
        "//labm8/test_data:hello_world_file",
        "//labm8/test_data:testdir_files",
>>>>>>> bb562b8d7... Refresh labm8 for new deps.:labm8/BUILD
    ],
    deps = [
        ":app",
        ":fs",
        ":system",
<<<<<<< HEAD:labm8/py/BUILD
        ":test",
=======
        "//labm8:test",
        "//third_party/py/absl",
>>>>>>> ad2622a47... Port tests to //labm8:test.:labm8/BUILD
    ],
)

py_library(
    name = "fmt",
    srcs = ["fmt.py"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//third_party/py/pandas",
    ],
)

py_test(
    name = "fmt_test",
    size = "small",
    srcs = ["fmt_test.py"],
    deps = [
        ":app",
        ":fmt",
<<<<<<< HEAD:labm8/py/BUILD
        ":test",
    ],
)

py_library(
    name = "google_sheets",
    srcs = ["google_sheets.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":app",
        "//third_party/py/gspread",
        "//third_party/py/oauth2client",
    ],
)

py_library(
    name = "gpu_scheduler",
    srcs = ["gpu_scheduler.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":app",
        ":humanize",
        "//third_party/py/appdirs",
        "//third_party/py/fasteners",
        "//third_party/py/gputil",
    ],
)

py_test(
    name = "gpu_scheduler_test",
    srcs = ["gpu_scheduler_test.py"],
    deps = [
        ":gpu_scheduler",
        ":test",
=======
        "//labm8:test",
        "//third_party/py/absl",
>>>>>>> ad2622a47... Port tests to //labm8:test.:labm8/BUILD
    ],
)

py_library(
    name = "io",
    srcs = ["io.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":labtypes",
        ":system",
    ],
)

py_test(
    name = "io_test",
    size = "small",
    srcs = ["io_test.py"],
    deps = [
        ":app",
        ":io",
        ":system",
<<<<<<< HEAD:labm8/py/BUILD
        ":test",
=======
        "//labm8:test",
        "//third_party/py/absl",
>>>>>>> ad2622a47... Port tests to //labm8:test.:labm8/BUILD
    ],
)

py_library(
    name = "jsonutil",
    srcs = ["jsonutil.py"],
    visibility = ["//visibility:public"],
    deps = [":fs"],
)

py_test(
    name = "jsonutil_test",
    size = "small",
    srcs = ["jsonutil_test.py"],
    deps = [
        ":app",
        ":fs",
        ":jsonutil",
        ":system",
<<<<<<< HEAD:labm8/py/BUILD
        ":test",
=======
        "//labm8:test",
        "//third_party/py/absl",
>>>>>>> ad2622a47... Port tests to //labm8:test.:labm8/BUILD
    ],
)

py_library(
    name = "labdate",
    srcs = ["labdate.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/pytz",
    ],
)

py_test(
    name = "labdate_test",
    size = "small",
    srcs = ["labdate_test.py"],
    deps = [
        ":app",
        ":labdate",
<<<<<<< HEAD:labm8/py/BUILD
        ":test",
=======
        "//labm8:test",
        "//third_party/py/absl",
>>>>>>> ad2622a47... Port tests to //labm8:test.:labm8/BUILD
    ],
)

py_library(
    name = "labmath",
    srcs = ["labmath.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":system",
        "//third_party/py/numpy",
        "//third_party/py/scipy",
    ],
)

py_test(
    name = "labmath_test",
    size = "small",
    srcs = ["labmath_test.py"],
    deps = [
        ":app",
        ":labmath",
<<<<<<< HEAD:labm8/py/BUILD
        ":test",
=======
        "//labm8:test",
        "//third_party/py/absl",
>>>>>>> ad2622a47... Port tests to //labm8:test.:labm8/BUILD
    ],
)

py_library(
    name = "labtypes",
    srcs = ["labtypes.py"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "labtypes_test",
    size = "small",
    srcs = ["labtypes_test.py"],
    deps = [
        ":app",
        ":labtypes",
<<<<<<< HEAD:labm8/py/BUILD
        ":test",
=======
        "//labm8:test",
        "//third_party/py/absl",
>>>>>>> ad2622a47... Port tests to //labm8:test.:labm8/BUILD
    ],
)

py_library(
    name = "latex",
    srcs = ["latex.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":io",
        "//third_party/py/pandas",
    ],
)

py_test(
    name = "latex_test",
    size = "small",
    srcs = ["latex_test.py"],
    deps = [
        ":app",
        ":latex",
<<<<<<< HEAD:labm8/py/BUILD
        ":test",
=======
        "//labm8:test",
        "//third_party/py/absl",
>>>>>>> ad2622a47... Port tests to //labm8:test.:labm8/BUILD
    ],
)

py_library(
    name = "lockfile",
    srcs = ["lockfile.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":app",
        ":humanize",
        ":labdate",
        ":pbutil",
        ":system",
<<<<<<< HEAD:labm8/py/BUILD
        "//labm8/py/internal:lockfile_pb_py",
=======
        "//labm8/internal:lockfile_pb_py",
>>>>>>> bb562b8d7... Refresh labm8 for new deps.:labm8/BUILD
    ],
)

py_test(
    name = "lockfile_test",
    size = "small",
    srcs = ["lockfile_test.py"],
    deps = [
        ":lockfile",
        #        ":fs",
        ":pbutil",
<<<<<<< HEAD:labm8/py/BUILD
        ":test",
<<<<<<< HEAD:labm8/py/BUILD
        # "//labm8/py/internal:lockfile_pb_py",
=======
        "//labm8:test",
        "//labm8/proto:lockfile_py_pb2",
        "//third_party/py/absl",
>>>>>>> ad2622a47... Port tests to //labm8:test.:labm8/BUILD
=======
        # "//labm8/internal:lockfile_pb_py",
>>>>>>> bb562b8d7... Refresh labm8 for new deps.:labm8/BUILD
    ],
)

py_library(
    name = "logutil",
    srcs = ["logutil.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":app",
        ":labdate",
<<<<<<< HEAD:labm8/py/BUILD
        "//labm8/py/internal:logging_pb_py",
=======
        "//labm8/internal:logging_pb_py",
>>>>>>> bb562b8d7... Refresh labm8 for new deps.:labm8/BUILD
        "//third_party/py/absl",
    ],
)

py_test(
    name = "logutil_test",
    size = "small",
    srcs = ["logutil_test.py"],
    deps = [
        ":app",
        ":logutil",
<<<<<<< HEAD:labm8/py/BUILD
        ":test",
=======
        "//labm8:test",
        "//third_party/py/absl",
>>>>>>> ad2622a47... Port tests to //labm8:test.:labm8/BUILD
    ],
)

py_library(
    name = "make",
    srcs = ["make.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":fs",
        ":system",
    ],
)

py_test(
    name = "make_test",
    size = "small",
    srcs = ["make_test.py"],
<<<<<<< HEAD:labm8/py/BUILD
    data = ["//labm8/py/test_data:makeproj_files"],
=======
    data = ["//labm8/test_data:makeproj_files"],
>>>>>>> bb562b8d7... Refresh labm8 for new deps.:labm8/BUILD
    deps = [
        ":app",
        ":fs",
        ":make",
<<<<<<< HEAD:labm8/py/BUILD
<<<<<<< HEAD:labm8/py/BUILD
        ":test",
=======
=======
        "//labm8:test",
>>>>>>> ad2622a47... Port tests to //labm8:test.:labm8/BUILD
        "//third_party/py/absl",
    ],
)

<<<<<<< HEAD:labm8/py/BUILD
# A Jupyter notebook launcher that includes a bunch of dependencies which I
# find useful. If you would like to include extra dependencies, create a new
# py_binary target with this file as the srcs, and add the dependencies, like
# this:
#
#    py_binary(
#      name = "notebooks",
#      srcs = ["//labm8:notebooks"],
#      deps = [
#          # Your dependencies here.
#      ],
#    )
py_binary(
    name = "notebooks",
    srcs = ["notebooks.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":bazelutil",
        ":labdate",
        ":pbutil",
        ":sqlutil",
        ":viz",
        "//third_party/py/absl",
        "//third_party/py/bokeh",
        "//third_party/py/graph_nets",
        "//third_party/py/jupyter",
        "//third_party/py/keras",
        "//third_party/py/matplotlib",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/seaborn",
<<<<<<< HEAD:labm8/py/BUILD
>>>>>>> 31e381197... Add new third party python packages.:labm8/BUILD
=======
        "//third_party/py/sonnet",
        "//third_party/py/tensorflow",
>>>>>>> 8f8ba2553... Add ML packages to labm8 notebook server.:labm8/BUILD
    ],
)

=======
>>>>>>> 77bad6326... Move //labm8:notebooks to //notebooks.:labm8/BUILD
py_library(
    name = "pbutil",
    srcs = ["pbutil.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/protobuf",
    ],
)

py_test(
    name = "pbutil_test",
    srcs = ["pbutil_test.py"],
    deps = [
        ":pbutil",
<<<<<<< HEAD:labm8/py/BUILD
        ":test",
<<<<<<< HEAD:labm8/py/BUILD
        "//labm8/py/test_data:test_protos_pb_py",
    ],
)

py_library(
    name = "pdutil",
    srcs = ["pdutil.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":app",
        ":sqlutil",
        "//third_party/py/pandas",
        "//third_party/py/tabulate",
    ],
)

py_test(
    name = "pdutil_test",
    srcs = ["pdutil_test.py"],
    deps = [
        ":pdutil",
        ":sqlutil",
        ":test",
        "//third_party/py/sqlalchemy",
=======
        "//labm8:test",
        "//labm8/proto:test_protos",
        "//third_party/py/absl",
>>>>>>> ad2622a47... Port tests to //labm8:test.:labm8/BUILD
=======
        "//:config_pb_py",
        "//labm8/test_data:test_protos_pb_py",
>>>>>>> bb562b8d7... Refresh labm8 for new deps.:labm8/BUILD
    ],
)

py_library(
    name = "pdutil",
    srcs = ["pdutil.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":app",
        ":sqlutil",
        "//third_party/py/pandas",
    ],
)

py_test(
    name = "pdutil_test",
    srcs = ["pdutil_test.py"],
    deps = [
        ":pdutil",
        ":sqlutil",
        ":test",
        "//third_party/py/sqlalchemy",
    ],
)

py_library(
    name = "ppar",
    srcs = ["ppar.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":app",
        ":bazelutil",
        ":pbutil",
        ":sqlutil",
        "//labm8/py:humanize",
    ],
)

py_test(
    name = "ppar_test",
    srcs = ["ppar_test.py"],
    data = [
<<<<<<< HEAD:labm8/py/BUILD
        "//labm8/py/test_data/ppar:proto_worker",
        "//labm8/py/test_data/ppar:proto_worker_requires_args",
        "//labm8/py/test_data/ppar:unexpected_output_proto_worker",
=======
        "//labm8/test_data/ppar:proto_worker",
        "//labm8/test_data/ppar:proto_worker_requires_args",
        "//labm8/test_data/ppar:unexpected_output_proto_worker",
>>>>>>> bb562b8d7... Refresh labm8 for new deps.:labm8/BUILD
    ],
    deps = [
        ":ppar",
<<<<<<< HEAD:labm8/py/BUILD
        ":test",
<<<<<<< HEAD:labm8/py/BUILD
        "//labm8/py/test_data/ppar:protos_pb_py",
=======
        "//labm8:test",
        "//labm8/data/test/ppar:protos_py_pb2",
        "//third_party/py/absl",
>>>>>>> ad2622a47... Port tests to //labm8:test.:labm8/BUILD
=======
        "//labm8/test_data/ppar:protos_pb_py",
>>>>>>> bb562b8d7... Refresh labm8 for new deps.:labm8/BUILD
        "//third_party/py/progressbar",
    ],
)

py_library(
    name = "prof",
    srcs = ["prof.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":app",
        ":humanize",
        ":labdate",
        ":labtypes",
        ":system",
    ],
)

py_test(
    name = "prof_test",
    size = "small",
    srcs = ["prof_test.py"],
    deps = [
        ":app",
        ":prof",
<<<<<<< HEAD:labm8/py/BUILD
        ":test",
    ],
)

py_library(
    name = "progress",
    srcs = ["progress.py"],
    visibility = ["//visibility:public"],
<<<<<<< HEAD:labm8/py/BUILD
    deps = [
        ":app",
        ":prof",
        "//third_party/py/tqdm",
    ],
=======
>>>>>>> 13ac44cd6... Add missing shell module.:labm8/BUILD
)

py_test(
    name = "progress_test",
    srcs = ["progress_test.py"],
    shard_count = 4,
    deps = [
        ":progress",
        ":test",
=======
        ":system",
        "//labm8:test",
        "//third_party/py/absl",
>>>>>>> ad2622a47... Port tests to //labm8:test.:labm8/BUILD
    ],
)

py_library(
    name = "shell",
    srcs = ["shell.py"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "shell_test",
    srcs = ["shell_test.py"],
    deps = [
        ":shell",
        ":test",
    ],
)

py_library(
    name = "sqlutil",
    srcs = ["sqlutil.py"],
    visibility = ["//visibility:public"],
    deps = [
<<<<<<< HEAD:labm8/py/BUILD
        ":humanize",
        ":labdate",
        ":pbutil",
<<<<<<< HEAD:labm8/py/BUILD
        ":progress",
=======
        ":humanize",
>>>>>>> ae1df45af... Add a pluralized table name generator.:labm8/BUILD
        ":text",
=======
        ":pbutil",
>>>>>>> 5e7765450... Add SQLAlchemy utility mixins.:labm8/BUILD
        "//third_party/py/absl",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "sqlutil_test",
    size = "small",
    srcs = ["sqlutil_test.py"],
<<<<<<< HEAD:labm8/py/BUILD
    deps = [
        ":sqlutil",
        ":test",
<<<<<<< HEAD:labm8/py/BUILD
        "//labm8/py/test_data:test_protos_pb_py",
=======
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":sqlutil",
        "//labm8:test",
        "//labm8/proto:test_protos",
        "//third_party/py/absl",
<<<<<<< HEAD:labm8/py/BUILD
        "//third_party/py/pytest",
>>>>>>> 4a3afd109... Add a CreateEngine() function.:labm8/BUILD
=======
>>>>>>> ad2622a47... Port tests to //labm8:test.:labm8/BUILD
=======
        "//labm8/test_data:test_protos_pb_py",
>>>>>>> bb562b8d7... Refresh labm8 for new deps.:labm8/BUILD
    ],
)

py_library(
    name = "system",
    srcs = ["system.py"],
    visibility = ["//visibility:public"],
<<<<<<< HEAD:labm8/py/BUILD
<<<<<<< HEAD:labm8/py/BUILD
=======
>>>>>>> a4e1bff54... Auto-format code.:labm8/BUILD
    deps = [
        ":app",
        ":fs",
    ],
<<<<<<< HEAD:labm8/py/BUILD
=======
    deps = [":fs", ":app"],
>>>>>>> 7851b2a76... Add extra dependencies on //labm8:fs.:labm8/BUILD
=======
>>>>>>> a4e1bff54... Auto-format code.:labm8/BUILD
)

py_test(
    name = "system_test",
    size = "small",
    srcs = ["system_test.py"],
    deps = [
        ":app",
        ":fs",
        ":system",
<<<<<<< HEAD:labm8/py/BUILD
        ":test",
=======
        "//labm8:test",
        "//third_party/py/absl",
>>>>>>> ad2622a47... Port tests to //labm8:test.:labm8/BUILD
    ],
)

py_library(
    name = "tar",
    srcs = ["tar.py"],
    visibility = ["//visibility:public"],
    deps = [":fs"],
)

py_library(
    name = "test",
    testonly = 1,
    srcs = ["test.py"],
    visibility = ["//visibility:public"],
    deps = [
<<<<<<< HEAD:labm8/py/BUILD
        ":app",
<<<<<<< HEAD:labm8/py/BUILD
        "//labm8/py/internal:pytest_plugin",
=======
=======
>>>>>>> bb562b8d7... Refresh labm8 for new deps.:labm8/BUILD
        "//:conftest",
        "//third_party/py/absl",
>>>>>>> ad2622a47... Port tests to //labm8:test.:labm8/BUILD
        "//third_party/py/pytest",
    ],
)

py_test(
    name = "test_test",
    srcs = ["test_test.py"],
    deps = [
        ":test",
    ],
)

py_library(
    name = "text",
    srcs = ["text.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":fs",
        ":system",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "text_test",
    size = "small",
    srcs = ["text_test.py"],
    deps = [
        ":app",
        ":test",
        ":text",
<<<<<<< HEAD:labm8/py/BUILD
=======
        "//labm8:test",
        "//third_party/py/absl",
>>>>>>> ad2622a47... Port tests to //labm8:test.:labm8/BUILD
    ],
)

py_library(
    name = "viz",
    srcs = ["viz.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":app",
        ":fs",
        "//third_party/py/jupyter",
        "//third_party/py/matplotlib",
    ],
)

py_test(
    name = "viz_test",
    size = "small",
    srcs = ["viz_test.py"],
    deps = [
        ":app",
        ":test",
        ":viz",
<<<<<<< HEAD:labm8/py/BUILD
        "//third_party/py/matplotlib",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/pytest",
        "//third_party/py/seaborn",
=======
        "//labm8:test",
        "//third_party/py/absl",
        "//third_party/py/matplotlib",
>>>>>>> ad2622a47... Port tests to //labm8:test.:labm8/BUILD
    ],
)
