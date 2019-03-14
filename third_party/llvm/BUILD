<<<<<<< HEAD:third_party/llvm/BUILD
# LLVM package.
=======
# This is an implementation of the feature extractor from:
#
#     ﻿Grewe, D., Wang, Z., & O’Boyle, M. (2013). Portable Mapping of Data
#     Parallel Programs to OpenCL for Heterogeneous Systems. In CGO. IEEE.
#     https://doi.org/10.1109/CGO.2013.6494993
#
# Copyright 2017, 2018, 2019 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
>>>>>>> e0d1223cb... Add license headers to //research.:research/grewe_2013_cgo/BUILD

load("@com_github_chriscummins_rules_bats//:bats.bzl", "bats_test")

sh_binary(
    name = "clang",
    srcs = ["clang.sh"],
    data = select({
        "//:darwin": [
            "@llvm_mac//:clang",
        ],
        "//conditions:default": [
            "@llvm_linux//:clang",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/sh:app",
    ],
)

bats_test(
    name = "clang_smoke_test",
    srcs = ["clang_smoke_test.bats"],
    data = [
<<<<<<< HEAD:third_party/llvm/BUILD
        ":clang",
        "//labm8/sh:test",
=======
        ":feature_extractor_binary",
        "//third_party/opencl/inlined",
>>>>>>> d9e584789... Move inlined header.:research/grewe_2013_cgo/BUILD
    ],
<<<<<<< HEAD:third_party/llvm/BUILD
)

sh_binary(
    name = "clang-format",
    srcs = ["clang-format.sh"],
    data = select({
        "//:darwin": [
            "@llvm_mac//:clang-format",
        ],
        "//conditions:default": [
            "@llvm_linux//:clang-format",
        ],
    }),
=======
>>>>>>> 7f3f24544... Set feature extractor as public.:gpu/portable_mapping_of_data_parallel_programs_to_opencl/BUILD
    visibility = ["//visibility:public"],
    deps = [
<<<<<<< HEAD:third_party/llvm/BUILD
        "//labm8/sh:app",
    ],
)

bats_test(
    name = "clang-format_smoke_test",
    srcs = ["clang-format_smoke_test.bats"],
    data = [
        ":clang-format",
        "//labm8/sh:test",
    ],
)

sh_binary(
    name = "llvm-as",
    srcs = ["llvm-as.sh"],
    data = select({
        "//:darwin": [
            "@llvm_mac//:llvm-as",
        ],
        "//conditions:default": [
            "@llvm_linux//:llvm-as",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/sh:app",
    ],
)

bats_test(
    name = "llvm-as_smoke_test",
    srcs = ["llvm-as_smoke_test.bats"],
    data = [
        ":llvm-as",
        "//labm8/sh:test",
    ],
)

sh_binary(
    name = "llvm-dis",
    srcs = ["llvm-dis.sh"],
    data = select({
        "//:darwin": [
            "@llvm_mac//:llvm-dis",
        ],
        "//conditions:default": [
            "@llvm_linux//:llvm-dis",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/sh:app",
=======
        "//labm8:app",
        "//labm8:bazelutil",
<<<<<<< HEAD:third_party/llvm/BUILD
>>>>>>> 89b790ba9... Merge absl logging, app, and flags modules.:research/grewe_2013_cgo/BUILD
    ],
)

bats_test(
    name = "llvm-dis_smoke_test",
    srcs = ["llvm-dis_smoke_test.bats"],
    data = [
        ":llvm-dis",
        "//labm8/sh:test",
=======
        "//labm8:fs",
>>>>>>> fa81e4ed1... Tidy up and convenience function.:research/grewe_2013_cgo/BUILD
    ],
)

sh_binary(
    name = "llvm-link",
    srcs = ["llvm-link.sh"],
    data = select({
        "//:darwin": [
            "@llvm_mac//:llvm-link",
        ],
        "//conditions:default": [
            "@llvm_linux//:llvm-link",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
<<<<<<< HEAD:third_party/llvm/BUILD
        "//labm8/sh:app",
=======
        ":feature_extractor",
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/pytest",
>>>>>>> dec37b744... Auto format BUILD file.:gpu/portable_mapping_of_data_parallel_programs_to_opencl/BUILD
    ],
)

bats_test(
    name = "llvm-link_smoke_test",
    srcs = ["llvm-link_smoke_test.bats"],
    data = [
        ":llvm-link",
        "//labm8/sh:test",
    ],
<<<<<<< HEAD:third_party/llvm/BUILD
)

sh_binary(
    name = "opt",
    srcs = ["opt.sh"],
    data = select({
        "//:darwin": [
            "@llvm_mac//:opt",
        ],
        "//conditions:default": [
            "@llvm_linux//:opt",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/sh:app",
    ],
)

bats_test(
    name = "opt_smoke_test",
    srcs = ["opt_smoke_test.bats"],
    data = [
        ":opt",
        "//labm8/sh:test",
    ],
)

# A C++ library for pulling in LLVM libraries. Depend on this target if you
# want to use the LLVM headers and library.
cc_library(
    name = "llvm",
=======
>>>>>>> d9e584789... Move inlined header.:research/grewe_2013_cgo/BUILD
    data = select({
        "//:darwin": ["@llvm_mac//:libdir"],
        "//conditions:default": ["@llvm_linux//:libdir"],
    }),
    linkopts = select({
        "//:darwin": [
            "-Wl,-rpath,$(location @llvm_mac//:libdir)/",
            "-L$(location @llvm_mac//:libdir)/",
        ],
        "//conditions:default": [
            "-Wl,-rpath,$(location @llvm_linux//:libdir)/",
            "-L$(location @llvm_linux//:libdir)/",
        ],
    }) + [
        "-pthread",
        "-ldl",
        "-lz",
        "-lclangARCMigrate",
        "-lclangFrontend",
        "-lclangFrontendTool",
        "-lclangDriver",
        "-lclangSerialization",
        "-lclangCodeGen",
        "-lclangParse",
        "-lclangSema",
        "-lclangRewriteFrontend",
        "-lclangRewrite",
        "-lclangStaticAnalyzerFrontend",
        "-lclangStaticAnalyzerCheckers",
        "-lclangStaticAnalyzerCore",
        "-lclangAnalysis",
        "-lclangEdit",
        "-lclangAST",
        "-lclangASTMatchers",
        "-lclangLex",
        "-lclangBasic",
        "-lclangTooling",
        "-lclangToolingCore",
    ] + select({
        "//:darwin": [],
        "//conditions:default": ["-lclang"],
    }) + [
        "-lLLVMLTO",
        "-lLLVMPasses",
        "-lLLVMObjCARCOpts",
        "-lLLVMMIRParser",
        "-lLLVMSymbolize",
        "-lLLVMDebugInfoPDB",
        "-lLLVMDebugInfoDWARF",
        "-lLLVMCoverage",
        "-lLLVMTableGen",
        "-lLLVMDlltoolDriver",
        "-lLLVMOrcJIT",
        "-lLLVMXCoreDisassembler",
        "-lLLVMXCoreCodeGen",
        "-lLLVMXCoreDesc",
        "-lLLVMXCoreInfo",
        "-lLLVMXCoreAsmPrinter",
        "-lLLVMSystemZDisassembler",
        "-lLLVMSystemZCodeGen",
        "-lLLVMSystemZAsmParser",
        "-lLLVMSystemZDesc",
        "-lLLVMSystemZInfo",
        "-lLLVMSystemZAsmPrinter",
        "-lLLVMSparcDisassembler",
        "-lLLVMSparcCodeGen",
        "-lLLVMSparcAsmParser",
        "-lLLVMSparcDesc",
        "-lLLVMSparcInfo",
        "-lLLVMSparcAsmPrinter",
        "-lLLVMPowerPCDisassembler",
        "-lLLVMPowerPCCodeGen",
        "-lLLVMPowerPCAsmParser",
        "-lLLVMPowerPCDesc",
        "-lLLVMPowerPCInfo",
        "-lLLVMPowerPCAsmPrinter",
        "-lLLVMPowerPCAsmPrinter",
        "-lLLVMNVPTXCodeGen",
        "-lLLVMNVPTXDesc",
        "-lLLVMNVPTXInfo",
        "-lLLVMNVPTXAsmPrinter",
        "-lLLVMMSP430CodeGen",
        "-lLLVMMSP430Desc",
        "-lLLVMMSP430Info",
        "-lLLVMMSP430AsmPrinter",
        "-lLLVMMipsDisassembler",
        "-lLLVMMipsCodeGen",
        "-lLLVMMipsAsmParser",
        "-lLLVMMipsDesc",
        "-lLLVMMipsInfo",
        "-lLLVMMipsAsmPrinter",
        "-lLLVMLanaiDisassembler",
        "-lLLVMLanaiCodeGen",
        "-lLLVMLanaiAsmParser",
        "-lLLVMLanaiDesc",
        "-lLLVMLanaiAsmPrinter",
        "-lLLVMLanaiInfo",
        "-lLLVMHexagonDisassembler",
        "-lLLVMHexagonCodeGen",
        "-lLLVMHexagonAsmParser",
        "-lLLVMHexagonDesc",
        "-lLLVMHexagonInfo",
        "-lLLVMBPFDisassembler",
        "-lLLVMBPFCodeGen",
        "-lLLVMBPFAsmParser",
        "-lLLVMBPFDesc",
        "-lLLVMBPFInfo",
        "-lLLVMBPFAsmPrinter",
        "-lLLVMARMDisassembler",
        "-lLLVMARMCodeGen",
        "-lLLVMARMAsmParser",
        "-lLLVMARMDesc",
        "-lLLVMARMInfo",
        "-lLLVMARMAsmPrinter",
        "-lLLVMARMUtils",
        "-lLLVMAMDGPUDisassembler",
        "-lLLVMAMDGPUCodeGen",
        "-lLLVMAMDGPUAsmParser",
        "-lLLVMAMDGPUDesc",
        "-lLLVMAMDGPUInfo",
        "-lLLVMAMDGPUAsmPrinter",
        "-lLLVMAMDGPUUtils",
        "-lLLVMAArch64Disassembler",
        "-lLLVMAArch64CodeGen",
        "-lLLVMAArch64AsmParser",
        "-lLLVMAArch64Desc",
        "-lLLVMAArch64Info",
        "-lLLVMAArch64AsmPrinter",
        "-lLLVMAArch64Utils",
        "-lLLVMObjectYAML",
        "-lLLVMLibDriver",
        "-lLLVMOption",
        "-lLLVMWindowsManifest",
        "-lLLVMFuzzMutate",
        "-lLLVMX86Disassembler",
        "-lLLVMX86AsmParser",
        "-lLLVMX86CodeGen",
        "-lLLVMGlobalISel",
        "-lLLVMSelectionDAG",
        "-lLLVMAsmPrinter",
        "-lLLVMDebugInfoCodeView",
        "-lLLVMDebugInfoMSF",
        "-lLLVMX86Desc",
        "-lLLVMMCDisassembler",
        "-lLLVMX86Info",
        "-lLLVMX86AsmPrinter",
        "-lLLVMX86Utils",
        "-lLLVMMCJIT",
        "-lLLVMLineEditor",
        "-lLLVMInterpreter",
        "-lLLVMExecutionEngine",
        "-lLLVMRuntimeDyld",
        "-lLLVMCodeGen",
        "-lLLVMTarget",
        "-lLLVMCoroutines",
        "-lLLVMipo",
        "-lLLVMInstrumentation",
        "-lLLVMVectorize",
        "-lLLVMScalarOpts",
        "-lLLVMLinker",
        "-lLLVMIRReader",
        "-lLLVMAsmParser",
        "-lLLVMInstCombine",
        "-lLLVMTransformUtils",
        "-lLLVMBitWriter",
        "-lLLVMAnalysis",
        "-lLLVMProfileData",
        "-lLLVMObject",
        "-lLLVMMCParser",
        "-lLLVMMC",
        "-lLLVMBitReader",
        "-lLLVMCore",
        "-lLLVMBinaryFormat",
        "-lLLVMSupport",
        "-lLLVMDemangle",
    ] + select({
        "//:darwin": ["-lncurses"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = select({
        "//:darwin": [
            "@llvm_headers_mac//:headers",
            "@llvm_mac//:libs",
        ],
        "//conditions:default": [
            "@llvm_headers_linux//:headers",
            "@llvm_linux//:libs",
        ],
    }),
)
