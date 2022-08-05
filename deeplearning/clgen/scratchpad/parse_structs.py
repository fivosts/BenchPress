"""
Scratchpad experimental analysis for anyhing related to CLDrive.
"""
import pathlib
import typing
import clang.cindex

from deeplearning.benchpress.util import environment
from deeplearning.benchpress.preprocessors import opencl
from deeplearning.benchpress.preprocessors import structs
from deeplearning.benchpress.util import plotter as plt

from deeplearning.benchpress.util import logging as l

clang.cindex.Config.set_library_path(environment.LLVM_LIB)
if environment.LLVM_VERSION != 6:
  # LLVM 9 needs libclang explicitly defined.
  clang.cindex.Config.set_library_file(environment.LLVM_LIB + "/libclang.so.{}".format(environment.LLVM_VERSION))

src1 ="""
struct my_struct{
  int a, b, c;
};
"""
src2 ="""
typedef struct my_struct {
  int x, y, z;
} structy;
"""

l.initLogger(name = "experiments")

try:
  unit = clang.cindex.TranslationUnit.from_source(f.name, args = builtin_cflags + cflags + extra_args)
except clang.cindex.TranslationUnitLoadError as e:
  raise ValueError(e)
