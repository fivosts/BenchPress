"""This module handles application's environment variables"""
import os

def check_exists(path):
  if not os.path.exists(path):
    raise ValueError("{} does not exist.".format(path))
  return path

try:
  LLVM_VERSION        = os.environ['LLVM_VERSION']
  LLVM                = check_exists(os.environ['LLVM'])
  LLVM_LIB            = check_exists(os.environ['LLVM_LIB'])
  LIBCXX_HEADERS      = check_exists(os.environ['LIBCXX_HEADERS'])
  OPENCL_HEADERS      = check_exists(os.environ['OPENCL_HEADERS'])
  CLANG               = check_exists(os.environ['CLANG'])
  OPT                 = check_exists(os.environ['OPT'])
  CLANG_FORMAT        = check_exists(os.environ['CLANG_FORMAT'])
  CLANG_HEADERS       = check_exists(os.environ['CLANG_HEADERS'])
  CLANG_REWRITER      = check_exists(os.environ['CLANG_REWRITER'])
  LIBCLC              = check_exists(os.environ['LIBCLC'])
  DASHBOARD_TEMPLATES = check_exists(os.environ['DASHBOARD_TEMPLATES'])
  DASHBOARD_STATIC    = check_exists(os.environ['DASHBOARD_STATIC'])
  DATA_CL_INCLUDE     = check_exists(os.environ['DATA_CL_INCLUDE'])
  AUX_INCLUDE         = check_exists(os.environ['AUX_INCLUDE'])
  CLGEN_FEATURES      = check_exists(os.environ['CLGEN_FEATURES'])
  CLGEN_INSTCOUNT     = check_exists(os.environ['CLGEN_INSTCOUNT'])
except Exception as e:
  raise e
  