"""This module handles application's environment variables"""
import os

def check_exists(path):
  if not os.path.exists(path):
    raise ValueError("{} does not exist.".format(path))
  return path

try:
  LLVM                = check_exists(os.environ['LLVM'])
  LLVM_LIB            = check_exists(os.environ['LLVM_LIB'])
  LIBCXX_HEADERS      = check_exists(os.environ['LIBCXX_HEADERS'])
  CLANG               = check_exists(os.environ['CLANG'])
  CLANG_FORMAT        = check_exists(os.environ['CLANG_FORMAT'])
  CLANG_HEADERS       = check_exists(os.environ['CLANG_HEADERS'])
  CLANG_REWRITER      = check_exists(os.environ['CLANG_REWRITER'])
  LIBCLC              = check_exists(os.environ['LIBCLC'])
  DASHBOARD_TEMPLATES = check_exists(os.environ['DASHBOARD_TEMPLATES'])
  DASHBOARD_STATIC    = check_exists(os.environ['DASHBOARD_STATIC'])
  DATA_CL_INCLUDE     = check_exists(os.environ['DATA_CL_INCLUDE'])
  CLGEN_FEATURES      = check_exists(os.environ['CLGEN_FEATURES'])
  CLGEN_REWRITER      = check_exists(os.environ['CLGEN_REWRITER'])
except Exception as e:
  raise e
  