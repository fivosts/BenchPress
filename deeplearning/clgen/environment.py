import os
try:
  LLVM                = os.environ['LLVM']
  LIBCXX_HEADERS      = os.environ['LIBCXX_HEADERS']
  CLANG               = os.environ['CLANG']
  CLANG_HEADERS       = os.environ['CLANG_HEADERS']
  CLANG_REWRITER      = os.environ['CLANG_REWRITER']
  LIBCLC              = os.environ['LIBCLC']
  DASHBOARD_TEMPLATES = os.environ['DASHBOARD_TEMPLATES']
  DASHBOARD_STATIC    = os.environ['DASHBOARD_STATIC']
  DATA_CL_INCLUDE     = os.environ['DATA_CL_INCLUDE']
except Exception as e:
  raise e
  