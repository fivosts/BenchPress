import os

try:
  CLANG          = os.environ['CLANG']
  CLANG_REWRITER = os.environ['CLANG_REWRITER']
  LLVM           = os.environ['LLVM']
except Exception as e:
  raise e