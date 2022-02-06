"""This module handles application's environment variables"""
import os
import ifcfg

def check_path_exists(path, must_exist = True):
  if not os.path.exists(path):
    if must_exist:
      raise ValueError("{} does not exist.".format(path))
    else:
      return None
  return path

try:
  LLVM_VERSION        = os.environ['LLVM_VERSION']
  LLVM                = check_path_exists(os.environ['LLVM'])
  LLVM_LIB            = check_path_exists(os.environ['LLVM_LIB'])
  LIBCXX_HEADERS      = check_path_exists(os.environ['LIBCXX_HEADERS'])
  OPENCL_HEADERS      = check_path_exists(os.environ['OPENCL_HEADERS'])
  CLANG               = check_path_exists(os.environ['CLANG'])
  OPT                 = check_path_exists(os.environ['OPT'])
  LLVM_EXTRACT        = check_path_exists(os.environ['LLVM_EXTRACT'])
  LLVM_DIS            = check_path_exists(os.environ['LLVM_DIS'])
  CLANG_FORMAT        = check_path_exists(os.environ['CLANG_FORMAT'])
  CLANG_HEADERS       = check_path_exists(os.environ['CLANG_HEADERS'])
  CLANG_REWRITER      = check_path_exists(os.environ['CLANG_REWRITER'])
  SEQ_CLANG_REWRITER  = check_path_exists(os.environ['SEQ_CLANG_REWRITER'])
  LIBCLC              = check_path_exists(os.environ['LIBCLC'])
  DASHBOARD_TEMPLATES = check_path_exists(os.environ['DASHBOARD_TEMPLATES'])
  DASHBOARD_STATIC    = check_path_exists(os.environ['DASHBOARD_STATIC'])
  DATA_CL_INCLUDE     = check_path_exists(os.environ['DATA_CL_INCLUDE'])
  AUX_INCLUDE         = check_path_exists(os.environ['AUX_INCLUDE'])
  GREWE               = check_path_exists(os.environ['GREWE'])
  CLDRIVE             = check_path_exists(os.environ['CLDRIVE'], must_exist = False)
  MUTEC               = check_path_exists(os.environ['MUTEC'], must_exist = False)
  SRCIROR_SRC         = check_path_exists(os.environ['SRCIROR_SRC'], must_exist = False)
  SRCIROR_IR          = check_path_exists(os.environ['SRCIROR_IR'], must_exist = False)
  CSMITH              = check_path_exists(os.environ['CSMITH'], must_exist = False)
  CLSMITH             = check_path_exists(os.environ['CLSMITH'], must_exist = False)
  CLSMITH_INCLUDE     = check_path_exists(os.environ['CLSMITH_INCLUDE'], must_exist = False)
  INSTCOUNT           = check_path_exists(os.environ['INSTCOUNT'])
  AUTOPHASE           = check_path_exists(os.environ['AUTOPHASE'])
  MASTER_PORT         = int(os.environ.get("MASTER_PORT", 8738))
  MASTER_ADDR         = os.environ.get("MASTER_ADDR", "127.0.0.1")
  LOCAL_RANK          = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
  WORLD_RANK          = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
  WORLD_SIZE          = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
  if "GLOO_SOCKET_IFNAME" not in os.environ:
    os.environ["GLOO_SOCKET_IFNAME"] = ifcfg.default_interface()['device']
  if "NCCL_SOCKET_IFNAME" not in os.environ:
    os.environ["NCCL_SOCKET_IFNAME"] = ifcfg.default_interface()['device']
except Exception as e:
  raise e
