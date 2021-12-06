"""This module handles application's environment variables"""
import os
import ifcfg

def check_path_exists(path):
  if not os.path.exists(path):
    raise ValueError("{} does not exist.".format(path))
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
  CLANG_FORMAT        = check_path_exists(os.environ['CLANG_FORMAT'])
  CLANG_HEADERS       = check_path_exists(os.environ['CLANG_HEADERS'])
  CLANG_REWRITER      = check_path_exists(os.environ['CLANG_REWRITER'])
  LIBCLC              = check_path_exists(os.environ['LIBCLC'])
  DASHBOARD_TEMPLATES = check_path_exists(os.environ['DASHBOARD_TEMPLATES'])
  DASHBOARD_STATIC    = check_path_exists(os.environ['DASHBOARD_STATIC'])
  DATA_CL_INCLUDE     = check_path_exists(os.environ['DATA_CL_INCLUDE'])
  AUX_INCLUDE         = check_path_exists(os.environ['AUX_INCLUDE'])
  GREWE               = check_path_exists(os.environ['GREWE'])
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
