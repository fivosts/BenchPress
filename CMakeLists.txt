cmake_minimum_required(VERSION 3.5)

project(CLGEN, 
        VERSION 0.0.1
        DESCRIPTION "Programming language model with machine learning")

cmake_policy(SET CMP0076 NEW)


# Base path of third_party deps
set(third_party ${CMAKE_BINARY_DIR}/third_party)

include(ExternalProject)
ExternalProject_Add(llvm
  URL https://releases.llvm.org/6.0.0/clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz
  PREFIX  ${third_party}
  TMP_DIR ${third_party}/llvm/logs/tmp
  DOWNLOAD_DIR ${third_party}/llvm
  SOURCE_DIR   ${third_party}/llvm/llvm
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
)
