cmake_minimum_required(VERSION 3.5)

project(CLGEN, 
        VERSION 0.0.1
        DESCRIPTION "Programming language model using machine learning")

cmake_policy(SET CMP0076 NEW)

# Base path of third_party deps
set(third_party ${CMAKE_BINARY_DIR}/third_party)

# Local folder with all included binaries and libraries
set(local_include ${CMAKE_BINARY_DIR}/local)

# Base path of python environment
set(python_path ${local_include}/env)

### Build and install system dependencies
include(ExternalProject)

# Build gcc-9 from source
ExternalProject_Add(gcc
  GIT
  GIT_TAG releases/gcc-9.3.0
  PREFIX       ${third_party}
  TMP_DIR      ${third_party}/tmp
  STAMP_DIR    ${third_party}/stamp
  SOURCE_DIR   ${third_party}/gcc/gcc
  BINARY_DIR   ${third_party}/gcc/build
  INSTALL_DIR  ${local_include}
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E ${third_party}/gcc/gcc/configure --prefix=${local_include} --enable-languages=c,c++,fortran,go
)

# Python 3.7.5 from source
ExternalProject_Add(python
  URL https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tar.xz
  URL_MD5 	   08ed8030b1183107c48f2092e79a87e2
  PREFIX       ${third_party}
  TMP_DIR      ${third_party}/tmp
  STAMP_DIR    ${third_party}/stamp
  DOWNLOAD_DIR ${third_party}/python
  SOURCE_DIR   ${third_party}/python/python
  BUILD_IN_SOURCE 1
  INSTALL_DIR  ${local_include}
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E ${third_party}/python/python/configure --prefix=${local_include}
)

# Clang and LLVM 6 pre-compiled binaries and libraries.
ExternalProject_Add(llvm
  URL https://releases.llvm.org/6.0.0/clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz
  PREFIX       ${third_party}
  TMP_DIR      ${third_party}/tmp
  STAMP_DIR    ${third_party}/stamp
  DOWNLOAD_DIR ${third_party}/llvm
  SOURCE_DIR   ${third_party}/llvm/llvm
  INSTALL_DIR  ${local_include}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND ${CMAKE_COMMAND} -E cp -R ${third_party}/llvm/llvm/* ${local_include}
)

# Boost lib 
ExternalProject_Add(boost
  URL https://dl.bintray.com/boostorg/release/1.73.0/source/boost_1_73_0.tar.gz
  URL_HASH SHA256=9995e192e68528793755692917f9eb6422f3052a53c5e13ba278a228af6c7acf
  PREFIX  ${third_party}
  TMP_DIR ${third_party}/tmp
  STAMP_DIR ${third_party}/stamp
  DOWNLOAD_DIR ${third_party}/boost
  SOURCE_DIR   ${third_party}/boost/boost
  BUILD_IN_SOURCE 1
  INSTALL_DIR  ${local_include}
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E ${third_party}/boost/boost/bootstrap.sh --prefix=${local_include}
  BUILD_COMMAND     ${CMAKE_COMMAND} -E ${third_party}/boost/boost/b2 && ${third_party}/boost/boost/b2 headers
  INSTALL_COMMAND   ${CMAKE_COMMAND} -E ${third_party}/boost/boost/b2 install
)

#include <OpenCL/*>
ExternalProject_Add(opencl-headers
  GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers.git
  GIT_TAG        9824efd0ec1adbfc23b40be6486a8ec6efc43822
  PREFIX       ${third_party}
  TMP_DIR      ${third_party}/tmp
  STAMP_DIR    ${third_party}/stamp
  SOURCE_DIR   ${third_party}/opencl_headers/opencl_headers
  INSTALL_DIR  ${local_include}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND ${CMAKE_COMMAND} -E cp -R ${third_party}/opencl_headers/opencl_headers/* ${local_include}
)

# TODO check if needed.
# ExternalProject_Add(opencl-icd-loader
#   GIT_REPOSITORY https://https://github.com/KhronosGroup/OpenCL-ICD-Loader
#   GIT_TAG        c7fda8bb042760b5ead8650c64445f5972a64ad7
#   DEPENDS      ${includes}
#   PREFIX       ${third_party}
#   TMP_DIR      ${third_party}/tmp
#   STAMP_DIR    ${third_party}/stamp
#   SOURCE_DIR   ${third_party}/opencl_icd_loader/opencl_icd_loader
#   BINARY_DIR   ${third_party}/opencl_icd_loader/build
#   INSTALL_DIR  ${local_include}
#   # CONFIGURE_COMMAND ${CMAKE_COMMAND}
#   BUILD_COMMAND     ${CMAKE_COMMAND} -E ${third_party}/boost/boost/b2 && ${third_party}/boost/boost/b2 headers
#   INSTALL_COMMAND   ${CMAKE_COMMAND} -E ${third_party}/boost/boost/b2 install
# )

# protoc compiler
ExternalProject_Add(protobuf
  GIT_REPOSITORY https://https://github.com/protocolbuffers/protobuf.git
  GIT_TAG        31ebe2ac71400344a5db91ffc13c4ddfb7589f92
  PREFIX       ${third_party}
  TMP_DIR      ${third_party}/tmp
  STAMP_DIR    ${third_party}/stamp
  SOURCE_DIR   ${third_party}/protobuf/protobuf
  BINARY_DIR   ${third_party}/protobuf/build
  INSTALL_DIR  ${local_include}
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E ${third_party}/protobuf/protobuf/configure --prefix=${local_include}
)
