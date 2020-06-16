cmake_minimum_required(VERSION 3.5)

project(CLGEN, 
        VERSION 0.0.1
        DESCRIPTION "Programming language model with machine learning")

cmake_policy(SET CMP0076 NEW)


# Base path of third_party deps
set(third_party ${CMAKE_BINARY_DIR}/third_party)

# Local folder with all included binaries and libraries
set(local_include ${CMAKE_BINARY_DIR}/local)
target_include_directories(includes ${local_include}/*)

include(ExternalProject)
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

ExternalProject_Add(boost
  URL https://dl.bintray.com/boostorg/release/1.73.0/source/boost_1_73_0.tar.gz
  URL_SHA 9995e192e68528793755692917f9eb6422f3052a53c5e13ba278a228af6c7acf
  PREFIX  ${third_party}
  TMP_DIR ${third_party}/tmp
  STAMP_DIR ${third_party}/stamp
  DOWNLOAD_DIR ${third_party}/boost
  SOURCE_DIR   ${third_party}/boost/boost
  INSTALL_DIR  ${local_include}
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E ${third_party}/boost/boost/bootstrap.sh --prefix=${local_include}
  BUILD_COMMAND     ${CMAKE_COMMAND} -E ${third_party}/boost/boost/b2 && ${third_party}/boost/boost/b2 headers
  INSTALL_COMMAND   ${CMAKE_COMMAND} -E ${third_party}/boost/boost/b2 install
)

ExternalProject_Add(opencl-headers
  GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers
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

ExternalProject_Add(opencl-icd-loader
  GIT_REPOSITORY https://https://github.com/KhronosGroup/OpenCL-ICD-Loader
  GIT_TAG        c7fda8bb042760b5ead8650c64445f5972a64ad7
  DEPENDS      ${includes}
  PREFIX       ${third_party}
  TMP_DIR      ${third_party}/tmp
  STAMP_DIR    ${third_party}/stamp
  SOURCE_DIR   ${third_party}/opencl_icd_loader/opencl_icd_loader
  BINARY_DIR   ${third_party}/opencl_icd_loader/build
  INSTALL_DIR  ${local_include}
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E ${third_party}/boost/boost/bootstrap.sh --prefix=${local_include}
  BUILD_COMMAND     ${CMAKE_COMMAND} -E ${third_party}/boost/boost/b2 && ${third_party}/boost/boost/b2 headers
  INSTALL_COMMAND   ${CMAKE_COMMAND} -E ${third_party}/boost/boost/b2 install
)
