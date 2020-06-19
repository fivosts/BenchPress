cmake_minimum_required(VERSION 3.5)

project(CLGEN, 
        VERSION 0.0.1
        DESCRIPTION "Programming language model using machine learning")
# TODO check
# https://cmake.org/cmake/help/v3.0/module/FindPythonInterp.html
cmake_policy(SET CMP0076 NEW)

OPTION(BUILD_FROM_SOURCE "Select to build necessary system wide packages from source locally (gcc, python, boost, protoc)" OFF)
set(PYTHON "/usr/bin/python3" CACHE FILEPATH "Provide specific python for application. Default: [/usr/bin/python3]. Ignored if BUILD_FROM_SOURCE is enabled." FORCE)
set(GCC    "/usr/bin/gcc"     CACHE FILEPATH "Provide specific gcc for application. Default: [/usr/bin/gcc]. Ignored if BUILD_FROM_SOURCE is enabled." FORCE)
set(PROTOC "/usr/bin/protoc"  CACHE FILEPATH "Provide specific protobuf compiler for application. Default: [/usr/bin/protoc]. Ignored if BUILD_FROM_SOURCE is enabled." FORCE)

# Base path of third_party deps
set(THIRD_PARTY ${CMAKE_BINARY_DIR}/third_party)
# Local folder with all included binaries and libraries
set(LOCAL ${CMAKE_BINARY_DIR}/local)

function(install_external_deps)
  ### Build and install third_party system dependencies
  include(ExternalProject)
  if (BUILD_FROM_SOURCE)
    # Build gcc-9 from source
    # Only 64-bit compiler build
    ExternalProject_Add(gcc
      GIT_REPOSITORY    git://gcc.gnu.org/git/gcc.git
      GIT_TAG           releases/gcc-9.3.0
      PREFIX            ${THIRD_PARTY}
      TMP_DIR           ${THIRD_PARTY}/tmp
      STAMP_DIR         ${THIRD_PARTY}/stamp
      SOURCE_DIR        ${THIRD_PARTY}/gcc/gcc
      BINARY_DIR        ${THIRD_PARTY}/gcc/build
      INSTALL_DIR       ${LOCAL}
      CONFIGURE_COMMAND ${THIRD_PARTY}/gcc/gcc/configure --prefix=${LOCAL} --disable-multilib --enable-languages=c,c++,fortran,go
    )
    set(GCC             ${LOCAL}/bin/gcc-9 PARENT_SCOPE)
  
    # Python 3.7.5 from source
    ExternalProject_Add(python
      URL               https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tar.xz
      URL_MD5           08ed8030b1183107c48f2092e79a87e2
      PREFIX            ${THIRD_PARTY}
      TMP_DIR           ${THIRD_PARTY}/tmp
      STAMP_DIR         ${THIRD_PARTY}/stamp
      DOWNLOAD_DIR      ${THIRD_PARTY}/python
      SOURCE_DIR        ${THIRD_PARTY}/python/python
      BUILD_IN_SOURCE   true
      INSTALL_DIR       ${LOCAL}
      CONFIGURE_COMMAND ${THIRD_PARTY}/python/python/configure --prefix=${LOCAL} --enable-optimizations
    )
    set(PYTHON          ${LOCAL}/bin/python3 PARENT_SCOPE)
  
    # Boost lib
    # TODO make sure this version is visible when compiling kernels
    ExternalProject_Add(boost
      URL               https://dl.bintray.com/boostorg/release/1.73.0/source/boost_1_73_0.tar.gz
      URL_HASH          SHA256=9995e192e68528793755692917f9eb6422f3052a53c5e13ba278a228af6c7acf
      PREFIX            ${THIRD_PARTY}
      TMP_DIR           ${THIRD_PARTY}/tmp
      STAMP_DIR         ${THIRD_PARTY}/stamp
      DOWNLOAD_DIR      ${THIRD_PARTY}/boost
      SOURCE_DIR        ${THIRD_PARTY}/boost/boost
      BUILD_IN_SOURCE   true
      INSTALL_DIR       ${LOCAL}
      CONFIGURE_COMMAND ${THIRD_PARTY}/boost/boost/bootstrap.sh --prefix=${LOCAL}
      BUILD_COMMAND     ${THIRD_PARTY}/boost/boost/b2 && ${THIRD_PARTY}/boost/boost/b2 headers
      INSTALL_COMMAND   ${THIRD_PARTY}/boost/boost/b2 install
    )
    # set(BOOST           ${LOCAL} PARENT_SCOPE)

    # protoc compiler
    ExternalProject_Add(protobuf
      GIT_REPOSITORY    https://github.com/protocolbuffers/protobuf.git
      GIT_TAG           31ebe2ac71400344a5db91ffc13c4ddfb7589f92
      GIT_PROGRESS      true
      BUILD_IN_SOURCE   true
      PREFIX            ${THIRD_PARTY}
      TMP_DIR           ${THIRD_PARTY}/tmp
      STAMP_DIR         ${THIRD_PARTY}/stamp
      SOURCE_DIR        ${THIRD_PARTY}/protobuf/protobuf
      INSTALL_DIR       ${LOCAL}
      CONFIGURE_COMMAND cd ${THIRD_PARTY}/protobuf/protobuf && ./autogen.sh && ${THIRD_PARTY}/protobuf/protobuf/configure --prefix=${LOCAL}
    )
    set(PROTOC          ${LOCAL}/bin/protoc PARENT_SCOPE)

  endif ()
  # Clang and LLVM 6 pre-compiled binaries and libraries.
  ExternalProject_Add(llvm
    URL               https://releases.llvm.org/6.0.0/clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz
    PREFIX            ${THIRD_PARTY}
    TMP_DIR           ${THIRD_PARTY}/tmp
    STAMP_DIR         ${THIRD_PARTY}/stamp
    DOWNLOAD_DIR      ${THIRD_PARTY}/llvm
    SOURCE_DIR        ${THIRD_PARTY}/llvm/llvm
    INSTALL_DIR       ${LOCAL}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   cp -Rn ${THIRD_PARTY}/llvm/llvm/ ${LOCAL}
  )
  set(LLVM            ${LOCAL}/llvm PARENT_SCOPE)

  #include <OpenCL/*>
  ExternalProject_Add(opencl-headers
    GIT_REPOSITORY    https://github.com/KhronosGroup/OpenCL-Headers.git
    GIT_TAG           9824efd0ec1adbfc23b40be6486a8ec6efc43822
    GIT_PROGRESS      true
    PREFIX            ${THIRD_PARTY}
    TMP_DIR           ${THIRD_PARTY}/tmp
    STAMP_DIR         ${THIRD_PARTY}/stamp
    SOURCE_DIR        ${THIRD_PARTY}/opencl_headers/opencl_headers
    INSTALL_DIR       ${LOCAL}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   cp -Rn ${THIRD_PARTY}/opencl_headers/opencl_headers/ ${LOCAL}
  )
  set(OPENCL_HEADERS  ${LOCAL}/opencl_headers] PARENT_SCOPE)

  # TODO check if needed.
  # ExternalProject_Add(opencl-icd-loader
  #   GIT_REPOSITORY    https://https://github.com/KhronosGroup/OpenCL-ICD-Loader
  #   GIT_TAG           c7fda8bb042760b5ead8650c64445f5972a64ad7
  #   GIT_PROGRESS      true
  #   DEPENDS           ${includes}
  #   PREFIX            ${THIRD_PARTY}
  #   TMP_DIR           ${THIRD_PARTY}/tmp
  #   STAMP_DIR         ${THIRD_PARTY}/stamp
  #   SOURCE_DIR        ${THIRD_PARTY}/opencl_icd_loader/opencl_icd_loader
  #   BINARY_DIR        ${THIRD_PARTY}/opencl_icd_loader/build
  #   INSTALL_DIR       ${LOCAL}
  #   # CONFIGURE_COMMAND 
  #   # BUILD_COMMAND     ${THIRD_PARTY}/boost/boost/b2 && ${THIRD_PARTY}/boost/boost/b2 headers
  #   # INSTALL_COMMAND   ${THIRD_PARTY}/boost/boost/b2 install
  # )
endfunction()

function(install_python_deps)
  add_custom_target(requirements
    ${PYTHON} -m pip install -r ${CMAKE_SOURCE_DIR}/requirements.txt
  )
endfunction()

function(clgen)
  string(CONCAT CLGEN 
         "PYTHON_BINARY=" \" ${PYTHON} \" "\n"
         "CLGEN_BINARY=\"deeplearning/clgen/clgen.py\"\n"
         "GCC=" ${GCC} "\n"
         "LLVM=" ${LLVM} "\n"
         "CLANG_REWRITER=" clang_rewriter "\n"
         "export LLVM\n"
         "export CLANG_REWRITER\n"
         "eval $PYTHON_BINARY $CLGEN_BINARY \"$@\"\n"
         "unset LLVM\n"
         "unset CLANG_REWRITER")
  file(WRITE ${CMAKE_SOURCE_DIR}/clgen ${CLGEN})
endfunction()

install_external_deps()
install_python_deps()
add_subdirectory(deeplearning/clgen/preprocessors)
clgen()