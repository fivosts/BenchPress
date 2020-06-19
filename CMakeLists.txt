cmake_minimum_required(VERSION 3.5)

project(CLGEN, 
        VERSION 0.0.1
        DESCRIPTION "Programming language model using machine learning")
# TODO check
# https://cmake.org/cmake/help/v3.0/module/FindPythonInterp.html
cmake_policy(SET CMP0076 NEW)

OPTION(BUILD_FROM_SOURCE "Select to build necessary system wide packages from source locally (gcc, python, boost, protoc)" OFF)
set(PYTHON "/usr/bin/python3" CACHE FILEPATH 
  "Provide specific python for application. Default: [/usr/bin/python3]. Ignored if BUILD_FROM_SOURCE is enabled."
  FORCE
)
set(GCC    "/usr/bin/gcc"     CACHE FILEPATH 
  "Provide specific gcc for application. Default: [/usr/bin/gcc]. Ignored if BUILD_FROM_SOURCE is enabled."
  FORCE
)
set(PROTOC "/usr/bin/protoc"  CACHE FILEPATH 
  "Provide specific protobuf compiler for application. Default: [/usr/bin/protoc]. Ignored if BUILD_FROM_SOURCE is enabled."
  FORCE
)

# Python application relative base path
set(APP_PATH    deeplearning/clgen)
# Base path of third_party deps
set(THIRD_PARTY ${CMAKE_BINARY_DIR}/third_party)
# Local bin folder for application binaries and libraries
set(LOCAL       ${CMAKE_BINARY_DIR}/local)

function(install_external_deps)

  ### Build and install third_party system dependencies
  #   Installs: gcc-9, python-3.7.5, clangtooling-6, boost-1.73.0 protobuf-compilers and opencl-headers
  include(ExternalProject)
  if (BUILD_FROM_SOURCE)
    # Build gcc-9 from source
    # Only 64-bit compiler build
    # ExternalProject_Add(gcc
    #   GIT_REPOSITORY    git://gcc.gnu.org/git/gcc.git
    #   GIT_TAG           releases/gcc-9.3.0
    #   PREFIX            ${THIRD_PARTY}
    #   TMP_DIR           ${THIRD_PARTY}/tmp
    #   STAMP_DIR         ${THIRD_PARTY}/stamp
    #   SOURCE_DIR        ${THIRD_PARTY}/gcc/gcc
    #   BINARY_DIR        ${THIRD_PARTY}/gcc/build
    #   INSTALL_DIR       ${LOCAL}
    #   CONFIGURE_COMMAND ${THIRD_PARTY}/gcc/gcc/configure --prefix=${LOCAL} --disable-multilib --enable-languages=c,c++,fortran,go
    # )
    # set(GCC             ${LOCAL}/bin/gcc-9 PARENT_SCOPE)
  
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
      CONFIGURE_COMMAND ${THIRD_PARTY}/python/python/configure --prefix=${LOCAL} --enable-optimizations --enable-loadable-sqlite-extensions
    )
    set(PYTHON          ${LOCAL}/bin/python3               PARENT_SCOPE)

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
      BUILD_COMMAND     ${THIRD_PARTY}/boost/boost/b2  --prefix=${LOCAL} && ${THIRD_PARTY}/boost/boost/b2 --prefix=${LOCAL} headers
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
  set(OPENCL_HEADERS  ${LOCAL}/opencl_headers PARENT_SCOPE)

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

function(setup_python)

  ### Initialize python environment for application
  # 1. Create virtual environment for python (depends on 'venv')
  # 2. Pip install requirements.txt
  # 3. Set python binary to virtual environment's bin folder

  set (PYTHON_ENV ${LOCAL}/env/bin/python)
  if (BUILD_FROM_SOURCE)
    file(MAKE_DIRECTORY ${LOCAL}/lib/python)
    add_custom_target(requirements ALL
      COMMAND ${PYTHON} -m venv ${LOCAL}/env
      COMMAND ${PYTHON_ENV} -m pip install --disable-pip-version-check -r ${CMAKE_SOURCE_DIR}/requirements.txt
      DEPENDS python
      COMMENT "Installing python package dependencies"
    )
  else ()
    add_custom_target(requirements ALL
      COMMAND ${PYTHON} -m venv ${LOCAL}/env
      COMMAND ${PYTHON_ENV} -m pip install --disable-pip-version-check -r ${CMAKE_SOURCE_DIR}/requirements.txt
      COMMENT "Installing python package dependencies"
    )
  endif ()
  set(PYTHON ${PYTHON_ENV} PARENT_SCOPE)
endfunction()

function(clgen)

  ### Auto-generates clgen binary that initializes app environment and deploys app.

  get_target_property(CLANG_REWRITER      clang_rewriter      path)
  get_target_property(DASHBOARD_STATIC    dashboard_static    path)
  get_target_property(DASHBOARD_TEMPLATES dashboard_templates path)
  get_target_property(LIBCLC              libclc              path)
  get_target_property(DATA_CL_INCLUDE     data_cl_include     path)

  string(CONCAT CLGEN 
         "PYTHON_BINARY="  ${PYTHON}  "\n"
         "CLGEN_BINARY=" ${CMAKE_CURRENT_SOURCE_DIR}/${APP_PATH} "/clgen.py\n"
         "GCC="                 ${GCC} "\n"
         "LLVM="                ${LLVM} "\n"
         "LIBCXX_HEADERS="      ${LLVM}/include/c++/v1 "\n"
         "LIBCLC="              ${LIBCLC} "\n"
         "DASHBOARD_TEMPLATES=" ${DASHBOARD_TEMPLATES} "\n"
         "DASHBOARD_STATIC="    ${DASHBOARD_STATIC} "\n"
         "DATA_CL_INCLUDE="     ${DATA_CL_INCLUDE} "\n"
         "CLANG="               ${LLVM}/bin/clang "\n"
         "CLANG_FORMAT="        ${LLVM}/bin/clang-format "\n"
         "CLANG_HEADERS="       ${LLVM}/lib/clang/6.0.0/include "\n"
         "CLANG_REWRITER="      ${CLANG_REWRITER} "\n"
         "export LLVM\n"
         "export LIBCXX_HEADERS\n"
         "export LIBCLC\n"
         "export DASHBOARD_TEMPLATES\n"
         "export DASHBOARD_STATIC\n"
         "export DATA_CL_INCLUDE\n"
         "export CLANG\n"
         "export CLANG_FORMAT\n"
         "export CLANG_HEADERS\n"
         "export CLANG_REWRITER\n"
         "eval PYTHONPATH=" ${CMAKE_CURRENT_SOURCE_DIR} " $PYTHON_BINARY $CLGEN_BINARY \"$@\"\n"
         "unset LLVM\n"
         "unset LIBCXX_HEADERS\n"
         "unset CLANG\n"
         "unset CLANG_FORMAT\n"
         "unset CLANG_HEADERS\n"
         "unset CLANG_REWRITER\n"
         "unset LIBCLC\n"
         "unset DASHBOARD_TEMPLATES\n"
         "unset DASHBOARD_STATIC\n"
         "unset DATA_CL_INCLUDE\n"
  )
  file(
    WRITE ${CMAKE_BINARY_DIR}/${APP_PATH}/clgen ${CLGEN}
    # FILE_PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  )
  add_custom_target(link_target ALL
      COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_BINARY_DIR}/${APP_PATH}/clgen ${CMAKE_CURRENT_SOURCE_DIR}/clgen
  )
endfunction()

install_external_deps()
setup_python()
add_subdirectory(${APP_PATH}/preprocessors)
add_subdirectory(${APP_PATH}/dashboard)
add_subdirectory(${APP_PATH}/data)
add_subdirectory(${APP_PATH}/proto)
clgen()