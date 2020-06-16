## Download tar
include(ExternalProject)

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(3rdParty_SUBDIR x64)
else()
  set(3rdParty_SUBDIR x86)
endif()

# https://cmake.org/cmake/help/latest/module/ExternalProject.html
ExternalProject_Add(llvm
  URL https://releases.llvm.org/6.0.0/clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz
  BUILD_IN_SOURCE 1
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E copy_directory ${3rdParty_SUBDIR}/${CMAKE_INSTALL_PREFIX}
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
)

# Unzip files
execute_process(
COMMAND ${CMAKE_COMMAND} -E tar xzf path/to/file.tar.gz
WORKING_DIRECTORY ${3rdParty_SUBDIR}/${CMAKE_INSTALL_PREFIX}/llvm
)