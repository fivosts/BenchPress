## Download tar
include(ExternalProject)

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(3rdParty_SUBDIR x64)
else()
  set(3rdParty_SUBDIR x86)
endif()

ExternalProject_Add( 3rdParty
  URL http://download.osgvisual.org/3rdParty_VS2013_v120_x86_x64_V9_small.7z
  URL_MD5 c9294a16b39783cba8c4c55df182b24b
  BUILD_IN_SOURCE 1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory ${3rdParty_SUBDIR}/
${CMAKE_INSTALL_PREFIX}
)

# Unzip files
execute_process(
COMMAND ${CMAKE_COMMAND} -E tar xzf path/to/file.tar.gz
WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/path/to/where/you/want/to/unpack
)