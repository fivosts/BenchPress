add_custom_target(data_cl_include ALL
  COMMAND cp -r ${CMAKE_CURRENT_LIST_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/data
  COMMENT "Installing data CL include folder"
)
set_target_properties(data_cl_include PROPERTIES
    path ${CMAKE_CURRENT_BINARY_DIR}/${APP_PATH}/data/include
)

add_custom_target(libclc ALL
  COMMAND cp -r ${CMAKE_CURRENT_LIST_DIR}/libclc ${CMAKE_CURRENT_BINARY_DIR}/data
  COMMENT "Installing lbclc"
)
set_target_properties(libclc PROPERTIES
    path ${CMAKE_CURRENT_BINARY_DIR}/${APP_PATH}/data/libclc/generic/include
)