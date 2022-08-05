add_custom_target(data_cl_include ALL
  COMMAND cp -r ${CMAKE_CURRENT_LIST_DIR}/include ${LOCAL}
  COMMENT "Installing data CL include headers"
)
set_target_properties(data_cl_include PROPERTIES
    path ${LOCAL}/include
)

add_custom_target(libclc ALL
  COMMAND cp -r ${CMAKE_CURRENT_LIST_DIR}/libclc ${LOCAL}
  COMMENT "Installing lbclc"
)
set_target_properties(libclc PROPERTIES
    path ${LOCAL}/libclc/generic/include
)
