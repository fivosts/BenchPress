add_custom_target(dashboard_static ALL
  COMMAND cp -r ${CMAKE_CURRENT_LIST_DIR}/static ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Installing dashboard static folder"
)
set_target_properties(dashboard_static PROPERTIES
    path ${CMAKE_CURRENT_BINARY_DIR}/static
)
add_custom_target(dashboard_templates ALL
  COMMAND cp -r ${CMAKE_CURRENT_LIST_DIR}/templates ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Installing dashboard templates folder"
)
set_target_properties(dashboard_templates PROPERTIES
    path ${CMAKE_CURRENT_BINARY_DIR}/templates
)