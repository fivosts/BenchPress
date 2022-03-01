if (PROTOC_FROM_SOURCE)
  add_custom_target(protobufs ALL
    COMMAND ${PROTOC} ${CMAKE_CURRENT_LIST_DIR}/clgen.proto           --proto_path=${PROJECT_SOURCE_DIR} --python_out=${PROJECT_SOURCE_DIR}
    COMMAND ${PROTOC} ${CMAKE_CURRENT_LIST_DIR}/corpus.proto          --proto_path=${PROJECT_SOURCE_DIR} --python_out=${PROJECT_SOURCE_DIR}
    COMMAND ${PROTOC} ${CMAKE_CURRENT_LIST_DIR}/internal.proto        --proto_path=${PROJECT_SOURCE_DIR} --python_out=${PROJECT_SOURCE_DIR}
    COMMAND ${PROTOC} ${CMAKE_CURRENT_LIST_DIR}/model.proto           --proto_path=${PROJECT_SOURCE_DIR} --python_out=${PROJECT_SOURCE_DIR}
    COMMAND ${PROTOC} ${CMAKE_CURRENT_LIST_DIR}/sampler.proto         --proto_path=${PROJECT_SOURCE_DIR} --python_out=${PROJECT_SOURCE_DIR}
    COMMAND ${PROTOC} ${CMAKE_CURRENT_LIST_DIR}/active_learning.proto --proto_path=${PROJECT_SOURCE_DIR} --python_out=${PROJECT_SOURCE_DIR}
    COMMAND ${PROTOC} ${CMAKE_CURRENT_LIST_DIR}/scrape_repos.proto    --proto_path=${PROJECT_SOURCE_DIR} --python_out=${PROJECT_SOURCE_DIR}
    COMMAND ${PROTOC} ${CMAKE_CURRENT_LIST_DIR}/github.proto          --proto_path=${PROJECT_SOURCE_DIR} --python_out=${PROJECT_SOURCE_DIR}
    COMMAND ${PROTOC} ${CMAKE_CURRENT_LIST_DIR}/evaluator.proto       --proto_path=${PROJECT_SOURCE_DIR} --python_out=${PROJECT_SOURCE_DIR}
    COMMAND ${PROTOC} ${CMAKE_CURRENT_LIST_DIR}/telemetry.proto       --proto_path=${PROJECT_SOURCE_DIR} --python_out=${PROJECT_SOURCE_DIR}
    DEPENDS protobuf
    COMMENT "Compiling clgen protobuf files"
  )
else ()
  add_custom_target(protobufs ALL
    COMMAND ${PROTOC} ${CMAKE_CURRENT_LIST_DIR}/clgen.proto           --proto_path=${PROJECT_SOURCE_DIR} --python_out=${PROJECT_SOURCE_DIR}
    COMMAND ${PROTOC} ${CMAKE_CURRENT_LIST_DIR}/corpus.proto          --proto_path=${PROJECT_SOURCE_DIR} --python_out=${PROJECT_SOURCE_DIR}
    COMMAND ${PROTOC} ${CMAKE_CURRENT_LIST_DIR}/internal.proto        --proto_path=${PROJECT_SOURCE_DIR} --python_out=${PROJECT_SOURCE_DIR}
    COMMAND ${PROTOC} ${CMAKE_CURRENT_LIST_DIR}/model.proto           --proto_path=${PROJECT_SOURCE_DIR} --python_out=${PROJECT_SOURCE_DIR}
    COMMAND ${PROTOC} ${CMAKE_CURRENT_LIST_DIR}/sampler.proto         --proto_path=${PROJECT_SOURCE_DIR} --python_out=${PROJECT_SOURCE_DIR}
    COMMAND ${PROTOC} ${CMAKE_CURRENT_LIST_DIR}/active_learning.proto --proto_path=${PROJECT_SOURCE_DIR} --python_out=${PROJECT_SOURCE_DIR}
    COMMAND ${PROTOC} ${CMAKE_CURRENT_LIST_DIR}/scrape_repos.proto    --proto_path=${PROJECT_SOURCE_DIR} --python_out=${PROJECT_SOURCE_DIR}
    COMMAND ${PROTOC} ${CMAKE_CURRENT_LIST_DIR}/github.proto          --proto_path=${PROJECT_SOURCE_DIR} --python_out=${PROJECT_SOURCE_DIR}
    COMMAND ${PROTOC} ${CMAKE_CURRENT_LIST_DIR}/evaluator.proto       --proto_path=${PROJECT_SOURCE_DIR} --python_out=${PROJECT_SOURCE_DIR}
    COMMAND ${PROTOC} ${CMAKE_CURRENT_LIST_DIR}/telemetry.proto       --proto_path=${PROJECT_SOURCE_DIR} --python_out=${PROJECT_SOURCE_DIR}
    COMMENT "Compiling clgen protobuf files"
  )
endif ()
