add_executable(clgen_features clgen_features.cpp)
add_executable(clgen_rewriter clgen_rewriter.cpp)

add_dependencies(clgen_features llvm)
add_dependencies(clgen_rewriter llvm)

set_target_properties(clgen_features PROPERTIES
    path ${CMAKE_CURRENT_BINARY_DIR}/clgen_features
)
set_target_properties(clgen_rewriter PROPERTIES
    path ${CMAKE_CURRENT_BINARY_DIR}/clgen_rewriter
)

target_include_directories(clgen_features PUBLIC ${LLVM}/include/)
target_include_directories(clgen_rewriter PUBLIC ${LLVM}/include/)

target_link_directories(clgen_features PUBLIC ${LLVM}/lib/)
target_link_directories(clgen_features PUBLIC ${LLVM}/lib/clang/6.0.0/lib/linux/)
target_link_directories(clgen_rewriter PUBLIC ${LLVM}/lib/)
target_link_directories(clgen_rewriter PUBLIC ${LLVM}/lib/clang/6.0.0/lib/linux/)

target_compile_options (clgen_features PRIVATE
    -I/usr/lib/llvm-9/include
    -std=c++11
    -fno-exceptions
    -D_GNU_SOURCE
    -D__STDC_CONSTANT_MACROS
    -D__STDC_FORMAT_MACROS
    -D__STDC_LIMIT_MACROS
    -std=c++11 -O3 -fno-rtti
)
target_compile_options (clgen_rewriter PRIVATE
    -I/usr/lib/llvm-9/include
    -std=c++11
    -fno-exceptions
    -D_GNU_SOURCE
    -D__STDC_CONSTANT_MACROS
    -D__STDC_FORMAT_MACROS
    -D__STDC_LIMIT_MACROS
    -std=c++11 -O3 -fno-rtti
)

target_link_libraries(clgen_features
  "-Wl,-rpath,"
  "-ldl"
  "-lclangTooling"
  "-lclangToolingCore"
  "-lclangFrontend"
  "-lclangDriver"
  "-lclangSerialization"
  "-lclangCodeGen"
  "-lclangParse"
  "-lclangSema"
  "-lclangAnalysis"
  "-lclangRewriteFrontend"
  "-lclangRewrite"
  "-lclangEdit"
  "-lclangAST"
  "-lclangLex"
  "-lclangBasic"
  "-lclang"
  "-ldl"
  "-lLLVM-9"
  "-pthread"
  "-lLLVMTarget"
  "-lLLVMMC"
  "-lLLVMObject"
  "-lLLVMCore"
)

target_link_libraries(clgen_rewriter
  "-Wl,-rpath,"
  "-ldl"
  "-lclangTooling"
  "-lclangToolingCore"
  "-lclangFrontend"
  "-lclangDriver"
  "-lclangSerialization"
  "-lclangCodeGen"
  "-lclangParse"
  "-lclangSema"
  "-lclangAnalysis"
  "-lclangRewriteFrontend"
  "-lclangRewrite"
  "-lclangEdit"
  "-lclangAST"
  "-lclangLex"
  "-lclangBasic"
  "-lclang"
  "-ldl"
  "-lLLVM-9"
  "-pthread"
  "-lLLVMTarget"
  "-lLLVMMC"
  "-lLLVMObject"
  "-lLLVMCore"
)
