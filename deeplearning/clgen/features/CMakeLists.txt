#######################################
# InstCount
#######################################
add_library(InstCount MODULE InstCount.cpp)

add_dependencies(InstCount llvm)

set_target_properties(InstCount PROPERTIES
    path ${LOCAL}/lib/InstCount.so
)

target_include_directories(InstCount PUBLIC ${LLVM}/include/)
target_link_directories(InstCount PUBLIC ${LLVM}/lib/)

target_compile_options (InstCount PRIVATE
    -I/usr/lib/llvm-9/include
    -std=c++17
    -fno-exceptions
    -D_GNU_SOURCE
    -D__STDC_CONSTANT_MACROS
    -D__STDC_FORMAT_MACROS
    -D__STDC_LIMIT_MACROS
    -std=c++17 -O3 -fno-rtti
)
add_custom_target(install_lib_ic ALL
    COMMAND mv libInstCount.so ${LOCAL}/lib/InstCount.so
    DEPENDS InstCount
)
#######################################
# Autophase
#######################################
add_library(Autophase MODULE Autophase.cpp)

add_dependencies(Autophase llvm)

set_target_properties(Autophase PROPERTIES
    path ${LOCAL}/lib/Autophase.so
)

target_include_directories(Autophase PUBLIC ${LLVM}/include/)
target_link_directories(Autophase PUBLIC ${LLVM}/lib/)

target_compile_options (Autophase PRIVATE
    -I/usr/lib/llvm-9/include
    -std=c++17
    -fno-exceptions
    -D_GNU_SOURCE
    -D__STDC_CONSTANT_MACROS
    -D__STDC_FORMAT_MACROS
    -D__STDC_LIMIT_MACROS
    -std=c++17 -O3 -fno-rtti
)
add_custom_target(install_lib_ap ALL
    COMMAND mv libAutophase.so ${LOCAL}/lib/Autophase.so
    DEPENDS Autophase
)
#######################################
# Grewe
#######################################
add_executable(Grewe Grewe.cpp)

add_dependencies(Grewe llvm6)
add_dependencies(Grewe data_cl_include)
add_dependencies(Grewe libclc)

get_target_property(DATA_CL_INCLUDE data_cl_include path)
get_target_property(LIBCLC libclc path)

add_compile_definitions(LLVM_VERSION=${LLVM_VERSION6})

add_compile_definitions(OPENCL_C_H="${DATA_CL_INCLUDE}/opencl-c.h")
add_compile_definitions(OPENCL_BASE_H="${DATA_CL_INCLUDE}/opencl-c-base.h")
add_compile_definitions(CL_H="${OPENCL_HEADERS}/CL/cl.h")
add_compile_definitions(STRUCTS_H="${DATA_CL_INCLUDE}/structs.h")

add_compile_definitions(OPENCL_HEADERS="${OPENCL_HEADERS}")
add_compile_definitions(LIBCLC="${LIBCLC}")
add_compile_definitions(LIBCXX_HEADERS="${LLVM6}/lib/clang/${LLVM_VERSION6}.0.0/include")
add_compile_definitions(AUX_INCLUDE="${DATA_CL_INCLUDE}/aux_include/")

set_target_properties(Grewe PROPERTIES
    RUNTIME_OUTPUT_NAME ${LOCAL}/bin/Grewe
)
set_target_properties(Grewe PROPERTIES
    path ${LOCAL}/bin/Grewe
)

target_include_directories(Grewe PUBLIC ${LLVM6}/include/)

target_link_directories(Grewe PUBLIC ${LLVM6}/lib/)
target_link_directories(Grewe PUBLIC ${LLVM6}/lib/clang/${LLVM_VERSION6}.0.0/lib/linux/)

add_custom_target(cl_header_file ALL
  COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/cl.h ${CMAKE_CURRENT_BINARY_DIR}/cl.h
  COMMENT "Copy cl.h header file"
)

target_compile_options (Grewe PRIVATE
    -std=c++11
    -fno-exceptions
    -D_GNU_SOURCE
    -D__STDC_CONSTANT_MACROS
    -D__STDC_FORMAT_MACROS
    -D__STDC_LIMIT_MACROS
    -std=c++11 -O3 -fno-rtti
)
target_link_libraries(Grewe
"-Wl,-rpath,"
"-lrt"
"-ldl"
"-lpthread"
"-lm"
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
"-lLLVMLTO"
"-lLLVMPasses"
"-lLLVMObjCARCOpts"
"-lLLVMMIRParser"
"-lLLVMSymbolize"
"-lLLVMDebugInfoPDB"
"-lLLVMDebugInfoDWARF"
"-lLLVMCoverage"
"-lLLVMTableGen"
"-lLLVMDlltoolDriver"
"-lLLVMOrcJIT"
"-lLLVMXCoreDisassembler"
"-lLLVMXCoreCodeGen"
"-lLLVMXCoreDesc"
"-lLLVMXCoreInfo"
"-lLLVMXCoreAsmPrinter"
"-lLLVMSystemZDisassembler"
"-lLLVMSystemZCodeGen"
"-lLLVMSystemZAsmParser"
"-lLLVMSystemZDesc"
"-lLLVMSystemZInfo"
"-lLLVMSystemZAsmPrinter"
"-lLLVMSparcDisassembler"
"-lLLVMSparcCodeGen"
"-lLLVMSparcAsmParser"
"-lLLVMSparcDesc"
"-lLLVMSparcInfo"
"-lLLVMSparcAsmPrinter"
"-lLLVMPowerPCDisassembler"
"-lLLVMPowerPCCodeGen"
"-lLLVMPowerPCAsmParser"
"-lLLVMPowerPCDesc"
"-lLLVMPowerPCInfo"
"-lLLVMPowerPCAsmPrinter"
"-lLLVMNVPTXCodeGen"
"-lLLVMNVPTXDesc"
"-lLLVMNVPTXInfo"
"-lLLVMNVPTXAsmPrinter"
"-lLLVMMSP430CodeGen"
"-lLLVMMSP430Desc"
"-lLLVMMSP430Info"
"-lLLVMMSP430AsmPrinter"
"-lLLVMMipsDisassembler"
"-lLLVMMipsCodeGen"
"-lLLVMMipsAsmParser"
"-lLLVMMipsDesc"
"-lLLVMMipsInfo"
"-lLLVMMipsAsmPrinter"
"-lLLVMLanaiDisassembler"
"-lLLVMLanaiCodeGen"
"-lLLVMLanaiAsmParser"
"-lLLVMLanaiDesc"
"-lLLVMLanaiAsmPrinter"
"-lLLVMLanaiInfo"
"-lLLVMHexagonDisassembler"
"-lLLVMHexagonCodeGen"
"-lLLVMHexagonAsmParser"
"-lLLVMHexagonDesc"
"-lLLVMHexagonInfo"
"-lLLVMBPFDisassembler"
"-lLLVMBPFCodeGen"
"-lLLVMBPFAsmParser"
"-lLLVMBPFDesc"
"-lLLVMBPFInfo"
"-lLLVMBPFAsmPrinter"
"-lLLVMARMDisassembler"
"-lLLVMARMCodeGen"
"-lLLVMARMAsmParser"
"-lLLVMARMDesc"
"-lLLVMARMInfo"
"-lLLVMARMAsmPrinter"
"-lLLVMARMUtils"
"-lLLVMAMDGPUDisassembler"
"-lLLVMAMDGPUCodeGen"
"-lLLVMAMDGPUAsmParser"
"-lLLVMAMDGPUDesc"
"-lLLVMAMDGPUInfo"
"-lLLVMAMDGPUAsmPrinter"
"-lLLVMAMDGPUUtils"
"-lLLVMAArch64Disassembler"
"-lLLVMAArch64CodeGen"
"-lLLVMAArch64AsmParser"
"-lLLVMAArch64Desc"
"-lLLVMAArch64Info"
"-lLLVMAArch64AsmPrinter"
"-lLLVMAArch64Utils"
"-lLLVMObjectYAML"
"-lLLVMLibDriver"
"-lLLVMOption"
"-lLLVMWindowsManifest"
"-lLLVMFuzzMutate"
"-lLLVMX86Disassembler"
"-lLLVMX86AsmParser"
"-lLLVMX86CodeGen"
"-lLLVMGlobalISel"
"-lLLVMSelectionDAG"
"-lLLVMAsmPrinter"
"-lLLVMDebugInfoCodeView"
"-lLLVMDebugInfoMSF"
"-lLLVMX86Desc"
"-lLLVMMCDisassembler"
"-lLLVMX86Info"
"-lLLVMX86AsmPrinter"
"-lLLVMX86Utils"
"-lLLVMMCJIT"
"-lLLVMLineEditor"
"-lLLVMInterpreter"
"-lLLVMExecutionEngine"
"-lLLVMRuntimeDyld"
"-lLLVMCodeGen"
"-lLLVMTarget"
"-lLLVMCoroutines"
"-lLLVMipo"
"-lLLVMInstrumentation"
"-lLLVMVectorize"
"-lLLVMScalarOpts"
"-lLLVMLinker"
"-lLLVMIRReader"
"-lLLVMAsmParser"
"-lLLVMInstCombine"
"-lLLVMTransformUtils"
"-lLLVMBitWriter"
"-lLLVMAnalysis"
"-lLLVMProfileData"
"-lLLVMObject"
"-lLLVMMCParser"
"-lLLVMMC"
"-lLLVMBitReader"
"-lLLVMCore"
"-lLLVMBinaryFormat"
"-lLLVMSupport"
"-lLLVMDemangle"
"-pthread"
"-lLLVMTarget"
"-lLLVMMC"
"-lLLVMObject"
"-lLLVMCore"
"-lncurses"
"-lLLVMSupport"
"-lz"
)