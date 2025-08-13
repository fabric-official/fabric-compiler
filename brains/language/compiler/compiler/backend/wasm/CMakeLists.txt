cmake_minimum_required(VERSION 3.13.4)
project(wasm_codegen LANGUAGES CXX)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)

add_executable(wasm_codegen wasm_codegen.cpp)

target_link_libraries(wasm_codegen
  PRIVATE
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRTargetLLVMIRExport
    LLVMCore
    LLVMSupport
    LLVMWebAssemblyCodeGen
    LLVMWebAssemblyAsmPrinter
)

set_target_properties(wasm_codegen PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)
