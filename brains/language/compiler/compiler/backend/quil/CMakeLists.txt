cmake_minimum_required(VERSION 3.13.4)
project(quil_emitter LANGUAGES CXX)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(quil_emitter quil_emitter.cpp)

target_link_libraries(quil_emitter
  PRIVATE
    MLIRIR
    MLIRParser
    MLIRSupport
    MLIRLLVMDialect
    LLVMSupport
)

# Pull in include paths, compile flags, etc.
llvm_update_compile_flags(quil_emitter)

