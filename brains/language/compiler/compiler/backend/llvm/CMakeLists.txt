cmake_minimum_required(VERSION 3.15)
project(fabric_llvm_codegen LANGUAGES CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find MLIR & LLVM (built with `-DLLVM_ENABLE_PROJECTS="mlir"`).
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "MLIR found at ${MLIR_DIR}")
message(STATUS "LLVM found at ${LLVM_DIR}")

# Include paths and defs
include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
add_definitions(${MLIR_DEFINITIONS} ${LLVM_DEFINITIONS})

# Our codegen executable
add_executable(llvm_codegen
  llvm_codegen.cpp
)

# Link in MLIR core, parser, pass infrastructure, transforms, and the LLVM IR translation
target_link_libraries(llvm_codegen PRIVATE
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTargetLLVMIR
  MLIRSupport
  LLVMCore
  LLVMSupport
)

# Helper macros to ensure all transitive libs are linked
mlir_check_all_link_libraries(llvm_codegen)

# Map and link necessary LLVM components (core + support)
llvm_map_components_to_libnames(llvm_codegen_libs core support)
target_link_libraries(llvm_codegen PRIVATE ${llvm_codegen_libs})



