cmake_minimum_required(VERSION 3.20)
project(FabricCompiler LANGUAGES C CXX)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH
  "${LLVM_CMAKE_DIR}"
  "${MLIR_CMAKE_DIR}"
  "${MLIR_DIR}"
)
include(HandleLLVMOptions)
include(AddLLVM)
include(AddMLIR)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

add_subdirectory(mlir/lib/Dialect/Fabric)

add_executable(fabc
  mlir/driver/main.cpp
)
target_link_libraries(fabc PRIVATE
  FabricDialect
  MLIRIR MLIRParser MLIRPass MLIRTransforms
  LLVMCore LLVMSupport
)

install(TARGETS fabc RUNTIME DESTINATION bin)









































