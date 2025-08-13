@echo off
cd /d D:\fab-project\compiler
echo Cleaning build...
rmdir /s /q build
mkdir build

echo Regenerating TableGen files...
cd mlir
call gen_fabric_ops.bat

echo Configuring with CMake...
cd /d D:\fab-project\compiler\build
cmake -G Ninja ^
  -DLLVM_DIR=..\llvm-project\install\lib\cmake\llvm ^
  -DMLIR_DIR=..\llvm-project\install\lib\cmake\mlir ^
  ..

echo Building Fabric Dialect...
ninja
echo Build complete.
pause
