# Dockerfile
FROM ubuntu:22.04

# 1) Install tools and dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 2) Fetch and build LLVM + MLIR
WORKDIR /opt
RUN git clone --depth 1 --branch llvmorg-15.0.7 https://github.com/llvm/llvm-project.git
RUN mkdir llvm-project/build && cd llvm-project/build && \
    cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="clang;mlir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/llvm-install && \
    ninja && \
    ninja install

# 3) Bring in your Fabric compiler sources
WORKDIR /opt
COPY compiler /opt/fabric-compiler

# 4) Configure & build your compiler against the installed LLVM/MLIR
WORKDIR /opt/fabric-compiler
RUN mkdir build && cd build && \
    cmake -G Ninja .. \
    -DLLVM_DIR=/opt/llvm-install/lib/cmake/llvm \
    -DMLIR_DIR=/opt/llvm-install/lib/cmake/mlir \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/fabric-install && \
    ninja && ninja install

# 5) Final image just holds the installed binaries
FROM ubuntu:22.04
COPY --from=0 /opt/fabric-install /usr/local
ENTRYPOINT ["/usr/local/bin/fabc"]
