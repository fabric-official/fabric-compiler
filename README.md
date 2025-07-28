# Fabric Compiler

The **Fabric Compiler** powers the Fabric 1.0 Universal AI Fabric, built on MLIR and LLVM for policy-enforced AI execution.

## Features
- Custom DSL parser and compiler frontend
- MLIR dialect for FabricAtoms and agent execution
- LLVM backend with quantum and policy-aware extensions
- CLI tools (`fab build`, `fab audit`, etc.)
- SGX-secured runtime

## Directory Structure
- `compiler/` - Frontend, parser, grammar
- `mlir/` - MLIR dialect and passes
- `llvm/` - LLVM integrations
- `runtime/` - Agent-VM and enclave runtime
- `cli/` - CLI tooling
- `docs/` - Specifications and RFCs
- `tests/` - Unit and integration tests

## License
Apache 2.0
