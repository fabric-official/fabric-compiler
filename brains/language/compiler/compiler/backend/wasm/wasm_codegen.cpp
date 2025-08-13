#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

static LogicalResult lowerToWasm(ModuleOp module)
{
    MLIRContext &ctx = module.getContext();
    ctx.getOrLoadDialect<LLVM::LLVMDialect>();
    ctx.getOrLoadDialect<func::FuncDialect>();

    // 1) Lower Func → LLVM → core → LLVM IR
    PassManager pm(&ctx);
    pm.addPass(createConvertFuncToLLVMPass());
    pm.addPass(createConvertStdToLLVMPass());
    pm.addPass(createLowerToLLVMPass());
    if (failed(pm.run(module)))
    {
        llvm::errs() << "MLIR lowering to LLVM dialect failed\n";
        return failure();
    }

    // 2) Translate MLIR module to LLVM IR
    std::unique_ptr<llvm::Module> llvmModule = translateModuleToLLVMIR(module);
    if (!llvmModule)
    {
        llvm::errs() << "Translation to LLVM IR failed\n";
        return failure();
    }

    // 3) Initialize WebAssembly backend
    LLVMInitializeWebAssemblyTarget();
    LLVMInitializeWebAssemblyTargetInfo();
    LLVMInitializeWebAssemblyTargetMC();
    LLVMInitializeWebAssemblyAsmPrinter();

    // 4) Create a wasm32 target machine
    std::string err;
    auto triple = llvm::Triple("wasm32-unknown-unknown-wasm");
    const llvm::Target *target = llvm::TargetRegistry::lookupTarget(triple.getTriple(), err);
    if (!target)
    {
        llvm::errs() << "Unable to find WASM target: " << err << "\n";
        return failure();
    }

    llvm::TargetOptions opts;
    auto rm = llvm::Optional<llvm::Reloc::Model>();
    std::unique_ptr<llvm::TargetMachine> tm(
        target->createTargetMachine(triple.getTriple(), "generic", "", opts, rm));
    if (!tm)
    {
        llvm::errs() << "Failed to create WASM TargetMachine\n";
        return failure();
    }

    llvmModule->setDataLayout(tm->createDataLayout());
    llvmModule->setTargetTriple(triple.getTriple());

    // 5) Emit a .wasm object to stdout
    llvm::legacy::PassManager passMgr;
    if (tm->addPassesToEmitFile(passMgr, llvm::outs(), nullptr, llvm::CGFT_ObjectFile))
    {
        llvm::errs() << "TargetMachine can't emit a file of this type\n";
        return failure();
    }
    passMgr.run(*llvmModule);

    return success();
}

int main(int argc, char **argv)
{
    llvm::InitLLVM y(argc, argv);
    mlir::registerAllDialects(mlir::DialectRegistry::getGlobal());
    mlir::registerAllPasses();

    if (argc < 2)
    {
        llvm::errs() << "Usage: wasm_codegen <input.mlir>\n";
        return 1;
    }

    llvm::SourceMgr sourceMgr;
    auto fileOrErr = mlir::openInputFile(argv[1]);
    if (!fileOrErr)
    {
        llvm::errs() << "Error opening file: " << argv[1] << "\n";
        return 1;
    }
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

    MLIRContext context;
    context.loadDialect<LLVM::LLVMDialect>();
    context.loadDialect<func::FuncDialect>();

    auto module = mlir::parseSourceFile<ModuleOp>(sourceMgr, &context);
    if (!module)
    {
        llvm::errs() << "Failed to parse MLIR module\n";
        return 1;
    }

    if (failed(lowerToWasm(*module)))
        return 1;

    return 0;
}
