#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Parser.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

static LogicalResult lowerToLLVMIR(mlir::ModuleOp module)
{
    MLIRContext &context = module.getContext();

    context.getOrLoadDialect<LLVM::LLVMDialect>();
    PassManager pm(&context);
    pm.addPass(createConvertStdToLLVMPass());
    pm.addPass(createConvertMathToLLVMPass());
    pm.addPass(createLowerToLLVMPass());

    if (failed(pm.run(module)))
    {
        llvm::errs() << "❌ Failed to run lowering passes\n";
        return failure();
    }

    std::unique_ptr<llvm::Module> llvmModule = translateModuleToLLVMIR(module);
    if (!llvmModule)
    {
        llvm::errs() << "❌ Translation to LLVM IR failed\n";
        return failure();
    }

    // Inject FabricAtom LLVM struct type for emission
    llvm::LLVMContext &llvmCtx = llvmModule->getContext();
    llvm::IRBuilder<> builder(llvmCtx);

    llvm::Type *i1 = llvm::Type::getInt1Ty(llvmCtx);
    llvm::Type *i8 = llvm::Type::getInt8Ty(llvmCtx);
    llvm::Type *i32 = llvm::Type::getInt32Ty(llvmCtx);
    llvm::Type *i64 = llvm::Type::getInt64Ty(llvmCtx);
    llvm::ArrayType *bitArray8 = llvm::ArrayType::get(i1, 8);
    llvm::PointerType *strPtr = llvm::Type::getInt8PtrTy(llvmCtx);
    llvm::PointerType *strPtrPtr = llvm::PointerType::get(strPtr, 0); // for entangled_with[]

    llvm::StructType *fabricAtomTy = llvm::StructType::create(
        llvmCtx,
        {
            bitArray8, // protons[8]
            bitArray8, // electrons[8]
            i32,       // energy_budget
            strPtr,    // policy_json
            strPtr,    // hash
            strPtr,    // atom_id
            i32,       // coord_id
            strPtrPtr, // entangled_with[]
            strPtr,    // channel
            strPtr     // collapse_mode
        },
        "struct.FabricAtom");

    llvm::outs() << "\n; --- FabricAtom LLVM Struct Definition ---\n";
    fabricAtomTy->print(llvm::outs());
    llvm::outs() << "\n; ----------------------------------------\n\n";

    // Emit the lowered IR
    llvmModule->print(llvm::outs(), nullptr);
    return success();
}

int main(int argc, char **argv)
{
    mlir::registerAllDialects(getGlobalDialectRegistry());
    mlir::registerAllPasses();

    if (argc < 2)
    {
        llvm::errs() << "Usage: llvm_codegen <input.mlir>\n";
        return 1;
    }

    MLIRContext context;
    context.loadDialect<LLVM::LLVMDialect>();
    llvm::SourceMgr sourceMgr;
    auto fileOrErr = mlir::openInputFile(argv[1]);
    if (failed(fileOrErr.takeError()))
    {
        llvm::errs() << "❌ Error opening file: " << argv[1] << "\n";
        return 1;
    }
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

    OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);
    if (!module)
    {
        llvm::errs() << "❌ Failed to parse MLIR file\n";
        return 1;
    }

    if (failed(lowerToLLVMIR(*module)))
    {
        return 1;
    }

    return 0;
}
