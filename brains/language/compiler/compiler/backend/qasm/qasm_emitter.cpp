// qasm_emitter.cpp
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc("<input MLIR file>"), llvm::cl::init("-"));

static llvm::cl::opt<std::string> outputFilename(
    "o", llvm::cl::desc("Specify output QASM file"), llvm::cl::init("-"));

/// Emit OpenQASM text from any fab_quantum ops found in the module.
static LogicalResult emitQasm(ModuleOp module)
{
    std::string qasmText;
    llvm::raw_string_ostream os(qasmText);

    // Header
    os << "OPENQASM 2.0;\n";
    os << "include \"qelib1.inc\";\n\n";

    // Walk all operations in topological order
    module.walk([&](Operation *op)
                {
                    StringRef dialect = op->getName().getDialectNamespace();
                    StringRef name = op->getName().getStringRef();

                    if (dialect != "fab_quantum")
                        return;

                    // Example mapping:
                    if (name == "fab.quantum.h")
                    {
                        // assumes single operand that names qreg index
                        auto reg = op->getOperand(0);
                        os << "h " << reg << ";\n";
                    }
                    else if (name == "fab.quantum.rx")
                    {
                        double angle = op->getAttrOfType<FloatAttr>("angle").getValueAsDouble();
                        os << "rx(" << angle << ") " << op->getOperand(0) << ";\n";
                    }
                    else if (name == "fab.quantum.cnot")
                    {
                        auto lhs = op->getOperand(0);
                        auto rhs = op->getOperand(1);
                        os << "cx " << lhs << ", " << rhs << ";\n";
                    }
                    else if (name == "fab.quantum.u3")
                    {
                        auto theta = op->getAttrOfType<FloatAttr>("theta").getValueAsDouble();
                        auto phi = op->getAttrOfType<FloatAttr>("phi").getValueAsDouble();
                        auto lambda = op->getAttrOfType<FloatAttr>("lambda").getValueAsDouble();
                        os << "u3(" << theta << ", " << phi << ", " << lambda << ") "
                           << op->getOperand(0) << ";\n";
                    }
                    else if (name == "fab.quantum.measure")
                    {
                        os << "measure " << op->getOperand(0) << " -> " << op->getResult(0)
                           << ";\n";
                    }
                    // add further gates here...
                });

    // Write QASM text out
    std::error_code ec;
    llvm::raw_fd_ostream out(outputFilename, ec);
    if (ec)
    {
        llvm::errs() << "Failed to open output '" << outputFilename
                     << "': " << ec.message() << "\n";
        return failure();
    }
    out << os.str();
    return success();
}

int main(int argc, char **argv)
{
    llvm::cl::ParseCommandLineOptions(argc, argv,
                                      "Fabric OpenQASM Emitter\n");

    MLIRContext context;
    context.getOrLoadDialect("fab_quantum");

    // Load MLIR from file/stdin
    auto file = openInputFile(inputFilename);
    if (!file)
    {
        llvm::errs() << "Error opening input file: " << inputFilename << "\n";
        return 1;
    }
    SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*file), SMLoc());

    auto module = parseSourceFile<ModuleOp>(sourceMgr, &context);
    if (!module)
    {
        llvm::errs() << "Failed to parse MLIR module\n";
        return 1;
    }

    if (failed(emitQasm(*module)))
        return 1;

    return 0;
}
