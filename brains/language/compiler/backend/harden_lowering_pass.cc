#include <string>
#include <unordered_set>
#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
using namespace mlir; namespace {
struct HardenLoweringPass:public PassWrapper<HardenLoweringPass,OperationPass<ModuleOp>>{
 void runOnOperation() override {
  ModuleOp m=getOperation();
  std::unordered_set<std::string> banned={"llvm.trap","llvm.memcpy","llvm.memmove","llvm.memset"};
  m.walk([&](Operation*op){std::string name=op->getName().getStringRef().str(); if(banned.count(name)){op->emitError()<<"banned intrinsic in lowered IR: "<<name; signalPassFailure();}});
 }};}
std::unique_ptr<Pass> createHardenLoweringPass(){return std::make_unique<HardenLoweringPass>();}
