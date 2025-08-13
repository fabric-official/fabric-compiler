// compiler/ir/qir.mlir
//===- qir.mlir --------------------------------------*- tablegen -*-===//
//
// Dialect Definition for Quantum IR (QIR-like)
//===----------------------------------------------------------------------===//

include "mlir/IR/Dialect.h"
include "mlir/IR/Builders.h"
include "mlir/IR/OpBase.td"

///===----------------------------------------------------------------------===//
/// Dialect
///===----------------------------------------------------------------------===///

def QIR_Dialect : Dialect<"qir", "qir", > {
  let summary = "Quantum IR dialect";
  let description = [{ Defines qubit management, gates, measurement, and classical interaction. }];
}

///===----------------------------------------------------------------------===//
/// Types
///===----------------------------------------------------------------------===///

def QIR_QubitType : Type<QIR_Dialect, MLIRType> {
  let description = "Type representing a quantum bit";
  let storageType = "mlir::OpaqueType";
}

def QIR_ResultType : Type<QIR_Dialect, MLIRType> {
  let description = "Type representing measurement result (bool)";
  let storageType = "mlir::IntegerType";
}

///===----------------------------------------------------------------------===//
/// Operations
///===----------------------------------------------------------------------===///

def QIR_AllocQubitOp : Op<QIR_Dialect> {
  let summary = "allocate a qubit";
  let results = (outs QIR_QubitType:$q);
  let assemblyFormat = [{$name}];
}

def QIR_FreeQubitOp : Op<QIR_Dialect> {
  let summary = "release a qubit";
  let arguments = (ins QIR_QubitType:$q);
  let assemblyFormat = [{$name ` ` $q}];
}

def QIR_XGateOp : Op<QIR_Dialect, [Pure]> {
  let summary = "apply Pauli-X gate";
  let arguments = (ins QIR_QubitType:$q);
  let assemblyFormat = [{$name ` ` $q}];
}

def QIR_HGateOp : Op<QIR_Dialect, [Pure]> {
  let summary = "apply Hadamard gate";
  let arguments = (ins QIR_QubitType:$q);
  let assemblyFormat = [{$name ` ` $q}];
}

def QIR_CNOTGateOp : Op<QIR_Dialect, [Pure]> {
  let summary = "apply CNOT gate";
  let arguments = (ins QIR_QubitType:$c, QIR_QubitType:$t);
  let assemblyFormat = [{$name ` ` $c `, ` $t}];
}

def QIR_MeasureOp : Op<QIR_Dialect> {
  let summary = "measure a qubit";
  let arguments = (ins QIR_QubitType:$q);
  let results   = (outs QIR_ResultType:$r);
  let assemblyFormat = [{$name ` ` $q ` : ` $r.type}];
}

def QIR_ResetOp : Op<QIR_Dialect> {
  let summary = "reset a qubit to |0⟩";
  let arguments = (ins QIR_QubitType:$q);
  let assemblyFormat = [{$name ` ` $q}];
}

def QIR_ReturnOp : Op<QIR_Dialect, [IsTerminator]> {
  let summary = "return from quantum function";
  let arguments = (ins Variadic<AnyType>:$rets);
  let assemblyFormat = [{$name `(` $rets `)`}];
}

///===----------------------------------------------------------------------===//
/// Dialect Registration
///===----------------------------------------------------------------------===///

def Dialect_QIR : DialectLibrary<"qir"> {
  let dialectClass = #qir::QIR_Dialect;
  let types = [
    #qir::QIR_QubitType,
    #qir::QIR_ResultType
  ];
  let operations = [
    #qir::QIR_AllocQubitOp,
    #qir::QIR_FreeQubitOp,
    #qir::QIR_XGateOp,
    #qir::QIR_HGateOp,
    #qir::QIR_CNOTGateOp,
    #qir::QIR_MeasureOp,
    #qir::QIR_ResetOp,
    #qir::QIR_ReturnOp
  ];
}

