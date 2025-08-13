//===- classical_ir.mlir --------------------------------------*- tablegen -*-===//
//
// Dialect Definition for Classical IR
//
//===----------------------------------------------------------------------===//

include "mlir/IR/Dialect.h"
include "mlir/IR/Builders.h"
include "mlir/IR/OpBase.td"

///===----------------------------------------------------------------------===//
/// Dialect
///===----------------------------------------------------------------------===///

def Classical_Dialect : Dialect<"classical", "classical", > {
  let summary = "Classical IR dialect";
  let description = [{ This dialect defines integer/float ops,
                       memory ops, control flow, calls and returns. }];
}

///===----------------------------------------------------------------------===//
/// Types
///===----------------------------------------------------------------------===///

def Classical_IntType : Type<Classical_Dialect, MLIRType> {
  let description = "Fixed-width signed integer type";
  let storageType = "mlir::IntegerType";
}

def Classical_FloatType : Type<Classical_Dialect, MLIRType> {
  let description = "Floating-point type";
  let storageType = "mlir::FloatType";
}

def Classical_MemRefType : Type<Classical_Dialect, MLIRType> {
  let description = "Pointer to memory (memref) with element type";
  let storageType = "mlir::MemRefType";
}

// ---- FabricAtom Type ----

def Classical_FabricAtomType : Type<Classical_Dialect, MLIRType> {
  let description = "FabricAtom composed of 8 protons and 8 mutable electrons";
  let storageType = "mlir::classical::FabricAtomType";
}

// ---- AtomPolicy Attribute ----

def Classical_AtomPolicyAttr : Attr<Classical_Dialect, "atom_policy"> {
  let description = "Per-bit mutation policy including mutability and energy_budget";
  let parameters = (ins StringRefAttr:$json);
  let assemblyFormat = "$json";
}

///===----------------------------------------------------------------------===//
/// Operations
///===----------------------------------------------------------------------===///

def Classical_AddOp : Op<Classical_Dialect, [Pure, Commutative]> {
  let summary = "integer or floating-point addition";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results   = (outs AnyType:$sum);
  let assemblyFormat = [{
    $name ` ` $lhs `, ` $rhs ` : ` $sum.type
  }];
}

def Classical_SubOp : Op<Classical_Dialect, [Pure]> {
  let summary = "integer or floating-point subtraction";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results   = (outs AnyType:$diff);
  let assemblyFormat = [{
    $name ` ` $lhs `, ` $rhs ` : ` $diff.type
  }];
}

def Classical_MulOp : Op<Classical_Dialect, [Pure]> {
  let summary = "integer or floating-point multiply";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results   = (outs AnyType:$prod);
  let assemblyFormat = [{
    $name ` ` $lhs `, ` $rhs ` : ` $prod.type
  }];
}

def Classical_DivOp : Op<Classical_Dialect, [Pure]> {
  let summary = "integer or floating-point division";
  let arguments = (ins AnyType:$num, AnyType:$den);
  let results   = (outs AnyType:$quo);
  let assemblyFormat = [{
    $name ` ` $num `, ` $den ` : ` $quo.type
  }];
}

def Classical_CmpOp : Op<Classical_Dialect, [Pure]> {
  let summary = "compare two values";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, StringAttr:$predicate);
  let results   = (outs i1:$cond);
  let assemblyFormat = [{
    $name ` ` $predicate `, ` $lhs `, ` $rhs
  }];
}

def Classical_LoadOp : Op<Classical_Dialect> {
  let summary = "load from memory";
  let arguments = (ins Classical_MemRefType:$ptr);
  let results   = (outs AnyType:$val);
  let assemblyFormat = [{
    $name ` ` $ptr ` : ` $ptr.type
  }];
}

def Classical_StoreOp : Op<Classical_Dialect> {
  let summary = "store to memory";
  let arguments = (ins AnyType:$val, Classical_MemRefType:$ptr);
  let results   = (outs);
  let assemblyFormat = [{
    $name ` ` $val `, ` $ptr ` : ` $ptr.type
  }];
}

def Classical_BranchOp : Op<Classical_Dialect, [HasBranchSuccs]> {
  let summary = "conditional branch";
  let arguments = (ins i1:$cond);
  let successors = (ins RegionSuccessor:$true, RegionSuccessor:$false);
  let assemblyFormat = [{
    $name ` ` $cond `, ^bb` $true.dest `, ^bb` $false.dest
  }];
}

def Classical_ReturnOp : Op<Classical_Dialect, [IsTerminator]> {
  let summary = "return from function";
  let arguments = (ins Variadic<AnyType>:$retval);
  let assemblyFormat = [{
    $name `(` $retval `)`
  }];
}

///===----------------------------------------------------------------------===//
/// Dialect Registration
///===----------------------------------------------------------------------===///

def Dialect_Classical : DialectLibrary<"classical"> {
  let dialectClass = #classical::Classical_Dialect;
  let types = [
    #classical::Classical_IntType,
    #classical::Classical_FloatType,
    #classical::Classical_MemRefType,
    #classical::Classical_FabricAtomType
  ];
  let operations = [
    #classical::Classical_AddOp,
    #classical::Classical_SubOp,
    #classical::Classical_MulOp,
    #classical::Classical_DivOp,
    #classical::Classical_CmpOp,
    #classical::Classical_LoadOp,
    #classical::Classical_StoreOp,
    #classical::Classical_BranchOp,
    #classical::Classical_ReturnOp
  ];
}