// src/ir/compiler.ts

import { ASTNode } from "../frontend/ast";
import { SemanticInfo } from "../frontend/checker";
import { IRModule, lowerToIR } from "./lowering";

/**
 * Compiles a Fabric AST to IRModule.
 * @param ast Parsed Fabric AST.
 * @param info Semantic metadata (type info, symbol tables, etc).
 * @returns Intermediate Representation (IR) module.
 */
export function compileToIR(ast: ASTNode, info: SemanticInfo): IRModule {
    return lowerToIR(ast, info);
}
