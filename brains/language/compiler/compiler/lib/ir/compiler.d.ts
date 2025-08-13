import { ASTNode } from "../frontend/ast";
import { SemanticInfo } from "../frontend/checker";
import { IRModule } from "./lowering";
/**
 * Compiles a Fabric AST to IRModule.
 * @param ast Parsed Fabric AST.
 * @param info Semantic metadata (type info, symbol tables, etc).
 * @returns Intermediate Representation (IR) module.
 */
export declare function compileToIR(ast: ASTNode, info: SemanticInfo): IRModule;
