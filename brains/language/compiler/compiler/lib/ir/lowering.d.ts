import { ASTNode } from "../frontend/ast";
import { SemanticInfo } from "../frontend/checker";
/**
 * Represents a lowered IR instruction or structure.
 * This should eventually contain opcodes, types, etc.
 */
export interface IRModule {
    functions: IRFunction[];
}
/**
 * Represents a single function in IR form.
 */
export interface IRFunction {
    name: string;
    instructions: string[];
}
/**
 * Lowers an AST + semantic metadata to an IR module.
 * @param ast Fabric AST
 * @param info Semantic info (e.g., type bindings)
 * @returns Lowered IR module
 */
export declare function lowerToIR(ast: ASTNode, info: SemanticInfo): IRModule;
