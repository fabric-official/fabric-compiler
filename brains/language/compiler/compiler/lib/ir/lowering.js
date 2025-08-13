"use strict";
// src/ir/lowering.ts
Object.defineProperty(exports, "__esModule", { value: true });
exports.lowerToIR = lowerToIR;
/**
 * Lowers an AST + semantic metadata to an IR module.
 * @param ast Fabric AST
 * @param info Semantic info (e.g., type bindings)
 * @returns Lowered IR module
 */
function lowerToIR(ast, info) {
    // Placeholder: convert function stubs
    const functions = [];
    // Example: traverse AST and generate dummy IR
    if (Array.isArray(ast.body)) {
        for (const node of ast.body) {
            if (node.type === "FunctionDecl") {
                functions.push({
                    name: node.name,
                    instructions: [`// Lowered function ${node.name}`]
                });
            }
        }
    }
    return { functions };
}
//# sourceMappingURL=lowering.js.map