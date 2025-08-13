"use strict";
// src/ir/compiler.ts
Object.defineProperty(exports, "__esModule", { value: true });
exports.compileToIR = compileToIR;
const lowering_1 = require("./lowering");
/**
 * Compiles a Fabric AST to IRModule.
 * @param ast Parsed Fabric AST.
 * @param info Semantic metadata (type info, symbol tables, etc).
 * @returns Intermediate Representation (IR) module.
 */
function compileToIR(ast, info) {
    return (0, lowering_1.lowerToIR)(ast, info);
}
//# sourceMappingURL=compiler.js.map