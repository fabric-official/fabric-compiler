"use strict";
// src/backend/generator.ts
Object.defineProperty(exports, "__esModule", { value: true });
exports.generate = generate;
const parser_1 = require("../frontend/parser");
const checker_1 = require("../frontend/checker");
const compiler_1 = require("../ir/compiler");
const emitter_1 = require("./emitter");
/**
 * Compiles Fabric source code to an output target (e.g. JS, WASM).
 * @param source Raw source code string
 * @param target Output format ("javascript" | "wasm" | "fabric-vm")
 * @returns Final compiled output with code (and optional sourcemap)
 */
function generate(source, target = "javascript") {
    const ast = (0, parser_1.parse)(source);
    const semanticInfo = (0, checker_1.checkSemantics)(ast);
    const ir = (0, compiler_1.compileToIR)(ast, semanticInfo);
    return (0, emitter_1.emitCode)(ir, target);
}
//# sourceMappingURL=generator.js.map