// src/backend/generator.ts

import { parse } from "../frontend/parser";
import { checkSemantics } from "../frontend/checker";
import { compileToIR } from "../ir/compiler";
import { emitCode, EmitResult, EmitTarget } from "./emitter";

/**
 * Compiles Fabric source code to an output target (e.g. JS, WASM).
 * @param source Raw source code string
 * @param target Output format ("javascript" | "wasm" | "fabric-vm")
 * @returns Final compiled output with code (and optional sourcemap)
 */
export function generate(source: string, target: EmitTarget = "javascript"): EmitResult {
    const ast = parse(source);
    const semanticInfo = checkSemantics(ast);
    const ir = compileToIR(ast, semanticInfo);
    return emitCode(ir, target);
}
