import { EmitResult, EmitTarget } from "./emitter";
/**
 * Compiles Fabric source code to an output target (e.g. JS, WASM).
 * @param source Raw source code string
 * @param target Output format ("javascript" | "wasm" | "fabric-vm")
 * @returns Final compiled output with code (and optional sourcemap)
 */
export declare function generate(source: string, target?: EmitTarget): EmitResult;
