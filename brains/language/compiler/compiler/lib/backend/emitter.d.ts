import { IRModule } from "../ir/lowering";
/**
 * Target languages for code emission.
 */
export type EmitTarget = "javascript" | "wasm" | "fabric-vm";
/**
 * Output of the emitter.
 */
export interface EmitResult {
    code: string;
    sourceMap?: string;
}
/**
 * Emits code from an IR module for a given target.
 * @param ir Intermediate representation module
 * @param target Output format (e.g. JS, WASM, Fabric VM)
 * @returns Emitted source code as a string
 */
export declare function emitCode(ir: IRModule, target?: EmitTarget): EmitResult;
