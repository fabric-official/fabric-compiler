// src/backend/emitter.ts

import { IRModule, AgentIR } from "../ir/lowering";

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
export function emitCode(ir: IRModule, target: EmitTarget = "javascript"): EmitResult {
    switch (target) {
        case "javascript":
            return emitJS(ir);
        case "wasm":
            return emitWASM(ir);
        case "fabric-vm":
            return emitFabricVM(ir);
        default:
            throw new Error(`Unsupported target: ${target}`);
    }
}

// --- Private Emitters ---

function emitJS(ir: IRModule): EmitResult {
    let code = `"use strict";\n\n`;

    // Emit agent metadata
    for (const agent of ir.agents || []) {
        code += `// Agent: ${agent.name} (${agent.model_id})\n`;
        code += `const ${agent.name}_metadata = {\n`;
        code += `  name: "${agent.name}",\n`;
        code += `  model_id: "${agent.model_id}",\n`;
        code += `  inputs: ${JSON.stringify(agent.inputs)},\n`;
        code += `  outputs: ${JSON.stringify(agent.outputs)},\n`;

        // Emit policy as-is
        if (agent.policy) {
            code += `  policy: ${JSON.stringify(agent.policy, null, 2)},\n`;
        }

        if (agent.auditTrail) {
            code += `  auditTrail: true,\n`;
        }

        code += `};\n\n`;
    }

    // Emit functions
    for (const fn of ir.functions) {
        code += `function ${fn.name}() {\n`;
        for (const line of fn.instructions) {
            code += `  ${line}\n`;
        }
        code += `}\n\n`;
    }

    return { code };
}

function emitWASM(ir: IRModule): EmitResult {
    return { code: `;; WebAssembly backend not implemented` };
}

function emitFabricVM(ir: IRModule): EmitResult {
    return { code: `;; Fabric VM backend not implemented` };
}
