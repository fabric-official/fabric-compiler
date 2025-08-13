// src/ir/lowering.ts

import {
    ASTNode,
    ModuleNode,
    AgentNode,
    ExecutionBlockNode,
    PolicyNode,
    PolicyEntry,
} from "../frontend/ast";
import { SemanticInfo } from "../frontend/checker";
import { IRModule, AgentIR, IRFunction } from "./schema";

/**
 * Lowers an AST + semantic metadata to an IR module.
 * @param ast Fabric AST (ModuleNode)
 * @param info Semantic info (unused for now)
 * @returns Lowered IR module
 */
export function lowerToIR(ast: ASTNode, info: SemanticInfo): IRModule {
    if (ast.kind !== "Module") {
        throw new Error(`Unsupported AST root kind: ${ast.kind}`);
    }

    const mod = ast as ModuleNode;

    const agents: AgentIR[] = [];
    const functions: IRFunction[] = [];

    for (const decl of mod.declarations) {
        if (decl.kind === "Agent") {
            const agent = decl as AgentNode;

            const irAgent: AgentIR = {
                name: agent.name,
                model_id: agent.modelId,
                inputs: agent.inputs.map(i => i.type),
                outputs: agent.outputs.map(o => o.type),
                policy: extractPolicy(agent.policy),
                auditTrail: agent.device === "ledger"
            };

            agents.push(irAgent);
        }

        if (decl.kind === "ExecutionBlock") {
            const block = decl as ExecutionBlockNode;
            functions.push({
                name: block.name,
                instructions: [`// ExecutionBlock for ${block.name}`]
            });
        }
    }

    return { agents, functions };
}

/**
 * Extracts policy as key-value pairs from AST PolicyNode
 */
function extractPolicy(policy?: PolicyNode): Record<string, any> | undefined {
    if (!policy) return undefined;

    const result: Record<string, any> = {};

    for (const entry of policy.entries) {
        if ((entry as any).key === "fairness") {
            result[entry.key] = (entry as any).weights;
        } else {
            result[entry.key] = (entry as any).value;
        }
    }

    return result;
}
