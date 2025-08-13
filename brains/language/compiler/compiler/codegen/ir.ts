// 📁 compiler/codegen/ir.ts
// Fabric Intermediate Representation (IR) lowering for compiler backend

import {
    AtomNode,
    CoordinationBlockNode,
    ModuleNode,
    DeclarationNode,
    ASTNode,
} from "../frontend/ast";

// === IR Data Structures ===

export interface IRAtom {
    type: "IRAtom";
    name: string;
    protons: number[];
    electrons: number[];
    energy_budget: number;
    coord_id?: string;
    entangled_with?: string[];
    channel?: "quantum" | "classical";
    collapse_mode?: "joint" | "sequential" | "stochastic";
}

export interface IRCoordinationBlock {
    type: "IRCoordinationBlock";
    name: string;
    agents: string[];
    protocol: string;
    on_commit: string;
}

export type IRDeclaration = IRAtom | IRCoordinationBlock;

export interface IRModule {
    type: "IRModule";
    name: string;
    declarations: IRDeclaration[];
}

// === IR Lowering ===

export function lowerFromAST(module: ModuleNode): IRModule {
    const declarations: IRDeclaration[] = [];

    for (const decl of module.declarations) {
        if (decl.kind === "Atom") {
            declarations.push(lowerAtom(decl));
        } else if (decl.kind === "CoordinationBlock") {
            declarations.push(lowerCoordBlock(decl));
        }
    }

    return {
        type: "IRModule",
        name: module.loc ? module.loc.start.line.toString() : "anonymous",
        declarations,
    };
}

function lowerAtom(atom: AtomNode): IRAtom {
    return {
        type: "IRAtom",
        name: atom.name,
        protons: atom.protons.map(b => b.value),
        electrons: atom.electrons.map(b => b.value),
        energy_budget: atom.policy.energy_budget,
        coord_id: atom.policy.coord_id || undefined,
        entangled_with: atom.entangled_with || [],
        channel: atom.policy.channel || "classical",
        collapse_mode: atom.policy.collapse_mode || "sequential",
    };
}

function lowerCoordBlock(block: CoordinationBlockNode): IRCoordinationBlock {
    return {
        type: "IRCoordinationBlock",
        name: block.name,
        agents: block.agents,
        protocol: block.protocol,
        on_commit: block.on_commit,
    };
}