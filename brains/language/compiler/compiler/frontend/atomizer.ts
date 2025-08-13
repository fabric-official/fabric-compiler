type BitLiteral = any;
type AtomBlock = any;
type PolicyBlock = any;
type EntanglementRef = any;
// fab-project/compiler/atomizer.ts

import { CompilerError } from "./errors";

export interface FabricAtomIR {
    name: string;
    protons: BitLiteral[];
    electrons: BitLiteral[];
    mutableIndices: number[];
    energyBudget: number;
    entangledWith: string[];
}

export function transformToFabricAtom(ast: AtomBlock): FabricAtomIR {
    const { name, fields } = ast;

    let protons: BitLiteral[] = [];
    let electrons: BitLiteral[] = [];
    let mutableIndices: number[] = [];
    let energyBudget = 0;
    let entangledWith: string[] = [];

    for (const field of fields) {
        switch (field.kind) {
            case "protons":
                protons = field.value;
                if (protons.length !== 8)
                    throw new CompilerError(`Atom '${name}': protons[] must have exactly 8 bits.`);
                break;
            case "electrons":
                electrons = field.value;
                if (electrons.length !== 8)
                    throw new CompilerError(`Atom '${name}': electrons[] must have exactly 8 bits.`);
                break;
            case "policy":
                const policy: PolicyBlock = field;
                mutableIndices = policy.mutable;
                energyBudget = policy.energy_budget;

                if (mutableIndices.some(i => i < 0 || i > 7))
                    throw new CompilerError(`Atom '${name}': policy.mutable contains invalid bit indices.`);
                if (energyBudget <= 0)
                    throw new CompilerError(`Atom '${name}': energy_budget must be > 0.`);
                break;
            case "entangled_with":
                const ent: EntanglementRef = field;
                entangledWith = ent.agents;
                break;
            default:
                throw new CompilerError(`Atom '${name}': unrecognized field '${field.kind}'.`);
        }
    }

    if (!protons.length || !electrons.length)
        throw new CompilerError(`Atom '${name}': both protons[] and electrons[] are required.`);

    return {
        name,
        protons,
        electrons,
        mutableIndices,
        energyBudget,
        entangledWith,
    };
}

