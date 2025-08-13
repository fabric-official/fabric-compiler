"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.transformToFabricAtom = transformToFabricAtom;
// fab-project/compiler/atomizer.ts
const errors_1 = require("./errors");
function transformToFabricAtom(ast) {
    const { name, fields } = ast;
    let protons = [];
    let electrons = [];
    let mutableIndices = [];
    let energyBudget = 0;
    let entangledWith = [];
    for (const field of fields) {
        switch (field.kind) {
            case "protons":
                protons = field.value;
                if (protons.length !== 8)
                    throw new errors_1.CompilerError(`Atom '${name}': protons[] must have exactly 8 bits.`);
                break;
            case "electrons":
                electrons = field.value;
                if (electrons.length !== 8)
                    throw new errors_1.CompilerError(`Atom '${name}': electrons[] must have exactly 8 bits.`);
                break;
            case "policy":
                const policy = field;
                mutableIndices = policy.mutable;
                energyBudget = policy.energy_budget;
                if (mutableIndices.some(i => i < 0 || i > 7))
                    throw new errors_1.CompilerError(`Atom '${name}': policy.mutable contains invalid bit indices.`);
                if (energyBudget <= 0)
                    throw new errors_1.CompilerError(`Atom '${name}': energy_budget must be > 0.`);
                break;
            case "entangled_with":
                const ent = field;
                entangledWith = ent.agents;
                break;
            default:
                throw new errors_1.CompilerError(`Atom '${name}': unrecognized field '${field.kind}'.`);
        }
    }
    if (!protons.length || !electrons.length)
        throw new errors_1.CompilerError(`Atom '${name}': both protons[] and electrons[] are required.`);
    return {
        name,
        protons,
        electrons,
        mutableIndices,
        energyBudget,
        entangledWith,
    };
}
//# sourceMappingURL=atomizer.js.map