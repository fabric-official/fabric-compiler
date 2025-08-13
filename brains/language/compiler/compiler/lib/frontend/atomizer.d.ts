type BitLiteral = any;
type AtomBlock = any;
export interface FabricAtomIR {
    name: string;
    protons: BitLiteral[];
    electrons: BitLiteral[];
    mutableIndices: number[];
    energyBudget: number;
    entangledWith: string[];
}
export declare function transformToFabricAtom(ast: AtomBlock): FabricAtomIR;
export {};
