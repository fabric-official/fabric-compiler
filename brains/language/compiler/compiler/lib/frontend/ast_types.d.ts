export interface ASTNode {
    kind: string;
    [key: string]: any;
}
export interface AtomBlock extends ASTNode {
    kind: "Atom";
    name: string;
    protons: BitLiteral[];
    electrons: BitLiteral[];
    policy: AtomPolicy;
}
export interface BitLiteral {
    kind: "Bit";
    value: 0 | 1;
}
export interface AtomPolicy {
    mutable: number[];
    energy_budget: number;
}
export interface PolicyBlock extends ASTNode {
    kind: "Policy";
    entries: any[];
}
export interface CoordinationBlock extends ASTNode {
    kind: "CoordinationBlock";
    entries: CoordinationEntry[];
}
export type CoordinationEntry = EntangleEntry | ChannelEntry | CollapseEntry;
export interface EntangleEntry extends ASTNode {
    kind: "EntangleEntry";
    atoms: string[];
}
export interface ChannelEntry extends ASTNode {
    kind: "ChannelEntry";
    name: string;
}
export interface CollapseEntry extends ASTNode {
    kind: "CollapseEntry";
    trigger: string;
}
