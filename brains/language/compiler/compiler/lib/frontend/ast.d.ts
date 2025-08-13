/** Source location for error reporting */
export interface SourceLocation {
    start: {
        line: number;
        column: number;
    };
    end: {
        line: number;
        column: number;
    };
}
/** Base AST node */
export interface Node {
    kind: string;
    loc?: SourceLocation;
}
/** Top-level module */
export interface ModuleNode extends Node {
    kind: "Module";
    name: string;
    imports: ImportNode[];
    declarations: TopLevelNode[];
}
/** Import statement */
export interface ImportNode extends Node {
    kind: "Import";
    modules: string[];
}
/** Device declaration */
export interface DeviceNode extends Node {
    kind: "Device";
    name: string;
    caps: CapNode[];
    policy: PolicyNode;
}
/** Capability spec */
export interface CapNode {
    kind: "Cap";
    loc?: SourceLocation;
    name: string;
    spec?: string;
    target?: string;
}
/** Policy block */
export interface PolicyNode extends Node {
    kind: "Policy";
    entries: GenericPolicyEntry[];
}
/** Generic flexible policy entry */
export interface GenericPolicyEntry {
    key: string;
    value: string | number | boolean;
}
/** Goal block */
export interface GoalNode extends Node {
    kind: "Goal";
    description: string;
    constraints?: ConstraintNode[];
    optimizeFor: string[];
}
export interface ConstraintNode {
    kind?: "Constraint";
    metric: string;
    comparator: string;
    value: number;
    unit: string;
}
/** Type alias */
export interface TypeAliasNode extends Node {
    kind: "TypeAlias";
    name: string;
    target: TypeExpr;
}
/** Type expression */
export type TypeExpr = {
    kind: "Identifier";
    name: string;
    size?: number;
} | {
    kind: "Union";
    options: TypeExpr[];
};
/** Agent declaration */
export interface AgentNode extends Node {
    kind: "Agent";
    name: string;
    id: string;
    modelId: string;
    inputs: IODecl[];
    outputs: IODecl[];
    learns?: string;
    explain?: string[];
    device?: string;
    fallback?: {
        impl: string;
        state: string;
    };
    policy?: PolicyNode;
}
export interface IODecl {
    alias?: string;
    type: string;
    device?: string;
}
/** Execution block */
export interface ExecutionBlockNode extends Node {
    kind: "ExecutionBlock";
    name: string;
    attrs: {
        [key: string]: string;
    };
    block: {
        blockType: string;
        agent: string;
        entry: string;
        inputs: IODecl[];
        outputs: IODecl[];
    };
    policy?: PolicyNode;
}
/** Polyglot block */
export interface PolyBlockNode extends Node {
    kind: "PolyBlock";
    name: string;
    lang: string;
    code: string;
    entry: string;
    inputs: IODecl[];
    outputs: IODecl[];
    container?: {
        [key: string]: string;
    };
    policy?: PolicyNode;
}
/** Workflow */
export interface WorkflowNode extends Node {
    kind: "Workflow";
    name: string;
    plan: string[];
    coordination: {
        consensus: boolean;
        conflict?: {
            metrics: string[];
        };
    };
    feedback: {
        metrics: string[];
        interval: string;
    };
    primitives?: WorkflowPrimNode[];
    onError?: OnErrorNode;
    schedule?: ScheduleNode;
    alert?: AlertNode;
}
export type WorkflowPrimNode = OnErrorNode | ScheduleNode | AlertNode;
/** Workflow primitive: retry on error */
export interface OnErrorNode extends Node {
    kind: "OnError";
    retries: number;
    backoff?: string | null;
}
/** Workflow primitive: cron schedule with optional backfill */
export interface ScheduleNode extends Node {
    kind: "Schedule";
    cron: string;
    backfill?: string | null;
}
/** Workflow primitive: SLA alert */
export interface AlertNode extends Node {
    kind: "Alert";
    metric: string;
    comparator: string;
    threshold: number;
    notify: string;
}
/** Coordination block */
export interface CoordinationBlockNode extends Node {
    kind: "CoordinationBlock";
    name: string;
    agents: string[];
    protocol: string;
    on_commit: string;
}
/** Audit trail */
export interface AuditTrailNode extends Node {
    kind: "AuditTrail";
    workflowRef?: string;
    name: string;
    snapshotOn: string[];
    store: string;
}
/** Atom block */
export interface FabricAtomNode extends Node {
    kind: "FabricAtom";
    name: string;
    protons: BitValue[];
    electrons: BitValue[];
    policy: AtomPolicy;
    entangled_with?: string[];
}
/** 0 or 1 */
export type BitValue = 0 | 1;
/** Atom policy for energy and mutability */
export interface AtomPolicy {
    mutable: number[];
    energy_budget: number;
}
/** Specialized policy entries for semantic checking */
export interface PolicyEntry extends GenericPolicyEntry {
    key: "policy";
    value: string;
}
export interface PrivacyEntry extends GenericPolicyEntry {
    key: "privacy";
    value: string;
}
export interface EnergyEntry extends GenericPolicyEntry {
    key: "energy";
    value: number;
    unit: string;
    scope?: string;
}
export interface FairnessEntry extends GenericPolicyEntry {
    key: "fairness";
    value: string;
    weights: {
        [group: string]: number;
    };
}
export interface ConsentRequiredEntry extends GenericPolicyEntry {
    key: "consent_required";
    value: boolean;
}
export interface PurposeEntry extends GenericPolicyEntry {
    key: "purpose";
    value: string;
}
export interface DPIAEntry extends GenericPolicyEntry {
    key: "dpia";
    value: boolean;
}
export interface DPIAReportEntry extends GenericPolicyEntry {
    key: "dpia_report";
    value: string;
}
/** All top-level declarations */
export type TopLevelNode = ImportNode | DeviceNode | PolicyNode | GoalNode | TypeAliasNode | AgentNode | ExecutionBlockNode | PolyBlockNode | WorkflowNode | CoordinationBlockNode | AuditTrailNode | FabricAtomNode;
