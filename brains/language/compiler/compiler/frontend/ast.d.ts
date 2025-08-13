/** Base AST node */
export interface Node {
    type: string;
    loc?: SourceLocation;
}
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
/** Top-level module */
export interface ModuleNode extends Node {
    type: "Module";
    name: string;
    imports: ImportNode[];
    body: TopLevelNode[];
}
/** Import statement */
export interface ImportNode extends Node {
    type: "Import";
    modules: string[];
}
/** Device declaration */
export interface DeviceNode extends Node {
    type: "Device";
    name: string;
    caps: CapNode[];
    policy: PolicyNode;
}
/** Capability spec */
export interface CapNode {
    name: string;
    qualifier?: string;
}
/** Policy block */
export interface PolicyNode extends Node {
    type: "Policy";
    entries: PolicyEntry[];
}
export type PolicyEntry = PrivacyEntry | EnergyEntry | FairnessEntry | ConsentRequiredEntry | PurposeEntry | DPIAEntry | DPIAReportEntry;
export interface PrivacyEntry {
    key: "privacy";
    value: string;
}
export interface EnergyEntry {
    key: "energy_budget";
    amount: number;
    unit: string;
    scope: string;
}
export interface FairnessEntry {
    key: "fairness";
    weights: {
        [key: string]: number;
    };
}
export interface ConsentRequiredEntry {
    key: "consentRequired";
    value: boolean;
}
export interface PurposeEntry {
    key: "purpose";
    value: string;
}
export interface DPIAEntry {
    key: "dpia";
    value: boolean;
}
export interface DPIAReportEntry {
    key: "dpiaReport";
    value: string;
}
/** Goal block */
export interface GoalNode extends Node {
    type: "Goal";
    description: string;
    constraints?: ConstraintNode[];
    optimizeFor?: string[];
}
export interface ConstraintNode {
    metric: string;
    comparator: string;
    value: number;
    unit: string;
}
/** Type definition */
export interface TypeAliasNode extends Node {
    type: "TypeAlias";
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
    type: "Agent";
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
    type: "ExecutionBlock";
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
    type: "PolyBlock";
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
    type: "Workflow";
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
    onError?: OnErrorNode;
    schedule?: ScheduleNode;
    alert?: AlertNode;
}
/** Workflow primitive: retry on error */
export interface OnErrorNode extends Node {
    type: "OnError";
    retries: number;
    backoff?: string;
}
/** Workflow primitive: cron schedule with optional backfill */
export interface ScheduleNode extends Node {
    type: "Schedule";
    cron: string;
    backfill?: string;
}
/** Workflow primitive: SLA alert */
export interface AlertNode extends Node {
    type: "Alert";
    metric: string;
    comparator: string;
    threshold: number;
    notifyChannel: string;
}
/** Coordination block */
export interface CoordinationBlockNode extends Node {
    type: "CoordinationBlock";
    name: string;
    agents: string[];
    protocol: string;
    onCommit: string;
}
/** Audit trail */
export interface AuditTrailNode extends Node {
    type: "AuditTrail";
    name: string;
    snapshotOn: string[];
    store: string;
}
export type TopLevelNode = ImportNode | DeviceNode | PolicyNode | GoalNode | TypeAliasNode | AgentNode | ExecutionBlockNode | PolyBlockNode | WorkflowNode | CoordinationBlockNode | AuditTrailNode;
