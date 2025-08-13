// 📁 compiler/frontend/parser.ts
// Transforms PEG parser output (CST) into Fabric DSL AST

// @ts-nocheck

import { parse } from "./grammar_parser";
import {
    ModuleNode, ImportNode, DeviceNode, PolicyNode,
    WorkflowNode, CoordinationBlockNode, QuantumCoordinationBlockNode,
    AuditTrailNode, Location, ASTNode, DeclarationNode, IODeclNode, CapNode,
    PolicyEntry, Constraint, TypeExpr,
    OnErrorNode, ScheduleNode, AlertNode, WorkflowPrimNode,
    AtomNode, EntangleEntry, ChannelEntry, CollapseEntry, CoordinationEntry
} from './ast';

import {
    buildGoal,
    buildTypeAlias,
    buildAgent,
    buildExecutionBlock,
    buildPolyBlock
} from './builder_stubs';

// Utility: compute Location from CST node
function loc(node: any): Location {
    return {
        start: { line: node.location.start.line, column: node.location.start.column },
        end: { line: node.location.end.line, column: node.location.end.column }
    };
}

export function parse(source: string, fileName = '<input>'): ModuleNode {
    const cst = pegParse(source, {});

    const imports: ImportNode[] = [];
    const declarations: DeclarationNode[] = [];

    for (const item of cst.TopLevel) {
        switch (item.type) {
            case 'Import': imports.push(buildImport(item)); break;
            case 'Device': declarations.push(buildDevice(item)); break;
            case 'Policy': declarations.push(buildPolicy(item)); break;
            case 'Goal': declarations.push(buildGoal(item)); break;
            case 'TypeDef': declarations.push(buildTypeAlias(item)); break;
            case 'Agent': declarations.push(buildAgent(item)); break;
            case 'ExecutionBlock': declarations.push(buildExecutionBlock(item)); break;
            case 'PolyBlock': declarations.push(buildPolyBlock(item)); break;
            case 'Workflow': declarations.push(buildWorkflow(item)); break;
            case 'CoordinationBlock': declarations.push(buildCoordinationBlock(item)); break;
            case 'QuantumCoordinationBlock': declarations.push(buildQuantumCoordinationBlock(item)); break;
            case 'AuditTrail': declarations.push(buildAuditTrail(item)); break;
            case 'Atom': declarations.push(buildAtom(item)); break;
            default: throw new Error(`Unknown top-level type: ${item.type}`);
        }
    }

    return {
        kind: 'Module',
        loc: loc(cst),
        imports,
        declarations
    };
}

function buildImport(node: any): ImportNode {
    return {
        kind: 'Import',
        loc: loc(node),
        modules: node.modules.map((m: any) => m.name)
    };
}

function buildDevice(node: any): DeviceNode {
    const caps: CapNode[] = node.body.caps.map((c: any) => ({
        kind: 'Cap',
        loc: loc(c),
        name: c.name,
        spec: c.spec || '',
        target: c.target
    }));
    return {
        kind: 'Device',
        loc: loc(node),
        name: node.name,
        caps,
        policy: buildPolicy(node.body.policy)
    };
}

function buildPolicy(node: any): PolicyNode {
    const entries: PolicyEntry[] = node.entries.map((e: any) => {
        if (e.type === 'fairness') {
            const weights: Record<string, number> = {};
            e.weights.forEach((w: any) => { weights[w.key] = parseFloat(w.value); });
            return { key: 'fairness', weights };
        } else if (e.type === 'kvPair') {
            return { key: e.key, value: e.value };
        } else {
            return { key: e.key, value: e.value };
        }
    });
    return {
        kind: 'Policy',
        loc: loc(node),
        entries
    };
}

function buildWorkflow(node: any): WorkflowNode {
    const prims: WorkflowPrimNode[] = (node.body.primitives || []).map((p: any) => {
        switch (p.primType) {
            case 'onError':
                const onErr: OnErrorNode = {
                    kind: 'OnError',
                    loc: loc(p),
                    retries: parseInt(p.times, 10),
                    backoff: p.backoffValue || null
                };
                return onErr;
            case 'schedule':
                const sched: ScheduleNode = {
                    kind: 'Schedule',
                    loc: loc(p),
                    cron: p.cronExpr,
                    backfill: p.backfillWindow || null
                };
                return sched;
            case 'alert':
                const alert: AlertNode = {
                    kind: 'Alert',
                    loc: loc(p),
                    metric: p.metric,
                    comparator: p.comp,
                    threshold: parseFloat(p.value),
                    notify: p.notifyChannel
                };
                return alert;
            default:
                throw new Error(`Unknown workflow primitive: ${p.primType}`);
        }
    });

    return {
        kind: 'Workflow',
        loc: loc(node),
        name: node.name,
        plan: node.body.plan,
        coordination: {
            consensus: node.body.coordination.consensus,
            conflict: node.body.coordination.conflict
        },
        feedback: {
            metrics: node.body.feedback.metrics,
            interval: node.body.feedback.interval
        },
        primitives: prims
    };
}

function buildCoordinationBlock(node: any): CoordinationBlockNode {
    return {
        kind: 'CoordinationBlock',
        loc: loc(node),
        name: node.name,
        agents: node.body.agents,
        protocol: node.body.protocol,
        on_commit: node.body.on_commit
    };
}

function buildQuantumCoordinationBlock(node: any): QuantumCoordinationBlockNode {
    const entries: CoordinationEntry[] = node.body.entries.map((entry: any) => {
        switch (entry.kind) {
            case 'EntangleEntry':
                return {
                    kind: 'EntangleEntry',
                    loc: loc(entry),
                    atoms: entry.atoms
                };
            case 'ChannelEntry':
                return {
                    kind: 'ChannelEntry',
                    loc: loc(entry),
                    name: entry.name
                };
            case 'CollapseEntry':
                return {
                    kind: 'CollapseEntry',
                    loc: loc(entry),
                    trigger: entry.trigger
                };
            default:
                throw new Error(`Unknown coordination entry kind: ${entry.kind}`);
        }
    });

    return {
        kind: 'QuantumCoordinationBlock',
        loc: loc(node),
        entries
    };
}

function buildAuditTrail(node: any): AuditTrailNode {
    return {
        kind: 'AuditTrail',
        loc: loc(node),
        workflowRef: node.body.workflowRef,
        snapshotOn: node.body.snapshotOn,
        store: node.body.store
    };
}

function buildAtom(node: any): AtomNode {
    return {
        kind: 'Atom',
        loc: loc(node),
        name: node.name,
        protons: node.body.protons,
        electrons: node.body.electrons,
        policy: node.body.policy,
        entangled_with: node.body.entangled_with
    };
}
