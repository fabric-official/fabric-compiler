"use strict";
// 📁 compiler/frontend/parser.ts
// Transforms PEG parser output (CST) into Fabric DSL AST
Object.defineProperty(exports, "__esModule", { value: true });
exports.parse = parse;
const builder_stubs_1 = require("./builder_stubs");
// Utility: compute Location from CST node
function loc(node) {
    return {
        start: { line: node.location.start.line, column: node.location.start.column },
        end: { line: node.location.end.line, column: node.location.end.column }
    };
}
function parse(source, fileName = '<input>') {
    const cst = pegParse(source, {});
    const imports = [];
    const declarations = [];
    for (const item of cst.TopLevel) {
        switch (item.type) {
            case 'Import':
                imports.push(buildImport(item));
                break;
            case 'Device':
                declarations.push(buildDevice(item));
                break;
            case 'Policy':
                declarations.push(buildPolicy(item));
                break;
            case 'Goal':
                declarations.push((0, builder_stubs_1.buildGoal)(item));
                break;
            case 'TypeDef':
                declarations.push((0, builder_stubs_1.buildTypeAlias)(item));
                break;
            case 'Agent':
                declarations.push((0, builder_stubs_1.buildAgent)(item));
                break;
            case 'ExecutionBlock':
                declarations.push((0, builder_stubs_1.buildExecutionBlock)(item));
                break;
            case 'PolyBlock':
                declarations.push((0, builder_stubs_1.buildPolyBlock)(item));
                break;
            case 'Workflow':
                declarations.push(buildWorkflow(item));
                break;
            case 'CoordinationBlock':
                declarations.push(buildCoordinationBlock(item));
                break;
            case 'QuantumCoordinationBlock':
                declarations.push(buildQuantumCoordinationBlock(item));
                break;
            case 'AuditTrail':
                declarations.push(buildAuditTrail(item));
                break;
            case 'Atom':
                declarations.push(buildAtom(item));
                break;
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
function buildImport(node) {
    return {
        kind: 'Import',
        loc: loc(node),
        modules: node.modules.map((m) => m.name)
    };
}
function buildDevice(node) {
    const caps = node.body.caps.map((c) => ({
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
function buildPolicy(node) {
    const entries = node.entries.map((e) => {
        if (e.type === 'fairness') {
            const weights = {};
            e.weights.forEach((w) => { weights[w.key] = parseFloat(w.value); });
            return { key: 'fairness', weights };
        }
        else if (e.type === 'kvPair') {
            return { key: e.key, value: e.value };
        }
        else {
            return { key: e.key, value: e.value };
        }
    });
    return {
        kind: 'Policy',
        loc: loc(node),
        entries
    };
}
function buildWorkflow(node) {
    const prims = (node.body.primitives || []).map((p) => {
        switch (p.primType) {
            case 'onError':
                const onErr = {
                    kind: 'OnError',
                    loc: loc(p),
                    retries: parseInt(p.times, 10),
                    backoff: p.backoffValue || null
                };
                return onErr;
            case 'schedule':
                const sched = {
                    kind: 'Schedule',
                    loc: loc(p),
                    cron: p.cronExpr,
                    backfill: p.backfillWindow || null
                };
                return sched;
            case 'alert':
                const alert = {
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
function buildCoordinationBlock(node) {
    return {
        kind: 'CoordinationBlock',
        loc: loc(node),
        name: node.name,
        agents: node.body.agents,
        protocol: node.body.protocol,
        on_commit: node.body.on_commit
    };
}
function buildQuantumCoordinationBlock(node) {
    const entries = node.body.entries.map((entry) => {
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
function buildAuditTrail(node) {
    return {
        kind: 'AuditTrail',
        loc: loc(node),
        workflowRef: node.body.workflowRef,
        snapshotOn: node.body.snapshotOn,
        store: node.body.store
    };
}
function buildAtom(node) {
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
//# sourceMappingURL=parser.js.map