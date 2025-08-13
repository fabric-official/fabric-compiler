"use strict";
// compiler/frontend/parser.ts
// Transforms PEG parser output (CST) into Fabric DSL AST
Object.defineProperty(exports, "__esModule", { value: true });
exports.parse = parse;
const grammar_1 = require("./grammar");
// Utility: compute Location from CST node
function loc(node) {
    return {
        start: { line: node.location.start.line, column: node.location.start.column },
        end: { line: node.location.end.line, column: node.location.end.column }
    };
}
function parse(source, fileName = '<input>') {
    const cst = (0, grammar_1.parse)(source);
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
                declarations.push(buildGoal(item));
                break;
            case 'TypeDef':
                declarations.push(buildTypeAlias(item));
                break;
            case 'Agent':
                declarations.push(buildAgent(item));
                break;
            case 'ExecutionBlock':
                declarations.push(buildExecutionBlock(item));
                break;
            case 'PolyBlock':
                declarations.push(buildPolyBlock(item));
                break;
            case 'Workflow':
                declarations.push(buildWorkflow(item));
                break;
            case 'CoordinationBlock':
                declarations.push(buildCoordinationBlock(item));
                break;
            case 'AuditTrail':
                declarations.push(buildAuditTrail(item));
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
// ... buildGoal, buildTypeAlias, buildAgent, buildExecutionBlock, buildPolyBlock unchanged ...
function buildWorkflow(node) {
    // primitives may not exist
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
function buildAuditTrail(node) {
    return {
        kind: 'AuditTrail',
        loc: loc(node),
        workflowRef: node.body.workflowRef,
        snapshotOn: node.body.snapshotOn,
        store: node.body.store
    };
}
//# sourceMappingURL=parser.js.map