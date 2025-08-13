"use strict";
// compiler/frontend/checker.ts
// Semantic validation for Fabric DSL AST
Object.defineProperty(exports, "__esModule", { value: true });
exports.runSemanticChecks = runSemanticChecks;
const errors_1 = require("./errors");
const version_map_1 = require("./version_map"); // maps module/ontology names to semver strings
// Compiler's supported core version
const SUPPORTED_VERSION = "1.0.0";
const SUPPORTED_MAJOR = parseInt(SUPPORTED_VERSION.split('.')[0], 10);
/**
 * Entry point: perform full semantic validation on a module AST.
 */
function runSemanticChecks(module) {
    const idRegistry = new Set(); // for unique IDs (agents, exec blocks, polyblocks)
    const blockNames = new Set(); // for unique block names in execution & polyBlocks
    // 1. Validate imports / version compatibility
    for (const imp of module.imports) {
        checkImportNode(imp);
    }
    // 2. Module-level policies
    if (module.body) {
        for (const decl of module.body) {
            if (decl.type === "Policy") {
                checkPolicyNode(decl, "module");
            }
        }
    }
    // 3. Collect executionBlock and polyBlock names before workflows
    for (const decl of module.body) {
        if (decl.type === "ExecutionBlock" || decl.type === "PolyBlock") {
            const name = decl.name;
            if (blockNames.has(name)) {
                throw new errors_1.SemanticError(`Duplicate block name "${name}"`, decl.loc);
            }
            blockNames.add(name);
        }
    }
    // 4. Validate each top-level declaration
    for (const decl of module.body) {
        switch (decl.type) {
            case "Device":
                checkDeviceNode(decl);
                break;
            case "Policy":
                // already checked module-level
                break;
            case "Goal":
                checkGoalNode(decl);
                break;
            case "TypeAlias":
                checkTypeAliasNode(decl);
                break;
            case "Agent":
                checkAgentNode(decl, idRegistry);
                break;
            case "ExecutionBlock":
                checkExecBlockNode(decl, idRegistry);
                if (decl.policy) {
                    checkPolicyNode(decl.policy, "exec");
                }
                break;
            case "PolyBlock":
                checkPolyBlockNode(decl, idRegistry);
                if (decl.policy) {
                    checkPolicyNode(decl.policy, "poly");
                }
                break;
            case "Workflow":
                checkWorkflowNode(decl, blockNames);
                break;
            case "CoordinationBlock":
                checkCoordinationBlockNode(decl, blockNames);
                break;
            case "AuditTrail":
                checkAuditTrailNode(decl, blockNames);
                break;
            default:
                throw new errors_1.SemanticError(`Unknown declaration type "${decl.type}"`, decl.loc);
        }
    }
}
/** IMPORTS */
function checkImportNode(node) {
    for (const name of node.modules) {
        const importedVersion = version_map_1.versionMap[name];
        if (!importedVersion) {
            throw new errors_1.SemanticError(`Unknown import "${name}"`, node.loc);
        }
        const major = parseInt(importedVersion.split('.')[0], 10);
        if (major !== SUPPORTED_MAJOR) {
            throw new errors_1.SemanticError(`Import "${name}" version ${importedVersion} is incompatible with supported version ${SUPPORTED_VERSION}`, node.loc);
        }
    }
}
/** DEVICE */
function checkDeviceNode(node) {
    if (node.caps.length === 0) {
        throw new errors_1.SemanticError("Device must declare at least one capability", node.loc);
    }
    for (const cap of node.caps) {
        if (!cap.name) {
            throw new errors_1.SemanticError(`Malformed capability with empty name`, node.loc);
        }
    }
    checkPolicyNode(node.policy, "device");
}
/** POLICY */
function checkPolicyNode(node, context) {
    const entries = node.entries;
    const seen = new Set();
    for (const e of entries) {
        if (seen.has(e.key)) {
            throw new errors_1.SemanticError(`Duplicate policy key "${e.key}"`, node.loc);
        }
        seen.add(e.key);
        switch (e.key) {
            case "privacy":
                if (!["anonymized", "pseudonymized", "raw"].includes(e.value)) {
                    throw new errors_1.SemanticError(`Invalid privacy level "${e.value}"`, node.loc);
                }
                break;
            case "energy_budget":
                if (typeof e.amount !== 'number' || !e.unit) {
                    throw new errors_1.SemanticError(`Invalid energy_budget entry`, node.loc);
                }
                break;
            case "fairness":
                const weights = e.weights;
                const total = Object.values(weights).reduce((a, b) => a + b, 0);
                if (Math.abs(total - 1) > 1e-6) {
                    throw new errors_1.SemanticError(`Fairness weights must sum to 1, got ${total}`, node.loc);
                }
                break;
            case "consentRequired":
            case "purpose":
            case "dpia":
            case "dpiaReport":
                if (!(context === "module" || context === "agent")) {
                    throw new errors_1.SemanticError(`"${e.key}" is only allowed in module or agent policies`, node.loc);
                }
                // additional type checks
                if (e.key === "consentRequired" || e.key === "dpia") {
                    if (typeof e.value !== "boolean") {
                        throw new errors_1.SemanticError(`"${e.key}" must be a boolean`, node.loc);
                    }
                }
                if (e.key === "purpose" || e.key === "dpiaReport") {
                    if (typeof e.value !== "string") {
                        throw new errors_1.SemanticError(`"${e.key}" must be a string`, node.loc);
                    }
                }
                break;
            default:
                throw new errors_1.SemanticError(`Unknown policy key "${e.key}"`, node.loc);
        }
    }
}
/** GOAL */
function checkGoalNode(node) {
    if (node.constraints) {
        for (const c of node.constraints) {
            if (!["<", ">", "<=", ">=", "=="].includes(c.comparator)) {
                throw new errors_1.SemanticError(`Invalid constraint operator "${c.comparator}"`, node.loc);
            }
        }
    }
    if (!node.optimizeFor || node.optimizeFor.length === 0) {
        throw new errors_1.SemanticError("Goal must specify at least one optimize target", node.loc);
    }
}
/** TYPE ALIAS */
function checkTypeAliasNode(_node) {
    // No additional semantic checks for type aliases
}
/** AGENT */
function checkAgentNode(node, idReg) {
    if (!node.id) {
        throw new errors_1.SemanticError("Agent missing id", node.loc);
    }
    if (idReg.has(node.id)) {
        throw new errors_1.SemanticError(`Duplicate id "${node.id}"`, node.loc);
    }
    idReg.add(node.id);
    if (!/^.+:\d+\.\d+\.\d+$/.test(node.modelId)) {
        throw new errors_1.SemanticError(`modelId "${node.modelId}" must follow "name:semver"`, node.loc);
    }
    if (node.inputs.length === 0 || node.outputs.length === 0) {
        throw new errors_1.SemanticError("Agent must declare inputs and outputs", node.loc);
    }
    if (node.fallback && !["preserve", "reset"].includes(node.fallback.state)) {
        throw new errors_1.SemanticError(`Invalid fallback state "${node.fallback.state}"`, node.loc);
    }
    if (node.policy) {
        checkPolicyNode(node.policy, "agent");
    }
}
/** EXECUTION & POLYBLOCK common checks */
function checkExecBlockNode(node, idReg) {
    const bid = node.attrs.id;
    if (!bid) {
        throw new errors_1.SemanticError("ExecutionBlock missing id", node.loc);
    }
    if (idReg.has(bid)) {
        throw new errors_1.SemanticError(`Duplicate id "${bid}"`, node.loc);
    }
    idReg.add(bid);
    const b = node.block;
    if (!["inference", "preprocess", "postprocess", "tuning"].includes(b.blockType)) {
        throw new errors_1.SemanticError(`Invalid block type "${b.blockType}"`, node.loc);
    }
    if (b.inputs.length === 0 || b.outputs.length === 0) {
        throw new errors_1.SemanticError("ExecutionBlock must declare inputs and outputs", node.loc);
    }
}
function checkPolyBlockNode(node, idReg) {
    checkExecBlockNode(node, idReg);
    if (!["cpp", "javascript", "python", "dart", "rust", "go"].includes(node.lang)) {
        throw new errors_1.SemanticError(`Unsupported polyBlock language "${node.lang}"`, node.loc);
    }
    if (!node.code.trim()) {
        throw new errors_1.SemanticError("polyBlock code must be non-empty", node.loc);
    }
}
/** WORKFLOW */
function checkWorkflowNode(node, blockNames) {
    // Plan must reference at least two blocks
    if (node.plan.length < 2) {
        throw new errors_1.SemanticError("Workflow plan must reference at least two blocks", node.loc);
    }
    for (const name of node.plan) {
        if (!blockNames.has(name)) {
            throw new errors_1.SemanticError(`Workflow references unknown block "${name}"`, node.loc);
        }
    }
    // Coordination
    if (typeof node.coordination.consensus !== 'boolean') {
        throw new errors_1.SemanticError("Workflow coordination.consensus must be boolean", node.loc);
    }
    // Feedback
    if (!node.feedback.metrics || node.feedback.metrics.length === 0) {
        throw new errors_1.SemanticError("Workflow feedback.metrics must list at least one metric", node.loc);
    }
    if (!node.feedback.interval) {
        throw new errors_1.SemanticError("Workflow feedback.interval must be specified", node.loc);
    }
    // Advanced DSL: onError retry
    const onError = node.onError;
    if (onError) {
        if (typeof onError.retries !== 'number' || onError.retries < 1) {
            throw new errors_1.SemanticError("onError retry count must be a positive integer", onError.loc);
        }
        if (onError.backoff && !/^\d+[smhd]$/.test(onError.backoff)) {
            throw new errors_1.SemanticError("onError backoff must be a duration like '10s', '5m'", onError.loc);
        }
    }
    // Advanced DSL: schedule cron
    const schedule = node.schedule;
    if (schedule) {
        if (!/^([\d\*\/\-,]+\s+){4}[\d\*\/\-,]+$/.test(schedule.cron)) {
            throw new errors_1.SemanticError("schedule cron expression must have 5 space-separated fields", schedule.loc);
        }
        if (schedule.backfill && !/^\d+[smhd]$/.test(schedule.backfill)) {
            throw new errors_1.SemanticError("schedule backfill window must be a duration like '7d', '12h'", schedule.loc);
        }
    }
    // Advanced DSL: alert if
    const alert = node.alert;
    if (alert) {
        const comparators = ['<', '>', '<=', '>=', '=='];
        if (!alert.metric) {
            throw new errors_1.SemanticError("alert must specify a metric", alert.loc);
        }
        if (!comparators.includes(alert.comparator)) {
            throw new errors_1.SemanticError(`alert comparator must be one of ${comparators.join(', ')}`, alert.loc);
        }
        if (typeof alert.threshold !== 'number') {
            throw new errors_1.SemanticError("alert threshold must be a number", alert.loc);
        }
        if (!alert.notifyChannel) {
            throw new errors_1.SemanticError("alert notifyChannel must be specified", alert.loc);
        }
    }
}
/** COORDINATION BLOCK */
function checkCoordinationBlockNode(node, blockNames) {
    if (node.agents.length === 0) {
        throw new errors_1.SemanticError("CoordinationBlock must list agents", node.loc);
    }
    for (const ag of node.agents) {
        if (!blockNames.has(ag)) {
            throw new errors_1.SemanticError(`CoordinationBlock references unknown agent/block "${ag}"`, node.loc);
        }
    }
    if (!node.protocol) {
        throw new errors_1.SemanticError("coordinationBlock protocol must be specified", node.loc);
    }
}
/** AUDIT TRAIL */
function checkAuditTrailNode(node, blockNames) {
    if (!blockNames.has(node.name)) {
        throw new errors_1.SemanticError(`AuditTrail references unknown workflow "${node.name}"`, node.loc);
    }
    if (node.snapshotOn.length === 0) {
        throw new errors_1.SemanticError("AuditTrail.snapshotOn must list at least one phase", node.loc);
    }
    if (!node.store) {
        throw new errors_1.SemanticError("AuditTrail.store must specify a ledger", node.loc);
    }
}
//# sourceMappingURL=checker.js.map