"use strict";
// compiler/frontend/checker.ts
// Semantic validation for Fabric DSL AST
Object.defineProperty(exports, "__esModule", { value: true });
exports.runSemanticChecks = runSemanticChecks;
const errors_1 = require("./errors");
const version_map_1 = require("./version_map");
const SUPPORTED_VERSION = "1.0.0";
const SUPPORTED_MAJOR = parseInt(SUPPORTED_VERSION.split('.')[0], 10);
// 👇 Type guard to distinguish real PolicyNode
function isPolicyNode(policy) {
    return policy && typeof policy === "object" && "kind" in policy && "entries" in policy;
}
function runSemanticChecks(module) {
    const idRegistry = new Set();
    const blockNames = new Set();
    for (const imp of module.imports) {
        checkImportNode(imp);
    }
    for (const decl of module.declarations) {
        if (decl.kind === "Policy") {
            checkPolicyNode(decl, "module");
        }
    }
    for (const decl of module.declarations) {
        if (decl.kind === "ExecutionBlock" ||
            decl.kind === "PolyBlock") {
            const name = decl.name;
            if (blockNames.has(name)) {
                throw new errors_1.SemanticError(`Duplicate block name "${name}"`, decl.loc);
            }
            blockNames.add(name);
        }
    }
    for (const decl of module.declarations) {
        switch (decl.kind) {
            case "Device":
                checkDeviceNode(decl);
                break;
            case "Policy":
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
                if ("policy" in decl && decl.policy && isPolicyNode(decl.policy))
                    checkPolicyNode(decl.policy, "exec");
                break;
            case "PolyBlock":
                checkPolyBlockNode(decl, idRegistry);
                if ("policy" in decl && decl.policy && isPolicyNode(decl.policy))
                    checkPolicyNode(decl.policy, "poly");
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
            case "FabricAtom":
                checkFabricAtomNode(decl);
                break;
            default:
                throw new errors_1.SemanticError(`Unknown declaration type "${decl.kind}"`, decl.loc);
        }
    }
}
function checkFabricAtomNode(node) {
    if (node.protons.length !== 8 || node.electrons.length !== 8) {
        throw new errors_1.SemanticError("FabricAtom must have exactly 8 protons and 8 electrons", node.loc);
    }
    for (const b of [...node.protons, ...node.electrons]) {
        if (b !== 0 && b !== 1) {
            throw new errors_1.SemanticError("FabricAtom bits must be binary (0 or 1)", node.loc);
        }
    }
    // ✅ Fixed: policy.mutable instead of node.mutableIndices
    for (const i of node.policy.mutable) {
        if (i < 0 || i >= 8) {
            throw new errors_1.SemanticError(`Mutable index ${i} out of range`, node.loc);
        }
    }
    // ✅ Fixed: policy.energy_budget instead of node.energy
    if (typeof node.policy.energy_budget !== 'number' || node.policy.energy_budget < 0) {
        throw new errors_1.SemanticError("FabricAtom energy_budget must be a non-negative number", node.loc);
    }
    // 🗑 Removed invalid `node.collapse` (not in AST)
}
function checkImportNode(node) {
    const importedVersion = version_map_1.versionMap[node.modules[0]];
    if (!importedVersion) {
        throw new errors_1.SemanticError(`Unknown import "${node.modules[0]}"`, node.loc);
    }
    const major = parseInt(importedVersion.split('.')[0], 10);
    if (major !== SUPPORTED_MAJOR) {
        throw new errors_1.SemanticError(`Import "${node.modules[0]}" version ${importedVersion} is incompatible`, node.loc);
    }
}
function checkDeviceNode(node) {
    if (node.caps.length === 0) {
        throw new errors_1.SemanticError("Device must declare at least one capability", node.loc);
    }
    checkPolicyNode(node.policy, "device");
}
function checkPolicyNode(node, _context) {
    const seen = new Set();
    for (const e of node.entries) {
        const key = e.key;
        if (seen.has(key)) {
            throw new errors_1.SemanticError(`Duplicate policy key "${key}"`, node.loc);
        }
        seen.add(key);
    }
}
function checkGoalNode(node) {
    if (!node.optimizeFor.length) {
        throw new errors_1.SemanticError("Goal must specify optimizeFor", node.loc);
    }
}
function checkTypeAliasNode(_node) { }
function checkAgentNode(node, idReg) {
    if (!node.id || idReg.has(node.id)) {
        throw new errors_1.SemanticError(`Invalid or duplicate agent id "${node.id}"`, node.loc);
    }
    idReg.add(node.id);
}
function checkExecBlockNode(node, idReg) {
    const bid = node.attrs.id;
    if (!bid || idReg.has(bid)) {
        throw new errors_1.SemanticError(`Invalid or duplicate block id "${bid}"`, node.loc);
    }
    idReg.add(bid);
}
function checkPolyBlockNode(node, idReg) {
    checkExecBlockNode(node, idReg);
    if (!node.code.trim()) {
        throw new errors_1.SemanticError("PolyBlock code must not be empty", node.loc);
    }
}
function checkWorkflowNode(node, blockNames) {
    for (const name of node.plan) {
        if (!blockNames.has(name)) {
            throw new errors_1.SemanticError(`Workflow references unknown block "${name}"`, node.loc);
        }
    }
}
function checkCoordinationBlockNode(node, blockNames) {
    for (const ag of node.agents) {
        if (!blockNames.has(ag)) {
            throw new errors_1.SemanticError(`CoordinationBlock references unknown agent/block "${ag}"`, node.loc);
        }
    }
}
function checkAuditTrailNode(node, blockNames) {
    if (!blockNames.has(node.name)) {
        throw new errors_1.SemanticError(`AuditTrail references unknown workflow "${node.name}"`, node.loc);
    }
}
//# sourceMappingURL=checker.js.map