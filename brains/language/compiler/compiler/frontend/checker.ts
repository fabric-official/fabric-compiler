// compiler/frontend/checker.ts
// Semantic validation for Fabric DSL AST

import {
    ModuleNode, ImportNode, DeviceNode, PolicyNode,
    GoalNode, TypeAliasNode, AgentNode, ExecutionBlockNode,
    PolyBlockNode, WorkflowNode, CoordinationBlockNode,
    AuditTrailNode, FabricAtomNode, PolicyEntry, PrivacyEntry, EnergyEntry,
    FairnessEntry, ConsentRequiredEntry, PurposeEntry,
    DPIAEntry, DPIAReportEntry
} from "./ast";
import { SemanticError } from "./errors";
import { versionMap } from "./version_map";

const SUPPORTED_VERSION = "1.0.0";
const SUPPORTED_MAJOR = parseInt(SUPPORTED_VERSION.split('.')[0], 10);

// 👇 Type guard to distinguish real PolicyNode
function isPolicyNode(policy: any): policy is PolicyNode {
    return policy && typeof policy === "object" && "kind" in policy && "entries" in policy;
}

export function runSemanticChecks(module: ModuleNode): void {
    const idRegistry = new Set<string>();
    const blockNames = new Set<string>();

    for (const imp of module.imports) {
        checkImportNode(imp);
    }

    for (const decl of module.declarations) {
        if ((decl as PolicyNode).kind === "Policy") {
            checkPolicyNode(decl as PolicyNode, "module");
        }
    }

    for (const decl of module.declarations) {
        if ((decl as ExecutionBlockNode).kind === "ExecutionBlock" ||
            (decl as PolyBlockNode).kind === "PolyBlock") {
            const name = (decl as any).name;
            if (blockNames.has(name)) {
                throw new SemanticError(`Duplicate block name "${name}"`, decl.loc);
            }
            blockNames.add(name);
        }
    }

    for (const decl of module.declarations) {
        switch ((decl as any).kind) {
            case "Device":
                checkDeviceNode(decl as DeviceNode);
                break;
            case "Policy":
                break;
            case "Goal":
                checkGoalNode(decl as GoalNode);
                break;
            case "TypeAlias":
                checkTypeAliasNode(decl as TypeAliasNode);
                break;
            case "Agent":
                checkAgentNode(decl as AgentNode, idRegistry);
                break;
            case "ExecutionBlock":
                checkExecBlockNode(decl as ExecutionBlockNode, idRegistry);
                if ("policy" in decl && decl.policy && isPolicyNode(decl.policy))
                    checkPolicyNode(decl.policy, "exec");
                break;
            case "PolyBlock":
                checkPolyBlockNode(decl as PolyBlockNode, idRegistry);
                if ("policy" in decl && decl.policy && isPolicyNode(decl.policy))
                    checkPolicyNode(decl.policy, "poly");
                break;
            case "Workflow":
                checkWorkflowNode(decl as WorkflowNode, blockNames);
                break;
            case "CoordinationBlock":
                checkCoordinationBlockNode(decl as CoordinationBlockNode, blockNames);
                break;
            case "AuditTrail":
                checkAuditTrailNode(decl as AuditTrailNode, blockNames);
                break;
            case "FabricAtom":
                checkFabricAtomNode(decl as FabricAtomNode);
                break;
            default:
                throw new SemanticError(`Unknown declaration type "${(decl as any).kind}"`, decl.loc);
        }
    }
}

function checkFabricAtomNode(node: FabricAtomNode) {
    if (node.protons.length !== 8 || node.electrons.length !== 8) {
        throw new SemanticError("FabricAtom must have exactly 8 protons and 8 electrons", node.loc);
    }

    for (const b of [...node.protons, ...node.electrons]) {
        if (b !== 0 && b !== 1) {
            throw new SemanticError("FabricAtom bits must be binary (0 or 1)", node.loc);
        }
    }

    // ✅ Fixed: policy.mutable instead of node.mutableIndices
    for (const i of node.policy.mutable) {
        if (i < 0 || i >= 8) {
            throw new SemanticError(`Mutable index ${i} out of range`, node.loc);
        }
    }

    // ✅ Fixed: policy.energy_budget instead of node.energy
    if (typeof node.policy.energy_budget !== 'number' || node.policy.energy_budget < 0) {
        throw new SemanticError("FabricAtom energy_budget must be a non-negative number", node.loc);
    }

    // 🗑 Removed invalid `node.collapse` (not in AST)
}

function checkImportNode(node: ImportNode) {
    const importedVersion = versionMap[node.modules[0]];
    if (!importedVersion) {
        throw new SemanticError(`Unknown import "${node.modules[0]}"`, node.loc);
    }
    const major = parseInt(importedVersion.split('.')[0], 10);
    if (major !== SUPPORTED_MAJOR) {
        throw new SemanticError(`Import "${node.modules[0]}" version ${importedVersion} is incompatible`, node.loc);
    }
}

function checkDeviceNode(node: DeviceNode) {
    if (node.caps.length === 0) {
        throw new SemanticError("Device must declare at least one capability", node.loc);
    }
    checkPolicyNode(node.policy, "device");
}

function checkPolicyNode(node: PolicyNode, _context: string) {
    const seen = new Set<string>();
    for (const e of node.entries) {
        const key = (e as any).key;
        if (seen.has(key)) {
            throw new SemanticError(`Duplicate policy key "${key}"`, node.loc);
        }
        seen.add(key);
    }
}

function checkGoalNode(node: GoalNode) {
    if (!node.optimizeFor.length) {
        throw new SemanticError("Goal must specify optimizeFor", node.loc);
    }
}

function checkTypeAliasNode(_node: TypeAliasNode) { }

function checkAgentNode(node: AgentNode, idReg: Set<string>) {
    if (!node.id || idReg.has(node.id)) {
        throw new SemanticError(`Invalid or duplicate agent id "${node.id}"`, node.loc);
    }
    idReg.add(node.id);
}

function checkExecBlockNode(node: ExecutionBlockNode, idReg: Set<string>) {
    const bid = node.attrs.id;
    if (!bid || idReg.has(bid)) {
        throw new SemanticError(`Invalid or duplicate block id "${bid}"`, node.loc);
    }
    idReg.add(bid);
}

function checkPolyBlockNode(node: PolyBlockNode, idReg: Set<string>) {
    checkExecBlockNode(node as any, idReg);
    if (!node.code.trim()) {
        throw new SemanticError("PolyBlock code must not be empty", node.loc);
    }
}

function checkWorkflowNode(node: WorkflowNode, blockNames: Set<string>) {
    for (const name of node.plan) {
        if (!blockNames.has(name)) {
            throw new SemanticError(`Workflow references unknown block "${name}"`, node.loc);
        }
    }
}

function checkCoordinationBlockNode(node: CoordinationBlockNode, blockNames: Set<string>) {
    for (const ag of node.agents) {
        if (!blockNames.has(ag)) {
            throw new SemanticError(`CoordinationBlock references unknown agent/block "${ag}"`, node.loc);
        }
    }
}

function checkAuditTrailNode(node: AuditTrailNode, blockNames: Set<string>) {
    if (!blockNames.has(node.name)) {
        throw new SemanticError(`AuditTrail references unknown workflow "${node.name}"`, node.loc);
    }
}
