// compiler/runtime/profile_enforcer.ts

import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';
import { AgentIR } from '../ir/schema';

/**
 * Load a .profile YAML or JSON file from the given path.
 * Supports relative or absolute paths.
 */
export function loadProfileFile(profilePath: string): Record<string, any> {
    const resolvedPath = path.resolve(profilePath);
    if (!fs.existsSync(resolvedPath)) {
        throw new Error(`Profile file not found: ${resolvedPath}`);
    }

    const content = fs.readFileSync(resolvedPath, 'utf-8');
    if (profilePath.endsWith('.yaml') || profilePath.endsWith('.yml')) {
        return yaml.load(content) as Record<string, any>;
    } else if (profilePath.endsWith('.json')) {
        return JSON.parse(content);
    } else {
        throw new Error(`Unsupported profile file format: ${profilePath}`);
    }
}

/**
 * Enforces rules from a .profile against a given AgentIR's policy.
 * Throws on violation, or logs warnings depending on profile config.
 */
export function enforceProfile(profile: Record<string, any>, agent: AgentIR): void {
    const policy = agent.policy || {};

    for (const [key, rule] of Object.entries(profile)) {
        const value = policy[key];

        if (rule.required && value === undefined) {
            throw new Error(`Missing required policy field: ${key}`);
        }

        if (rule.enum && !rule.enum.includes(value)) {
            throw new Error(`Policy field '${key}' has invalid value '${value}'. Allowed: ${rule.enum.join(', ')}`);
        }

        if (rule.pattern && typeof value === 'string' && !new RegExp(rule.pattern).test(value)) {
            throw new Error(`Policy field '${key}' does not match pattern ${rule.pattern}`);
        }

        if (rule.warnIfMissing && value === undefined) {
            console.warn(`⚠️  Warning: Optional policy field '${key}' is missing.`);
        }
    }
}
