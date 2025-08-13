// frontend/version_map.ts

/**
 * Maps well-known modules and ontologies to their semver-compatible versions.
 * This can be extended to support remote fetching or dynamic version resolution.
 */
export const versionMap: Record<string, string> = {
    "core": "1.0.0",
    "devices": "1.0.0",
    "privacy": "1.2.3",
    "analytics": "2.0.0-beta.1",
    "fabric:std": "1.0.0",
    "fabric:ml": "0.9.5",
    // Add more module/ontology aliases as needed
};
