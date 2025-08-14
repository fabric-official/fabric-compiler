Language Brain (Unified)
This package wraps the original Fabric language compiler behind a core, non-forkable Language Brain that speaks gRPC (with JSON/HTTP mapping). It keeps multilingual inputs intact, produces deterministic artifacts plus an attestation envelope, and exposes three service methods:

Compile — parse → lower → build targets (IR, Solidity, etc.)

Atomize — tokenize/segment multilingual sources (en/fr/zh/…)

PolicyLint — static checks on policy blocks & build flags

Ships with the bundled compiler (no external fabricc required). The daemon auto-detects a build runner and normalizes outputs to reproducible form.

Features
Multilingual in: lang: "en" | "fr" | "zh" | … per source file; preserved in diagnostics.

Deterministic out: stable ordering, path normalization, CRLF→LF, UTF-8 w/o BOM.

Attestation: envelope binds input digests, toolchain version, target list, and output digests.

Targets: IR (.ir.json tuple-steps), optional Solidity (.sol) via flags.targets: ["sol"].

Hardened: default-deny policy keys, numeric range checks, max string/step limits.

gRPC + JSON: pure gRPC and REST-style JSON POST supported.

Quick Start
1) Build & run the daemon
bash
Copy
Edit
# Build daemon image
docker build -f docker/language-brain.Dockerfile -t language-brain:1 .

# Run daemon (gRPC + JSON on :8891)
docker run --rm -p 8891:8891 language-brain:1
Health check:

bash
Copy
Edit
curl -s http://localhost:8891/healthz
# { "status": "ok" }
2) Compile over HTTP/JSON
IR (default)

bash
Copy
Edit
curl -s -X POST \
  http://localhost:8891/fabric.core.language.v1.LanguageBrain/Compile \
  -H "Content-Type: application/json" \
  -d '{
        "src":[
          {"path":"agents/Hello.fab","content":"agent Hello { policy { royalty_bps: 400, energy_budget: 5, rollback_max: 1 } run { steps: [ { emit: \"hello\" }, { emit: \"world\" } ] } }","lang":"en"}
        ],
        "flags":{"reproducible":true}
      }' | jq .
Solidity target

bash
Copy
Edit
curl -s -X POST \
  http://localhost:8891/fabric.core.language.v1.LanguageBrain/Compile \
  -H "Content-Type: application/json" \
  -d '{
        "src":[{"path":"agents/MyAgent.fab","content":"...","lang":"en"}],
        "flags":{"targets":["sol"],"reproducible":true}
      }' | jq .
The daemon forwards --target sol to the bundled compiler and validates .sol artifacts exist.

3) Register as a core (non-forkable)
bash
Copy
Edit
fab publish \
  --manifest brains/language/model.yaml \
  --chain base \
  --token USDC \
  --royalty 0 \
  --non-forkable \
  --origin Fabric
4) Use the CLI against the remote Language Brain
bash
Copy
Edit
# Route CLI builds to the daemon
fab build --remote http://localhost:8891 --in agents/Hello.fab --out out/hello.ir.json
API (gRPC)
Proto surface (informative)

proto
Copy
Edit
syntax = "proto3";
package fabric.core.language.v1;

service LanguageBrain {
  rpc Compile   (CompileRequest)   returns (CompileResponse);
  rpc Atomize   (AtomizeRequest)   returns (AtomizeResponse);
  rpc PolicyLint(PolicyLintRequest)returns (PolicyLintResponse);
}

message Source {
  string path    = 1;   // logical path (normalized internally)
  string content = 2;   // UTF-8 source
  string lang    = 3;   // "en","fr","zh",...
}

message CompileFlags {
  repeated string targets       = 1; // e.g. ["ir","sol"]
  bool reproducible             = 2; // deterministic mode
  map<string,string> defines    = 3; // optional -Dk=v
  repeated string warningsAsErr = 4; // promote codes
}

message CompileRequest {
  repeated Source src = 1;
  CompileFlags flags  = 2;
}

message Artifact {
  string path     = 1; // e.g. "out/Hello.ir.json" or "out/Hello.sol"
  string kind     = 2; // "ir" | "sol" | ...
  bytes  content  = 3; // raw bytes (JSON/UTF-8 for IR, text for .sol)
  string sha256   = 4; // hex digest of content
}

message Diagnostic {
  string path   = 1;
  string lang   = 2;   // input language
  string level  = 3;   // "error" | "warning" | "info"
  string code   = 4;   // e.g. "POLICY_UNKNOWN_KEY"
  string message= 5;
  int32  line   = 6;
  int32  column = 7;
}

message Attestation {
  string toolchain      = 1; // git sha / version
  string timestamp_utc  = 2;
  string seed           = 3; // build seed when reproducible=true
  repeated string targets = 4;
  message Input { string path=1; string sha256=2; }
  repeated Input inputs  = 5;
  repeated Artifact outputs = 6; // paths + digests only in envelope
}

message CompileResponse {
  repeated Artifact artifacts  = 1;
  Attestation attestation      = 2;
  repeated Diagnostic diagnostics = 3;
}

message AtomizeRequest  { repeated Source src = 1; }
message AtomizeResponse {
  message Token { string kind=1; string text=2; int32 line=3; int32 column=4; }
  map<string, repeated Token> tokens = 1; // key = Source.path
}

message PolicyLintRequest  { repeated Source src = 1; }
message PolicyLintResponse { repeated Diagnostic diagnostics = 1; }
HTTP/JSON mapping

bash
Copy
Edit
POST /fabric.core.language.v1.LanguageBrain/Compile
POST /fabric.core.language.v1.LanguageBrain/Atomize
POST /fabric.core.language.v1.LanguageBrain/PolicyLint
All endpoints accept/return JSON; the daemon also serves pure gRPC on the same port.

Determinism & Attestation
When flags.reproducible = true:

Input normalization: CRLF→LF, strip BOM, sort sources by path.

Stable compiler flags / locale / timezone (UTC).

Artifact normalization: stable key ordering for JSON, UTF-8 w/o BOM.

Attestation ties everything together:

json
Copy
Edit
{
  "toolchain": "language-brain@1.0.0+compiler@abc123",
  "timestamp_utc": "2025-08-14T02:22:00Z",
  "seed": "00000000…",
  "targets": ["ir","sol"],
  "inputs": [
    {"path":"agents/Hello.fab","sha256":"…"}
  ],
  "outputs": [
    {"path":"out/Hello.ir.json","kind":"ir","sha256":"…"},
    {"path":"out/Hello.sol","kind":"sol","sha256":"…"}
  ]
}
Hardened policy & IR limits
Policy default-deny: only royalty_bps, energy_budget, rollback_max, optional group.

Ranges: royalty_bps ∈ [0,10000], energy_budget ∈ [1,1e9], rollback_max ∈ [0,10].

IR limits: max string length 8192; max total steps 5000; rejects nested policy objects.

Violations return diagnostics[] with level:"error" and a non-zero status.

Targets
IR (default): tuple-step form consumed by AgentVM, e.g.

json
Copy
Edit
{ "program": { "steps": [ ["emit","hello"], ["emit","world"] ] } }
Solidity: add flags.targets: ["sol"]. The daemon forwards --target sol and verifies .sol is produced. Solidity files are returned as separate artifacts.

Bundled Compiler
This package carries the full original compiler under brains/language/compiler/. The daemon chooses a runner in this order:

bin/fabricc (executable)

package.json scripts (build, compile, or fabricc)

CMakeLists.txt (configure + build)

No external fabricc in PATH is required.

CLI wiring (developers)
Local one-shot:

bash
Copy
Edit
fab build --remote http://localhost:8891 --in agents/Hello.fab --out out/hello.ir.json
VS Code task:

json
Copy
Edit
{ "label":"Build (remote)", "type":"shell",
  "command":"fab build --remote http://localhost:8891 --in agents/Hello.fab --out out/hello.ir.json" }
Operational notes
Port: 8891 (configurable via PORT env).

Auth: if AUTH_BEARER is set, requests must include Authorization: Bearer <token>.

Logs: structured JSON to stdout; set LOG_LEVEL=debug|info|warn|error.

Health: GET /healthz → { "status": "ok" }.

Limits: override with env MAX_STR, MAX_STEPS (use cautiously).

Examples
Minimal Hello
json
Copy
Edit
{
  "src":[{"path":"agents/Hello.fab","content":"agent Hello { policy { royalty_bps: 400, energy_budget: 5, rollback_max: 1 } run { steps: [ { emit: \"hello\" }, { emit: \"world\" } ] } }","lang":"en"}],
  "flags":{"reproducible":true}
}
Policy Lint only
bash
Copy
Edit
curl -s -X POST http://localhost:8891/fabric.core.language.v1.LanguageBrain/PolicyLint \
  -H "Content-Type: application/json" \
  -d '{"src":[{"path":"agents/Bad.fab","content":"agent B { policy { royalty_bps:1, energy_budget:1, backdoor:true } }","lang":"en"}]}'
Returns an error diagnostic for backdoor.

Versioning & License
Versioned as language-brain@X.Y.Z.

Attestation includes toolchain identifiers for supply-chain tracking.

Licensed under MIT (see LICENSE).

FAQ
Q: Do I need Node.js or external tools inside the container?
A: No. The daemon bundles the compiler and auto-detects how to build.

Q: Can I request multiple targets at once?
A: Yes, e.g. ["ir","sol"]. You’ll get multiple artifacts with digests in the attestation.

Q: What fails a build?
A: Policy schema violations, IR size limits, missing target artifacts, or compiler errors. You’ll receive non-zero status and structured diagnostics.

