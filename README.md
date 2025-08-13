# Language Brain (Unified)

This package wraps the original compiler into a **core non-forkable Language Brain** with gRPC.
- Keeps multilingual inputs (en/fr/zh/…)
- Emits deterministic artifacts and attestation envelope
- Exposes `Compile/Atomize/PolicyLint`

## Bring-up
```bash
# Build daemon
docker build -f docker/language-brain.Dockerfile -t language-brain:1 .
docker run --rm -p 8891:8891 language-brain:1

# Register as core
fab publish --manifest brains/language/model.yaml --chain base --token USDC --royalty 0 --non-forkable --origin Fabric
```

## Wire CLI to remote Language Brain
Use `fab build --remote` to route to gRPC.


## Solidity (sol) target
Request Solidity output by passing `targets: ["sol"]` to `CompileFlags`. The daemon forwards `--target sol` to the compiler and validates `.sol` artifacts exist.

Example:
```bash
curl -X POST http://localhost:8891/fabric.core.language.v1.LanguageBrain/Compile   -d '{"src":[{"path":"agents/MyAgent.fab","content":"...","lang":"en"}],"flags":{"targets":["sol"],"reproducible":true}}'
```


## Bundled Compiler
This package includes the full original compiler source under `brains/language/compiler/`.
The daemon uses an internal runner that detects `bin/fabricc`, `package.json` scripts, or `CMakeLists.txt`
to build and emit artifacts without requiring an external `fabricc` in PATH.
