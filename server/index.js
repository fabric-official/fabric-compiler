const path  = require("path");
const fs    = require("fs");
const os    = require("os");
const grpc  = require("@grpc/grpc-js");
const protoLoader = require("@grpc/proto-loader");
const { compileGuarded, normalizeToTuple, sha256File } = require("./runner");
const { fromTupleIRToSol } = require("./solidityGen");

const ROOT   = process.env.FAB_ROOT || path.resolve(__dirname, "..");
const SCHEMA = process.env.FAB_SCHEMA || path.join(ROOT, "brains","language","compiler","compiler","backend","schema","policy.schema.json");
const PROTO  = path.join(ROOT,"proto","fabric","core","language","v1","language.proto");
const PORT   = process.env.PORT || "8891";

const pkgDef = protoLoader.loadSync(PROTO, { keepCase:true, longs:String, enums:String, defaults:true, oneofs:true });
const proto  = grpc.loadPackageDefinition(pkgDef).fabric.core.language.v1;

function lintPolicy(source) {
  const errors = [];
  const txt = String(source || "");
  const idxPolicy = txt.toLowerCase().indexOf("policy");
  if (idxPolicy < 0) { errors.push("policy{} missing"); return errors; }
  const open = txt.indexOf("{", idxPolicy); if (open < 0) { errors.push("policy{} missing opening {"); return errors; }
  let depth=0,end=-1;
  for (let i=open;i<txt.length;i++){ const c=txt[i]; if (c==="{") depth++; else if (c==="}") { depth--; if (depth===0){ end=i; break; } } }
  if (end < 0) { errors.push("policy{} missing closing }"); return errors; }
  const inner = txt.slice(open+1,end);
  const allowed = new Set(["royalty_bps","energy_budget","rollback_max"]);
  const keyRe = /([A-Za-z_]\w*)\s*:/g; let m;
  while ((m=keyRe.exec(inner))){ const k=m[1]; if (!allowed.has(k)) errors.push(`Unknown policy key: ${k}`); }
  if (/\{/.test(inner)) errors.push("Nested policy not allowed");
  const num = (name)=>{ const r=new RegExp("\\b"+name+"\\s*:\\s*(\\d+)","i"); const m=inner.match(r); return m?parseInt(m[1],10):null; };
  const rbp = num("royalty_bps"), enb = num("energy_budget"), rbm = num("rollback_max");
  if (rbp==null || enb==null) errors.push("Missing required keys royalty_bps or energy_budget");
  if (rbp!=null && (rbp<0 || rbp>10000)) errors.push("royalty_bps out of range");
  if (enb!=null && (enb<1 || enb>1_000_000_000)) errors.push("energy_budget out of range");
  if (rbm!=null && (rbm<0 || rbm>10)) errors.push("rollback_max out of range");
  // emit flood
  const emits = [...txt.matchAll(/emit\s*[:"]\s*"([^"]*)"/gi)].map(x=>x[1]);
  if (emits.length > 5000) errors.push("Too many steps (>5000 emits)");
  if (emits.some(s=>s.length>8192)) errors.push("Overlong emit string (>8192 chars)");
  return errors;
}

async function handleCompile(call, cb) {
  try {
    const req = call.request || {};
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "fab-brain-"));
    const srcDir = path.join(tmp, "src"); const outDir = path.join(tmp, "out");
    fs.mkdirSync(srcDir, { recursive:true }); fs.mkdirSync(outDir, { recursive:true });

    const files = (req.src || []).map((s,i)=>{
      const p = path.join(srcDir, s.path && s.path.trim() ? s.path : `in${i||""}.fab`);
      fs.mkdirSync(path.dirname(p), { recursive:true });
      fs.writeFileSync(p, s.content || "", { encoding:"utf8" });
      return p;
    });

    if (files.length===0) throw new Error("No source files provided");
    const mainFab = files[0];
    const outIr   = path.join(outDir, "program.ir.json");

    // Lint first (same rules as CLI)
    const lintErrs = lintPolicy(fs.readFileSync(mainFab,"utf8"));
    if (lintErrs.length) return cb(null, { ir_json:"", artifacts:[], attestation_json: JSON.stringify({ ok:false, errors:lintErrs }) });

    // Compile (guarded) and normalize to tuple
    await compileGuarded({ root: ROOT, schemaPath: SCHEMA, srcPath: mainFab, outIr });
    const irTxt = fs.readFileSync(outIr,"utf8");
    const irObj = JSON.parse(irTxt);
    const tuple = normalizeToTuple(irObj);
    fs.writeFileSync(outIr, JSON.stringify(tuple,null,2), { encoding:"utf8" });

    // Optional targets
    const artifacts = ["out/program.ir.json"];
    if (req.flags && Array.isArray(req.flags.targets) && req.flags.targets.includes("sol")) {
      const solPath = fromTupleIRToSol(JSON.stringify(tuple), path.join(outDir,"sol"), "FabricAgent");
      artifacts.push("out/sol/" + path.basename(solPath));
    }

    // Attestation (deterministic fields)
    const att = {
      schema_sha256:  fs.existsSync(SCHEMA) ? sha256File(SCHEMA) : null,
      input_sha256:   sha256File(mainFab),
      ir_sha256:      sha256File(outIr),
      node:           process.version,
      fab_guard:      tryReadLocalVersion(),
      created_at_utc: new Date().toISOString()
    };

    cb(null, { ir_json: JSON.stringify(tuple), artifacts, attestation_json: JSON.stringify(att) });
  } catch (e) {
    cb(null, { ir_json:"", artifacts:[], attestation_json: JSON.stringify({ ok:false, error: String(e.message||e) }) });
  }
}

function tryReadLocalVersion(){
  try {
    const pkg = require(path.join(ROOT,"node_modules","fabric-fab-guard-cli","package.json"));
    return pkg.version || null;
  } catch { return null; }
}

function handleAtomize(call, cb) {
  try {
    const txt = String((call.request && call.request.ir_json) || "");
    const ir  = JSON.parse(txt);
    const steps = (ir && ir.program && Array.isArray(ir.program.steps)) ? ir.program.steps : [];
    const atoms = steps.map(s => Array.isArray(s) ? JSON.stringify(s) : JSON.stringify(s));
    cb(null, { atoms });
  } catch(e) {
    cb(null, { atoms: [] });
  }
}

function handlePolicyLint(call, cb) {
  const content = String((call.request && call.request.content) || "");
  const errors = lintPolicy(content);
  cb(null, { ok: errors.length===0, errors });
}

function main(){
  const server = new grpc.Server();
  server.addService(proto.LanguageBrain.service, {
    Compile:    handleCompile,
    Atomize:    handleAtomize,
    PolicyLint: handlePolicyLint
  });
  server.bindAsync(`0.0.0.0:${PORT}`, grpc.ServerCredentials.createInsecure(), (err,port)=>{
    if (err) { console.error(err); process.exit(1); }
    console.log(`LanguageBrain listening on :${port}`);
    server.start();
  });
}

if (require.main === module) main();
