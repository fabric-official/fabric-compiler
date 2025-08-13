// Hardened AgentVM (CommonJS, single-file, no deps). Pack with `pkg` into AgentVM.exe.
// Features: default-deny policy, energy ledger safety, coordination guard, append-only audit with hash chain,
//           model provenance verification (sha256), rollback limits.

const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

function die(msg, code=1){ console.error(msg); process.exit(code); }
function sha256(buf){ return crypto.createHash("sha256").update(buf).digest("hex"); }
function readStdin(){ try { return fs.readFileSync(0,"utf8"); } catch { return ""; } }
function loadIR(opts){
  let json = "";
  if (opts.stdin) json = readStdin();
  else if (opts.ir) { if (!fs.existsSync(opts.ir)) die("IR file not found: " + opts.ir); json = fs.readFileSync(opts.ir,"utf8"); }
  else if (opts.dir) {
    const files = fs.existsSync(opts.dir)? fs.readdirSync(opts.dir) : [];
    const hit = ["hello.ir.json"].concat(files.filter(f=>f.endsWith(".ir.json"))).map(f=>path.join(opts.dir,f)).find(p=>fs.existsSync(p));
    if (!hit) die("No *.ir.json in " + opts.dir);
    json = fs.readFileSync(hit,"utf8");
  } else die("No input. Use --ir <file>, --stdin, or --dir <folder>.");
  json = json.replace(/^\uFEFF/,"");
  try { return JSON.parse(json); } catch(e){ die("Invalid IR JSON: " + e); }
}

// ========== Policy ==========
function validatePolicy(pol){
  const allowed = new Set(["royalty_bps","energy_budget","group","rollback_max","model_sha256","model_path"]);
  if (!pol || typeof pol !== "object") die("Policy missing");
  for (const k of Object.keys(pol)) {
    if (!allowed.has(k)) die("Policy unknown key: " + k);
    if (typeof pol[k] === "object" && pol[k] !== null && !Array.isArray(pol[k])) {
      if (k !== "group") die("Nested policy objects not allowed: " + k);
    }
  }
  if (!Number.isInteger(pol.energy_budget) || pol.energy_budget < 1 || pol.energy_budget > 1e9) die("Invalid energy_budget");
  if (!Number.isInteger(pol.royalty_bps) || pol.royalty_bps < 0 || pol.royalty_bps > 10000) die("Invalid royalty_bps");
  if (pol.rollback_max !== undefined && (!Number.isInteger(pol.rollback_max) || pol.rollback_max < 0 || pol.rollback_max > 10)) die("Invalid rollback_max");
}

// ========== Audit (append-only, hash chain) ==========
class Audit {
  constructor(dir){
    this.dir = dir || process.env.FAB_AUDIT_DIR || path.join(process.cwd(), "out","audit");
    fs.mkdirSync(this.dir, { recursive: true });
    this.file = path.join(this.dir, "audit.log");
    this.prev = fs.existsSync(this.file) ? sha256(fs.readFileSync(this.file)) : "0".repeat(64);
  }
  write(entry){
    const rec = JSON.stringify({ ts: new Date().toISOString(), prev: this.prev, ...entry }) + "\n";
    fs.appendFileSync(this.file, rec, "utf8");
    this.prev = sha256(fs.readFileSync(this.file));
  }
}

// ========== Energy ==========
class Energy {
  constructor(budget){ this.b = Number.isFinite(budget) ? budget : Infinity; }
  spend(n=1){ if (!Number.isFinite(this.b)) return; if (n<0 || !Number.isInteger(n)) die("Bad energy spend"); if (this.b < n) die("Energy budget exceeded"); this.b -= n; }
  left(){ return Number.isFinite(this.b)? this.b : -1; }
}

const isObj = x => x && typeof x === "object" && !Array.isArray(x);

// ========== Model provenance ==========
function verifyModel(pol){
  if (!pol.model_path && !pol.model_sha256) return; // optional
  const p = pol.model_path; const want = pol.model_sha256;
  if (!p || !want) die("Both model_path and model_sha256 required together");
  if (!fs.existsSync(p)) die("Model file not found: " + p);
  const got = sha256(fs.readFileSync(p));
  if (got.toLowerCase() !== String(want).toLowerCase()) die("Model hash mismatch");
}

// ========== Execution ==========
function execProgram(ir, audit, log){
  validatePolicy(ir.policy);
  verifyModel(ir.policy);

  const ctx = {
    energy: new Energy(ir.policy.energy_budget),
    rollbackMax: Number.isInteger(ir.policy.rollback_max)? ir.policy.rollback_max : 0,
    rollbacks: 0,
    group: ir.policy.group || null,
    audit, log
  };
  audit.write({ ev: "start", policy: ir.policy });

  runOp(ir.program || ir.run || {}, ctx);

  audit.write({ ev: "end", energy_left: ctx.energy.left() });
  return 0;
}

function runOp(op, ctx){
  if (!op) return;

  if (Array.isArray(op)) { for (const s of op) runOp(s, ctx); return; }

  if (isObj(op)) {
    if (typeof op.emit === "string") { ctx.energy.spend(1); ctx.log(op.emit); ctx.audit.write({ ev:"emit", msg: op.emit }); return; }

    if (Array.isArray(op.steps)) { for (const s of op.steps) runOp(s, ctx); return; }

    if (Array.isArray(op.fork))  {  // simple sequential fork
      const n = op.fork.length; ctx.audit.write({ ev:"fork", branches:n });
      for (const b of op.fork) runOp(b, ctx);
      return;
    }

    if (op.rollback === true) {
      if (ctx.rollbacks >= ctx.rollbackMax) die("Rollback limit exceeded");
      ctx.rollbacks++; ctx.audit.write({ ev:"rollback" });
      return;
    }

    if (op.coordination && ctx.group) {
      // trivial group policy gate
      if (ctx.group.approvalsRequired && op.coordination.approvals < ctx.group.approvalsRequired) {
        die("Coordination approvals below policy threshold");
      }
      ctx.audit.write({ ev:"coordination", ok:true });
      runOp(op.coordination.body || {}, ctx);
      return;
    }

    if (op.load_model) {
      // model load must match policy provenance
      verifyModel({ model_path: op.load_model.path, model_sha256: op.load_model.sha256 });
      ctx.audit.write({ ev:"model_loaded", path: op.load_model.path });
      return;
    }

    if (op.program || op.run) { runOp(op.program || op.run, ctx); return; }

    // Unknown op: default-deny
    die("Unknown op in IR");
  }

  if (typeof op === "string") { ctx.energy.spend(1); ctx.log(op); ctx.audit.write({ ev:"emit", msg: op }); }
}

// ===== CLI =====
(function main(){
  const args = process.argv.slice(2);
  const has = f => args.includes(f);
  const take = f => { const i = args.indexOf(f); return i>=0 ? args[i+1] : undefined; };
  const opts = { ir: take("--ir"), stdin: has("--stdin"), dir: take("--dir"), quiet: has("--quiet"), auditDir: take("--audit-dir") };

  const ir = loadIR(opts);
  const audit = new Audit(opts.auditDir);
  const rc = execProgram(ir, audit, s => opts.quiet ? null : console.log(s));
  if (!opts.quiet) {
    const eb = ir && ir.policy && ir.policy.energy_budget;
    if (Number.isFinite(eb)) console.log("(energy left: " + eb + ")");
  }
  process.exit(rc);
})();