#!/usr/bin/env node
// Guarded wrapper around fab.exe: validates policy, sanitizes AST/IR, default-deny.
// Usage: fab-guard.cjs build --in <file> --out <file> [--schema <path>]

const cp = require("child_process");
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

function die(msg, code=1){ console.error(msg); process.exit(code); }
function sha256(buf){ return crypto.createHash("sha256").update(buf).digest("hex"); }

function parseArgs(argv){
  const a = argv.slice(2);
  const out = { cmd: (a[0]||"").toLowerCase(), args: [] };
  out.args = a.filter((_,i)=>i>0);
  return out;
}

function readJson(str){
  const s = String(str).replace(/^\uFEFF/,"");
  return JSON.parse(s);
}

function validatePolicy(pol) {
  // Default-deny + strict keys:
  const allowed = new Set(["royalty_bps","energy_budget","group","rollback_max","model_sha256","model_path"]);
  if (!pol || typeof pol !== "object") die("Policy missing or not an object");
  for (const k of Object.keys(pol)) {
    if (!allowed.has(k)) die("Policy contains unknown key: " + k);
    if (typeof pol[k] === "object" && pol[k] !== null && !Array.isArray(pol[k])) {
      // No nested objects other than "group"
      if (k !== "group") die("Nested policy object not allowed for key: " + k);
    }
  }
  if (!Number.isInteger(pol.energy_budget) || pol.energy_budget < 1 || pol.energy_budget > 1e9) {
    die("Invalid energy_budget");
  }
  if (!Number.isInteger(pol.royalty_bps) || pol.royalty_bps < 0 || pol.royalty_bps > 10000) {
    die("Invalid royalty_bps");
  }
  if (pol.rollback_max !== undefined && (!Number.isInteger(pol.rollback_max) || pol.rollback_max < 0 || pol.rollback_max > 10)) {
    die("Invalid rollback_max");
  }
}

function sanitizeIR(ir) {
  // Whitelist top-level fields
  const allowedTop = new Set(["policy","agents","program","run","meta"]);
  for (const k of Object.keys(ir)) if (!allowedTop.has(k)) delete ir[k];

  // Enforce program size/shape limits
  const MAX_STEPS = 5000, MAX_STRING = 8*1024;
  let steps = 0, bad = false;

  function walk(x){
    if (bad) return;
    if (typeof x === "string") {
      if (x.length > MAX_STRING) bad = true;
      return;
    }
    if (Array.isArray(x)) { for (const v of x) walk(v); return; }
    if (x && typeof x === "object") {
      if ("emit" in x) steps++;
      if ("steps" in x && Array.isArray(x.steps)) steps += x.steps.length;
      if (steps > MAX_STEPS) { bad = true; return; }
      for (const k of Object.keys(x)) walk(x[k]);
    }
  }
  walk(ir.program || ir.run || {});
  if (bad) die("IR exceeds allowed step or string limits");
  return ir;
}

function runFabAndGetIR(stdinBuf, args) {
  // We *prefer* stdin path to avoid writing temp intermediates
  const proc = cp.spawn(process.env.FAB_PATH_REAL || process.env.FAB_PATH || "D:\\\\Fabric\\\\fabric_Lang\\\\bin\\\\fab.exe", args, { stdio: ["pipe","pipe","pipe"] });
  let out = "", err = "";
  proc.stdout.on("data", d => out += d.toString());
  proc.stderr.on("data", d => err += d.toString());
  proc.on("close", code => {
    if (code !== 0) {
      console.error(err || out);
      process.exit(code || 1);
    } else {
      try {
        const ir = readJson(out);
        if (!ir.policy) die("Compiler emitted IR without policy");
        validatePolicy(ir.policy);
        const cleaned = sanitizeIR(ir);
        process.stdout.write(JSON.stringify(cleaned, null, 2));
      } catch(e) {
        die("Guard failed: " + e);
      }
    }
  });
  if (stdinBuf) proc.stdin.end(stdinBuf); else proc.stdin.end();
}

(function main(){
  const { cmd, args } = parseArgs(process.argv);
  if (cmd !== "build") die('usage: build (--stdin | --in <file>) [--out <file>] [--schema <path>]');
  const ai = (flag) => { const i = args.indexOf(flag); return i>=0 ? args[i+1] : undefined; };
  const has = (flag) => args.includes(flag);
  const fromStdin = has("--stdin");
  const inFile = ai("--in");
  const outFile = ai("--out");
  const schema = ai("--schema") || process.env.FAB_SCHEMA || "D:\\\\Fabric\\\\fabric_Lang\\\\brains\\\\language\\\\compiler\\\\compiler\\\\backend\\\\schema\\\\policy.schema.json";

  const passArgs = ["build"];
  if (fromStdin) { passArgs.push("--stdin"); } else if (inFile) { passArgs.push("--in", inFile); }
  if (schema && fs.existsSync(schema)) passArgs.push("--schema", schema);

  if (fromStdin) {
    // read stdin fully first (so we can hash/check size if needed)
    const chunks = [];
    process.stdin.on("data", d => chunks.push(Buffer.from(d)));
    process.stdin.on("end", ()=> runFabAndGetIR(Buffer.concat(chunks), passArgs));
  } else {
    runFabAndGetIR(null, passArgs);
  }
})();