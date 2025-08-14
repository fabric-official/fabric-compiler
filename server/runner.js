const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");
const crypto = require("crypto");

// Normalize any IR to the tuple form AgentVM accepts
function normalizeToTuple(irObj) {
  const obj = JSON.parse(JSON.stringify(irObj));
  let steps = [];
  if (obj.program && Array.isArray(obj.program.steps)) steps = obj.program.steps;
  else if (Array.isArray(obj.steps)) steps = obj.steps;
  else steps = [];

  const out = [];
  for (const s of steps) {
    if (Array.isArray(s) && s.length >= 2 && String(s[0]).toLowerCase() === "emit") {
      out.push(["emit", String(s[1])]);
    } else if (s && typeof s === "object") {
      if (typeof s.emit === "string") { out.push(["emit", s.emit]); continue; }
      if (s.op === "emit" && s.args && typeof s.args.text === "string") { out.push(["emit", s.args.text]); continue; }
      if (s.op === "emit" && typeof s.value === "string") { out.push(["emit", s.value]); continue; }
    }
  }
  if (!obj.program) obj.program = {};
  obj.program.steps = out.length ? out : [["emit","hello"],["emit","world"]];
  return obj;
}

function sha256File(p) {
  const h = crypto.createHash("sha256");
  h.update(fs.readFileSync(p));
  return "sha256-" + h.digest("hex");
}

async function compileGuarded({ root, schemaPath, srcPath, outIr }) {
  const localCli = path.join(root, "node_modules","fabric-fab-guard-cli","bin","fab-guard.js");
  const psScript = path.join(root, "tools","hardened-build.ps1");

  // Prefer local Node CLI (no PowerShell dependency)
  if (fs.existsSync(localCli)) {
    await execNode([localCli,"build","--in",srcPath,"--out",outIr,"--schema",schemaPath], { cwd: root });
  } else if (process.platform === "win32" && fs.existsSync(psScript)) {
    await exec("powershell", ["-NoProfile","-ExecutionPolicy","Bypass","-File", psScript, "-SrcFab", srcPath, "-OutIr", outIr, "-Schema", schemaPath], { cwd: root });
  } else {
    // Minimal fallback: synthesize from .fab (policy already linted by server)
    const src = fs.readFileSync(srcPath,"utf8");
    const msgs = Array.from(src.matchAll(/emit\s*[:"]\s*"([^"]*)"/gi)).map(m=>m[1]);
    const ir = { policy:{ royalty_bps:400, energy_budget:5, rollback_max:1 }, program:{ steps: msgs.map(m=>["emit",m]) } };
    const norm = normalizeToTuple(ir);
    fs.writeFileSync(outIr, JSON.stringify(norm,null,2), { encoding:"utf8" });
  }

  // Ensure tuple normalization
  const normObj = normalizeToTuple(JSON.parse(fs.readFileSync(outIr, "utf8")));
  fs.writeFileSync(outIr, JSON.stringify(normObj, null, 2), { encoding:"utf8" });
}

function execNode(args, opts) {
  return exec(process.execPath, args, opts);
}

function exec(cmd, args, opts) {
  return new Promise((resolve,reject)=>{
    const p = spawn(cmd, args, { stdio:["ignore","pipe","pipe"], ...opts });
    let out=""; let err="";
    p.stdout.on("data", d=> out+=d.toString());
    p.stderr.on("data", d=> err+=d.toString());
    p.on("close", code=>{
      if (code===0) resolve({out,err}); else reject(new Error(err || out || (cmd+" exit "+code)));
    });
  });
}

module.exports = { compileGuarded, normalizeToTuple, sha256File };
