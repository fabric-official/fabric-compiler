#!/usr/bin/env node
// Minimal "brain" shim for local testing.
// Usage patterns supported:
//   AgentVM.cjs --ir <file>
//   AgentVM.cjs -i <file>
//   AgentVM.cjs --input <file>
//   AgentVM.cjs <file>
//   (stdin)  type file | AgentVM.cjs --stdin

const fs = require("fs");
const path = require("path");

function die(msg, code=1){ console.error(msg); process.exit(code); }

function readFromStdinSync(){
  const BUFS = [];
  let b;
  try {
    while ((b = fs.readFileSync(0)) && b.length) BUFS.push(b);
  } catch(e){}
  return Buffer.concat(BUFS).toString("utf8");
}

function pickInput(argv){
  const take = f => { const i = argv.indexOf(f); return i>=0 ? argv[i+1] : undefined; };
  if (argv.includes("--stdin")) return { src: readFromStdinSync(), from: "stdin" };
  const file = take("--ir") || take("-i") || take("--input") || argv.find(a => !a.startsWith("-"));
  if (!file) die("AgentVM shim: no input. Use --ir <file> or pipe via --stdin.");
  const p = path.resolve(file);
  if (!fs.existsSync(p)) die(`AgentVM shim: file not found: ${p}`);
  return { src: fs.readFileSync(p,"utf8"), from: p };
}

function summarize(ir){
  const out = [];
  try {
    const obj = JSON.parse(ir);
    out.push("AgentVM shim: loaded IR OK");
    if (obj.policy) out.push(" policy: " + JSON.stringify(obj.policy));
    if (obj.agents) out.push(" agents: " + Object.keys(obj.agents).join(", "));
    if (obj.program || obj.run) out.push(" program present");
  } catch(e){
    die("AgentVM shim: invalid IR JSON: " + e);
  }
  return out.join("\n");
}

(function main(){
  const argv = process.argv.slice(2);
  const {src, from} = pickInput(argv);
  const summary = summarize(src);
  console.log(summary + "\n(Shim only)  replace me with real AgentVM.exe when available.");
})();