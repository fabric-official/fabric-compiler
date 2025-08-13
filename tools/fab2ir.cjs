#!/usr/bin/env node
const fs = require("fs");
const path = require("path");

function die(m){ console.error(m); process.exit(1); }

function parsePolicy(src){
  const out = {};
  const m = src.match(/policy\s*\{([^}]*)\}/s);
  if (!m) return out;
  const body = m[1];
  function num(name, def){
    const r = new RegExp("\\b"+name+"\\s*:\\s*([0-9]+)","i").exec(body);
    return r ? parseInt(r[1],10) : def;
  }
  out.royalty_bps   = num("royalty_bps", 0);
  out.energy_budget = num("energy_budget", 100);
  const rm = /rollback_max\s*:\s*([0-9]+)/i.exec(body);
  if (rm) out.rollback_max = parseInt(rm[1],10);
  return out;
}

function parseEmits(src){
  const emits = [];
  // steps: [ { emit: "..." }, ... ]
  const re1 = /emit\s*:\s*"([^"]*)"/g;
  let m;
  while ((m = re1.exec(src))) emits.push(m[1]);

  // also catch simple: emit "..."
  const re2 = /emit\s*"([^"]*)"/g;
  while ((m = re2.exec(src))) emits.push(m[1]);

  return emits;
}

(function main(){
  const inFile = process.argv[2];
  if (!inFile) die("usage: node fab2ir.cjs <file.fab> <out.json>");
  const outFile = process.argv[3] || "out.ir.json";
  if (!fs.existsSync(inFile)) die("No such file: " + inFile);

  const src = fs.readFileSync(inFile, "utf8").replace(/^\uFEFF/,"");
  const policy = parsePolicy(src);
  const emits  = parseEmits(src);

  const program = { steps: emits.map(s => ({ emit: s })) };
  const ir = { policy, agents: { }, program };

  fs.writeFileSync(outFile, JSON.stringify(ir, null, 2));
  console.log("IR -> " + path.resolve(outFile));
})();