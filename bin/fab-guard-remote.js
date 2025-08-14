#!/usr/bin/env node
const fs = require("fs");
const path = require("path");
const grpc  = require("@grpc/grpc-js");
const protoLoader = require("@grpc/proto-loader");

const args = process.argv.slice(2);
const urlIdx = args.indexOf("--url");
const url = urlIdx>=0 ? args[urlIdx+1] : "localhost:8891";
const inIdx = args.indexOf("--in");
const outIdx = args.indexOf("--out");
const targetIdx = args.indexOf("--target");
const inFile  = inIdx>=0 ? args[inIdx+1] : null;
const outFile = outIdx>=0 ? args[outIdx+1] : null;
const targets = targetIdx>=0 ? [args[targetIdx+1]] : [];

if (!inFile || !outFile) { console.error("usage: fab-guard-remote --url host:8891 --in input.fab --out out.json [--target sol]"); process.exit(2); }

const ROOT = path.resolve(__dirname, "..");
const PROTO  = path.join(ROOT,"proto","fabric","core","language","v1","language.proto");
const pkgDef = protoLoader.loadSync(PROTO, { keepCase:true, longs:String, enums:String, defaults:true, oneofs:true });
const proto  = grpc.loadPackageDefinition(pkgDef).fabric.core.language.v1;

const client = new proto.LanguageBrain(url, grpc.credentials.createInsecure());
const content = fs.readFileSync(inFile,"utf8");

client.Compile({ src:[{ path: path.basename(inFile), content, lang:"en" }], flags:{ targets, reproducible:true } }, (err,res)=>{
  if (err) { console.error("RPC error:", err.message || err); process.exit(1); }
  if (!res || !res.ir_json) { console.error("No IR in reply:", res && res.attestation_json); process.exit(1); }
  fs.mkdirSync(path.dirname(outFile), { recursive:true });
  fs.writeFileSync(outFile, res.ir_json, { encoding:"utf8" });
  console.log("Artifacts:", (res.artifacts||[]).join(", "));
  console.log("Attestation:", res.attestation_json);
});
