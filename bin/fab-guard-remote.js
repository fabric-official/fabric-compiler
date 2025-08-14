#!/usr/bin/env node
const fs = require("fs");
const grpc = require("@grpc/grpc-js");
const protoLoader = require("@grpc/proto-loader");
const path = require("path");

const args = require("yargs/yargs")(process.argv.slice(2))
  .option("url", { type:"string", demandOption:true })
  .option("in",  { type:"string", demandOption:true })
  .option("out", { type:"string", demandOption:true })
  .option("target", { type:"string", array:true, default: [] })
  .option("auth", { type:"string", default: "" }) // "Bearer abc"
  .strict(false).argv;

const PROTO = path.join(__dirname, "..", "proto", "fabric", "core", "language", "v1", "language.proto");
const pkgDef = protoLoader.loadSync(PROTO, { keepCase:true, longs:String, enums:String, defaults:true, oneofs:true });
const proto  = grpc.loadPackageDefinition(pkgDef).fabric.core.language.v1;

const md = new grpc.Metadata();
if (args.auth) md.set("authorization", args.auth);

const client = new proto.LanguageBrain(args.url, grpc.credentials.createInsecure());

const src = [{ path: path.basename(args.in), content: fs.readFileSync(args.in, "utf8"), lang: "en" }];
const flags = { targets: args.target, reproducible: true };

client.Compile({ src, flags }, md, (err, res) => {
  if (err) {
    console.error(err.message || String(err));
    process.exit(1);
  }
  const irPath = res?.artifacts?.ir_path;
  if (!irPath || !fs.existsSync(irPath)) {
    console.error("No IR path returned from daemon.");
    process.exit(1);
  }
  const txt = fs.readFileSync(irPath, "utf8");
  fs.writeFileSync(args.out, txt, { encoding: "utf8" });
  if (flags.targets.includes("sol")) {
    const sol = res?.artifacts?.sol_path;
    if (!sol || !fs.existsSync(sol)) { console.error("Solidity requested but not produced."); process.exit(1); }
  }
  console.log("OK");
});