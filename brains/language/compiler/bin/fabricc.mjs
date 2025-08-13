#!/usr/bin/env node
import { promises as fs } from "node:fs";
import path from "node:path";
function parseArgs(argv){const o={out:"out",targets:[],atomized:false};
  for(let i=2;i<argv.length;i++){const a=argv[i];
    if(a==="--out")o.out=argv[++i];
    else if(a==="--target")o.targets.push(argv[++i]);
    else if(a==="--atomized")o.atomized=true;}
  return o; }
const sol = `// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
contract HelloFabric { function ping() public pure returns (string memory) { return "pong"; } }`;
async function main(){
  const {out,targets}=parseArgs(process.argv);
  await fs.mkdir(out,{recursive:true});
  if(targets.includes("sol")) { await fs.writeFile(path.join(out,"HelloFabric.sol"), sol); }
  else { await fs.writeFile(path.join(out,"artifact.txt"), "hello from stub fabricc"); }
  await fs.writeFile(path.join(out,"atom_graph.json"), JSON.stringify({nodes:[]}));
  await fs.writeFile(path.join(out,"policy_report.json"), JSON.stringify({ok:true}));
}
main().catch(e=>{console.error(e);process.exit(1)});
