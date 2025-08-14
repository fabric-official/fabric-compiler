const path = require("path");
const fs = require("fs");

function fromTupleIRToSol(irJson, outDir, contractName="FabricAgent") {
  let ir;
  try { ir = JSON.parse(irJson); } catch(e){ throw new Error("Invalid IR JSON"); }
  const steps = (ir && ir.program && Array.isArray(ir.program.steps)) ? ir.program.steps : [];
  // Expect [["emit","..."], ...]; tolerate objects too
  const messages = [];
  for (const s of steps) {
    if (Array.isArray(s) && s.length >= 2 && String(s[0]).toLowerCase() === "emit") {
      messages.push(String(s[1]));
    } else if (s && typeof s === "object") {
      if (s.emit) messages.push(String(s.emit));
      else if (s.op === "emit" && s.args && typeof s.args.text === "string") messages.push(String(s.args.text));
      else if (s.op === "emit" && typeof s.value === "string") messages.push(String(s.value));
    }
  }

  const lines = [
    "// SPDX-License-Identifier: MIT",
    "pragma solidity ^0.8.24;",
    "",
    `contract ${contractName} {`,
    "    event Emitted(string text);",
    "    function replay() external {"
  ];
  for (const m of messages) {
    const esc = m.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n");
    lines.push(`        emit Emitted("${esc}");`);
  }
  lines.push("    }","}");
  const out = lines.join("\n");
  const outPath = path.join(outDir, `${contractName}.sol`);
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(outPath, out, { encoding: "utf8" });
  return outPath;
}

module.exports = { fromTupleIRToSol };
