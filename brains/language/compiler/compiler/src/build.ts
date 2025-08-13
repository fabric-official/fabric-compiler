import fs from "fs";
import path from "path";
import { spawnSync } from "child_process";
import chalk from "chalk";

// Path to compiler binary
const COMPILER_BIN = path.resolve(__dirname, "../../../compiler/bin/fabc");

export async function buildAgent(opts: { input: string; atomized?: boolean }) {
    const inputPath = path.resolve(opts.input);
    const outJson = path.resolve("build/atomized.json");

    if (!fs.existsSync(inputPath)) {
        throw new Error(`Agent file not found: ${inputPath}`);
    }

    // Atomized mode flag
    const args = ["--input", inputPath];
    if (opts.atomized) args.push("--atomized");

    console.log(chalk.blueBright(`[FAB] Compiling ${path.basename(inputPath)}...`));
    const res = spawnSync(COMPILER_BIN, args, { encoding: "utf-8" });

    if (res.status !== 0) {
        console.error(chalk.red(`❌ Compiler error:\n${res.stderr}`));
        process.exit(1);
    }

    if (!fs.existsSync(outJson)) {
        console.error(chalk.red("❌ Atomized output not found at build/atomized.json"));
        process.exit(1);
    }

    const json = fs.readFileSync(outJson, "utf-8");
    const atoms = JSON.parse(json);

    console.log(chalk.green(`✅ Atomized FabricAtoms: ${atoms.length} atoms emitted.`));
    atoms.forEach((a: any, i: number) => {
        console.log(`  #${i} → ID: ${a.id}, Policy: ${JSON.stringify(a.policy)}`);
    });
}
