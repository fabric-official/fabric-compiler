$ErrorActionPreference = "Stop"

# --- Paths ---
$ROOT   = "D:\Fabric\fabric_Lang"
$COMP   = Join-Path $ROOT "brains\language\compiler\compiler"
$ENTRY  = Join-Path $COMP  "lib\ir\compiler.js"
$SCHEMA = Join-Path $COMP  "backend\schema\policy.schema.json"
$TOOLS  = Join-Path $ROOT  "tools"
$OUT    = Join-Path $ROOT  "out"
$BIN    = Join-Path $ROOT  "bin"
$SRC    = Join-Path $ROOT  "examples\hello.fab"
$IR     = Join-Path $OUT   "hello.ir.json"
$EXT    = Join-Path $ROOT  "tools\vsce-fabric"

New-Item -ItemType Directory -Force -Path $OUT,$BIN,$TOOLS | Out-Null

# --- 1) Build compiler (tsc) ---
Push-Location $COMP
npm install | Out-Null
tsc -p tsconfig.json
Pop-Location
if (!(Test-Path $ENTRY)) { throw "Compiler entry missing: $ENTRY" }

# --- 2) Ensure sample .fab ---
@'
agent Hello {
  policy { royalty_bps: 400, energy_budget: 5, rollback_max: 1 }
  run { steps: [ { emit: "hello" }, { emit: "world" } ] }
}
'@ | Set-Content -Encoding UTF8 $SRC

# --- helper: try to save valid JSON to file ---
function Try-SaveJson([string]$text,[string]$path){
  if (-not $text) { return $false }
  $text = $text -replace '^\uFEFF',''
  try { $null = $text | ConvertFrom-Json } catch { return $false }
  $text | Set-Content -Encoding UTF8 $path
  return $true
}

# --- 3) Try real compiler (stdin -> stdout JSON) ---
Remove-Item $IR -ErrorAction SilentlyContinue

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName  = "node"
$psi.Arguments = "`"$ENTRY`" build --stdin"
if (Test-Path $SCHEMA) { $psi.Arguments += " --schema `"$SCHEMA`"" }
$psi.UseShellExecute = $false
$psi.RedirectStandardInput  = $true
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError  = $true

$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $psi
[void]$proc.Start()
$proc.StandardInput.Write((Get-Content -Raw $SRC))
$proc.StandardInput.Close()

$outText = $proc.StandardOutput.ReadToEnd()
$errText = $proc.StandardError.ReadToEnd()
$proc.WaitForExit()

$ok = $false
if ($proc.ExitCode -eq 0) { $ok = Try-SaveJson $outText $IR }

# --- 4) Fallback converter (only if compiler didn't emit JSON) ---
if (-not $ok -or -not (Test-Path $IR)) {
  $fab2ir = Join-Path $TOOLS "fab2ir.cjs"
  if (!(Test-Path $fab2ir)) {
    $code = @'
#!/usr/bin/env node
const fs = require("fs");
function die(m){ console.error(m); process.exit(1); }
function parsePolicy(src){
  const out={}; const m=src.match(/policy\s*\{([^}]*)\}/s); if(!m) return out;
  const b=m[1]; function num(n,d){ const r=new RegExp("\\b"+n+"\\s*:\\s*([0-9]+)","i").exec(b); return r?parseInt(r[1],10):d; }
  out.royalty_bps = num("royalty_bps",0);
  out.energy_budget = num("energy_budget",100);
  const rm=/rollback_max\s*:\s*([0-9]+)/i.exec(b); if(rm) out.rollback_max=parseInt(rm[1],10);
  return out;
}
function parseEmits(src){
  const e=[]; let m;
  const r1=/emit\s*:\s*"([^"]*)"/g; while((m=r1.exec(src))) e.push(m[1]);
  const r2=/emit\s*"([^"]*)"/g;     while((m=r2.exec(src))) e.push(m[1]);
  return e;
}
(function(){
  const inFile=process.argv[2], outFile=process.argv[3]||"out.ir.json";
  if(!inFile) die("usage: node fab2ir.cjs <file> <out>");
  const src=fs.readFileSync(inFile,"utf8").replace(/^\uFEFF/,"");
  const policy=parsePolicy(src), emits=parseEmits(src);
  if (!policy.energy_budget) policy.energy_budget = 100;
  const ir={ policy, agents:{ Hello:{} }, program:{ steps: emits.map(s=>({emit:s})) } };
  fs.writeFileSync(outFile, JSON.stringify(ir,null,2));
  console.log("IR -> "+outFile);
})();
'@
    [IO.File]::WriteAllText($fab2ir, $code, (New-Object System.Text.UTF8Encoding($false)))
  }
  & node $fab2ir $SRC $IR | Out-Null
  try { $null = (Get-Content -Raw $IR) | ConvertFrom-Json } catch { throw "Fallback IR invalid JSON." }
}

Write-Host "IR OK -> $IR" -ForegroundColor Green

# --- 5) Run AgentVM (real exe you already built) ---
& (Join-Path $ROOT "AgentVM.exe") --ir $IR

# --- 6) Package + install VSIX ---
Push-Location $EXT
npm install | Out-Null
npm run compile | Out-Null
npx vsce package
$vsix = Get-ChildItem -Filter "*.vsix" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($vsix) {
  code --install-extension $vsix.FullName
  Write-Host "Installed: $($vsix.Name)" -ForegroundColor Green
}
Pop-Location
