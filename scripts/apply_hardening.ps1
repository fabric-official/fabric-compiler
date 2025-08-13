param([string]$RepoRoot=".")
$ErrorActionPreference = "Stop"
function Write-Step($m){ Write-Host "==> $m" -ForegroundColor Cyan }

$bundle = Split-Path -Parent $MyInvocation.MyCommand.Path

$map = @{
  "language\grammar\grammar.patch"              = "brains\language\compiler\grammar\grammar.fb"
  "language\frontend\ast_sanitize.ts"           = "brains\language\compiler\frontend\ast_sanitize.ts"
  "language\backend\harden_lowering_pass.cc"    = "brains\language\compiler\backend\harden_lowering_pass.cc"
  "language\atomize\bounds.ts"                  = "brains\language\compiler\atomize\bounds.ts"
  "runtime\policy_kernel.cpp"                   = "runtime\policy_kernel.cpp"
  "runtime\audit_logger.cpp"                    = "runtime\audit_logger.cpp"
  "economy\contracts\RoyaltyTreasury.sol"       = "economy\contracts\RoyaltyTreasury.sol"
  "economy\contracts\DaoTimelockMultisig.sol"   = "economy\contracts\DaoTimelockMultisig.sol"
  "economy\contracts\PaymasterHardened.sol"     = "economy\contracts\PaymasterHardened.sol"
  "economy\contracts\XpChain.sol"               = "economy\contracts\XpChain.sol"
  "brains\model_guard.ts"                       = "brains\model_guard.ts"
  "cicd\security_gates.yml"                     = ".github\workflows\security_gates.yml"
}

Write-Step "Copying hardening files"
foreach ($k in $map.Keys) {
  $src = Join-Path $bundle $k
  if (!(Test-Path $src)) { Write-Host "WARN: missing $src" -ForegroundColor Yellow; continue }
  $dst = Join-Path $RepoRoot $map[$k]
  $dstDir = Split-Path -Parent $dst
  if (!(Test-Path $dstDir)) { New-Item -ItemType Directory -Force -Path $dstDir | Out-Null }
  Copy-Item -Force $src $dst
}

# Simple marker append to grammar (acts as a lightweight patch marker)
$target = Join-Path $RepoRoot "brains\language\compiler\grammar\grammar.fb"
if (Test-Path $target) {
  Write-Step "Marking grammar as hardened"
  Add-Content -Encoding UTF8 -Path $target -Value "// --- HARDENED POLICY KEYS APPLIED ---"
}

Write-Host "HARDENING BUNDLE APPLIED. Rebuild & run tests." -ForegroundColor Green
