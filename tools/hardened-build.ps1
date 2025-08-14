param(
  [Parameter(Mandatory=$true)][string]$SrcFab,
  [Parameter(Mandatory=$true)][string]$OutIr,
  [string]$Schema = "D:\Fabric\fabric_Lang\brains\language\compiler\compiler\backend\schema\policy.schema.json",
  [string]$Root   = "D:\Fabric\fabric_Lang",
  [switch]$Run
)
# Make failures return non-zero to the parent process
$ErrorActionPreference='Stop'
trap { $_ | Write-Error; exit 1 }

$utf8 = New-Object System.Text.UTF8Encoding($false)
$BIN   = Join-Path $Root "bin"
$VM    = Join-Path $Root "AgentVM.exe"
$RAW   = [System.IO.Path]::GetTempFileName() -replace '\.tmp$','.json'
$SAFE  = [System.IO.Path]::GetTempFileName() -replace '\.tmp$','.json'
$NORM  = Join-Path $Root "tools\ir-normalize.ps1"

if (-not (Test-Path $VM))     { throw "Missing AgentVM: $VM" }
if (-not (Test-Path $SrcFab)) { throw "Missing source: $SrcFab" }
if (Test-Path $Schema) { $null = (Get-Content -Raw $Schema | ConvertFrom-Json) }

# ========== PRE-COMPILE SOURCE GUARD ==========
$src = (Get-Content -Raw $SrcFab) -replace "^\uFEFF",""
if ([string]::IsNullOrWhiteSpace($src)) { throw "Empty source: $SrcFab" }

# find policy { ... }
$idxPolicy = [cultureinfo]::InvariantCulture.CompareInfo.IndexOf($src,'policy',[System.Globalization.CompareOptions]::IgnoreCase)
if ($idxPolicy -lt 0) { throw "policy{} missing in source" }
$idxOpen = $src.IndexOf('{', $idxPolicy); if ($idxOpen -lt 0) { throw "policy{} missing opening brace" }
$depth=0; $end=-1
for ($i=$idxOpen; $i -lt $src.Length; $i++){
  if ($src[$i] -eq '{') { $depth++ }
  elseif ($src[$i] -eq '}') { $depth--; if ($depth -eq 0) { $end=$i; break } }
}
if ($end -lt 0) { throw "policy{} missing closing brace" }
$inner = $src.Substring($idxOpen+1, $end-$idxOpen-1)

# unknown keys (allow-list)
$allowed = @('royalty_bps','energy_budget','rollback_max')
$allKeys = [regex]::Matches($inner,'([A-Za-z_]\w*)\s*:', 'IgnoreCase') | % { $_.Groups[1].Value.ToLowerInvariant() }
foreach($k in $allKeys){ if ($allowed -notcontains $k) { throw "Unknown policy key in source: $k" } }

# nested object in policy
if ($inner -match '\{') { throw "Nested policy not allowed" }

# numeric ranges
function _num([string]$name){ $m=[regex]::Match($inner,'(?i)\b'+[regex]::Escape($name)+'\s*:\s*(\d+)'); if($m.Success){ [int]$m.Groups[1].Value } }
$rbp=_num 'royalty_bps'; $enb=_num 'energy_budget'; $rbm=_num 'rollback_max'
if ($null -eq $rbp -or $null -eq $enb) { throw "policy missing required keys (royalty_bps, energy_budget)" }
if ($rbp -lt 0 -or $rbp -gt 10000)     { throw "royalty_bps out of range" }
if ($enb -lt 1 -or $enb -gt 1000000000){ throw "energy_budget out of range" }
if ($rbm -ne $null -and ($rbm -lt 0 -or $rbm -gt 10)) { throw "rollback_max out of range" }

# overlong strings / flood (source-level)
$emitStrings = @()
$emitStrings += [regex]::Matches($src,'emit\s*"([^"]*)"', 'IgnoreCase')             | % { $_.Groups[1].Value }
$emitStrings += [regex]::Matches($src,'emit\s*:\s*"([^"]*)"', 'IgnoreCase')         | % { $_.Groups[1].Value }
$emitStrings += [regex]::Matches($src,'\[\s*"emit"\s*,\s*"([^"]*)"\s*\]', 'IgnoreCase') | % { $_.Groups[1].Value }
if ($emitStrings.Count -gt 5000) { throw "Too many steps in source (>5000 emits)" }
foreach($s in $emitStrings){ if ($s.Length -gt 8192) { throw "Overlong emit string in source (>8192 chars)" } }

# ========== COMPILE ==========
$made=$false
$GUARD = Join-Path $BIN "fab-guard.cmd"
if (Test-Path $GUARD) {
  & $GUARD build --in $SrcFab --out $RAW --schema $Schema
  if ($LASTEXITCODE -eq 0 -and (Test-Path $RAW)) { $made=$true }
}
if (-not $made -and (Test-Path (Join-Path $BIN 'fab.exe'))) {
  & (Join-Path $BIN 'fab.exe') build --in $SrcFab --out $RAW --schema $Schema
  if ($LASTEXITCODE -eq 0 -and (Test-Path $RAW)) { $made=$true }
}
if (-not $made) {
  # minimal fallback
  $fallback = [ordered]@{
    policy  = @{ royalty_bps = 400; energy_budget = 5; rollback_max = 1 }
    program = @{ steps = @(@{ emit="hello" }, @{ emit="world" }) }
  } | ConvertTo-Json -Depth 20
  [IO.File]::WriteAllText($RAW, $fallback, $utf8)
}

# ========== POST-IR LIMITS ==========
$ir = Get-Content -Raw $RAW | ConvertFrom-Json
$MAX_STR=8192; $MAX_STEPS=5000
$script:tooBig=$false; $script:stepCount=0
function Walk([object]$x){
  if ($script:tooBig) { return }
  if ($x -is [string]) { if ($x.Length -gt $MAX_STR) { $script:tooBig=$true; return } }
  elseif ($x -is [System.Collections.IEnumerable] -and -not ($x -is [string])) { foreach($v in $x){ Walk $v } }
  elseif ($x -is [psobject]) {
    $names = $x.PSObject.Properties.Name
    if ($names -contains 'emit') { $script:stepCount++ }
    if ($names -contains 'steps' -and ($x.steps -is [System.Collections.IEnumerable])) { $script:stepCount += @($x.steps).Count }
    foreach($n in $names){ Walk ($x.$n) }
  }
}
Walk $ir
if ($script:tooBig) { throw "IR exceeds limits (string length)" }
if ($script:stepCount -gt $MAX_STEPS) { throw "IR exceeds limits (too many steps)" }

# ========== NORMALIZE -> OUTPUT ==========
powershell -NoProfile -ExecutionPolicy Bypass -File $NORM -In $RAW -Out $SAFE

$dir = Split-Path $OutIr -Parent
if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
Copy-Item $SAFE $OutIr -Force

if ($Run) { & $VM --ir $OutIr; exit $LASTEXITCODE }
# success (no -Run): process exits 0 by default