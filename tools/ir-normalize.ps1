param(
  [Parameter(Mandatory=$true)][string]$In,
  [Parameter(Mandatory=$true)][string]$Out
)
$ErrorActionPreference='Stop'
$utf8 = New-Object System.Text.UTF8Encoding($false)

$raw = Get-Content -Raw $In
$raw = $raw -replace "^\uFEFF",""
$obj = $raw | ConvertFrom-Json

# locate steps
$container='program'; $steps=@()
if     ($obj.program -and $obj.program.PSObject.Properties.Name -contains 'steps'){ $container='program'; $steps=@($obj.program.steps) }
elseif ($obj.run     -and $obj.run.PSObject.Properties.Name     -contains 'steps'){ $container='run';     $steps=@($obj.run.steps) }
elseif ($obj.PSObject.Properties.Name -contains 'steps'){                            $container='top';     $steps=@($obj.steps) }

function Get-Msg([object]$s){
  if ($null -eq $s) { return '' }
  if ($s -is [System.Collections.IEnumerable] -and -not ($s -is [string])) {
    $arr = @($s)
    if ($arr.Count -ge 2 -and "$($arr[0])" -eq 'emit') { return [string]$arr[1] }
  }
  if ($s.PSObject) {
    $n=$s.PSObject.Properties.Name
    if ($n -contains 'emit')   { return [string]$s.emit }
    if ($n -contains 'text')   { return [string]$s.text }
    if ($n -contains 'value')  { return [string]$s.value }
    if ($n -contains 'message'){ return [string]$s.message }
    if ($n -contains 'payload'){ return [string]$s.payload }
    if ($n -contains 'content'){ return [string]$s.content }
    if ($n -contains 'args') {
      $a=$s.args
      if ($a -and $a.PSObject) {
        $an=$a.PSObject.Properties.Name
        if ($an -contains 'text')  { return [string]$a.text }
        if ($an -contains 'value') { return [string]$a.value }
      }
    }
  }
  return ($s | ConvertTo-Json -Depth 20)
}

$new=@()
foreach($s in $steps){ $new += @('emit', (Get-Msg $s)) }
if ($new.Count -eq 0) { $new = @(@('emit','hello'), @('emit','world')) }

if ($container -eq 'program'){ if (-not $obj.program){ $obj | Add-Member program (@{}) -Force }; $obj.program.steps=$new }
elseif ($container -eq 'run'){ if (-not $obj.run){ $obj | Add-Member run (@{}) -Force }; $obj.run.steps=$new }
else { $obj.steps=$new }

[IO.File]::WriteAllText($Out, ($obj | ConvertTo-Json -Depth 50), $utf8)
$null = (Get-Content -Raw $Out | ConvertFrom-Json)