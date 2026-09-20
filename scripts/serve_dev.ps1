<#
.SYNOPSIS
  Launch llama-server for the dev tutor config, using tutor/settings.py to
  resolve the binary path and argv (no flag-building logic duplicated here).

.PARAMETER Config
  Path to a TOML config file. Defaults to config/dev.toml relative to the
  repo root (this script's parent directory).
#>
param(
    [string] $Config
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

if (-not $Config) {
    $Config = Join-Path $RepoRoot "config\dev.toml"
}

$Lines = & python -m tutor.settings --argv $Config
if ($LASTEXITCODE -ne 0) {
    throw "tutor.settings failed to resolve config: $Config"
}

$Binary = $Lines[0]
$Argv = @($Lines | Select-Object -Skip 1)

if (-not (Test-Path $Binary)) {
    throw "server binary not found: $Binary"
}

& $Binary @Argv
exit $LASTEXITCODE
