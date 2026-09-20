<#
.SYNOPSIS
  Launch the tutor app's HTTP server (FastAPI/uvicorn), binding to the host
  and port from the config's [app] table.

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

Set-Location $RepoRoot
python -m tutor.app.main --config $Config
exit $LASTEXITCODE
