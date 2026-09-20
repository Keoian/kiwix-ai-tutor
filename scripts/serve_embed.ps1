<#
.SYNOPSIS
  Launch a SECOND llama-server, started with --embedding, for the dense
  sidecar's embedding model. This runs ALONGSIDE (not instead of) the chat
  server started by serve_dev.ps1 -- they are two separate processes on two
  separate ports, both read from the same config file's [runtime]/[embedding]
  tables. Never hard-codes a binary or model path; both come from config via
  tutor/settings.py --argv-embedding.

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

$Lines = & python -m tutor.settings --argv-embedding $Config
if ($LASTEXITCODE -ne 0) {
    throw "tutor.settings failed to resolve [embedding] config: $Config"
}

$Binary = $Lines[0]
$Argv = @($Lines | Select-Object -Skip 1)

if (-not (Test-Path $Binary)) {
    throw "server binary not found: $Binary"
}

& $Binary @Argv
exit $LASTEXITCODE
