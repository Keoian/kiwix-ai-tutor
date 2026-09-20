<#
.SYNOPSIS
    Launch the tutor UI in kiosk (fullscreen, chromeless) mode on Windows.

.DESCRIPTION
    Tries Microsoft Edge first, falling back to Chrome if Edge is not
    installed. Both are launched in kiosk mode against the given URL
    (defaults to the loopback tutor app).
#>

param(
    [string]$Url = "http://127.0.0.1:8420/"
)

$edgePaths = @(
    "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
    "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe"
)

$chromePaths = @(
    "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe"
)

function Find-Browser($paths) {
    foreach ($path in $paths) {
        if ($path -and (Test-Path $path)) {
            return $path
        }
    }
    return $null
}

$edge = Find-Browser $edgePaths
$chrome = Find-Browser $chromePaths

if ($edge) {
    & $edge --kiosk $Url --edge-kiosk-type=fullscreen
} elseif ($chrome) {
    & $chrome --kiosk $Url
} else {
    Write-Error "Neither Microsoft Edge nor Google Chrome was found."
    exit 1
}
