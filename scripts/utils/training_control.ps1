param(
    [Parameter(Mandatory=$true)]
    [string]$RunDir,

    [Parameter(Mandatory=$true)]
    [ValidateSet("pause", "resume", "stop", "save", "preview", "status")]
    [string]$Action
)

$controlDir = Join-Path $RunDir "control"
New-Item -ItemType Directory -Force -Path $controlDir | Out-Null

$pauseFile = Join-Path $controlDir "PAUSE"
$stopFile = Join-Path $controlDir "STOP"
$saveFile = Join-Path $controlDir "SAVE_NOW"
$previewFile = Join-Path $controlDir "PREVIEW_NOW"

switch ($Action) {
    "pause" {
        New-Item -ItemType File -Force -Path $pauseFile | Out-Null
        Write-Host "Pause requested: $pauseFile"
    }
    "resume" {
        if (Test-Path $pauseFile) {
            Remove-Item $pauseFile -Force
        }
        Write-Host "Resume requested."
    }
    "stop" {
        New-Item -ItemType File -Force -Path $stopFile | Out-Null
        Write-Host "Clean stop requested: $stopFile"
    }
    "save" {
        New-Item -ItemType File -Force -Path $saveFile | Out-Null
        Write-Host "Manual save requested: $saveFile"
    }
    "preview" {
        New-Item -ItemType File -Force -Path $previewFile | Out-Null
        Write-Host "Manual preview requested: $previewFile"
    }
    "status" {
        Write-Host "Control dir: $controlDir"
        Write-Host "Paused:  $(Test-Path $pauseFile)"
        Write-Host "Stop:    $(Test-Path $stopFile)"
        Write-Host "Save:    $(Test-Path $saveFile)"
        Write-Host "Preview: $(Test-Path $previewFile)"
    }
}

