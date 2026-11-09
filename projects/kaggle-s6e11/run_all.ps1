param(
    [switch]$Quick
)

$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$pythonArgs = @()
if (-not (Test-Path -LiteralPath $python)) {
    $python = "py"
    $pythonArgs = @("-3")
}

$arguments = @("train_models.py", "--model", "all")
if ($Quick) {
    $arguments += "--quick"
}

& $python @pythonArgs @arguments

if (-not $Quick) {
    & $python @pythonArgs "make_report_assets.py"
}
