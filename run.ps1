param([int]$Port = 8501)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Create the project environment first. See RUNNING.md.'
}
& $projectPython -m streamlit run app/streamlit_app.py --server.port $Port --server.address 127.0.0.1
