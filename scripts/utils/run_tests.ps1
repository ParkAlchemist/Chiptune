$ErrorActionPreference = "Stop"

cd C:\Users\sklem\Chiptune
$env:PYTHONPATH = "."

pytest -v

