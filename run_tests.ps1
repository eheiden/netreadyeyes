$ErrorActionPreference = "Stop"
python .\scripts\run_pre_push_tests.py
exit $LASTEXITCODE
