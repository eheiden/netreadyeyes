@echo off
setlocal
python scripts\run_pre_push_tests.py
exit /b %ERRORLEVEL%
