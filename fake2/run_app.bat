@echo off
setlocal

REM Ensure the python path includes job_fraud_detection so `src` resolves
set PYTHONPATH=%~dp0job_fraud_detection;%PYTHONPATH%

python -c "import sys; print('PYTHONPATH=', sys.path)" >nul

python -m job_fraud_detection.app

