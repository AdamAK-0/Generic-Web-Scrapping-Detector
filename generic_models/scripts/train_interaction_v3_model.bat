@echo off
setlocal
for %%I in ("%~dp0..\..") do set PROJECT_ROOT=%%~fI
cd /d "%PROJECT_ROOT%"

set PYTHON_EXE=python
if exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" set PYTHON_EXE=%PROJECT_ROOT%\.venv\Scripts\python.exe

"%PYTHON_EXE%" -m generic_models.train_interaction_v3_model ^
  --artifacts-dir generic_models\artifacts ^
  --sessions-per-scenario 18

endlocal
