@echo off
setlocal enableextensions enabledelayedexpansion

chcp 1251 >NUL

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "VENV_PYTHON=%ROOT%venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo [ERROR] venv not found at %VENV_PYTHON%
    pause
    exit /b 1
)

set "PYTHONIOENCODING=cp1251"
set "PYTHONPATH=%ROOT%src;%PYTHONPATH%"

::echo ==== [%date% %time%] WAITING for Monday 09:50 MSK ====
::"%VENV_PYTHON%" -X utf8=0 -m moex_tools.cli wait || exit /b %ERRORLEVEL%

echo ==== [%date% %time%] START: make-clean ====
"%VENV_PYTHON%" -X utf8=0 -m moex_tools.cli make-clean

echo ==== [%date% %time%] START: low-volatility ====
"%VENV_PYTHON%" -X utf8=0 -m moex_tools.cli low-volatility

echo ==== [%date% %time%] DONE. ====
