@echo off
rem Busca Python: el embebido del portable, si no el lanzador py, si no python del PATH
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONDONTWRITEBYTECODE=1
if exist "%~dp0python\python.exe" ( "%~dp0python\python.exe" "%~dp0update_llama.py" %* & goto :fin )
where py >nul 2>nul && ( py -3 "%~dp0update_llama.py" %* & goto :fin )
python "%~dp0update_llama.py" %*
:fin
pause
