@echo off
rem ==========================================================================
rem  LLAMA MANAGER - doble clic para arrancar
rem   1. Si no hay .\python, lo prepara (Python embebido + Flask), una sola vez
rem   2. Busca actualizaciones en GitHub y las aplica (updater.py)
rem   3. Arranca el panel en http://localhost:8080
rem  Todo va dentro de UN bloque ( ... ): cmd lo lee entero antes de ejecutarlo,
rem  asi el updater puede sustituir este mismo .bat sin romperlo a mitad.
rem ==========================================================================
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONDONTWRITEBYTECODE=1
rem set LLAMA_MANAGER_PORT=8080
rem set LLAMA_MANAGER_MODELS=D:\modelos
rem set LLAMA_MANAGER_NO_UPDATE=1
title LLAMA MANAGER
(
  if not exist "python\python.exe" (
    echo Primera ejecucion: preparando Python embebido en .\python - tarda un par de minutos...
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bootstrap.ps1"
  )
  if exist "python\python.exe" (
    "python\python.exe" updater.py
    "python\python.exe" server.py
  ) else (
    echo No se pudo preparar Python: revisa el mensaje de arriba.
  )
  if errorlevel 1 pause
  exit /b
)
