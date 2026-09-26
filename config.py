# -*- coding: utf-8 -*-
"""
config.py — Constantes globales de LLAMA MANAGER
Edita aquí las rutas y el puerto antes de arrancar.
"""
import os
import platform
from pathlib import Path

# ── Plataforma ────────────────────────────────────────────────────────────
# Único punto de verdad: el resto del proyecto (instances.py,
# system_monitor.py, api/*) importa IS_WINDOWS/EXE_NAME de aquí en vez de
# comprobar platform.system() suelto en cada sitio.
IS_WINDOWS = platform.system() == "Windows"
EXE_NAME   = "llama-server.exe" if IS_WINDOWS else "llama-server"

# Directorio base del que cuelgan el MOTOR y los datos (presets.json, logs/,
# skills/). Dos casos:
#  • Congelado (PyInstaller → .exe): los .py viven en <exe>/_internal/, así que
#    la referencia fiable a "dónde está este programa" es la carpeta del .exe.
#    Ahí colocamos llama-vulkan/, presets.json, logs/ y skills/ — portable tal cual.
#  • No congelado (python server.py): la carpeta del proyecto — igual que antes,
#    sin cambio de comportamiento.
import sys as _sys
if getattr(_sys, "frozen", False):          # dentro de un .exe de PyInstaller
    BASE_DIR = Path(_sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent

# Motores disponibles (builds de llama-server) — clave usada en la UI/API,
# valor = carpeta que contiene el binario de llama-server + sus DLL/.so.
#
#  • "vulkan"   → <BASE_DIR>/llama-vulkan   (build oficial de GitHub, fijado en versions.py)
#  • "rocm"     → <BASE_DIR>/llama-rocm     (build oficial ROCm/HIP de GitHub, mismo build)
#  • "rocm-lms" → backend ROCm que trae LM Studio (solo Windows, si está instalado).
#                 Se autodetecta la carpeta más reciente — LM Studio la renombra
#                 en cada actualización (2.28.2 → 2.37.0 → 2.43.0...), así que ya
#                 no hay ruta fija con el usuario "Bravo" dentro.
# Un motor solo se registra si su carpeta trae llama-server: así el portable
# funciona en equipos sin ROCm y sin LM Studio sin tocar nada.

def _newest(glob_root: Path, pattern: str):
    """Carpeta más reciente (por nombre de versión) que casa con pattern."""
    try:
        hits = [p for p in glob_root.glob(pattern) if p.is_dir()]
    except OSError:
        return None
    def _key(p):
        import re as _re
        return [int(x) for x in _re.findall(r"\d+", p.name)]
    return max(hits, key=_key) if hits else None

_LMS_BACKENDS = Path.home() / ".lmstudio" / "extensions" / "backends"

ENGINES = {"vulkan": BASE_DIR / "llama-vulkan"}
# Variables de entorno extra por motor, fusionadas con el entorno actual al
# lanzar --list-devices o una instancia. Ej: {"HSA_OVERRIDE_GFX_VERSION": "11.0.0"}.
# La clave especial "PATH" se ANTEPONE al PATH heredado (no lo sustituye).
ENGINE_ENV = {"vulkan": {}}

if (BASE_DIR / "llama-rocm" / EXE_NAME).exists():
    ENGINES["rocm"] = BASE_DIR / "llama-rocm"
    ENGINE_ENV["rocm"] = {}

if IS_WINDOWS:
    _lms = _newest(_LMS_BACKENDS, "llama.cpp-win-x86_64-amd-rocm-avx2-*")
    if _lms and (_lms / EXE_NAME).exists():
        ENGINES["rocm-lms"] = _lms
        _vendor = _newest(_LMS_BACKENDS / "vendor", "win-llama-rocm-vendor-v*")
        # amdhip64, rocblas, hipblas(lt), amd_comgr + kernels Tensile de LM Studio
        ENGINE_ENV["rocm-lms"] = {"PATH": str(_vendor / "bin")} if _vendor else {}

DEFAULT_ENGINE = "vulkan"

LLAMA_DIR    = ENGINES[DEFAULT_ENGINE]  # compat: motor activo al arrancar
# ~/.lmstudio/models existe igual en Windows (C:\Users\<user>\.lmstudio) y en
# Linux (/home/<user>/.lmstudio) — Path.home() resuelve el equivalente en
# cada SO sin tener que hardcodear el usuario ni la unidad.
MODELS_DIR   = Path(os.environ["LLAMA_MANAGER_MODELS"]) if os.environ.get("LLAMA_MANAGER_MODELS") \
               else Path.home() / ".lmstudio" / "models"
# Carpetas EXTRA con GGUF, barridas por list_models() junto a MODELS_DIR —
# donde aterrizar descargas directas (wget/hf) para que aparezcan en la UI
# sin moverlas. Borrar sigue protegido: delete_model() solo borra dentro de
# MODELS_DIR, así que nada de aquí se toca con el botón de borrar.
EXTRA_MODELS_DIRS = [Path.home() / "llm-downloads"]
# ~/models: en los nodos Linux los GGUF viven ahí (p.ej. SpectrumIA02). Se
# añade solo si la carpeta existe, para no romper nada en Windows.
if not IS_WINDOWS and (Path.home() / "models").is_dir():
    EXTRA_MODELS_DIRS.append(Path.home() / "models")
LOGS_DIR     = BASE_DIR / "logs"
PRESETS_FILE = BASE_DIR / "presets.json"
SKILLS_DIR   = BASE_DIR / "skills"
MANAGER_PORT = int(os.environ.get("LLAMA_MANAGER_PORT", "8080"))
# Puerto que escucha ggml-rpc-server cuando arrancas un esclavo. Si en el otro
# host llama_manager ya ocupa ese puerto, cámbialo aquí (y en el campo de la UI).
RPC_PORT     = 50052

LLAMA_EXE = LLAMA_DIR / EXE_NAME  # compat: usar instances.engine_exe() para el motor activo real

LOGS_DIR.mkdir(exist_ok=True)
SKILLS_DIR.mkdir(exist_ok=True)

KV_BYTES = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}
