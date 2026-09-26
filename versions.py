# -*- coding: utf-8 -*-
"""
versions.py — ÚNICO sitio donde se fijan las versiones de LLAMA MANAGER.

Cuando salga un llama.cpp nuevo:
  1. Cambia LLAMA_CPP_BUILD aquí (p.ej. "b11250").
  2. Ejecuta update_llama.ps1 (Windows) o update_llama.sh (Linux): descarga
     los motores de ESE build desde GitHub Releases a llama-vulkan/ y llama-rocm/.
  3. Reinicia llama_manager. El panel avisa si el build instalado no coincide.

Importante para RPC (maestro/esclavo en red): todos los equipos deben correr
EXACTAMENTE el mismo build de llama.cpp. Por eso el build está fijado aquí y
no se usa "latest".
"""

# Versión de la propia aplicación (panel + API)
APP_VERSION = "4.6.0"

# Repositorio de la aplicación: al arrancar, LlamaManager.bat ejecuta updater.py,
# que compara el último commit de esta rama con el instalado (.installed_commit)
# y, si hay uno nuevo, se baja el código y lo aplica. Desactivar: variable de
# entorno LLAMA_MANAGER_NO_UPDATE=1. Repo privado: pon un token (solo lectura)
# en github_token.txt junto a este fichero.
UPDATE_REPO = "Absaiz/Llama-Manager"
UPDATE_BRANCH = "main"

# Build de llama.cpp que espera esta versión del manager. Se comprueba contra
# `llama-server --version` al arrancar (ver instances.engine_info()).
LLAMA_CPP_BUILD = "b11193"

# Versión del runtime ROCm/HIP con el que GitHub publica los binarios ROCm.
# Cambia cuando llama.cpp cambia el runtime (antes 7.14, ahora 10.0). Si
# update_llama falla con 404 en el zip ROCm, mira el nombre real del asset en
# https://github.com/ggml-org/llama.cpp/releases y ajusta esto.
ROCM_RUNTIME = "10.0"

# Python embebido del portable (python-build-standalone de Astral, en GitHub).
# El portable no necesita Python instalado en el equipo destino.
PYTHON_VERSION = "3.11.13"
PYTHON_STANDALONE_TAG = "20250612"
PYTHON_URL = ("https://github.com/astral-sh/python-build-standalone/releases/download/"
              "{tag}/cpython-{ver}+{tag}-{triple}-install_only.tar.gz")
PYTHON_TRIPLE = {"windows": "x86_64-pc-windows-msvc", "linux": "x86_64-unknown-linux-gnu"}

# Paquetes Python del manager (psutil y ddgs son opcionales: nº de núcleos y
# búsqueda web del agente de pruebas)
PY_PACKAGES = ["flask", "psutil", "ddgs"]

RELEASES_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{build}/{asset}"

# Nombre del asset de GitHub por (sistema, motor). {build} y {rocm} se rellenan.
ASSETS = {
    ("windows", "vulkan"): "llama-{build}-bin-win-vulkan-x64.zip",
    ("windows", "rocm"):   "llama-{build}-bin-win-rocm-{rocm}-x64.zip",
    ("windows", "cpu"):    "llama-{build}-bin-win-cpu-x64.zip",
    ("linux",   "vulkan"): "llama-{build}-bin-ubuntu-vulkan-x64.tar.gz",
    ("linux",   "rocm"):   "llama-{build}-bin-ubuntu-rocm-{rocm}-x64.tar.gz",
    ("linux",   "cpu"):    "llama-{build}-bin-ubuntu-x64.tar.gz",
}


def asset_url(system: str, engine: str, build: str = None) -> str:
    """URL de descarga del motor `engine` para `system` ("windows"/"linux")."""
    build = build or LLAMA_CPP_BUILD
    asset = ASSETS[(system, engine)].format(build=build, rocm=ROCM_RUNTIME)
    return RELEASES_URL.format(build=build, asset=asset)


def python_url(system: str) -> str:
    return PYTHON_URL.format(tag=PYTHON_STANDALONE_TAG, ver=PYTHON_VERSION,
                             triple=PYTHON_TRIPLE[system])


def build_number(text: str):
    """Extrae el nº de build de 'b11193' o de la salida de --version
    ('version: 0.5.0-dev (build 11193, commit ...)'). None si no hay."""
    import re
    if not text:
        return None
    m = re.search(r"build\s+(\d+)", text) or re.search(r"\bb(\d{3,})\b", text)
    return int(m.group(1)) if m else None


if __name__ == "__main__":
    # Lo usan update_llama.ps1 / update_llama.sh para no duplicar versiones:
    #   python versions.py                 -> b11193
    #   python versions.py url windows vulkan
    import sys
    if len(sys.argv) >= 4 and sys.argv[1] == "url":
        print(asset_url(sys.argv[2], sys.argv[3]))
    else:
        print(LLAMA_CPP_BUILD)
