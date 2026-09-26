# -*- coding: utf-8 -*-
# /// script
# requires-python = ">=3.10"
# dependencies = ["flask"]
# ///
"""
LLAMA MANAGER — Gestor de instancias llama-server (versión en versions.py)
=============================================================================
Estructura modular:
  server.py          ← punto de entrada (este fichero)
  versions.py        ← APP_VERSION + build de llama.cpp fijado (LLAMA_CPP_BUILD)
  config.py          ← constantes: rutas, puerto, motores
  system_monitor.py  ← RAM + VRAM (rocm-smi / nvidia-smi / WMI)
  gguf_meta.py       ← leer metadatos GGUF + inferir capabilities
  instances.py       ← estado global de instancias, arranque/parada
  api/
    state.py         ← /api/state, /api/log, /api/mmprojs, /api/skills
    control.py       ← /api/start, /api/stop, /api/kill*, /api/proxy, /api/profile, /api/shutdown
    chat.py          ← /api/chat, /api/vision, /api/chatlog, /api/cctest
    tools.py         ← /api/estimate
    proxy_route.py   ← /v1/*
  api/cloud.py       ← /api/cloud/providers|keys|models|chat (probar APIs online)
  ui/templates/
    index.html       ← UI Jinja2 con sidebar
  skills/
    opencvsharp4.md  ← documentación inyectable en tests
    *.md             ← añade más skills aquí
"""

import os
import sys
from pathlib import Path

# Asegurar que el directorio del proyecto está en el path
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, render_template, request

from config import BASE_DIR, IS_WINDOWS, MANAGER_PORT, PRESETS_FILE
from versions import APP_VERSION, LLAMA_CPP_BUILD
from instances import (PROXY, engine_exe, engine_info, get_engine, load_persisted_engine,
                       load_presets)
from system_monitor import start_monitor

# Blueprints
from api.state       import state_bp
from api.control     import control_bp
from api.chat        import chat_bp
from api.tools       import tools_bp
from api.proxy_route import proxy_bp
from api.agent       import agent_bp
from api.downloads   import hf_bp
from api.cloud       import cloud_bp
from api.peers       import peers_bp

# ── Flask ──────────────────────────────────────────────────────────────────
# Rutas ABSOLUTAS (config.BASE_DIR): en modo congelado (.exe) los .py viven en
# _internal/ y Flask no resuelve un template_folder relativo contra __main__;
# dando la ruta completa evitamos ese fallo de PyInstaller. En modo desarrollo
# BASE_DIR = la carpeta del proyecto, así que esto es idéntico a antes.
_TPL = BASE_DIR / "ui" / "templates"
_ST = BASE_DIR / "ui" / "static"
app = Flask(
    __name__,
    template_folder=str(_TPL),
    static_folder=str(_ST) if _ST.is_dir() else None,
    static_url_path="/static",
)

app.register_blueprint(state_bp)
app.register_blueprint(control_bp)
app.register_blueprint(chat_bp)
app.register_blueprint(tools_bp)
app.register_blueprint(proxy_bp)
app.register_blueprint(agent_bp)
app.register_blueprint(hf_bp)
app.register_blueprint(cloud_bp)
app.register_blueprint(peers_bp)


# ── CORS mínimo ─────────────────────────────────────────────────────────────
# /api/host es de solo lectura (GPUs/RAM/instancias, sin secretos) y está
# pensado para que OTRO llama_manager, en otro equipo de la red, lo lea desde
# su propio navegador. Un fetch() a http://otro-equipo:8080/api/host con el
# origen distinto falla si no hay cabeceras CORS, así que las inyectamos aquí,
# solo para esa ruta. El resto del manager sigue sin exponer CORS (sus rutas
# de control no deben ser alcanzables desde otra página).
@app.after_request
def _cors_peers(resp):
    if request.path == "/api/host":
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/")
def index():
    return render_template("index.html")


# ── Arranque ───────────────────────────────────────────────────────────────

def main():
    # En un .exe congelado PyInstaller hereda el codec de consola de Windows
    # (cp1252 por defecto) y cualquier carácter fuera de ese conjunto (→, ·,
    # …) lanza UnicodeEncodeError. Forzamos UTF-8 para que la salida sea
    # idéntica en cualquier máquina. En modo desarrollo esto es un no-op
    # (Python ya trae stdout en UTF-8 desde 3.7 con PYTHONUTF8, y si no
    # reconfigure() es un fallback a la codificación actual, sin romper).
    import sys as _s
    for _stream in (_s.stdout, _s.stderr):
        if hasattr(_stream, "reconfigure"):
            try: _stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception: pass

    # Cargar proxy y motor (vulkan/rocm) persistidos
    presets = load_presets()
    _p = presets.get("_proxy")
    if _p:
        PROXY["port"] = _p.get("port")
    load_persisted_engine()

    exe = engine_exe()
    if not exe.exists():
        print(f"AVISO: no encuentro {exe} — revisa ENGINES en config.py o cambia de motor en la UI")
    elif not IS_WINDOWS and not os.access(exe, os.X_OK):
        print(f'AVISO: {exe} no tiene permiso de ejecución — ejecuta: chmod +x "{exe}"')
    else:
        info = engine_info()
        if not info.get("match"):
            print(f"AVISO: motor {info['engine']} = {info.get('installed')} · esperado {LLAMA_CPP_BUILD} "
                  f"(versions.py) — ejecuta update_llama para igualarlo")

    start_monitor()

    plat = "Windows" if IS_WINDOWS else "Linux/otro"
    print(f"LLAMA MANAGER v{APP_VERSION} [{plat}] · llama.cpp {LLAMA_CPP_BUILD} → http://localhost:{MANAGER_PORT}  (motor: {get_engine()})")
    print(f"Proxy estable → http://<IP>:{MANAGER_PORT}/v1")
    print(f"Claude Code   → ANTHROPIC_BASE_URL=http://<IP>:{MANAGER_PORT}")

    # Doble clic → abre el panel en el navegador (así no hace falta teclear la URL).
    # Se salta con LLAMA_MANAGER_NO_BROWSER=1 (p.ej. arranque por script o ssh, sin UI).
    if not os.environ.get("LLAMA_MANAGER_NO_BROWSER"):
        import threading, webbrowser
        threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{MANAGER_PORT}")).start()

    app.run(host="0.0.0.0", port=MANAGER_PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
