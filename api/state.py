# -*- coding: utf-8 -*-
"""api/state.py — /api/state, /api/log, /api/mmprojs, /api/skills"""
from pathlib import Path

from flask import Blueprint, jsonify, request

from config import ENGINES, LOGS_DIR, MODELS_DIR, SKILLS_DIR
from instances import (ANTHROPIC_CACHE, INSTANCES, PROXY, RPC_SERVERS,
                       anthropic_ready, cleanup_dead, engine_info, find_draft_sidecars,
                       get_engine, health_check,
                       list_devices, list_llama_pids, list_mmprojs, list_models,
                       load_presets)
from system_monitor import SYS, attach_live_usage
from versions import APP_VERSION, LLAMA_CPP_BUILD

state_bp = Blueprint("state", __name__)


@state_bp.route("/api/state")
def api_state():
    cleanup_dead()
    devices, err = list_devices()

    # Lectura de VRAM: TOTAL de Vulkan (fiable, coincide con el Administrador
    # de tareas en todas las pruebas) + USADO de WMI (dedicada+compartida,
    # también coincide con el Administrador de tareas). Se probó usar solo
    # Vulkan para todo, pero su consulta de memoria libre NO ve el uso de
    # OTROS procesos (p.ej. un llama-server ya cargado con el modelo) — solo
    # refleja el estado de la propia consulta aislada. Así que hace falta
    # WMI para el "usado" en vivo, emparejado por orden con los dispositivos
    # Vulkan (el total de WMI, vía AdapterRAM, no es fiable para emparejar
    # por tamaño — desbordamiento de 32 bits en tarjetas >4GB).
    attach_live_usage(devices)

    managed_pids = set()
    instances = []
    for port, inst in INSTANCES.items():
        managed_pids.add(inst["proc"].pid)
        h = health_check(port)
        instances.append({
            "port":     port,
            "model":    inst["model_name"],
            "alias":    inst["alias"],
            "device":   inst["device"],
            "ctx":      inst["ctx"],
            "ngl":      inst["ngl"],
            "engine":   inst.get("engine", "vulkan"),
            "rpc":      inst.get("rpc"),
            "pid":      inst["proc"].pid,
            "health":   h,
            "uptime_s": int(__import__("time").time() - inst["started"]),
            "anthropic": anthropic_ready(port) if h == "ok" else False,
        })

    # Esclavos (ggml-rpc-server) activos en ESTE host — para que la UI muestre
    # los que están corriendo y permita pararlos.
    import time as _t
    slaves = []
    for p, s in RPC_SERVERS.items():
        if s["proc"].poll() is None:
            slaves.append({
                "port": p, "engine": s.get("engine", get_engine()),
                "pid": s["proc"].pid,
                "uptime_s": int(_t.time() - s["started"]),
            })

    orphans = [p for p in list_llama_pids() if p not in managed_pids]

    return jsonify({
        "devices":       devices,
        "devices_error": err,
        "models":        list_models(MODELS_DIR),
        "instances":     instances,
        "slaves":        slaves,
        "orphans":       orphans,
        "presets":       load_presets(),
        "sys":           SYS,
        "proxy_port":    PROXY["port"],
        "models_dir":    str(MODELS_DIR),
        "engine":        get_engine(),
        "engines":       list(ENGINES.keys()),
        "versions":      {"app": APP_VERSION, "llama_cpp": LLAMA_CPP_BUILD,
                          "engine": engine_info(), "commit": _installed_commit()},
    })


def _installed_commit():
    """Commit de GitHub aplicado por updater.py (vacío si nunca se actualizó)."""
    from config import BASE_DIR
    try:
        return (BASE_DIR / ".installed_commit").read_text(encoding="utf-8").strip()[:7]
    except OSError:
        return ""


@state_bp.route("/api/versions")
def api_versions():
    """Build instalado de cada motor vs el fijado en versions.py."""
    return jsonify({"app": APP_VERSION, "llama_cpp": LLAMA_CPP_BUILD,
                    "engines": {n: engine_info(n) for n in ENGINES}})


@state_bp.route("/api/drafts")
def api_drafts():
    """Sidecars de speculative (mtp-*.gguf, dspark-*, dflash-*) junto al modelo."""
    model = request.args.get("model", "")
    return jsonify({"drafts": find_draft_sidecars(model) if model else []})


@state_bp.route("/api/devices")
def api_devices():
    """Lista de dispositivos con soporte opcional para RPC (--rpc).

    Sin parámetro: solo GPUs locales (igual que /api/state).
    Con ?rpc=ip:puerto[,ip2:puerto2]: llama.cpp incluye los servidores RPC
    en la enumeración (aparecen como RPC0, RPC1...). La UI usa esto para
    mostrar checkboxes de TODAS las GPUs disponibles cuando hay campo maestro.
    """
    rpc = request.args.get("rpc", "").strip() or None
    devices, err = list_devices(rpc=rpc)
    return jsonify({"devices": devices, "error": err, "rpc": rpc})


@state_bp.route("/api/log")
def api_log():
    port = int(request.args["port"])
    inst = INSTANCES.get(port)
    if inst:
        path = inst["log_file"]
    else:
        logs = sorted(LOGS_DIR.glob(f"port{port}_*.log"))
        if not logs:
            return jsonify({"log": "(sin logs para ese puerto)"})
        path = logs[-1]
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        return jsonify({"log": text[-12000:]})
    except Exception as e:
        return jsonify({"log": f"Error: {e}"})


@state_bp.route("/api/mmprojs")
def api_mmprojs():
    return jsonify({"mmprojs": list_mmprojs(MODELS_DIR)})


@state_bp.route("/api/skills")
def api_skills():
    """Lista los skills disponibles en la carpeta skills/."""
    skills = []
    for p in sorted(SKILLS_DIR.glob("*.md")):
        skills.append({
            "name": p.stem,
            "filename": p.name,
            "size": p.stat().st_size,
        })
    return jsonify({"skills": skills})


@state_bp.route("/api/skills/<filename>")
def api_skill_content(filename):
    """Devuelve el contenido de un skill concreto."""
    p = SKILLS_DIR / filename
    if not p.exists() or not p.suffix == ".md":
        return jsonify({"error": "skill no encontrado"}), 404
    return jsonify({"content": p.read_text(encoding="utf-8", errors="replace")})
