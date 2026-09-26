# -*- coding: utf-8 -*-
"""api/control.py — /api/start, /api/stop, /api/kill, /api/killall, /api/proxy,
                    /api/shutdown, /api/profile"""
import subprocess

from flask import Blueprint, jsonify, request

from config import ENGINES, IS_WINDOWS, MODELS_DIR, RPC_PORT
from instances import (ANTHROPIC_CACHE, INSTANCES, PROXY, RPC_SERVERS, delete_model,
                       force_kill, get_engine, list_llama_pids, load_presets,
                       save_presets, set_engine, start_instance, stop_instance,
                       start_slave, stop_slave)

from runlog import manager_log

control_bp = Blueprint("control", __name__)


@control_bp.route("/api/start", methods=["POST"])
def api_start():
    cfg = request.get_json(force=True)
    desc = (f"ARRANCAR puerto {cfg.get('port')} · motor {cfg.get('engine') or 'activo'} · "
            f"device {cfg.get('device') or 'auto'} · modelo {cfg.get('model')}")
    manager_log(desc)
    try:
        pid, err = start_instance(cfg)
    except Exception as e:
        manager_log(f"  EXCEPCIÓN al arrancar puerto {cfg.get('port')}: {e}", exc=e)
        return jsonify({"error": f"Excepción interna al arrancar: {type(e).__name__}: {e} (ver logs/manager.log)"}), 500
    if err:
        manager_log(f"  ERROR: {err}")
        return jsonify({"error": err}), 400
    manager_log(f"  OK → PID {pid}")
    return jsonify({"ok": True, "pid": pid})


@control_bp.route("/api/stop", methods=["POST"])
def api_stop():
    body  = request.get_json(force=True)
    port  = int(body["port"])
    force = bool(body.get("force", False))
    manager_log(f"PARAR puerto {port}" + (" (forzado)" if force else ""))
    ok, err = stop_instance(port, force)
    if not ok:
        manager_log(f"  ERROR: {err}")
        return jsonify({"error": err}), 404
    return jsonify({"ok": True})


@control_bp.route("/api/kill", methods=["POST"])
def api_kill():
    pid = int(request.get_json(force=True)["pid"])
    if pid not in list_llama_pids():
        return jsonify({"error": "PID no es llama-server.exe"}), 400
    force_kill(pid)
    return jsonify({"ok": True})


@control_bp.route("/api/killall", methods=["POST"])
def api_killall():
    for pid in list_llama_pids():
        force_kill(pid)
    INSTANCES.clear()
    ANTHROPIC_CACHE.clear()
    return jsonify({"ok": True})


@control_bp.route("/api/proxy", methods=["POST"])
def api_proxy():
    port = request.get_json(force=True).get("port")
    PROXY["port"] = int(port) if port else None
    presets = load_presets()
    presets["_proxy"] = {"port": PROXY["port"]}
    save_presets(presets)
    return jsonify({"ok": True, "proxy_port": PROXY["port"]})


@control_bp.route("/api/profile", methods=["POST"])
def api_profile():
    body   = request.get_json(force=True)
    action = body.get("action")
    slot   = (body.get("slot") or "").strip()
    presets = load_presets()

    if action == "list":
        names = sorted(
            k[len("_profile_"):] for k in presets if k.startswith("_profile_")
        )
        default_slot = (presets.get("_default_profile") or {}).get("slot")
        if default_slot not in names:
            default_slot = None
        return jsonify({"ok": True, "profiles": names, "default": default_slot})

    if action == "save":
        if not slot:
            return jsonify({"ok": False, "error": "Nombre de perfil vacío"})
        presets[f"_profile_{slot}"] = body.get("cfg", {})
        save_presets(presets)
        return jsonify({"ok": True})

    if action == "load":
        cfg = presets.get(f"_profile_{slot}")
        if not cfg:
            return jsonify({"ok": False, "error": f"Perfil {slot} no guardado todavía"})
        return jsonify({"ok": True, "cfg": cfg})

    if action == "delete":
        key = f"_profile_{slot}"
        if key not in presets:
            return jsonify({"ok": False, "error": f"Perfil {slot} no existe"})
        del presets[key]
        if (presets.get("_default_profile") or {}).get("slot") == slot:
            del presets["_default_profile"]
        save_presets(presets)
        return jsonify({"ok": True})

    if action == "set_default":
        if f"_profile_{slot}" not in presets:
            return jsonify({"ok": False, "error": f"Perfil {slot} no existe — guárdalo primero"})
        presets["_default_profile"] = {"slot": slot}
        save_presets(presets)
        return jsonify({"ok": True, "default": slot})

    if action == "clear_default":
        presets.pop("_default_profile", None)
        save_presets(presets)
        return jsonify({"ok": True})

    return jsonify({"error": "acción desconocida"}), 400


@control_bp.route("/api/engine", methods=["GET", "POST"])
def api_engine():
    if request.method == "GET":
        return jsonify({
            "engine": get_engine(), "engines": list(ENGINES.keys()),
            "slaves": [p for p, s in RPC_SERVERS.items()
                       if s["proc"].poll() is None],
        })
    body = request.get_json(force=True)
    ok, err = set_engine(body.get("engine"))
    if not ok:
        return jsonify({"error": err}), 400
    return jsonify({"ok": True, "engine": get_engine()})


# ── Modo esclavo (ggml-rpc-server): exponer las GPUs de ESTE host ──────────
@control_bp.route("/api/slave", methods=["POST"])
def api_slave_start():
    body   = request.get_json(force=True)
    engine = (body.get("engine") or "").strip() or None
    port   = int(body.get("port") or RPC_PORT)
    sp, err = start_slave(engine=engine, port=port)
    if err:
        return jsonify({"error": err}), 400
    return jsonify({"ok": True, "port": sp, "engine": engine or get_engine()})


@control_bp.route("/api/slave/<int:port>", methods=["DELETE"])
def api_slave_stop(port):
    force = bool((request.get_json(silent=True) or {}).get("force", False))
    ok, err = stop_slave(port, force=force)
    if not ok:
        return jsonify({"error": err}), 404
    return jsonify({"ok": True})


@control_bp.route("/api/model", methods=["DELETE"])
def api_delete_model():
    body = request.get_json(force=True)
    ok, result = delete_model(body.get("path", ""), MODELS_DIR)
    if not ok:
        return jsonify({"error": result}), 400
    return jsonify({"ok": True, "deleted": result})


@control_bp.route("/api/shutdown", methods=["POST"])
def api_shutdown():
    try:
        if IS_WINDOWS:
            subprocess.Popen(
                ["shutdown", "/s", "/t", "10", "/c", "Apagado solicitado desde LLAMA MANAGER"],
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:
            # shutdown -h +1 en Linux necesita normalmente privilegios de root
            # (o polkit configurado) — si falla por permisos, el error queda
            # en la respuesta en vez de perderse en el Popen.
            subprocess.Popen(["shutdown", "-h", "+1", "Apagado solicitado desde LLAMA MANAGER"])
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
