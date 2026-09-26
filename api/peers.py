# -*- coding: utf-8 -*-
"""
api/peers.py — Nodos remotos: ver el estado de OTRO llama_manager desde aquí.

Dos cosas, ambas pensadas para la red local (sin autenticación, LAN de confianza):

  • GET /api/host
      Estado LIGERO y estable de ESTE host, pensado para que otro llama_manager
      lo lea a distancia: hostname, SO, CPU, RAM total/usada, motor activo,
      GPUs (nombre + VRAM total/libre/usada) e instancias vivas. Es intencional-
      mente una versión recortada de /api/state: no trae la lista completa de
      modelos ni los presets, porque desde fuera solo interesa "¿qué GPUs hay,
      cuánta VRAM libre queda y qué está corriendo?".

  • GET/POST /api/peers  (+ DELETE /api/peers/<id>)
      La lista de nodos remotos que quieres seguir (ip:puerto). Se persiste en
      presets.json bajo "_peers", igual que el proxy o los perfiles, así que
      al abrir la pestaña se reconectan solos. El estado real de cada remoto lo
      trae el NAVEGADOR con fetch() a http://ip:puerto/api/host (por eso el
      server.py inyecta cabeceras CORS — ver comentario en after_request), y
      aquí solo guardamos la dirección.

stdlib puro: platform + subprocess para CPU/hostname, urllib NO se usa porque
el cliente es el navegador del equipo remoto, no este proceso.
"""
import platform
import re
import subprocess
from collections import OrderedDict

from flask import Blueprint, jsonify, request

from config import IS_WINDOWS
from instances import INSTANCES, get_engine, list_devices
from system_monitor import SYS, attach_live_usage

peers_bp = Blueprint("peers", __name__)

# ── Datos estáticos del host (con caché) ───────────────────────────────────
# hostname/SO/CPU no cambian en caliente: se calculan una vez y se reutilizan.
_HOST_CACHE = {}


def _cpu_name():
    """Nombre de CPU. Windows: WMI (Win32_Processor). Linux: /proc/cpuinfo.
    Fallback "desconocido" — es solo etiqueta, nunca rompe el endpoint."""
    try:
        if IS_WINDOWS:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"],
                capture_output=True, text=True, errors="replace", timeout=6,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            name = out.stdout.strip()
            if name:
                return name
        else:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = re.match(r"\s*model name\s*:\s*(.+)", line)
                    if m:
                        return m.group(1).strip()
    except Exception:
        pass
    return "CPU desconocida"


def host_fingerprint():
    """Identidad estable de ESTE equipo (hostname + SO + CPU). Se calcula una
    sola vez y se cachea a nivel de proceso — es solo informativo."""
    if _HOST_CACHE.get("fingerprint"):
        return dict(_HOST_CACHE["fingerprint"])
    sysname = f"{platform.system()} {platform.release()}"
    fp = {
        "hostname": platform.node() or platform.socket(),
        "os":       sysname,
        "cpu":      _cpu_name(),
        "arch":     platform.machine(),
    }
    _HOST_CACHE["fingerprint"] = fp
    return dict(fp)


# ── Estado ligero del host (lo que lee el OTRO equipo) ─────────────────────
@peers_bp.route("/api/host")
def api_host():
    """GPU + RAM + instancias de ESTE nodo, en un JSON pequeño y estable.

    La VRAM total viene de --list-devices (Vulkan/ROCm, fiable para el tamaño)
    y la usada se empareja por orden con system_monitor (WMI/rocm-smi), igual
    que en /api/state — el mismo criterio, sin duplicar la lógica.
    """
    devices, dev_err = list_devices()

    # Usada en vivo de system_monitor, emparejada por posición con los
    # dispositivos Vulkan. Si no coincide el nº (p.ej. iGPU extra en el
    # contador de rendimiento), caemos a free_mib del propio motor.
    attach_live_usage(devices)
    gpus = []
    for d in devices:
        used = d["used_mib"]
        gpus.append({
            "id":         d["id"],
            "name":       d["name"],
            "total_mib":  d["total_mib"],
            "used_mib":   used,
            "free_mib":   max(d["total_mib"] - used, 0),
        })

    # Instancias vivas gestionadas por ESTE manager (lo que está corriendo).
    instances = []
    for port, inst in INSTANCES.items():
        if inst["proc"].poll() is not None:
            continue
        instances.append({
            "port":   port,
            "model":  inst.get("model_name"),
            "alias":  inst.get("alias"),
            "device": inst.get("device"),
            "engine": inst.get("engine", "vulkan"),
            "rpc":    inst.get("rpc"),
        })

    return jsonify({
        "hostname":      host_fingerprint(),
        "engine":        get_engine(),
        "devices_error": dev_err,
        "ram_total_mib": int(SYS.get("ram_total", 0)),
        "ram_used_mib":  int(SYS.get("ram_used", 0)),
        "gpus":          gpus,
        "instances":     instances,
    })


# ── Lista de nodos remotos persistida (presets.json → "_peers") ─────────────
def _load_peers():
    """Lista de remotos [{id, base}] ordenada por inserción. base = 'ip:puerto'
    sin protocolo (el navegador monta el http:// delante)."""
    from instances import load_presets
    raw = load_presets().get("_peers") or []
    peers = []
    for item in raw:
        if isinstance(item, dict) and item.get("base"):
            base = re.sub(r"^https?://", "", str(item["base"]).strip()).rstrip("/")
            # tolera que alguien escriba el puerto de gestión con barra extra
            peers.append({"id": str(item.get("id") or base), "base": base})
    return peers


def _save_peers(peers):
    from instances import load_presets, save_presets
    presets = load_presets()
    presets["_peers"] = peers
    save_presets(presets)


def _normalize_base(base: str) -> str:
    """'192.168.0.57' / 'http://192.168.0.57:8080/' → '192.168.0.57:8080'.
    Si no da puerto, asume el 8080 de gestión (MANAGER_PORT)."""
    base = re.sub(r"^https?://", "", str(base or "").strip()).rstrip("/")
    if not base:
        return ""
    from config import MANAGER_PORT
    if ":" not in base.rsplit("/", 1)[-1]:
        base = f"{base}:{MANAGER_PORT}"
    return base


@peers_bp.route("/api/peers", methods=["GET", "POST"])
def api_peers():
    if request.method == "GET":
        return jsonify({"peers": _load_peers()})

    body = request.get_json(force=True, silent=True) or {}
    base = _normalize_base(body.get("base", ""))
    if not base:
        return jsonify({"error": "Dime la IP (y puerto si no es 8080) del otro nodo"}), 400

    peers = _load_peers()
    # Si ya existe la misma base, no duplicamos — refrescamos su posición.
    for p in peers:
        if p["base"] == base:
            return jsonify({"ok": True, "peers": peers, "added": False})
    peers.append({"id": base, "base": base})
    _save_peers(peers)
    return jsonify({"ok": True, "peers": peers, "added": True})


@peers_bp.route("/api/peers/<peer_id>", methods=["DELETE"])
def api_peer_delete(peer_id):
    peers = _load_peers()
    before = len(peers)
    peers = [p for p in peers if p["id"] != peer_id]
    if len(peers) == before:
        return jsonify({"error": "nodo no encontrado"}), 404
    _save_peers(peers)
    return jsonify({"ok": True, "peers": peers})
