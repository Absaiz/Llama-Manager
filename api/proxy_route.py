# -*- coding: utf-8 -*-
"""api/proxy_route.py — Proxy estable /v1/* → instancia elegida por "model"

Antes: todo /v1/* iba SIEMPRE a la instancia fijada en PROXY["port"]
(la que seleccionabas a mano en la UI), aunque hubiera varias instancias
vivas en otros puertos.

Ahora:
  - GET /v1/models devuelve el catálogo AGREGADO de todas las instancias
    vivas (antes solo reenviaba a PROXY["port"] y veías un único modelo).
  - Cualquier otra petición con un campo "model" en el body (chat/completions,
    completions, embeddings, /v1/messages de Anthropic...) se enruta a la
    instancia cuyo alias o nombre de modelo coincida, vía
    instances.find_instance_port().
  - Si no viene "model", o no coincide con ninguna instancia activa, se cae
    al comportamiento de siempre: reenviar a PROXY["port"]. Esto mantiene
    compatibilidad con clientes que mandan un "model" de relleno (p.ej.
    Claude Code con model="local").
"""
import json
import urllib.error
import urllib.request

from flask import Blueprint, Response, jsonify, request

from instances import INSTANCES, PROXY, cleanup_dead, find_instance_port
try:
    from api.chat import _probe_hipfire   # detección hipfire por puerto (cacheada)
except ImportError:
    from chat import _probe_hipfire

proxy_bp = Blueprint("proxy", __name__)

_SKIP_HEADERS = {"host", "content-length", "connection", "accept-encoding"}


def _aggregated_models():
    """Catálogo OpenAI-style de TODOS los modelos cargados ahora mismo,
    uno por instancia viva (no solo el de PROXY["port"])."""
    cleanup_dead()
    data = []
    for port, inst in INSTANCES.items():
        model_id = inst.get("alias") or inst.get("model_name")
        data.append({
            "id": model_id,
            "object": "model",
            "owned_by": "llama-manager",
            "port": port,
        })
    return {"object": "list", "data": data}


@proxy_bp.route("/v1/<path:sub>", methods=["GET", "POST"])
def proxy_v1(sub):
    # Catálogo agregado — no se reenvía a ninguna instancia en concreto.
    if request.method == "GET" and sub.rstrip("/") == "models":
        return jsonify(_aggregated_models())

    data = request.get_data() if request.method == "POST" else None

    target = None
    if data:
        try:
            body = json.loads(data)
            model_name = body.get("model") if isinstance(body, dict) else None
        except Exception:
            model_name = None
        if model_name:
            target = find_instance_port(model_name)

    if target is None:
        target = PROXY["port"]

    if not target:
        return jsonify({
            "error": "Proxy sin destino: selecciona instancia en el gestor, "
                     "o manda \"model\" con el alias de una instancia activa",
        }), 503

    url     = f"http://127.0.0.1:{target}/v1/{sub}"
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _SKIP_HEADERS}
    headers.setdefault("Content-Type", "application/json")

    # hipfire (motor Qwen): sin estos dos ajustes, peticiones cortas fallan
    # ("open think span") y un "model" ajeno al tag (p.ej. "local", o el alias
    # del manager) corta la stream tras 1 chunk. llama.cpp ignora ambos campos.
    is_hf, hf_tag = _probe_hipfire(target)
    if is_hf and data and request.method == "POST":
        try:
            body = json.loads(data)
            if isinstance(body, dict):
                kws = body.get("chat_template_kwargs") or {}
                body["chat_template_kwargs"] = {**(kws if isinstance(kws, dict) else {}),
                                                "enable_thinking": False}
                if hf_tag:
                    body["model"] = hf_tag
                data = json.dumps(body).encode()
                headers["Content-Length"] = str(len(data))
        except Exception:
            pass

    req     = urllib.request.Request(url, data=data, headers=headers, method=request.method)

    try:
        r = urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as e:
        return Response(e.read(), status=e.code,
                        content_type=e.headers.get("Content-Type", "application/json"))
    except Exception as e:
        return jsonify({"error": f"Instancia :{target} no responde: {e}"}), 502

    def stream():
        try:
            while True:
                chunk = r.read(1024)
                if not chunk:
                    break
                yield chunk
        finally:
            r.close()

    return Response(
        stream(), status=r.status,
        content_type=r.headers.get("Content-Type", "application/json"),
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
