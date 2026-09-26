# -*- coding: utf-8 -*-
"""api/cloud.py — Probar modelos ONLINE vía API (OpenAI-compatible) antes de comprar hardware.

Endpoints:
  GET  /api/cloud/providers   → presets listos (base URL + nombre) para la UI
  POST /api/cloud/keys        → guarda/borra el API key por base URL (en memoria, sesión del proceso)
  GET  /api/cloud/models      → lista modelos de una API (<base>/models, Bearer si hay clave guardada)
  POST /api/cloud/chat        → chat SSE-streaming contra <base>/chat/completions

Todos los proveedores soportados usan el protocolo OpenAI (/v1/...), así que
la base URL siempre termina en "/v1" y a ella le añadimos "models" o
"chat/completions". Solo stdlib (urllib) — sin dependencias nuevas.
"""
import json
import threading
import urllib.parse
import urllib.request

from flask import Blueprint, Response, jsonify, request

from versions import APP_VERSION

cloud_bp = Blueprint("cloud", __name__)

# Abre sin pasar por el proxy del entorno (http_proxy/https_proxy): aquí se llama a un
# destino explícito que el usuario fijó de propósito — el proxy lo rompería en redes con
# proxy local/corporativo o loopback. ProxyHandler({}) ignora todas las variables/proxies.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# ── Presets: base URL ya termina en /v1 (se le añade "models" o            #
#    "chat/completions"). Todos hablan protocolo OpenAI → un solo código.   #
PROVIDERS = [
    {"id": "openrouter",  "name": "OpenRouter  (catálogo amplio)",     "url": "https://openrouter.ai/api/v1"},
    {"id": "groq",        "name": "Groq  (LPU, muy rápido)",           "url": "https://api.groq.com/openai/v1"},
    {"id": "mistral",     "name": "Mistral AI",                        "url": "https://api.mistral.ai/v1"},
    {"id": "cerebras",    "name": "Cerebras  (CS-2, muy rápido)",      "url": "https://api.cerebras.ai/v1"},
    {"id": "openai",      "name": "OpenAI",                            "url": "https://api.openai.com/v1"},
    {"id": "together",    "name": "Together AI",                       "url": "https://api.together.xyz/v1"},
    {"id": "deepseek",    "name": "DeepSeek",                          "url": "https://api.deepseek.com/v1"},
]

# API keys por base URL, solo en memoria (vive mientras corra el proceso; no
# se escribe a disco de forma permanente — es un panel local de pruebas).
KEYS   = {}      # url_normalizada → api key
_LOCK  = threading.Lock()


def _norm(url: str) -> str:
    """Base URL canónica: sin barra final, host normalizado (para KEYS)."""
    u = (url or "").strip().rstrip("/")
    p = urllib.parse.urlparse(u)
    return f"{p.scheme.lower()}://{p.netloc}{(p.path or '').lower()}"


def _headers(url: str, extra=None):
    h = {"User-Agent": "llama-manager/" + APP_VERSION, **(extra or {})}
    key = None
    with _LOCK:
        key = KEYS.get(_norm(url))
    if key:
        # OpenAI-compatibles (OpenRouter, Groq, Mistral…) usan Bearer; los que
        # esperan X-API-Key también lo aceptan vía Authorization en la práctica.
        h["Authorization"] = f"Bearer {key}"
    return h


def _http_json(base_url: str, path_suffix: str = "", timeout=60):
    """GET <base><suffix> JSON; las cabeceras se resuelven SIEMPRE contra la
    base URL (donde vive la clave en KEYS), nunca contra la URL con sufijo."""
    req = urllib.request.Request(base_url.rstrip("/") + path_suffix, headers=_headers(base_url))
    with _OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


# ── Presets para la UI ─────────────────────────────────────────────────────

@cloud_bp.route("/api/cloud/providers")
def api_providers():
    with _LOCK:
        have = {_norm(u) for u in KEYS}
    out = []
    for p in PROVIDERS:
        d = dict(p)
        d["has_key"] = _norm(p["url"]) in have   # la UI puede marcar "clave guardada"
        out.append(d)
    return jsonify({"providers": out})


# ── Gestión de claves (en memoria) ─────────────────────────────────────────

@cloud_bp.route("/api/cloud/keys", methods=["POST"])
def api_keys():
    j  = request.get_json(force=True) or {}
    url  = (j.get("url") or "").strip()
    key  = (j.get("key") or "").strip()
    scheme = urllib.parse.urlparse(url).scheme.lower() if url else ""
    if not url or scheme not in ("http", "https"):
        return jsonify({"error": "URL inválida"}), 400
    with _LOCK:
        n = _norm(url)
        if key:
            KEYS[n] = key       # guarda o reemplaza la clave de esa API
        else:
            KEYS.pop(n, None)   # clave vacía → borrar guardada
        has = n in KEYS
    return jsonify({"ok": True, "has_key": has})


# ── Listar modelos de una API ───────────────────────────────────────────────

@cloud_bp.route("/api/cloud/models")
def api_models():
    url = (request.args.get("url") or "").strip().rstrip("/")
    if not url:
        return jsonify({"error": "falta la URL base"}), 400
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        return jsonify({"error": "URL inválida (usa http/https)"}), 400
    try:
        data = _http_json(url, "/models")   # clave se resuelve contra <url>, no contra <url>/models
    except Exception as e:
        # Error de autenticación (401), red, etc. → lo devuelve tal cual; la UI muestra el detalle.
        return jsonify({"error": f"no pude listar modelos: {e}"})
    arr = data.get("data", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    models = []
    for m in arr:
        mid  = m.get("id") or "" if isinstance(m, dict) else str(m)
        name = m.get("name") if isinstance(m, dict) else None
        if mid:
            d = {"id": mid}
            # OpenRouter devuelve owner/name útiles para mostrar "org/modelo" legible
            for extra in ("owner", "context_length"):
                v = m.get(extra) if isinstance(m, dict) else None
                if v is not None:
                    d[extra] = v
            if name and name != mid:
                d["name"] = name
            models.append(d)
    # más largo primero → los modelos grandes (los que importan para comparar) arriba en el <select>
    models.sort(key=lambda x: str(x.get("id", "")), reverse=True)
    with _LOCK:
        has_key = _norm(url) in KEYS
    return jsonify({"url": url, "models": models[:400], "has_key": has_key})


# ── Chat streaming (SSE pasarela al proveedor) ─────────────────────────────

def _stream_remote(base_url, model, messages, inline_key=None):
    """Yields chunks SSE del /chat/completions del proveedor remoto.
    Misma forma de emisión que api/chat.py para reutilizar el parser de la UI."""
    payload = {"model": model, "messages": messages, "stream": True}
    headers = {**_headers(base_url), "Content-Type": "application/json",
               "User-Agent": "llama-manager/" + APP_VERSION}
    if inline_key:                      # clave del payload tiene preferencia sobre la guardada
        headers["Authorization"] = f"Bearer {inline_key}"
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(), headers=headers, method="POST",
    )
    try:
        with _OPENER.open(req, timeout=600) as r:      # 10 min: APIs grandes/ocupadas tardan; sin proxy del entorno
            while True:
                chunk = r.read(1024 * 8)     # buffer mayor que /api/chat: el tráfico es de red, no local
                if not chunk:
                    break
                yield chunk
    except Exception as e:   # HTTPError incluido → misma forma SSE de error que chat.py
        detail = ""
        try:
            code = e.getcode()               # type: ignore[union-attr]
            detail = f"HTTP {code}: " + e.read().decode("utf-8", "replace")[:500]  # type: ignore[union-attr]
        except Exception:
            pass
        yield ("data: " + json.dumps({"error": detail or str(e)}) + "\n\n").encode()


@cloud_bp.route("/api/cloud/chat", methods=["POST"])
def api_chat():
    j  = request.get_json(force=True) or {}
    url   = (j.get("url") or "").strip().rstrip("/")
    model = (j.get("model") or "").strip()
    msgs  = j.get("messages") or []
    scheme = urllib.parse.urlparse(url).scheme.lower() if url else ""
    if not url or scheme not in ("http", "https"):
        return jsonify({"error": "falta la URL base (o inválida)"}), 400
    inline_key = (j.get("key") or "").strip()
    # Clave del payload tiene preferencia sobre guardada en KEYS: así se puede
    # probar una API sin dejar la clave aparcada globalmente.
    if not model and msgs:
        return jsonify({"error": "falta el modelo"}), 400
    gen = _stream_remote(url, model, msgs, inline_key or None)
    return Response(gen, mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
