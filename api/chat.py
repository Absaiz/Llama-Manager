# -*- coding: utf-8 -*-
"""api/chat.py — /api/chat, /api/vision, /api/chatlog, /api/cctest"""
import json
import time
import urllib.error
import urllib.request

from flask import Blueprint, Response, jsonify, request

from config import LOGS_DIR
from instances import INSTANCES

chat_bp = Blueprint("chat", __name__)

# ── hipfire: exige tag exacto + thinking off. Cacheamos detección por puerto
# para no repetir peticiones de sonda en cada chat/stream. ────────────────────
_HF_PROBE_CACHE = {}   # port -> (is_hipfire, tag)

def _probe_hipfire(port):
    cached = _HF_PROBE_CACHE.get(port)
    if cached is not None:
        return cached
    result = (False, None)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=4) as r:
            h = json.loads(r.read())
        # hipfire /health trae "native"; llama-server no. Con eso + su PID
        # (campo "pid") leemos el tag exacto del proceso desde /proc/cmdline.
        if isinstance(h, dict) and "native" in h:
            tag = None
            pid = h.get("pid")
            if h.get("model"):
                tag = h["model"]
            elif pid:
                try:
                    cmdl = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
                    for i, a in enumerate(cmdl):
                        if a == b"--model" and i + 1 < len(cmdl):
                            tag = cmdl[i + 1].decode()
                            break
                except Exception:
                    pass
            if not tag:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=4) as r:
                        d = json.loads(r.read())
                    cands = [m["id"] for m in d.get("data", []) if "draft" not in m.get("id", "")]
                    tag = cands[0] if cands else None
                except Exception:
                    pass
            result = (True, tag)
    except Exception:
        result = (False, None)
    _HF_PROBE_CACHE[port] = result
    return result

def _hipfire_tag(port):
    _, tag = _probe_hipfire(port)
    return tag


def _stream_completions(port: int, payload: dict):
    """Genera chunks SSE desde /v1/chat/completions de llama-server."""
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=data, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            while True:
                chunk = r.read(1024)
                if not chunk:
                    break
                yield chunk
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", "replace")[:500]
        except Exception:
            detail = ""
        yield f"data: {json.dumps({'error': f'HTTP {e.code}: {detail}'})}\n\n".encode()
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n".encode()


@chat_bp.route("/api/chat", methods=["POST"])
def api_chat():
    body = request.get_json(force=True)
    port = int(body["port"])
    pay  = {"model": "local", "messages": body["messages"], "stream": True}
    if body.get("max_tokens"):
        pay["max_tokens"] = int(body["max_tokens"])
    # hipfire: exige tag exacto + thinking off (ver _probe_hipfire). llama.cpp
    # ignora ambos campos, así que solo se inyectan si la instancia es hipfire.
    is_hf, tag = _probe_hipfire(port)
    if is_hf:
        pay["chat_template_kwargs"] = {"enable_thinking": False}
        if tag:
            pay["model"] = tag
    return Response(
        _stream_completions(port, pay), mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@chat_bp.route("/api/vision", methods=["POST"])
def api_vision():
    body       = request.get_json(force=True)
    port       = int(body["port"])
    prompt     = body.get("prompt", "Describe la imagen.")
    image_b64  = body.get("image_b64", "")
    image_type = body.get("image_type", "image/jpeg")

    if not image_b64:
        return jsonify({"error": "image_b64 vacío"}), 400

    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{image_b64}"}},
            {"type": "text", "text": prompt},
        ],
    }]
    pay = {"model": "local", "messages": messages, "stream": True}
    if body.get("max_tokens"):
        pay["max_tokens"] = int(body["max_tokens"])
    is_hf, tag = _probe_hipfire(port)
    if is_hf:
        pay["chat_template_kwargs"] = {"enable_thinking": False}
        if tag:
            pay["model"] = tag

    return Response(
        _stream_completions(port, pay), mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@chat_bp.route("/api/chatlog", methods=["POST"])
def api_chatlog():
    b    = request.get_json(force=True)
    port = int(b.get("port", 0))
    kind = b.get("kind", "chat")
    inst = INSTANCES.get(port)
    model = (inst["alias"] or inst["model_name"]) if inst else "?"
    ts    = time.strftime("%Y-%m-%d %H:%M:%S")
    t     = b.get("timings") or {}
    tline = (
        f"prompt {t.get('prompt_per_second',0):.1f} t/s ({t.get('prompt_n','?')} tok) | "
        f"gen {t.get('predicted_per_second',0):.1f} t/s ({t.get('predicted_n','?')} tok) | "
        f"TTFT {b.get('ttft','?')} s | total {b.get('tt','?')} s"
    ) if t else "(sin timings)"

    if kind == "code_test":
        fname = LOGS_DIR / f"codetest_port{port}_{time.strftime('%Y%m%d_%H%M%S')}.md"
        fname.write_text(
            f"# Test de código — {ts}\n\nModelo: {model} (:{port})\nVelocidad: {tline}\n\n"
            f"## Respuesta del modelo\n\n{b.get('answer','')}\n",
            encoding="utf-8",
        )
        saved = fname.name
    else:
        fname = LOGS_DIR / f"chat_port{port}.log"
        with open(fname, "a", encoding="utf-8") as f:
            f.write(
                f"\n{'='*70}\n[{ts}] {kind.upper()} — {model} (:{port}) — {tline}\n"
                f"--- PROMPT ---\n{b.get('prompt','')[:2000]}\n"
                f"--- RESPUESTA ---\n{b.get('answer','')}\n"
            )
        saved = fname.name

    if kind in ("bench", "code_test") and t:
        with open(LOGS_DIR / "bench_results.csv", "a", encoding="utf-8") as f:
            if f.tell() == 0:
                f.write("fecha;tipo;puerto;modelo;prompt_tps;prompt_tok;gen_tps;gen_tok;ttft_s;total_s\n")
            f.write(
                f"{ts};{kind};{port};{model};"
                f"{t.get('prompt_per_second',0):.1f};{t.get('prompt_n','')};"
                f"{t.get('predicted_per_second',0):.1f};{t.get('predicted_n','')};"
                f"{b.get('ttft','')};{b.get('tt','')}\n"
            )
    return jsonify({"ok": True, "file": saved})


@chat_bp.route("/api/cctest", methods=["POST"])
def api_cctest():
    port = int(request.get_json(force=True)["port"])
    body = {
        "model": "local", "max_tokens": 256,
        "tools": [{"name": "get_weather",
                   "description": "Devuelve el tiempo de una ciudad",
                   "input_schema": {"type": "object",
                                    "properties": {"city": {"type": "string"}},
                                    "required": ["city"]}}],
        "tool_choice": {"type": "auto"},
        "messages": [{"role": "user", "content": "¿Qué tiempo hace en Burgos? Usa la herramienta."}],
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/messages",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:400]
        except Exception:
            pass
        if e.code == 404:
            return jsonify({"ok": False, "stage": "endpoint",
                            "msg": "404: build sin soporte /v1/messages."})
        return jsonify({"ok": False, "stage": "http", "msg": f"HTTP {e.code}: {detail}"})
    except Exception as e:
        return jsonify({"ok": False, "stage": "conn", "msg": str(e)})

    stop        = data.get("stop_reason")
    blocks      = data.get("content", []) or []
    tool_blocks = [c for c in blocks if isinstance(c, dict) and c.get("type") == "tool_use"]
    text_blocks = [c.get("text","") for c in blocks if isinstance(c,dict) and c.get("type")=="text"]
    usage       = data.get("usage", {}) or {}

    if tool_blocks or stop == "tool_use":
        tb = tool_blocks[0] if tool_blocks else {}
        return jsonify({"ok": True, "stage": "tool_use", "stop_reason": stop,
                        "tool_name": tb.get("name"), "tool_input": tb.get("input"),
                        "usage": usage, "msg": "Handshake OK. Lista para Claude Code."})
    return jsonify({"ok": False, "stage": "no_tool", "stop_reason": stop,
                    "text": " ".join(text_blocks)[:400], "usage": usage,
                    "msg": "Responde pero no invocó la herramienta. Revisa --jinja."})
