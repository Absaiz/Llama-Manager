# -*- coding: utf-8 -*-
"""api/downloads.py — Buscar y descargar modelos GGUF desde Hugging Face.

Endpoints:
  GET  /api/hf/search?q=...   → repos GGUF (ordenados por descargas)
  GET  /api/hf/files?repo=... → ficheros .gguf/mmproj del repo, con tamaño
  POST /api/hf/download       → arranca descarga en background (thread)
  GET  /api/hf/progress       → estado de todas las descargas
  POST /api/hf/cancel         → cancela una descarga activa

Las descargas aterrizan en la primera carpeta de EXTRA_MODELS_DIRS
(~/llm-downloads por defecto, ver config.py) — list_models() ya barre esa
carpeta, así que el modelo aparece en la UI sin moverlo. Solo stdlib
(urllib) + threading: sin dependencias nuevas.
"""
import json
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

from flask import Blueprint, jsonify, request

from versions import APP_VERSION

from config import EXTRA_MODELS_DIRS, MODELS_DIR

hf_bp = Blueprint("downloads", __name__)

HF_API = "https://huggingface.co/api"
UA = {"User-Agent": "llama-manager/" + APP_VERSION}

DOWNLOADS = {}          # key → estado de descarga
_LOCK = threading.Lock()


class _Cancelled(Exception):
    pass


def _dest_root() -> Path:
    """Carpeta donde aterrizan las descargas: la primera EXTRA_MODELS_DIRS
    (los modelos aparecen solos en la UI) o MODELS_DIR como último recurso."""
    root = EXTRA_MODELS_DIRS[0] if EXTRA_MODELS_DIRS else MODELS_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root


def _key(repo: str, fname: str) -> str:
    return f"{repo}::{fname}"


def _hf_get(url: str, timeout: int = 25):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


# ── Búsqueda ────────────────────────────────────────────────────────────────

@hf_bp.route("/api/hf/search")
def api_hf_search():
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify({"error": "escribe algo que buscar"})
    try:
        limit = min(max(int(request.args.get("limit", 25)), 1), 100)
    except ValueError:
        limit = 25
    url = (f"{HF_API}/models?search={urllib.parse.quote(q)}"
           f"&filter=gguf&sort=downloads&direction=-1&limit={limit}")
    try:
        data = _hf_get(url)
    except Exception as e:
        return jsonify({"error": f"error consultando Hugging Face: {e}"})
    results = [{
        "repo": m.get("id"),
        "downloads": m.get("downloads", 0),
        "likes": m.get("likes", 0),
        "updated": (m.get("lastModified") or "")[:10],
    } for m in data if m.get("id")]
    return jsonify({"results": results, "dest_root": str(_dest_root())})


# ── Ficheros de un repo ─────────────────────────────────────────────────────

@hf_bp.route("/api/hf/files")
def api_hf_files():
    repo = (request.args.get("repo") or "").strip()
    if not repo or "/" not in repo:
        return jsonify({"error": "repo inválido (usa org/nombre)"})
    try:
        data = _hf_get(
            f"{HF_API}/models/{urllib.parse.quote(repo, safe='/')}/tree/main?recursive=true"
        )
    except Exception as e:
        return jsonify({"error": f"error listando ficheros: {e}"})
    files = []
    for f in data:
        if f.get("type") != "file":
            continue
        name = f.get("path", "")
        if name.lower().endswith(".gguf") or "mmproj" in name.lower():
            files.append({"name": name, "size": f.get("size", 0) or 0})
    files.sort(key=lambda x: x["name"].lower())
    return jsonify({"repo": repo, "files": files})


# ── Descargas (background) ──────────────────────────────────────────────────

def _worker(key: str, repo: str, fname: str, dest: Path):
    """Descarga en streaming a <dest>.part y lo renombra al terminar.
    El estado (done/speed/status) se actualiza en DOWNLOADS[key]."""
    url = f"https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(fname)}"
    with _LOCK:
        d = DOWNLOADS[key]

    # Tamaño total: HEAD con Range (funciona aunque HF no devuelva Content-Length)
    total = None
    try:
        req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            cr = r.headers.get("Content-Range", "")
            if "/" in cr:
                total = int(cr.split("/")[-1])
    except Exception:
        pass

    tmp = dest.with_name(dest.name + ".part")
    done = 0
    t0 = time.time()
    last_report = 0.0
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as out:
            if total is None:
                cl = r.headers.get("Content-Length")
                if cl:
                    total = int(cl)
                    with _LOCK:
                        d["total"] = total
            while True:
                chunk = r.read(1024 * 256)
                if not chunk:
                    break
                with _LOCK:
                    if d.get("cancel"):
                        raise _Cancelled()
                out.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last_report > 0.5:
                    with _LOCK:
                        d["done"] = done
                        d["speed"] = int(done / max(now - t0, 0.001) / 1024)
                    last_report = now
        # Verificación: el .part debe pesar lo esperado antes de renombrar
        if tmp.stat().st_size != done:
            raise OSError(f"tamaño inesperado al terminar ({tmp.stat().st_size} de {done})")
        tmp.replace(dest)
        with _LOCK:
            d["status"] = "done"
            d["done"] = done
            d["finished"] = time.time()
    except _Cancelled:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        with _LOCK:
            d["status"] = "cancelled"
    except Exception as e:
        with _LOCK:
            d["status"] = "error"
            d["error"] = str(e)


@hf_bp.route("/api/hf/download", methods=["POST"])
def api_hf_download():
    j = request.get_json(silent=True) or {}
    repo = (j.get("repo") or "").strip()
    fname = (j.get("file") or "").strip().lstrip("/")
    if not repo or "/" not in repo or not fname:
        return jsonify({"error": "falta repo o file"})
    if "\\" in fname or fname.startswith(".."):
        return jsonify({"error": "nombre de fichero inválido"})

    key = _key(repo, fname)
    with _LOCK:
        cur = DOWNLOADS.get(key)
        if cur and cur["status"] == "downloading":
            return jsonify({"error": "ya hay una descarga activa de ese fichero"})
        if cur and cur["status"] == "done":
            return jsonify({"ok": True, "already": True, "dest": cur["dest"]})

    sub = _dest_root() / repo.replace("/", "__")
    sub.mkdir(parents=True, exist_ok=True)
    dest = sub / Path(fname).name
    if dest.exists():
        return jsonify({"ok": True, "already": True, "dest": str(dest)})

    with _LOCK:
        DOWNLOADS[key] = {
            "key": key, "status": "downloading",
            "repo": repo, "file": fname, "dest": str(dest),
            "total": None, "done": 0, "speed": 0,
            "cancel": False, "started": time.time(),
        }
    threading.Thread(target=_worker, args=(key, repo, fname, dest), daemon=True).start()
    return jsonify({"ok": True, "key": key, "dest": str(dest)})


@hf_bp.route("/api/hf/progress")
def api_hf_progress():
    with _LOCK:
        items = sorted(DOWNLOADS.values(), key=lambda d: d.get("started", 0), reverse=True)
        out = [{k: v for k, v in d.items() if k != "cancel"} for d in items[:20]]
    return jsonify({"downloads": out, "dest_root": str(_dest_root())})


@hf_bp.route("/api/hf/cancel", methods=["POST"])
def api_hf_cancel():
    j = request.get_json(silent=True) or {}
    key = (j.get("key") or "").strip()
    with _LOCK:
        d = DOWNLOADS.get(key)
        if d:
            d["cancel"] = True
    return jsonify({"ok": True})
