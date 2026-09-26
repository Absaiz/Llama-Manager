# -*- coding: utf-8 -*-
"""
updater.py — Auto-actualización de LLAMA MANAGER desde GitHub.

Lo ejecuta LlamaManager.bat ANTES de arrancar el servidor:
  1. Pregunta a GitHub por el último commit de UPDATE_REPO/UPDATE_BRANCH (versions.py).
  2. Si es distinto del instalado (.installed_commit), descarga ese commit en
     zip y copia sus ficheros encima de esta carpeta. NUNCA toca: presets.json,
     logs\\, python\\, llama-*\\ (motores), github_token.txt.
     No borra ficheros locales que no estén en el repo.
  3. Si el nuevo versions.py pide otro build de llama.cpp (LLAMA_CPP_BUILD),
     actualiza los motores instalados (llama-vulkan, llama-rocm) con update_llama.
  4. Sin internet, repo inaccesible o cualquier error → avisa y sigue: la app
     arranca con lo que ya hay. Nunca bloquea el arranque.

Opciones:
  python updater.py            comprobar y aplicar
  python updater.py --check    solo comprobar (código de salida 10 = hay actualización)
  python updater.py --force    re-descargar aunque el commit coincida
  LLAMA_MANAGER_NO_UPDATE=1    desactiva la comprobación
Si la carpeta es un clon git (.git), no toca nada: usa git pull.
"""
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARKER = HERE / ".installed_commit"
TOKEN_FILE = HERE / "github_token.txt"
PROTECTED_TOP = {"presets.json", "logs", "python", ".installed_commit", "github_token.txt", ".git"}
PROTECTED_PREFIX = ("llama-",)          # llama-vulkan, llama-rocm, llama-cpu (+ .old)
TIMEOUT = 8


def log(msg):
    print(f"[updater] {msg}", flush=True)
    try:
        (HERE / "logs").mkdir(exist_ok=True)
        with open(HERE / "logs" / "updater.log", "a", encoding="utf-8") as f:
            import time
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    except OSError:
        pass


def _headers(accept="application/vnd.github+json"):
    h = {"User-Agent": "llama-manager-updater", "Accept": accept}
    tok = os.environ.get("GITHUB_TOKEN") or (TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else "")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _get(url, accept="application/vnd.github+json", timeout=TIMEOUT) -> bytes:
    req = urllib.request.Request(url, headers=_headers(accept))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def read_settings():
    ns = {}
    exec(compile((HERE / "versions.py").read_text(encoding="utf-8"), "versions.py", "exec"), ns)
    return ns


def remote_commit(repo, branch):
    data = json.loads(_get(f"https://api.github.com/repos/{repo}/commits/{branch}"))
    return data["sha"], (data.get("commit", {}).get("message") or "").strip().splitlines()[0:1]


def local_commit():
    try:
        return MARKER.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def is_protected(rel: Path) -> bool:
    top = rel.parts[0]
    return top in PROTECTED_TOP or top.startswith(PROTECTED_PREFIX)


def apply_zip(data: bytes) -> int:
    n = 0
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(tmp)
        roots = [p for p in Path(tmp).iterdir() if p.is_dir()]
        if len(roots) != 1 or not (roots[0] / "server.py").exists():
            raise RuntimeError("el zip descargado no parece LLAMA MANAGER (falta server.py)")
        src = roots[0]
        for p in src.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(src)
            if is_protected(rel):
                continue
            dst = HERE / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() and dst.read_bytes() == p.read_bytes():
                continue
            # escribir a .tmp y renombrar: si se corta a medias no queda un .py truncado
            tmpf = dst.with_name(dst.name + ".upd")
            shutil.copy2(p, tmpf)
            os.replace(tmpf, dst)
            n += 1
    return n


def installed_build(engine_dir: Path):
    exe = engine_dir / ("llama-server.exe" if os.name == "nt" else "llama-server")
    if not exe.exists():
        return None
    try:
        kw = {"creationflags": 0x08000000} if os.name == "nt" else {}
        r = subprocess.run([str(exe), "--version"], capture_output=True, text=True,
                           errors="replace", timeout=30, cwd=str(exe.parent), **kw)
        import re
        m = re.search(r"build\s+(\d+)", (r.stdout or "") + (r.stderr or ""))
        return int(m.group(1)) if m else None
    except Exception:
        return None


def sync_engines(settings):
    want = settings.get("LLAMA_CPP_BUILD", "")
    try:
        want_n = int(want.lstrip("b"))
    except ValueError:
        return
    todo = []
    for eng, folder in (("vulkan", "llama-vulkan"), ("rocm", "llama-rocm")):
        d = HERE / folder
        if eng == "vulkan" and not d.exists():
            todo.append(eng)                       # sin motor: se instala Vulkan
            continue
        if d.exists():
            have = installed_build(d)
            if have != want_n:
                log(f"motor {eng}: instalado b{have} · pedido {want} → actualizando")
                todo.append(eng)
    if not todo:
        return
    sys.path.insert(0, str(HERE))
    import importlib
    import update_llama
    importlib.reload(update_llama)
    for eng in todo:
        try:
            update_llama.install(eng, want, HERE)
        except Exception as e:
            log(f"no se pudo actualizar el motor {eng}: {e} (sigue con el que hay)")


def main():
    args = set(sys.argv[1:])
    if os.environ.get("LLAMA_MANAGER_NO_UPDATE"):
        log("comprobación desactivada (LLAMA_MANAGER_NO_UPDATE)")
        return 0
    if (HERE / ".git").exists():
        log("carpeta clonada con git: no auto-actualizo (usa git pull)")
        return 0
    s = read_settings()
    repo, branch = s.get("UPDATE_REPO"), s.get("UPDATE_BRANCH", "main")
    if not repo:
        return 0
    try:
        sha, msg = remote_commit(repo, branch)
    except urllib.error.HTTPError as e:
        hint = (" — repo privado o inexistente: hazlo público o pon un token de solo lectura "
                "en github_token.txt") if e.code in (401, 403, 404) else ""
        log(f"no puedo consultar {repo}: HTTP {e.code}{hint}")
        return 0
    except Exception as e:
        log(f"sin conexión con GitHub ({e.__class__.__name__}): arranco con la versión instalada")
        return 0

    cur = local_commit()
    if sha == cur and "--force" not in args:
        log(f"al día ({s.get('APP_VERSION')} · {sha[:7]})")
        sync_engines(s)
        return 0
    log(f"actualización disponible: {cur[:7] or 'ninguna'} → {sha[:7]}" + (f" «{msg[0]}»" if msg else ""))
    if "--check" in args:
        return 10
    try:
        data = _get(f"https://api.github.com/repos/{repo}/zipball/{sha}", timeout=120)
        n = apply_zip(data)
        MARKER.write_text(sha, encoding="utf-8")
        s = read_settings()
        log(f"actualizado a {s.get('APP_VERSION')} ({sha[:7]}): {n} fichero(s) cambiados")
    except Exception as e:
        log(f"ERROR aplicando la actualización: {e} — arranco con la versión instalada")
        return 0
    sync_engines(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
