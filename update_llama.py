# -*- coding: utf-8 -*-
"""
update_llama.py — Descarga los motores de llama.cpp del build fijado en versions.py.

Uso (desde la carpeta de llama_manager, con el manager PARADO o sin instancias
cargadas: Windows bloquea las DLL en uso):

    update_llama.bat                      -> vulkan (y rocm si ya existe llama-rocm\\)
    update_llama.bat vulkan rocm          -> motores concretos
    python update_llama.py --build b11250 -> otro build sin tocar versions.py

Para cambiar de versión de forma permanente: edita LLAMA_CPP_BUILD en
versions.py y ejecuta esto en TODOS los equipos (RPC exige el mismo build).
Solo usa la librería estándar de Python.
"""
import argparse
import io
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from versions import LLAMA_CPP_BUILD, asset_url  # noqa: E402

SYSTEM = "windows" if platform.system() == "Windows" else "linux"
EXE = "llama-server.exe" if SYSTEM == "windows" else "llama-server"
FOLDER = {"vulkan": "llama-vulkan", "rocm": "llama-rocm", "cpu": "llama-cpu"}


def download(url: str) -> bytes:
    print(f"  descargando {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "llama-manager-updater"})
    with urllib.request.urlopen(req, timeout=120) as r:
        total = int(r.headers.get("Content-Length") or 0)
        buf, done = io.BytesIO(), 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            buf.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 2**20:7.1f} / {total / 2**20:.1f} MiB", end="", flush=True)
        print()
        return buf.getvalue()


def extract(data: bytes, url: str, dest: Path, exe: str = EXE):
    if url.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(dest)
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
            t.extractall(dest)
    # Algunos paquetes traen una subcarpeta (llama-bXXXX/); aplanar
    found = next(dest.rglob(exe), None)
    if found is None:
        raise RuntimeError(f"el paquete no contiene {exe}")
    return found.parent


def install(engine: str, build: str, base: Path, system: str = SYSTEM) -> bool:
    exe = "llama-server.exe" if system == "windows" else "llama-server"
    url = asset_url(system, engine, build)
    target = base / FOLDER[engine]
    print(f"\n[{engine}] {build} → {target}")
    try:
        data = download(url)
    except Exception as e:
        print(f"  ERROR descargando: {e}")
        print("  (si es 404, revisa el nombre del asset en GitHub y ROCM_RUNTIME en versions.py)")
        return False
    with tempfile.TemporaryDirectory(dir=base) as tmp:
        src = extract(data, url, Path(tmp), exe)
        old = target.with_name(target.name + ".old")
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)
        if target.exists():
            try:
                target.rename(old)
            except OSError as e:
                print(f"  ERROR: no puedo sustituir {target} ({e}). "
                      f"¿Hay un llama-server o ggml-rpc-server corriendo? Páralo y repite.")
                return False
        shutil.move(str(src), str(target))
        if system != "windows":
            for f in target.iterdir():
                if f.is_file() and not f.suffix:
                    f.chmod(0o755)
        shutil.rmtree(old, ignore_errors=True)   # sin basura: solo si todo fue bien
    print(f"  OK: {target / exe}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("engines", nargs="*", help="vulkan, rocm, cpu (defecto: vulkan + rocm si ya existe)")
    ap.add_argument("--build", default=LLAMA_CPP_BUILD, help=f"build de llama.cpp (defecto {LLAMA_CPP_BUILD})")
    ap.add_argument("--dir", default=str(HERE), help="carpeta de llama_manager")
    a = ap.parse_args()
    base = Path(a.dir)
    engines = a.engines or (["vulkan"] + (["rocm"] if (base / "llama-rocm").exists() else []))
    bad = [e for e in engines if e not in FOLDER]
    if bad:
        sys.exit(f"motor desconocido: {bad} (válidos: {list(FOLDER)})")
    ok = all([install(e, a.build, base) for e in engines])
    print("\nListo." if ok else "\nTerminado con errores.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
