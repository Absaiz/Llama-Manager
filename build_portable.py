# -*- coding: utf-8 -*-
"""
build_portable.py — Genera una carpeta LLAMA MANAGER autocontenida (y un .zip).

    build_portable.bat                         -> ..\\LlamaManager-Portable  (Windows, Vulkan)
    build_portable.bat --engines vulkan rocm   -> incluye también ROCm (~1,2 GB)
    build_portable.bat --with-presets          -> copia tus perfiles (este equipo)
    python build_portable.py --target linux    -> portable para nodos Linux

Contenido del portable:
    python\\            Python embebido (versions.PYTHON_VERSION) + Flask
    llama-vulkan\\      llama.cpp del build fijado en versions.py
    *.py, api\\, ui\\, fixed\\, skills\\   la aplicación
    LlamaManager.bat   doble clic para arrancar (abre el navegador en :8080)

La carpeta resultante se copia tal cual a otro equipo (USB, red...). No
instala nada, no toca el registro y no necesita Python en el destino. Los
modelos se buscan en %USERPROFILE%\\.lmstudio\\models (o en la variable
LLAMA_MANAGER_MODELS si la defines en LlamaManager.bat).
"""
import argparse
import io
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import update_llama  # noqa: E402
from versions import (APP_VERSION, LLAMA_CPP_BUILD, PY_PACKAGES,  # noqa: E402
                      PYTHON_VERSION, python_url)

HOST = "windows" if platform.system() == "Windows" else "linux"

APP_FILES = ["LlamaManager.bat", "server.py", "config.py", "versions.py", "instances.py", "runlog.py",
             "gguf_meta.py", "system_monitor.py", "update_llama.py", "build_portable.py",
             "update_llama.bat", "build_portable.bat", "updater.py", "bootstrap.ps1",
             "README.md"]
APP_DIRS = ["api", "ui", "fixed", "skills"]
SKIP = {"__pycache__"}
SKIP_SUFFIX = {".pyc", ".bak", ".log"}

# Dependencias puras de Flask: si no se puede usar pip (equipo sin internet o
# construyendo para otro SO) se copian del Python que ejecuta este script.
PURE_PKGS = ["flask", "werkzeug", "jinja2", "itsdangerous", "click", "blinker",
             "markupsafe", "colorama"]


def copy_tree(src: Path, dst: Path):
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        if any(part in SKIP for part in rel.parts) or p.suffix in SKIP_SUFFIX:
            continue
        q = dst / rel
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=True)
        else:
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)


def py_exe(root: Path, target: str) -> Path:
    return root / "python" / ("python.exe" if target == "windows" else "bin/python3")


def site_packages(root: Path, target: str) -> Path:
    if target == "windows":
        return root / "python" / "Lib" / "site-packages"
    maj, mnr = PYTHON_VERSION.split(".")[:2]
    return root / "python" / "lib" / f"python{maj}.{mnr}" / "site-packages"


def install_python(root: Path, target: str):
    url = python_url(target)
    print(f"\n[python {PYTHON_VERSION}] {target}")
    data = update_llama.download(url)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
        t.extractall(root)          # crea root/python/


def install_packages(root: Path, target: str):
    print("\n[paquetes] " + ", ".join(PY_PACKAGES))
    if target == HOST:
        r = subprocess.run([str(py_exe(root, target)), "-m", "pip", "install",
                            "--disable-pip-version-check", "--no-warn-script-location",
                            *PY_PACKAGES])
        if r.returncode == 0:
            return
        print("  pip falló (¿sin internet?) → copio Flask del Python local")
    else:
        print(f"  construyendo para {target} desde {HOST}: se copian las dependencias puras")
    copy_pure_packages(site_packages(root, target))


def copy_pure_packages(dest: Path):
    import importlib.metadata as md
    dest.mkdir(parents=True, exist_ok=True)
    for name in PURE_PKGS:
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            if name == "colorama":
                continue
            sys.exit(f"  falta '{name}' en este Python — instala flask aquí primero")
        for f in dist.files or []:
            src = Path(dist.locate_file(f))
            # extensiones C (.so/.pyd) fuera: markupsafe cae a su versión en Python puro
            if not src.is_file() or src.suffix in (".so", ".pyd", ".pyc") or "__pycache__" in f.parts:
                continue
            if str(f).startswith(".."):      # scripts de bin/, no hacen falta
                continue
            q = dest / f
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, q)
        print(f"  + {name} {dist.version}")


def trim_python(root: Path):
    """Quita lo que el manager no usa del Python embebido (~80 MB): símbolos de
    depuración, cabeceras C, Tk/IDLE."""
    py = root / "python"
    freed = 0
    for p in list(py.rglob("*.pdb")):
        freed += p.stat().st_size
        p.unlink()
    for rel in ["include", "tcl", "Lib/idlelib", "Lib/tkinter", "Lib/turtledemo", "Lib/test",
                "lib/tcl8", "lib/tk8.6", "lib/tcl8.6", "share"]:
        d = py / rel
        if d.is_dir():
            freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            shutil.rmtree(d)
    print(f"\n[recorte python] {freed / 2**20:.0f} MiB fuera")


def write_launchers(root: Path, target: str):
    if target == "windows":
        return          # se usa el LlamaManager.bat del repo (updater + arranque)
    else:
        sh = root / "llama_manager.sh"
        sh.write_text(
            "#!/usr/bin/env bash\n"
            "# LLAMA MANAGER portable\n"
            "cd \"$(dirname \"$0\")\"\n"
            "export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1\n"
            "# export LLAMA_MANAGER_PORT=8080 LLAMA_MANAGER_MODELS=$HOME/models\n"
            "./python/bin/python3 updater.py || true\n"
            "exec ./python/bin/python3 server.py \"$@\"\n", encoding="utf-8")
        sh.chmod(0o755)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE.parent / "LlamaManager-Portable"))
    ap.add_argument("--target", choices=["windows", "linux"], default=HOST)
    ap.add_argument("--engines", nargs="+", default=["vulkan"], choices=["vulkan", "rocm", "cpu"])
    ap.add_argument("--with-presets", action="store_true", help="copiar presets.json (perfiles) de este equipo")
    ap.add_argument("--no-zip", action="store_true")
    a = ap.parse_args()

    root = Path(a.out).resolve()
    if root == HERE:
        sys.exit("--out no puede ser la propia carpeta de llama_manager")
    if root.exists():
        print(f"borrando {root} anterior")
        shutil.rmtree(root)
    root.mkdir(parents=True)

    print(f"LLAMA MANAGER {APP_VERSION} · llama.cpp {LLAMA_CPP_BUILD} · portable {a.target} → {root}")
    for f in APP_FILES:
        if (HERE / f).exists():
            shutil.copy2(HERE / f, root / f)
    for d in APP_DIRS:
        if (HERE / d).exists():
            copy_tree(HERE / d, root / d)
    (root / "logs").mkdir()
    if a.with_presets and (HERE / "presets.json").exists():
        shutil.copy2(HERE / "presets.json", root / "presets.json")
    else:
        (root / "presets.json").write_text('{\n  "_engine": {"name": "vulkan"}\n}\n', encoding="utf-8")

    install_python(root, a.target)
    install_packages(root, a.target)
    trim_python(root)
    for e in a.engines:
        if not update_llama.install(e, LLAMA_CPP_BUILD, root, system=a.target):
            sys.exit(f"no se pudo descargar el motor {e}")
    write_launchers(root, a.target)

    if not a.no_zip:
        name = f"LlamaManager-{APP_VERSION}-llama{LLAMA_CPP_BUILD}-{a.target}-{'-'.join(a.engines)}"
        z = root.parent / f"{name}.zip"
        print(f"\n[zip] {z}")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    zi = zipfile.ZipInfo.from_file(p, Path(name) / p.relative_to(root))
                    zi.compress_type = zipfile.ZIP_DEFLATED
                    if a.target == "linux" and os.access(p, os.X_OK):
                        zi.external_attr = 0o755 << 16
                    with open(p, "rb") as fh:
                        zf.writestr(zi, fh.read())
        print(f"  {z.stat().st_size / 2**20:.0f} MiB")
    print("\nPortable listo. Arranca con", "LlamaManager.bat" if a.target == "windows" else "./llama_manager.sh")


if __name__ == "__main__":
    main()
