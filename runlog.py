# -*- coding: utf-8 -*-
"""
runlog.py — Logs de instancias con fecha/hora real y cabecera de hardware.

• LogPump: sustituye al "stdout=log_handle" directo. La salida de llama-server
  pasa por un hilo que antepone la fecha/hora de pared a CADA línea:
      [2026-09-25 18:20:27.123] 0.05.373.162 I srv load_model: ...
  (el "0.05.373.162" es el reloj relativo del propio llama.cpp; se conserva).
  Al terminar el proceso escribe el código de salida y la hora de fin.
• system_header(): bloque de cabecera con SO, CPU, RAM, GPU(s)+driver, motor
  (Vulkan/ROCm) y build de llama-server. Lo caro (CPU, drivers, versión del
  binario) se cachea para no retrasar cada arranque.
"""
import os
import platform
import socket
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from config import IS_WINDOWS


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


# ── Bomba de log ─────────────────────────────────────────────────────────────

class LogPump:
    """Fichero de log compartido entre el manager (eventos) y el hilo lector
    (salida del proceso). Thread-safe. close() desde fuera es seguro: si el
    hilo lector sigue vivo, él cierra el fichero al llegar a EOF."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._f = open(self.path, "w", encoding="utf-8", errors="replace")
        self._lock = threading.Lock()
        self._thread = None
        self._closed = False

    # API compatible con el antiguo file handle
    def write(self, text: str):
        with self._lock:
            if self._closed:
                return
            try:
                self._f.write(text)
                self._f.flush()
            except Exception:
                pass

    def flush(self):
        pass  # write() ya hace flush

    def event(self, msg: str):
        """Línea de evento del manager, con fecha."""
        self.write(f"[{now_str()}] ### {msg}\n")

    def attach(self, proc: subprocess.Popen, label: str = "proceso"):
        """Arranca el hilo que lee proc.stdout (PIPE, binario) y lo vuelca
        con marca de tiempo. IMPORTANTE: el hilo SIEMPRE sigue leyendo aunque
        falle la escritura, para que el pipe no se llene y bloquee al hijo."""
        def _run():
            try:
                for raw in iter(proc.stdout.readline, b""):
                    line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                    self.write(f"[{now_str()}] {line}\n" if line else "\n")
            except Exception:
                pass
            try:
                code = proc.wait(timeout=10)
            except Exception:
                code = proc.poll()
            self.event(f"{label} terminado (código {code})")
            self._really_close()

        self._thread = threading.Thread(target=_run, daemon=True, name=f"logpump-{self.path.name}")
        self._thread.start()

    def _really_close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                try:
                    self._f.close()
                except Exception:
                    pass

    def close(self):
        # Con hilo vivo, lo cierra el propio hilo tras escribir el código de salida.
        if self._thread is not None and self._thread.is_alive():
            return
        self._really_close()


# ── Información de sistema (cacheada) ────────────────────────────────────────

_CACHE = {}


def _run(args, timeout=10):
    kw = {}
    if IS_WINDOWS:
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                           errors="replace", **kw)
        return (r.stdout or "") + (r.stderr or "")
    except Exception:
        return ""


def _cpu_name() -> str:
    if IS_WINDOWS:
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except Exception:
            pass
    else:
        try:
            for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or "desconocido"


def _cores() -> str:
    logical = os.cpu_count() or 0
    physical = None
    try:
        import psutil  # opcional
        physical = psutil.cpu_count(logical=False)
    except Exception:
        pass
    return f"{physical} núcleos / {logical} hilos" if physical else f"{logical} hilos"


def _ram_gib() -> str:
    try:
        if IS_WINDOWS:
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return f"{m.ullTotalPhys / 2**30:.1f} GiB (libre {m.ullAvailPhys / 2**30:.1f} GiB)"
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            info[k] = int(v.split()[0])
        return f"{info['MemTotal'] / 2**20:.1f} GiB (libre {info.get('MemAvailable', 0) / 2**20:.1f} GiB)"
    except Exception:
        return "?"


def _os_name() -> str:
    if IS_WINDOWS:
        ver = platform.version()          # 10.0.26100
        build = ver.split(".")[-1] if ver else ""
        name = "Windows 11" if build.isdigit() and int(build) >= 22000 else f"Windows {platform.release()}"
        return f"{name} (build {ver}, {platform.machine()})"
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                pretty = line.split("=", 1)[1].strip().strip('"')
                return f"{pretty} (kernel {platform.release()}, {platform.machine()})"
    except Exception:
        pass
    return f"{platform.system()} {platform.release()} ({platform.machine()})"


def _gpu_drivers() -> list:
    """[(nombre, versión driver)] de las GPUs del sistema (no depende del motor)."""
    out = []
    if IS_WINDOWS:
        txt = _run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_VideoController | "
                    "ForEach-Object { $_.Name + '|' + $_.DriverVersion + '|' + $_.DriverDate }"], timeout=15)
        for line in txt.splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2 and parts[0]:
                date = parts[2][:8] if len(parts) > 2 else ""
                if len(date) == 8 and date.isdigit():
                    date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
                out.append((parts[0], parts[1] + (f", {date}" if date else "")))
    else:
        txt = _run(["sh", "-c", "lspci 2>/dev/null | grep -Ei 'vga|3d|display'"])
        for line in txt.splitlines():
            out.append((line.split(": ", 1)[-1].strip(), ""))
    return out


def _static_info() -> dict:
    if "static" not in _CACHE:
        _CACHE["static"] = {
            "host": socket.gethostname(),
            "os": _os_name(),
            "cpu": _cpu_name(),
            "cores": _cores(),
            "gpus": _gpu_drivers(),
            "python": platform.python_version(),
        }
    return _CACHE["static"]


def engine_build(exe: Path, env: dict = None) -> str:
    """'version: 1234 (abcdef)' de llama-server --version, cacheado por mtime."""
    exe = Path(exe)
    try:
        key = ("ver", str(exe), exe.stat().st_mtime)
    except OSError:
        return "?"
    if key not in _CACHE:
        kw = {"cwd": str(exe.parent), "env": env}
        if IS_WINDOWS:
            kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            r = subprocess.run([str(exe), "--version"], capture_output=True, text=True,
                               timeout=20, errors="replace", **kw)
            txt = (r.stdout or "") + (r.stderr or "")
        except Exception as e:
            txt = f"error: {e}"
        lines = [l.strip() for l in txt.splitlines()
                 if l.strip().startswith(("version", "built with"))]
        _CACHE[key] = " · ".join(lines) or (txt.strip().splitlines() or ["?"])[-1][:200]
    return _CACHE[key]


ENGINE_LABEL = {"vulkan": "Vulkan", "rocm": "ROCm / HIP", "rocm-lms": "ROCm / HIP (LM Studio)", "cuda": "CUDA", "cpu": "CPU"}


def system_header(*, title: str, engine: str, exe: Path, env: dict = None,
                  devices=None, used=None, split=None, rpc=None,
                  model: str = None, extra_lines=None) -> str:
    """Cabecera legible para el log.
    devices: lista de list_devices() [{id,name,total_mib,free_mib}]
    used:    lista de ids usados (None → todas, modo auto)."""
    s = _static_info()
    L = []
    bar = "=" * 78
    L.append(bar)
    L.append(f" {title}")
    L.append(bar)
    L.append(f" Fecha        : {now_str()}")
    L.append(f" Equipo       : {s['host']}")
    L.append(f" SO           : {s['os']}")
    L.append(f" CPU          : {s['cpu']} — {s['cores']}")
    L.append(f" RAM          : {_ram_gib()}")
    if s["gpus"]:
        for i, (n, d) in enumerate(s["gpus"]):
            L.append(f" {'GPU sistema' if i == 0 else '':<12} : {n}" + (f"  [driver {d}]" if d else ""))
    L.append(f" Motor        : {engine} → backend {ENGINE_LABEL.get(engine, engine)}")
    L.append(f" Binario      : {exe}")
    L.append(f" Build        : {engine_build(exe, env)}")
    if model:
        L.append(f" Modelo       : {model}")
    if devices:
        used_set = set(used) if used else None
        L.append(f" Dispositivos visibles para {ENGINE_LABEL.get(engine, engine)}"
                 + ("" if used_set else "  (modo auto: se usan todas)") + ":")
        for d in devices:
            mark = "►" if (used_set is None or d["id"] in used_set) else " "
            L.append(f"   {mark} {d['id']:<9} {d['name']}  "
                     f"({d['total_mib'] / 1024:.1f} GiB, libre {d['free_mib'] / 1024:.1f} GiB)")
        if used_set:
            missing = [u for u in used if u not in {d['id'] for d in devices}]
            if missing:
                L.append(f"   (no listados localmente: {', '.join(missing)})")
    else:
        L.append(" Dispositivos : (no se pudieron enumerar)")
    if split:
        L.append(f" Reparto -ts  : {split}")
    if rpc:
        L.append(f" RPC (red)    : {rpc}")
    for x in extra_lines or []:
        L.append(f" {x}")
    L.append(bar)
    return "\n".join(L) + "\n"


# ── Log del propio manager (operaciones + errores) ───────────────────────────

_MGR_LOCK = threading.Lock()


def manager_log(msg: str, exc: BaseException = None):
    """Añade una línea con fecha a logs/manager.log (y traza si hay excepción).
    Nunca lanza: el log no debe romper la operación que registra."""
    try:
        from config import LOGS_DIR
        text = f"[{now_str()}] {msg}\n"
        if exc is not None:
            import traceback
            text += "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        with _MGR_LOCK:
            with open(Path(LOGS_DIR) / "manager.log", "a", encoding="utf-8", errors="replace") as f:
                f.write(text)
    except Exception:
        pass
