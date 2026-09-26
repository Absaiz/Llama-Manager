# -*- coding: utf-8 -*-
"""
instances.py — Estado global de instancias llama-server y utilidades de control.
"""
import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from config import (DEFAULT_ENGINE, ENGINE_ENV, ENGINES, EXE_NAME, IS_WINDOWS,
                    EXTRA_MODELS_DIRS, LOGS_DIR, PRESETS_FILE, RPC_PORT)
from gguf_meta import get_cached_meta
from runlog import LogPump, engine_build, system_header
from versions import LLAMA_CPP_BUILD, build_number

INSTANCES       = {}   # port → {proc, model_name, model_path, alias, device, ctx, ngl, engine, started, log_file, log_handle}
ANTHROPIC_CACHE = {}   # port → bool
PROXY           = {"port": None}
ENGINE_STATE    = {"name": DEFAULT_ENGINE}  # motor activo (vulkan/rocm) para --list-devices y nuevos arranques
RPC_SERVERS     = {}   # port → {proc, engine, log_file, log_handle, started} — esclavos ggml-rpc-server activos


# ── Rutas portables entre SO ─────────────────────────────────────────────────
# Los presets/UI pueden contener rutas de otro host (p.ej. un preset Windows
# "C:\\ToolsIA\\llama_manager\\fixed\\chat_template.jinja" corriendo en Linux,
# o rutas "C:\\Users\\Bravo\\.lmstudio\\models\\..." para mmproj/modelos).
# _portable() resuelve la ruta tal cual si existe; si no, la remapa al
# equivalente local usando dos anclajes:
#   • la carpeta del propio proyecto (todo lo que va DESPUÉS de "llama_manager")
#   • los directorios de modelos conocidos (busca por nombre de archivo)
# Así los presets viajan entre Windows y Linux sin editarse.

def _portable(raw: str) -> Path:
    # En Linux las barras inversas no son separador: "C:\a\b" sería un solo
    # componente. Se normalizan a "/" para poder parsear partes en ambos SO.
    norm = str(raw).strip().replace("\\", "/")
    p = Path(norm)
    if not norm:
        return p
    # Relativa → primero respecto a la carpeta del manager (p.ej.
    # "fixed/chat_template.jinja"). Siempre se devuelve ABSOLUTA: llama-server
    # se lanza con cwd = carpeta del motor y una ruta relativa fallaría ahí.
    if not p.is_absolute():
        cand = Path(__file__).resolve().parent / p
        if cand.exists():
            return cand
    if p.exists():
        return p.resolve()
    # Anclaje 1: ruta RELATIVA al árbol llama_manager (cualquier host) —
    # "C:\ToolsIA\llama_manager\fixed\x.jinja" -> <este árbol>/fixed/x.jinja.
    # Se prueban sufijos progresivos: el directorio padre (ToolsIA vs home)
    # no se puede presuponer.
    parts = [pt for pt in p.parts if pt not in (p.anchor, "/", "\\")]
    if parts:
        root = Path(__file__).resolve().parent
        seen = set()
        for n in range(1, len(parts) + 1):
            cand = root.joinpath(*parts[-n:])
            key = str(cand).lower()
            if key in seen:
                continue
            seen.add(key)
            if cand.exists():
                return cand
    # Anclaje 2: archivo de modelo/mmproj — buscar por nombre en las carpetas
    # de modelos de este host (incl. la carpeta del build llama en ~/llama)
    bases = [Path.home() / ".lmstudio" / "models"] + list(EXTRA_MODELS_DIRS) + [Path.home() / "llama"]
    for base in bases:
        c = base / p.name
        if c.exists():
            return c
        try:
            for hit in base.rglob(p.name):  # carpeta del modelo, un nivel más abajo
                return hit
        except OSError:
            pass
    return p  # sin resolver: se reporta como "no existe" con la ruta original


# ── Motor (llama-vulkan / llama-rocm) ────────────────────────────────────────

def get_engine():
    return ENGINE_STATE["name"]


def engine_exe(name: str = None) -> Path:
    name = name if name in ENGINES else ENGINE_STATE["name"]
    engine_dir = ENGINES.get(name) or ENGINES[DEFAULT_ENGINE]
    return engine_dir / EXE_NAME


def engine_env(name: str = None) -> dict:
    """Entorno del proceso actual + overrides de ENGINE_ENV[name] (ver config.py).
    La clave "PATH", si está presente, se antepone al PATH heredado en vez de
    sustituirlo — así el motor sigue viendo el resto del sistema."""
    name = name if name in ENGINES else ENGINE_STATE["name"]
    env = os.environ.copy()
    overrides = dict(ENGINE_ENV.get(name) or {})
    extra_path = overrides.pop("PATH", None)
    if extra_path:
        env["PATH"] = extra_path + os.pathsep + env.get("PATH", "")
    env.update(overrides)
    return env


# ── Capacidades del binario (flags que acepta) ──────────────────────────────
# llama.cpp renombra/quita flags a menudo (b11xxx: --mmap/--no-mmap/--mlock/-dio
# eliminados → --load-mode; --webui → --ui; enable_thinking por kwargs → --reasoning).
# En vez de fijar "si build >= N", se lee `llama-server --help` una vez por
# binario (cacheado por mtime) y se construye el comando con lo que ESE binario
# acepta. Así conviven el build oficial nuevo y el backend viejo de LM Studio.

_CAPS_CACHE = {}
_FLAG_RE = re.compile(r"(?<![\w-])(--?[a-zA-Z][\w-]*)")


def engine_caps(name: str = None) -> set:
    name = name if name in ENGINES else ENGINE_STATE["name"]
    exe = engine_exe(name)
    try:
        key = (str(exe), exe.stat().st_mtime)
    except OSError:
        return set()
    if key not in _CAPS_CACHE:
        try:
            r = subprocess.run([str(exe), "--help"], capture_output=True, text=True,
                               errors="replace", timeout=30, cwd=str(exe.parent),
                               env=engine_env(name), **_popen_kwargs())
            txt = (r.stdout or "") + (r.stderr or "")
        except Exception:
            txt = ""
        _CAPS_CACHE[key] = set(_FLAG_RE.findall(txt))
    return _CAPS_CACHE[key]


def engine_info(name: str = None) -> dict:
    """Build instalado del motor vs el esperado en versions.py (para la UI)."""
    name = name if name in ENGINES else ENGINE_STATE["name"]
    exe = engine_exe(name)
    if not exe.exists():
        return {"engine": name, "exe": str(exe), "installed": None,
                "expected": LLAMA_CPP_BUILD, "match": False, "error": "no instalado"}
    ver = engine_build(exe, engine_env(name))
    inst_n, exp_n = build_number(ver), build_number(LLAMA_CPP_BUILD)
    return {"engine": name, "exe": str(exe), "installed": ver,
            "installed_build": inst_n, "expected": LLAMA_CPP_BUILD,
            "match": bool(inst_n and inst_n == exp_n)}


def set_engine(name: str):
    if name not in ENGINES:
        return False, f"Motor desconocido: {name} (válidos: {', '.join(ENGINES)})"
    ENGINE_STATE["name"] = name
    presets = load_presets()
    presets["_engine"] = {"name": name}
    save_presets(presets)
    return True, None


def load_persisted_engine():
    """Restaura el motor guardado la última vez, sin volver a escribir el fichero."""
    presets = load_presets()
    name = (presets.get("_engine") or {}).get("name")
    if name in ENGINES:
        ENGINE_STATE["name"] = name


# ── Utilidades de proceso ────────────────────────────────────────────────────
# En Windows usamos taskkill/tasklist (herramientas nativas de cmd) y
# CREATE_NO_WINDOW (constante que SOLO existe en subprocess en Windows — si
# se pasa en Linux, revienta con AttributeError antes de llegar a ejecutar
# nada). En Linux no hay tasklist/taskkill: usamos pgrep para listar y
# os.killpg + SIGKILL para matar el grupo completo de proceso (equivalente al
# "/T" — árbol completo — de taskkill). Para que killpg funcione, el proceso
# tiene que haberse lanzado con start_new_session=True (ver start_instance).

def _popen_kwargs():
    """kwargs de subprocess.Popen/run específicos de plataforma para lanzar
    el motor de forma silenciosa (sin ventana de consola en Windows) y en su
    propio grupo de proceso (para poder matar el árbol entero luego)."""
    if IS_WINDOWS:
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def force_kill(pid):
    if IS_WINDOWS:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True, **_popen_kwargs(),
        )
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass


def list_llama_pids():
    if IS_WINDOWS:
        try:
            out = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq llama-server.exe", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, errors="replace", timeout=10, **_popen_kwargs(),
            )
            pids = []
            for line in out.stdout.splitlines():
                parts = [p.strip('"') for p in line.split('","')]
                if len(parts) >= 2 and parts[1].isdigit():
                    pids.append(int(parts[1]))
            return pids
        except Exception:
            return []
    # Linux/macOS: pgrep -f encuentra el PID buscando en la línea de comandos
    # completa, así que da igual si el binario se llama exactamente
    # "llama-server" o va con ruta/sufijo (p.ej. llama-server-vulkan).
    try:
        out = subprocess.run(
            ["pgrep", "-f", "llama-server"],
            capture_output=True, text=True, errors="replace", timeout=10,
        )
        return [int(p) for p in out.stdout.split() if p.isdigit()]
    except Exception:
        return []


def list_devices(rpc=None, engine=None):
    engine = engine if engine in ENGINES else get_engine()
    exe = engine_exe(engine)
    if not exe.exists():
        return [], f"No se encuentra {exe} (motor: {engine})"
    args = [str(exe), "--list-devices"]
    # Si se da una lista de esclavos RPC, los incluye en la enumeración
    # (aparecen como RPC0, RPC1... con su memoria).
    servers = ",".join(s.strip() for s in (rpc or "").split(",") if s.strip())
    if servers:
        args += ["--rpc", servers]
    try:
        out = subprocess.run(
            args,
            capture_output=True, text=True, errors="replace", timeout=30,
            cwd=str(exe.parent),  # las DLL/.so del motor están junto al exe
            env=engine_env(engine), **_popen_kwargs(),
        )
        text = out.stdout + out.stderr
    except Exception as e:
        return [], f"Error ejecutando llama-server: {e}"
    devices = []
    # Formato esperado: "Vulkan0: <nombre> (24564 MiB, 20000 MiB free)"
    for m in re.finditer(
        r"^\s*(\w+\d+):\s*(.+?)\s*\((\d+)\s*Mi?B,\s*(\d+)\s*Mi?B free\)", text, re.M
    ):
        devices.append({
            "id": m.group(1), "name": m.group(2),
            "total_mib": int(m.group(3)), "free_mib": int(m.group(4)),
        })
    if not devices:
        # No se reconoció ningún dispositivo — casi siempre es que el motor
        # (p.ej. rocm sin runtime HIP instalado/soportado para la GPU) no
        # encontró nada, o el formato de salida no coincide con el patrón de
        # arriba. Devolvemos la salida cruda (recortada) como "error" para
        # que se vea en la UI qué dijo realmente el binario, en vez de un
        # "Sin GPUs" mudo.
        raw = text.strip()
        if raw:
            return [], f"[{engine}] --list-devices no devolvió dispositivos reconocibles. Salida: {raw[:600]}"
        return [], f"[{engine}] --list-devices no devolvió salida (código {out.returncode})"
    return devices, None


DRAFT_PREFIXES = ("mtp-", "dspark-", "dflash-", "eagle3-")


def _is_companion(fname: str) -> bool:
    low = fname.lower()
    return "mmproj" in low or low.startswith(DRAFT_PREFIXES)


def find_draft_sidecars(model_path) -> list:
    """GGUF de draft (MTP/DSpark/DFlash/EAGLE3) en la carpeta del modelo.
    llama.cpp los reconoce por prefijo (common/preset.cpp): mtp-*.gguf, etc.
    Qwen3.8-Flash-Next (arch qwen4exp) publica el MTP como sidecar aparte."""
    try:
        folder = _portable(str(model_path)).parent
        return sorted(str(p) for p in folder.glob("*.gguf")
                      if p.name.lower().startswith(DRAFT_PREFIXES))
    except Exception:
        return []


def _scan_models_root(root: Path, models: list):
    if not root.exists():
        return
    for p in root.rglob("*.gguf"):
        # mmproj y "sidecars" de speculative (mtp-*, dspark-*, dflash-*) no son
        # modelos cargables por sí solos — antes aparecían en la lista y se
        # podían elegir como modelo por error.
        if _is_companion(p.name):
            continue
        mp = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", p.name)
        if mp and mp.group(1) != "00001":
            continue
        try:
            size_gb = round(p.stat().st_size / (1024 ** 3), 2)
        except OSError:
            size_gb = 0
        models.append({"path": str(p), "name": str(p.relative_to(root)), "size_gb": size_gb})


def list_models(models_dir: Path):
    # MODELS_DIR + carpetas EXTRA_MODELS_DIRS (descargas directas, ver config.py).
    # Se deduplica por ruta real para que enlaces solapados no dupliquen entradas.
    roots = [models_dir] + [r for r in EXTRA_MODELS_DIRS if r != models_dir]
    models = []
    for root in roots:
        _scan_models_root(root, models)
    seen, uniq = set(), []
    for m in models:
        key = os.path.realpath(m["path"])
        if key not in seen:
            seen.add(key)
            uniq.append(m)
    uniq.sort(key=lambda m: m["name"].lower())
    return uniq


def delete_model(path_str: str, models_dir: Path):
    """
    Borra un .gguf de MODELS_DIR (y sus shards -00001-of-000NN- hermanos si
    es un modelo partido en varios ficheros). Protege contra:
      - rutas fuera de MODELS_DIR (o que ni siquiera existan)
      - ficheros que no sean .gguf
      - modelos que estén cargados ahora mismo en alguna instancia activa
    Devuelve (ok, detalle) donde detalle es una lista de ficheros borrados
    si ok=True, o un mensaje de error si ok=False.
    """
    if not path_str:
        return False, "Ruta vacía"
    p = Path(path_str)
    try:
        p_resolved = p.resolve()
        models_resolved = models_dir.resolve()
    except Exception:
        return False, "Ruta inválida"

    if models_resolved != p_resolved and models_resolved not in p_resolved.parents:
        return False, "La ruta no está dentro de la carpeta de modelos"
    if p_resolved.suffix.lower() != ".gguf":
        return False, "Solo se pueden borrar ficheros .gguf"
    if not p_resolved.exists():
        return False, "El fichero no existe (¿ya se borró?)"

    for port, inst in INSTANCES.items():
        if inst.get("model_path") == str(p_resolved):
            return False, f"En uso por la instancia del puerto {port} — detenla antes de borrar"

    mp = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", p_resolved.name)
    deleted = []
    try:
        if mp:
            pattern = p_resolved.name.replace(f"-{mp.group(1)}-of-", "-*-of-")
            shards = sorted(p_resolved.parent.glob(pattern)) or [p_resolved]
            for shard in shards:
                shard.unlink()
                deleted.append(shard.name)
        else:
            p_resolved.unlink()
            deleted.append(p_resolved.name)
    except Exception as e:
        return False, f"Error al borrar: {e}" + (f" (borrados antes del fallo: {', '.join(deleted)})" if deleted else "")

    # Si la carpeta del modelo queda vacía (típico: una carpeta por modelo
    # descargada de HF), la eliminamos también para no dejar basura.
    try:
        parent = p_resolved.parent
        if parent != models_resolved and not any(parent.iterdir()):
            parent.rmdir()
    except Exception:
        pass

    return True, deleted


def list_mmprojs(models_dir: Path):
    results = []
    if models_dir.exists():
        for p in models_dir.rglob("mmproj*.gguf"):
            try:
                size_kb = round(p.stat().st_size / 1024)
            except OSError:
                size_kb = 0
            results.append({
                "path": str(p),
                "name": str(p.relative_to(models_dir)),
                "size_kb": size_kb,
            })
    results.sort(key=lambda x: x["name"].lower())
    return results


# ── Resolución de instancia por nombre de modelo (proxy multi-modelo) ───────

def find_instance_port(model_name: str):
    """
    Busca, entre las instancias vivas, cuál sirve `model_name` (el valor que
    manda el cliente en el campo "model" de /v1/chat/completions, /v1/messages,
    etc.). Se prueba en este orden:
      1) alias exacto (insensible a mayúsculas) — lo normal si arrancaste la
         instancia con --alias.
      2) nombre de fichero del modelo exacto (p.ej. "Qwen3.8-27B-Q4_K_M.gguf").
      3) "stem" del fichero (sin extensión) exacto o como substring — cubre
         el caso de pedir "qwen3.8-27b" sin el sufijo de cuantización.
    Devuelve el puerto (int) o None si no hay ninguna instancia que encaje
    (el llamador debe caer entonces al PROXY["port"] de siempre).
    """
    if not model_name:
        return None
    needle = str(model_name).strip().lower()
    if not needle:
        return None

    for port, inst in INSTANCES.items():
        if (inst.get("alias") or "").strip().lower() == needle:
            return port

    for port, inst in INSTANCES.items():
        if (inst.get("model_name") or "").lower() == needle:
            return port

    for port, inst in INSTANCES.items():
        stem = Path(inst.get("model_name") or "").stem.lower()
        if stem == needle or needle in stem:
            return port

    return None


# ── Health + Anthropic check ─────────────────────────────────────────────────

def health_check(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1.5) as r:
            return "ok" if r.status == 200 else "loading"
    except urllib.error.HTTPError as e:
        return "loading" if e.code == 503 else "down"
    except Exception:
        return "down"


def anthropic_ready(port):
    if port in ANTHROPIC_CACHE:
        return ANTHROPIC_CACHE[port]
    try:
        payload = json.dumps({"messages": [{"role": "user", "content": "x"}]}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/messages/count_tokens",
            data=payload, headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=3):
            ok = True
    except urllib.error.HTTPError as e:
        ok = (e.code != 404)
    except Exception:
        return False
    ANTHROPIC_CACHE[port] = ok
    return ok


# ── Presets ──────────────────────────────────────────────────────────────────

def load_presets():
    if PRESETS_FILE.exists():
        try:
            return json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_presets(presets):
    PRESETS_FILE.write_text(
        json.dumps(presets, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ── Limpieza de instancias muertas ───────────────────────────────────────────

def cleanup_dead():
    for port in list(INSTANCES.keys()):
        if INSTANCES[port]["proc"].poll() is not None:
            try:
                INSTANCES[port]["log_handle"].close()
            except Exception:
                pass
            del INSTANCES[port]
            ANTHROPIC_CACHE.pop(port, None)
    cleanup_dead_slaves()


# ── Modo esclavo: ggml-rpc-server ────────────────────────────────────────────
# Este host EXPONE sus GPUs a un maestro remoto. Se lanza
# ggml-rpc-server.exe (vive en la carpeta del motor, igual que llama-server)
# escuchando en 0.0.0.0:RPC_PORT. El maestro lo usa con --rpc ip-este-host:puerto.
# Los procesos hijos mueren junto al manager si se mata (taskkill /T / killpg),
# y cleanup_dead_slaves() limpia los que se caen solos.

def start_slave(engine: str = None, port: int = RPC_PORT) -> tuple:
    """Lanza ggml-rpc-server exponiendo los devices del motor. Devuelve (port, err)."""
    port = int(port)
    if port in RPC_SERVERS and RPC_SERVERS[port]["proc"].poll() is None:
        return None, f"Ya hay un esclavo escuchando en el puerto {port}"

    engine = engine if engine in ENGINES else get_engine()
    exe_dir = ENGINES.get(engine) or ENGINES[DEFAULT_ENGINE]
    rpc_name = "ggml-rpc-server.exe" if IS_WINDOWS else "ggml-rpc-server"
    exe = exe_dir / rpc_name
    if not exe.exists():
        return None, (f"El motor '{engine}' no trae {rpc_name} — "
                      f"necesita un build de llama.cpp con -DGGML_RPC=ON")

    log_file   = LOGS_DIR / f"rpc{port}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_handle = LogPump(log_file)
    try:
        slave_devs, _ = list_devices(engine=engine)
        log_handle.write(system_header(
            title=f"ESCLAVO ggml-rpc-server — puerto {port}",
            engine=engine, exe=exe, env=engine_env(engine), devices=slave_devs))
    except Exception as e:
        log_handle.event(f"(no se pudo generar la cabecera de hardware: {e})")

    popen_kwargs = {}
    if IS_WINDOWS:
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    cmd = [str(exe), "--host", "0.0.0.0", "--port", str(port)]
    log_handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] CMD: " + " ".join(cmd) + "\n\n")
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(exe.parent),
            env=engine_env(engine), **popen_kwargs,
        )
    except Exception as e:
        log_handle.event(f"ERROR al lanzar: {e}")
        log_handle.close()
        return None, f"No se pudo lanzar ggml-rpc-server: {e}"
    log_handle.attach(proc, label="ggml-rpc-server")

    # Verificación rápida: si el puerto estaba ocupado o el device no carga,
    # el proceso muere al instante; le damos un margen corto para detectarlo.
    try:
        proc.wait(timeout=2)
        if log_handle._thread:
            log_handle._thread.join(timeout=3)  # que vuelque la salida antes de leerla
        tail = ""
        try:
            tail = Path(log_file).read_text(encoding="utf-8", errors="replace")[-800:]
        except OSError:
            pass
        return None, f"ggml-rpc-server terminó al arrancar (código {proc.returncode}). Log: {tail.strip() or '(vacío)'}"
    except subprocess.TimeoutExpired:
        pass  # sobrevivió los 2 s → sigue vivo, bien
    log_handle.event(f"ggml-rpc-server lanzado (PID {proc.pid})")

    RPC_SERVERS[port] = {
        "proc": proc, "engine": engine,
        "log_file": log_file, "log_handle": log_handle, "started": time.time(),
    }
    return port, None


def stop_slave(port: int, force: bool = False):
    srv = RPC_SERVERS.get(port)
    if not srv:
        return False, "Esclavo no encontrado"
    proc = srv["proc"]
    try:
        srv["log_handle"].event("Parada " + ("FORZADA" if force else "solicitada") + " desde llama_manager")
    except Exception:
        pass
    try:
        if force:
            force_kill(proc.pid)
        else:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                force_kill(proc.pid)
    finally:
        try:
            srv["log_handle"].close()
        except Exception:
            pass
        RPC_SERVERS.pop(port, None)
    return True, None


def cleanup_dead_slaves():
    for port in list(RPC_SERVERS.keys()):
        if RPC_SERVERS[port]["proc"].poll() is not None:
            try:
                RPC_SERVERS[port]["log_handle"].close()
            except Exception:
                pass
            del RPC_SERVERS[port]


# ── Traducción de flags entre versiones de llama.cpp ────────────────────────

LOAD_MODES = ("auto", "none", "mmap", "mlock", "mmap+mlock", "dio")


def normalize_load_mode(cfg: dict) -> str:
    """load_mode del preset; si es un preset viejo, se deduce de mmap/mlock:
    mmap=on → auto · mmap=off → none · +mlock → mlock / mmap+mlock."""
    lm = (cfg.get("load_mode") or "").strip()
    if lm in LOAD_MODES:
        return lm
    mmap, mlock = cfg.get("mmap", True), cfg.get("mlock", False)
    if mmap is None:
        mmap = True
    if mlock:
        return "mmap+mlock" if mmap else "mlock"
    return "auto" if mmap else "none"


def load_mode_args(mode: str, has, notes: list) -> list:
    if mode == "auto":
        return []                       # es el default en todas las versiones
    if has("--load-mode"):
        return ["--load-mode", mode]
    # Binario viejo (p.ej. backend de LM Studio): flags clásicos
    notes.append(f"motor sin --load-mode: load-mode={mode} traducido a flags antiguos")
    return {"none": ["--no-mmap"], "mmap": [], "mlock": ["--no-mmap", "--mlock"],
            "mmap+mlock": ["--mlock"], "dio": ["--direct-io"]}.get(mode, [])


# flags eliminados en b11xxx → traducción (None = descartar)
_REMOVED = {
    "--no-mmap": ("load", "none"), "--mmap": ("load", "mmap"),
    "--mlock": ("load", "mlock"), "--direct-io": ("load", "dio"), "-dio": ("load", "dio"),
    "--no-direct-io": None, "-ndio": None,
}
_REMOVED_WITH_VALUE = {
    "--draft": "--spec-draft-n-max", "--draft-n": "--spec-draft-n-max",
    "--draft-max": "--spec-draft-n-max", "--draft-min": "--spec-draft-n-min",
    "--draft-n-min": "--spec-draft-n-min", "--defrag-thold": None, "-dt": None,
}


def _apply_legacy(mode, flag):
    """Aplica un flag antiguo sobre el load-mode acumulado (en orden, el último manda)."""
    locked = mode in ("mlock", "mmap+mlock")
    if flag == "--no-mmap":
        return "mlock" if locked else "none"
    if flag == "--mmap":
        return "mmap+mlock" if locked else "mmap"
    if flag == "--mlock":
        return "mlock" if mode == "none" else "mmap+mlock"
    if flag in ("--direct-io", "-dio"):
        return "dio"
    return mode          # --no-direct-io / -ndio: sin efecto


def sanitize_extra(extra: str):
    """Limpia 'Flags extra' de presets viejos. Devuelve (tokens, load_mode|None, notas).
    - --no-mmap/--mmap/--mlock/-dio → se convierten en un único --load-mode
    - --load-mode repetido → se queda el último (llama.cpp avisa DEPRECATED)
    - --draft-max N & co. → --spec-draft-n-max N
    Se procesan EN ORDEN: lo último que aparece manda, igual que haría llama.cpp."""
    toks = extra.split()
    out, notes = [], []
    load, seen_lm = None, False
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in ("--load-mode", "-lm") and i + 1 < len(toks):
            if seen_lm:
                notes.append(f"'--load-mode' repetido en Flags extra: manda el último ({toks[i+1]})")
            seen_lm = True
            load = toks[i + 1]
            i += 2
            continue
        if t in _REMOVED:
            load = _apply_legacy(load, t)
            notes.append(f"'{t}' ya no existe en llama.cpp → traducido a --load-mode")
            i += 1
            continue
        if t in _REMOVED_WITH_VALUE:
            new = _REMOVED_WITH_VALUE[t]
            val = toks[i + 1] if i + 1 < len(toks) else ""
            if new:
                out += [new, val]
                notes.append(f"'{t}' eliminado → {new}")
            else:
                notes.append(f"'{t}' eliminado → descartado")
            i += 2
            continue
        out.append(t)
        i += 1
    if load is not None and load not in LOAD_MODES:
        notes.append(f"--load-mode '{load}' no válido → auto")
        load = "auto"
    return out, load, notes


# flags que llevan valor (para deduplicar extra contra lo que ya generó la UI)
_VALUE_FLAGS = {"--cache-ram", "-cram", "--spec-type", "--spec-draft-n-max", "-md",
                "--ctx-checkpoints", "-ctxcp", "--lazy-mode", "--reasoning",
                "--reasoning-effort", "--image-min-tokens", "--fit", "-ot", "--override-tensor"}
_BOOL_FLAGS = {"--spec-draft-backend-sampling", "--backend-sampling", "-bs",
               "--no-ui", "--no-webui", "--reasoning-preserve", "--no-reasoning-preserve"}


def _dedupe_against(cmd: list, extra: list, notes: list) -> list:
    out, i = [], 0
    while i < len(extra):
        t = extra[i]
        if t in _VALUE_FLAGS and t in cmd and i + 1 < len(extra):
            if t in ("-ot", "--override-tensor"):
                # se fusiona con el -ot de la UI
                j = cmd.index(t)
                merged = list(dict.fromkeys(cmd[j + 1].split(",") + extra[i + 1].split(",")))
                cmd[j + 1] = ",".join(merged)
            else:
                notes.append(f"'{t} {extra[i+1]}' de Flags extra ignorado: ya lo fija el panel")
            i += 2
            continue
        if t in _BOOL_FLAGS and t in cmd:
            i += 1
            continue
        out.append(t)
        i += 1
    return out


# ── Arranque de instancia ────────────────────────────────────────────────────

def start_instance(cfg: dict):
    """
    Construye el comando y lanza llama-server.
    Devuelve (pid, error_str). error_str es None si ok.
    """
    port = int(cfg["port"])
    if port in INSTANCES:
        return None, f"Puerto {port} ya en uso"

    engine = cfg.get("engine") or get_engine()
    if engine not in ENGINES:
        engine = get_engine()

    model_path = _portable(cfg["model"])
    if not model_path.exists():
        return None, f"El modelo no existe: {cfg['model']}"

    exe = engine_exe(engine)
    if not exe.exists():
        return None, f"No se encuentra {exe} (motor: {engine})"
    if not IS_WINDOWS and not os.access(exe, os.X_OK):
        return None, f'{exe} existe pero no tiene permiso de ejecución — prueba chmod +x "{exe}"'

    ngl   = int(cfg.get("ngl", 999))
    ts    = (cfg.get("ts") or "").strip()
    alias = (cfg.get("alias") or "").strip()
    ctk   = cfg.get("ctk", "q8_0")
    ctv   = cfg.get("ctv", "q8_0")
    fa    = bool(cfg.get("fa", True))

    if (ctk != "f16" or ctv != "f16") and not fa:
        return None, "KV cuantizado requiere Flash Attention"

    # Recorte de seguridad: si piden más contexto del que el modelo soporta
    # de verdad, lo bajamos aquí también — no solo en la UI — para que
    # cualquier arranque (manual, por API, o desde un preset viejo) quede
    # protegido igual, no solo el que pasa por el formulario web.
    ctx_req = int(cfg.get("ctx", 4096))
    ctx = ctx_req
    ctx_clamped_from = None
    meta = get_cached_meta(model_path)
    if meta:
        arch = meta.get("general.architecture", "")
        model_max_ctx = meta.get(f"{arch}.context_length")
        if model_max_ctx and ctx_req > model_max_ctx:
            ctx = model_max_ctx
            ctx_clamped_from = ctx_req

    # ── Modo maestro: repartir el modelo con esclavos ggml-rpc-server ────────
    # Se activa con el check "rpc_on" de la UI + la lista "host:port,...".
    # Presets antiguos sin "rpc_on": activo si traen lista rpc (compatibilidad).
    # Requiere el motor compilado con -DGGML_RPC=ON y la MISMA versión de
    # llama.cpp en maestro y esclavos.
    rpc_on = cfg.get("rpc_on")
    if rpc_on is None:
        rpc_on = bool((cfg.get("rpc") or "").strip())
    rpc = (cfg.get("rpc") or "").strip() if rpc_on else ""
    rpc_servers = []
    if rpc_on:
        rpc_servers = [s.strip() for s in rpc.replace(" ", "").split(",") if s.strip()]
        rpc_servers = [s for s in rpc_servers if ":" in s]
        if not rpc_servers:
            return None, "RPC activado pero sin esclavos válidos — usa host:port,host2:port2"

    # --device: con RPC activo NUNCA se pasa. llama.cpp parsea --device antes
    # de registrar los RPC ("invalid device: RPC0") y además una lista explícita
    # excluiría a los esclavos; sin --device reparte entre GPUs locales + RPC.
    # Sin RPC: lista explícita = solo esas GPUs; vacío = auto.
    device = "" if rpc_on else (cfg.get("device") or "").strip()
    launch_devs = None  # para la cabecera del log
    cmd = [str(exe)]
    if rpc_servers:
        cmd += ["--rpc", ",".join(rpc_servers)]
    if device:
        # Validar que los VulkanN/ROCmN existen AHORA. Justo tras parar/matar
        # una instancia el driver puede tardar unos segundos en devolver las
        # GPUs y llama-server muere con "invalid device: Vulkan0".
        wanted = [d.strip() for d in device.split(",") if d.strip()]
        missing, dev_err = wanted, None
        for attempt in range(4):
            devs, dev_err = list_devices(engine=engine)
            launch_devs = devs
            known = {d["id"] for d in devs}
            missing = [d for d in wanted if d not in known]
            if not missing:
                break
            time.sleep(2)
        if missing:
            return None, (f"GPU(s) no visibles para el motor '{engine}': {', '.join(missing)}. "
                          f"Si acabas de parar otra instancia, espera unos segundos y reintenta "
                          f"(el driver aún no ha liberado la GPU)."
                          + (f" Detalle: {dev_err}" if dev_err else ""))
        cmd += ["--device", device]
    cmd += [
        "-m", str(model_path), "-ngl", str(ngl),
        "-c", str(ctx),
        "--host", "0.0.0.0", "--port", str(port),
        "--parallel", str(int(cfg.get("parallel", 1))),
        "--jinja", "-fa", "on" if fa else "off",
        "-b", str(int(cfg.get("batch", 2048))),
        "-ub", str(int(cfg.get("ubatch", 512))),
    ]
    if alias:
        cmd += ["--alias", alias]
    if ts and "," in device:
        cmd += ["-ts", ts]
    if ctk != "f16":
        cmd += ["-ctk", ctk]
    if ctv != "f16":
        cmd += ["-ctv", ctv]

    def _flag(field, flag):
        v = str(cfg.get(field) or "").strip().replace(",", ".")
        if v:
            cmd.extend([flag, v])

    _flag("threads",  "-t")
    _flag("seed",     "--seed")
    _flag("npredict", "-n")
    _flag("temp",     "--temp")
    _flag("topp",     "--top-p")
    _flag("topk",     "--top-k")
    _flag("minp",     "--min-p")
    _flag("repp",     "--repeat-penalty")
    _flag("presp",    "--presence-penalty")
    _flag("freqp",    "--frequency-penalty")

    caps  = engine_caps(engine)
    notes = []   # traducciones/avisos que se escriben en el log de la instancia

    def has(flag):
        # Sin --help legible (caps vacío) se asume build moderno.
        return (not caps) or flag in caps

    # ── Carga del modelo: --load-mode (b11xxx) ≈ --mmap/--no-mmap/--mlock (viejo)
    load_mode = normalize_load_mode(cfg)
    extra_tokens, extra_load_mode, extra_notes = sanitize_extra(cfg.get("extra") or "")
    notes += extra_notes
    if extra_load_mode:
        load_mode = extra_load_mode
    cmd += load_mode_args(load_mode, has, notes)

    lazy = (cfg.get("lazy_mode") or "").strip()
    if lazy in ("on", "auto", "off") and has("--lazy-mode"):
        if lazy == "on" and load_mode in ("none", "mlock", "dio"):
            notes.append("--lazy-mode on requiere mmap: se omite (load-mode=%s)" % load_mode)
        else:
            cmd += ["--lazy-mode", lazy]

    ncpumoe = int(cfg.get("ncpumoe", 0) or 0)
    if ncpumoe > 0:
        cmd += ["--n-cpu-moe", str(ncpumoe)]

    # Qwen4 (qwen4exp / Qwen3.8-Flash-Next): la tabla PLE (per_layer_token_embd)
    # es enorme y solo se lee por filas → a RAM, no a VRAM.
    ot = [t for t in (cfg.get("ot") or "").replace(" ", "").split(",") if t]
    if cfg.get("ple_cpu"):
        ot.insert(0, "per_layer_token_embd=CPU")
    if ot:
        cmd += ["-ot", ",".join(dict.fromkeys(ot))]

    # ── Razonamiento ────────────────────────────────────────────────────────
    reasoning = cfg.get("reasoning")
    if reasoning not in ("on", "off", "auto"):
        reasoning = "off" if cfg.get("thinking") is False else "auto"   # presets viejos
    if reasoning != "auto":
        if has("--reasoning"):
            cmd += ["--reasoning", reasoning]
        elif reasoning == "off":
            cmd += ["--reasoning-budget", "0"]
    effort = cfg.get("effort", "auto")
    if effort not in ("", "auto", None):
        if has("--reasoning-effort"):
            cmd += ["--reasoning-effort", effort]
        else:
            cmd += ["--chat-template-kwargs", json.dumps({"reasoning_effort": effort})]
    rp = cfg.get("reasoning_preserve") or ""
    if rp in ("on", "off") and has("--reasoning-preserve"):
        cmd += ["--reasoning-preserve" if rp == "on" else "--no-reasoning-preserve"]

    mmproj = (cfg.get("mmproj") or "").strip()
    if mmproj:
        mp = _portable(mmproj)
        cmd += ["--mmproj", str(mp)]
        _flag("image_min_tokens", "--image-min-tokens")

    # ── Flags avanzados seleccionables por checkbox en la UI ────────────────
    if cfg.get("no_webui"):
        cmd += ["--no-ui"] if (caps and "--no-ui" in caps) else ["--no-webui"]
    if cfg.get("fit_off"):
        cmd += ["--fit", "off"]
    if cfg.get("backend_sampling"):
        cmd += ["--backend-sampling"]

    # ── Speculative decoding ────────────────────────────────────────────────
    # MTP embebido (Qwen3.5/3.6/3.8-27B -MTP): basta --spec-type draft-mtp.
    # MTP en sidecar (Qwen3.8-Flash-Next / qwen4exp: mtp-*.gguf junto al modelo):
    # además -md <sidecar>. spec_draft="auto" → primer mtp-*.gguf de la carpeta.
    draft = (cfg.get("spec_draft") or "").strip()
    if draft == "auto":
        side = [d for d in find_draft_sidecars(model_path) if Path(d).name.lower().startswith("mtp-")]
        draft = side[0] if side else ""
        if not side and cfg.get("spec_mtp"):
            notes.append("MTP: no hay mtp-*.gguf junto al modelo — se usa el MTP embebido (si lo tiene)")
    if cfg.get("spec_mtp"):
        cmd += ["--spec-type", "draft-mtp"]
    if draft:
        cmd += ["-md", str(_portable(draft))]
        if not cfg.get("spec_mtp"):
            notes.append("draft/sidecar indicado sin 'Speculative MTP': llama.cpp deduce el tipo del GGUF")
    if cfg.get("spec_mtp") or draft:
        _flag("spec_n_max", "--spec-draft-n-max")

    # ── Caché de prompts (clave con modelos híbridos/recurrentes como qwen4exp)
    _flag("cache_ram", "--cache-ram")
    _flag("ctx_ckpt",  "--ctx-checkpoints")

    ctf = (cfg.get("ctf_path") or "").strip()
    if cfg.get("use_ctf") and ctf:
        cmd += ["--chat-template-file", str(_portable(ctf))]

    if cfg.get("dry"):
        _flag("dry_multiplier",     "--dry-multiplier")
        _flag("dry_base",           "--dry-base")
        _flag("dry_allowed_length", "--dry-allowed-length")
        _flag("dry_penalty_last_n", "--dry-penalty-last-n")

    if extra_tokens:
        # Remapear rutas ajenas al SO que viajan dentro de "extra"
        _path_flags = {"--chat-template-file", "--mmproj", "-md", "--model-draft", "--spec-draft-model"}
        for i, t in enumerate(extra_tokens):
            if i > 0 and extra_tokens[i - 1] in _path_flags:
                extra_tokens[i] = str(_portable(t))
        # Quitar de "extra" lo que ya ha puesto la UI (evita duplicados tipo
        # --cache-ram dos veces o --spec-type repetido)
        extra_tokens = _dedupe_against(cmd, extra_tokens, notes)
        cmd += extra_tokens

    log_file   = LOGS_DIR / f"port{port}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_handle = LogPump(log_file)
    if launch_devs is None:
        try:
            launch_devs, _ = list_devices(engine=engine)
        except Exception:
            launch_devs = []
    try:
        log_handle.write(system_header(
            title=f"ARRANQUE llama-server — puerto {port}" + (f" — alias {alias}" if alias else ""),
            engine=engine, exe=exe, env=engine_env(engine),
            devices=launch_devs,
            used=[d.strip() for d in device.split(",") if d.strip()] or None,
            split=ts if (ts and "," in device) else None,
            rpc=",".join(rpc_servers) or None,
            model=str(model_path),
            extra_lines=[f"Contexto     : {ctx}   ngl: {ngl}   parallel: {int(cfg.get('parallel', 1))}"],
        ))
    except Exception as e:
        log_handle.event(f"(no se pudo generar la cabecera de hardware: {e})")
    if ctx_clamped_from:
        log_handle.write(f"NOTA: contexto pedido {ctx_clamped_from} > máximo del modelo "
                          f"({ctx}) — recortado automáticamente.\n")
    for n in notes:
        log_handle.write(f"NOTA: {n}\n")
    log_handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] CMD: " + " ".join(cmd) + "\n\n")

    popen_kwargs = {}
    if IS_WINDOWS:
        # CREATE_NEW_PROCESS_GROUP: le da su propio grupo para poder matar el
        # árbol completo luego con taskkill /T sin afectar a llama_manager.
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        # Equivalente en POSIX: nueva sesión → nuevo grupo de proceso, que es
        # lo que force_kill() necesita para os.killpg(). Sin esto, matar solo
        # el PID del proceso principal podría dejar huérfanos a sus hijos.
        popen_kwargs["start_new_session"] = True

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(exe.parent),
            env=engine_env(engine), **popen_kwargs,
        )
    except Exception as e:
        log_handle.event(f"ERROR al lanzar: {e}")
        log_handle.close()
        return None, f"No se pudo lanzar: {e}"

    log_handle.event(f"llama-server lanzado (PID {proc.pid})")
    log_handle.attach(proc, label="llama-server")

    INSTANCES[port] = {
        "proc": proc,
        "model_name": model_path.name,
        "model_path": str(model_path),
        "alias": alias,
        "device": device or ("auto+RPC" if rpc_on else "auto"), "ctx": ctx, "ngl": ngl, "engine": engine,
        "rpc": rpc or None,  # si es maestro: lista de esclavos host:port
        "started": time.time(), "log_file": log_file, "log_handle": log_handle,
    }

    # Persistir preset (con el ctx ya recortado, para que relanzar el mismo
    # preset no vuelva a pedir un valor que el modelo no soporta)
    cfg = {**cfg, "ctx": ctx, "engine": engine, "rpc_on": bool(rpc_on),
           "load_mode": load_mode, "extra": " ".join(extra_tokens)}
    presets = load_presets()
    presets[str(port)] = {k: cfg.get(k) for k in (
        "port", "model", "device", "ctx", "parallel", "ngl", "ts", "alias", "fa",
        "ctk", "ctv", "threads", "seed", "mmap", "mlock", "batch", "ubatch",
        "ncpumoe", "extra", "npredict", "temp", "topp", "topk", "minp",
        "repp", "presp", "freqp", "thinking", "effort", "mmproj", "engine",
        "no_webui", "fit_off", "spec_mtp", "use_ctf", "ctf_path",
        "dry", "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n",
        "rpc", "rpc_on",
        "load_mode", "lazy_mode", "ple_cpu", "ot", "reasoning", "reasoning_preserve",
        "backend_sampling", "spec_draft", "spec_n_max", "cache_ram", "ctx_ckpt",
        "image_min_tokens",
    )}
    save_presets(presets)
    return proc.pid, None


def stop_instance(port: int, force: bool = False):
    inst = INSTANCES.get(port)
    if not inst:
        return False, "Instancia no encontrada"
    proc = inst["proc"]
    try:
        inst["log_handle"].event("Parada " + ("FORZADA" if force else "solicitada") + " desde llama_manager")
    except Exception:
        pass
    try:
        if force:
            force_kill(proc.pid)
        else:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                force_kill(proc.pid)
    finally:
        try:
            inst["log_handle"].close()
        except Exception:
            pass
        INSTANCES.pop(port, None)
        ANTHROPIC_CACHE.pop(port, None)
    return True, None
