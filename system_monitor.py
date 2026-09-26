# -*- coding: utf-8 -*-
"""
system_monitor.py — Monitor de RAM y VRAM en tiempo real.
Soporta AMD (rocm-smi, o sysfs amdgpu en Linux sin rocm-smi instalado),
NVIDIA (nvidia-smi), y fallback WMI/Get-Counter en Windows.
"""
import ctypes
import glob
import re
import subprocess
import threading
import time
from pathlib import Path

from config import IS_WINDOWS

SYS = {"ram_total": 0, "ram_used": 0, "gpus": [], "vram_source": None}

_device_cache = []
_device_cache_lock = threading.Lock()


def _run(cmd, timeout=8):
    """subprocess.run con creationflags SOLO en Windows — en Linux/macOS ese
    atributo ni siquiera existe en el módulo subprocess, así que pasarlo
    ahí revienta con AttributeError antes de lanzar nada."""
    # errors="replace": la consola de Windows devuelve cp850/cp1252 (acentos)
    # y con UTF-8 estricto el hilo lector reventaba con UnicodeDecodeError.
    kwargs = {"capture_output": True, "text": True, "errors": "replace", "timeout": timeout}
    if IS_WINDOWS:
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return subprocess.run(cmd, **kwargs)


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength",               ctypes.c_ulong),
        ("dwMemoryLoad",           ctypes.c_ulong),
        ("ullTotalPhys",           ctypes.c_ulonglong),
        ("ullAvailPhys",           ctypes.c_ulonglong),
        ("ullTotalPageFile",       ctypes.c_ulonglong),
        ("ullAvailPageFile",       ctypes.c_ulonglong),
        ("ullTotalVirtual",        ctypes.c_ulonglong),
        ("ullAvailVirtual",        ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual",ctypes.c_ulonglong),
    ]


def _get_ram_windows():
    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    total = st.ullTotalPhys // (1024 ** 2)
    used  = (st.ullTotalPhys - st.ullAvailPhys) // (1024 ** 2)
    return total, used


def _get_ram_linux():
    """Lee /proc/meminfo — sin dependencias externas, funciona en cualquier
    Linux con procfs (todo lo que soportamos: Debian, etc.)."""
    info = {}
    with open("/proc/meminfo", "r", encoding="utf-8") as f:
        for line in f:
            key, _, rest = line.partition(":")
            info[key.strip()] = rest.strip()
    total_kb = int(info.get("MemTotal", "0 kB").split()[0])
    # MemAvailable (kernel >= 3.14) estima mejor la memoria realmente
    # disponible que MemFree a secas (cuenta caches reclamables). Si no
    # existe (kernel muy viejo), caemos a MemFree.
    avail_raw = info.get("MemAvailable") or info.get("MemFree", "0 kB")
    avail_kb = int(avail_raw.split()[0])
    total = total_kb // 1024
    used  = (total_kb - avail_kb) // 1024
    return total, used


def get_ram():
    try:
        return _get_ram_windows() if IS_WINDOWS else _get_ram_linux()
    except Exception:
        return 0, 0


def _try_rocm_smi():
    try:
        out = _run(["rocm-smi", "--showmeminfo", "vram", "--csv"])
        results = []
        for line in out.stdout.splitlines():
            m_total = re.search(r"GPU\[(\d+)\].*?VRAM Total Memory.*?:\s*(\d+)", line)
            m_used  = re.search(r"GPU\[(\d+)\].*?VRAM Total Used.*?:\s*(\d+)", line)
            if m_total:
                idx = int(m_total.group(1))
                while len(results) <= idx:
                    results.append({"gpu_idx": idx, "total_mib": 0, "used_mib": 0})
                results[idx]["total_mib"] = int(m_total.group(2)) // (1024**2)
            if m_used:
                idx = int(m_used.group(1))
                results[idx]["used_mib"] = int(m_used.group(2)) // (1024**2)
        if any(r["total_mib"] > 0 for r in results):
            return results
    except Exception:
        pass
    return None


def _try_rocm_smi_showmeminfo():
    try:
        out = _run(["rocm-smi", "--showmeminfo", "vram"])
        results = {}
        for line in out.stdout.splitlines():
            mt = re.search(r"GPU\[(\d+)\].*?VRAM Total Memory.*?:\s*(\d+)", line)
            mu = re.search(r"GPU\[(\d+)\].*?VRAM Total Used.*?:\s*(\d+)", line)
            if mt:
                idx = int(mt.group(1))
                results.setdefault(idx, {"gpu_idx": idx, "total_mib": 0, "used_mib": 0})
                results[idx]["total_mib"] = int(mt.group(2)) // (1024**2)
            if mu:
                idx = int(mu.group(1))
                results.setdefault(idx, {"gpu_idx": idx, "total_mib": 0, "used_mib": 0})
                results[idx]["used_mib"] = int(mu.group(2)) // (1024**2)
        if results:
            return [results[k] for k in sorted(results)]
    except Exception:
        pass
    return None


def _try_amdgpu_sysfs():
    """Lee VRAM de tarjetas/iGPU AMD vía sysfs del driver amdgpu — no
    necesita rocm-smi instalado (típico en setups solo-Vulkan como HouseMini,
    donde no hay stack ROCm). Cada tarjeta expone mem_info_vram_total/used
    en bytes bajo /sys/class/drm/cardN/device/."""
    try:
        results = []
        for card_dir in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
            total_f = Path(card_dir) / "mem_info_vram_total"
            used_f  = Path(card_dir) / "mem_info_vram_used"
            if not (total_f.exists() and used_f.exists()):
                continue
            total_b = int(total_f.read_text().strip())
            used_b  = int(used_f.read_text().strip())
            if total_b <= 0:
                continue
            results.append({
                "gpu_idx":   len(results),
                "total_mib": total_b // (1024**2),
                "used_mib":  used_b // (1024**2),
            })
        return results or None
    except Exception:
        return None


def _try_nvidia_smi():
    try:
        out = _run(["nvidia-smi", "--query-gpu=memory.total,memory.used",
                    "--format=csv,noheader,nounits"])
        results = []
        for i, line in enumerate(out.stdout.strip().splitlines()):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit():
                results.append({"gpu_idx": i,
                                 "total_mib": int(parts[0]),
                                 "used_mib":  int(parts[1])})
        if results:
            return results
    except Exception:
        pass
    return None


def _try_get_counter_vram():
    """
    Lee '\\GPU Adapter Memory(*)\\Dedicated Usage' y '...\\Shared Usage'
    directamente con Get-Counter — el mismo contador de rendimiento crudo
    que usa el Administrador de tareas por debajo, sin pasar por la capa
    CIM/WMI (Win32_PerfFormattedData_...) que a veces queda desfasada.
    Cada instancia se identifica por su LUID (algo como
    "luid_0x00000000_0x0000abcd_phys_0"), que exponemos en "raw_id" para
    poder verificar a ojo el orden si algún día hay que depurarlo — Windows
    no da una forma sencilla de mapear LUID → nombre de tarjeta sin más
    pasos, así que seguimos emparejando por orden con Vulkan, pero al menos
    ahora hay un identificador estable que enseñar si algo no cuadra.
    Solo se intenta en Windows — PowerShell no existe en Linux.
    """
    try:
        ps_script = (
            "$ErrorActionPreference='Stop';"
            "$d=(Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage').CounterSamples;"
            "$s=(Get-Counter '\\GPU Adapter Memory(*)\\Shared Usage').CounterSamples;"
            "$names=$d | Select-Object -ExpandProperty InstanceName | Sort-Object;"
            "foreach($n in $names){"
            "  $du=($d | Where-Object {$_.InstanceName -eq $n}).CookedValue;"
            "  $su=($s | Where-Object {$_.InstanceName -eq $n}).CookedValue;"
            "  '{0}|{1}|{2}' -f $n,[int64]$du,[int64]$su}"
        )
        out = _run(["powershell", "-NoProfile", "-Command", ps_script], timeout=15)
        results = []
        for i, line in enumerate(out.stdout.strip().splitlines()):
            parts = line.strip().split("|")
            if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                ded_mib = int(parts[1]) // (1024**2)
                shr_mib = int(parts[2]) // (1024**2)
                if ded_mib == 0 and shr_mib == 0:
                    # Casi seguro un adaptador virtual/software (Basic Render
                    # Driver, RDP...) sin uso real — Windows lo expone en
                    # este mismo contador junto a las GPUs físicas, y si lo
                    # dejamos entrar descuadra el nº de entradas frente a las
                    # GPUs reales que ve Vulkan.
                    continue
                results.append({"gpu_idx": len(results), "total_mib": 0,
                                 "used_mib": ded_mib + shr_mib,
                                 "dedicated_mib": ded_mib, "shared_mib": shr_mib,
                                 "raw_id": parts[0]})
        if results:
            return results
    except Exception:
        pass
    return None


def _try_wmi_vram():
    try:
        ps_script = (
            "$adap = Get-CimInstance Win32_VideoController | "
            "Select-Object -ExpandProperty AdapterRAM; "
            "$perf = Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory; "
            "for($i=0;$i -lt $perf.Count;$i++){"
            "  '{0}|{1}|{2}' -f ($adap[$i]/1MB),$perf[$i].DedicatedUsage,$perf[$i].SharedUsage}"
        )
        out = _run(["powershell", "-NoProfile", "-Command", ps_script], timeout=15)
        results = []
        for i, line in enumerate(out.stdout.strip().splitlines()):
            parts = line.strip().split("|")
            if len(parts) >= 3:
                total = int(float(parts[0])) if parts[0].replace(".", "").isdigit() else 0
                # Task Manager suma dedicada + compartida en "Memoria de GPU" —
                # antes solo cogíamos DedicatedUsage y descartábamos SharedUsage,
                # lo que en iGPU (donde la mayor parte del uso cae en la
                # compartida) hacía parecer la GPU casi vacía cuando no lo estaba.
                dedicated = int(parts[1]) if parts[1].isdigit() else 0
                shared    = int(parts[2]) if parts[2].isdigit() else 0
                used = (dedicated + shared) // (1024**2)
                results.append({"gpu_idx": i, "total_mib": total, "used_mib": used})
        if any(r["total_mib"] > 0 for r in results):
            return results
    except Exception:
        pass
    return None


def _try_wmi_dedicated_only():
    try:
        out = _run([
            "powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory | "
            "ForEach-Object { '{0}|{1}' -f $_.DedicatedUsage,$_.SharedUsage }",
        ], timeout=15)
        results = []
        for i, line in enumerate(out.stdout.strip().splitlines()):
            parts = line.strip().split("|")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                used = (int(parts[0]) + int(parts[1])) // (1024**2)
                results.append({"gpu_idx": i, "total_mib": 0, "used_mib": used})
        if results:
            return results
    except Exception:
        pass
    return None


def get_vram_usage():
    # Orden de intentos por plataforma: en Windows probamos primero los
    # contadores nativos (más precisos, incluyen memoria compartida de
    # iGPU); en Linux vamos directo a rocm-smi/sysfs/nvidia-smi — probar
    # "powershell" o WMI ahí solo desperdicia un intento (falla siempre con
    # FileNotFoundError, capturado y descartado, pero es ruido innecesario).
    candidates = []
    if IS_WINDOWS:
        candidates.append(("get-counter", _try_get_counter_vram))
    candidates.append(("rocm-smi", _try_rocm_smi))
    candidates.append(("rocm-smi(meminfo)", _try_rocm_smi_showmeminfo))
    if not IS_WINDOWS:
        candidates.append(("amdgpu-sysfs", _try_amdgpu_sysfs))
    candidates.append(("nvidia-smi", _try_nvidia_smi))
    if IS_WINDOWS:
        candidates.append(("wmi", _try_wmi_vram))
        candidates.append(("wmi(dedicated+shared)", _try_wmi_dedicated_only))

    for name, fn in candidates:
        result = fn()
        if result:
            SYS["vram_source"] = name
            return result
    SYS["vram_source"] = "ninguna fuente disponible"
    return []


def update_device_cache(devices):
    global _device_cache
    with _device_cache_lock:
        _device_cache = list(devices)


def monitor_loop():
    while True:
        try:
            total, used = get_ram()
            vram = get_vram_usage()
            SYS["ram_total"] = total
            SYS["ram_used"]  = used
            SYS["gpus"]      = vram
        except Exception:
            pass
        time.sleep(5)


def start_monitor():
    threading.Thread(target=monitor_loop, daemon=True).start()
