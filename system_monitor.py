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


# ── Windows: DXGI (LUID → nombre/total) + Get-Counter (uso por LUID) ──────
_DXGI_CACHE = None


def _dxgi_adapters():
    """Enumera los adaptadores con DXGI vía ctypes (sin dependencias).
    Devuelve {(luid_high, luid_low): {name, dedicated_total_mib, software}}.
    Es lo que permite saber QUÉ tarjeta es cada instancia del contador de
    rendimiento (que solo trae el LUID) en vez de emparejar a ciegas por
    orden — el orden de los LUID no tiene nada que ver con el de Vulkan0/1.
    Los adaptadores no cambian en caliente, así que se cachea."""
    global _DXGI_CACHE
    if _DXGI_CACHE is not None:
        return _DXGI_CACHE
    result = {}
    try:
        from ctypes import wintypes

        class GUID(ctypes.Structure):
            _fields_ = [("d1", ctypes.c_ulong), ("d2", ctypes.c_ushort),
                        ("d3", ctypes.c_ushort), ("d4", ctypes.c_ubyte * 8)]

        class LUID(ctypes.Structure):
            _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", ctypes.c_long)]

        class DXGI_ADAPTER_DESC1(ctypes.Structure):
            _fields_ = [
                ("Description", ctypes.c_wchar * 128),
                ("VendorId", ctypes.c_uint), ("DeviceId", ctypes.c_uint),
                ("SubSysId", ctypes.c_uint), ("Revision", ctypes.c_uint),
                ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t),
                ("SharedSystemMemory", ctypes.c_size_t),
                ("AdapterLuid", LUID),
                ("Flags", ctypes.c_uint),
            ]

        # IID_IDXGIFactory1 = 770aae78-f26f-4dba-a829-253c83d1b387
        iid = GUID(0x770aae78, 0xf26f, 0x4dba,
                   (ctypes.c_ubyte * 8)(0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87))
        factory = ctypes.c_void_p()
        hr = ctypes.windll.dxgi.CreateDXGIFactory1(ctypes.byref(iid), ctypes.byref(factory))
        if hr != 0 or not factory:
            _DXGI_CACHE = {}
            return _DXGI_CACHE

        def vcall(obj, idx, restype, *argtypes):
            vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
            return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtbl[idx])

        HRESULT = ctypes.c_long
        i = 0
        while True:
            adapter = ctypes.c_void_p()
            # IDXGIFactory1::EnumAdapters1 = vtable[12]
            hr = vcall(factory, 12, HRESULT, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(
                factory, i, ctypes.byref(adapter))
            if hr != 0 or not adapter:
                break  # DXGI_ERROR_NOT_FOUND → fin de la lista
            desc = DXGI_ADAPTER_DESC1()
            # IDXGIAdapter1::GetDesc1 = vtable[10]
            if vcall(adapter, 10, HRESULT, ctypes.POINTER(DXGI_ADAPTER_DESC1))(
                    adapter, ctypes.byref(desc)) == 0:
                key = (desc.AdapterLuid.HighPart & 0xFFFFFFFF, desc.AdapterLuid.LowPart)
                result[key] = {
                    "name": desc.Description.strip(),
                    "dedicated_total_mib": desc.DedicatedVideoMemory // (1024**2),
                    # DXGI_ADAPTER_FLAG_SOFTWARE = 2 (Basic Render Driver)
                    "software": bool(desc.Flags & 2),
                }
            vcall(adapter, 2, ctypes.c_ulong)(adapter)  # Release
            i += 1
        vcall(factory, 2, ctypes.c_ulong)(factory)  # Release
    except Exception:
        result = {}
    _DXGI_CACHE = result
    return result


_LUID_RE = re.compile(r"luid_0x([0-9a-f]+)_0x([0-9a-f]+)_phys_\d+", re.I)


def _try_get_counter_vram():
    """
    Lee '\\GPU Adapter Memory(*)\\Dedicated Usage' y '...\\Shared Usage' con
    Get-Counter — el mismo contador crudo que usa el Administrador de tareas.
    Cada instancia trae el LUID del adaptador; lo cruzamos con DXGI para
    saber el NOMBRE y la VRAM dedicada total de cada una, y así emparejar
    con los dispositivos Vulkan por nombre (ver attach_live_usage) en vez de
    por posición. Antes se ordenaba por LUID y se asignaba a Vulkan0/1 por
    orden, lo que en la torre (R9700 + W7700) podía cruzar las lecturas, y
    se descartaban las tarjetas con uso 0, lo que descuadraba el recuento
    y hacía caer a la lectura de Vulkan (que no ve otros procesos).
    Solo se intenta en Windows — PowerShell no existe en Linux.
    """
    try:
        ps_script = (
            "$ErrorActionPreference='Stop';"
            "$d=(Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage').CounterSamples;"
            "$s=(Get-Counter '\\GPU Adapter Memory(*)\\Shared Usage').CounterSamples;"
            "foreach($x in $d){'D|{0}|{1}' -f $x.InstanceName,[int64]$x.CookedValue};"
            "foreach($x in $s){'S|{0}|{1}' -f $x.InstanceName,[int64]$x.CookedValue}"
        )
        out = _run(["powershell", "-NoProfile", "-Command", ps_script], timeout=15)
        usage = {}  # (high, low) → {"ded": bytes, "shr": bytes, "raw": instancia}
        for line in out.stdout.strip().splitlines():
            parts = line.strip().split("|")
            if len(parts) != 3 or not parts[2].lstrip("-").isdigit():
                continue
            m = _LUID_RE.search(parts[1])
            if not m:
                continue
            key = (int(m.group(1), 16), int(m.group(2), 16))
            u = usage.setdefault(key, {"ded": 0, "shr": 0, "raw": parts[1]})
            # Varias instancias phys_N con el mismo LUID = adaptador enlazado → se suman
            u["ded" if parts[0] == "D" else "shr"] += max(int(parts[2]), 0)
        if not usage:
            return None

        adapters = _dxgi_adapters()
        results = []
        for key in sorted(usage):
            u = usage[key]
            a = adapters.get(key)
            if adapters:
                # Con DXGI disponible: fuera software y LUIDs desconocidos
                # (Basic Render Driver, adaptadores de RDP...). Ya NO se filtra
                # por "uso 0": una GPU secundaria sin nada cargado puede
                # marcar 0 y es real.
                if a is None or a["software"]:
                    continue
            elif u["ded"] == 0 and u["shr"] == 0:
                continue  # sin DXGI, mantenemos el filtro antiguo
            ded_mib = u["ded"] // (1024**2)
            shr_mib = u["shr"] // (1024**2)
            results.append({
                "gpu_idx": len(results),
                "name": a["name"] if a else None,
                "total_mib": a["dedicated_total_mib"] if a else 0,
                "used_mib": ded_mib,          # VRAM real: solo dedicada
                "dedicated_mib": ded_mib,
                "shared_mib": shr_mib,        # esto es RAM del sistema, va aparte
                "raw_id": u["raw"],
            })
        return results or None
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


def _norm_name(n):
    n = (n or "").lower()
    n = re.sub(r"\(r\)|\(tm\)|®|™", "", n)
    return re.sub(r"[^a-z0-9]+", " ", n).strip()


def attach_live_usage(devices):
    """Añade a cada dispositivo de --list-devices su uso en vivo
    (used_mib, dedicated_mib, shared_mib, raw_id, vram_live).

    1) Si la fuente trae nombres (Windows + DXGI), se empareja POR NOMBRE;
       con dos tarjetas idénticas se desempata por el total más parecido y
       luego por orden.
    2) Si no hay nombres (rocm-smi, sysfs, nvidia-smi), se mantiene el
       emparejamiento por posición, solo si el nº de entradas coincide.
    3) Lo que quede sin pareja cae al "free" de Vulkan (vram_live=False).

    En iGPU (el total de Vulkan incluye memoria compartida, muy por encima de
    la dedicada que declara DXGI) el uso sí es dedicada + compartida, que es
    lo que ocupa realmente el heap que Vulkan reporta.
    """
    monitor = list(SYS.get("gpus", []))
    pairs = {}
    if monitor and any(m.get("name") for m in monitor):
        free = list(range(len(monitor)))
        for i, d in enumerate(devices):
            dn = _norm_name(d.get("name"))
            cands = [j for j in free
                     if monitor[j].get("name")
                     and (_norm_name(monitor[j]["name"]) in dn or dn in _norm_name(monitor[j]["name"]))]
            if not cands:
                continue
            cands.sort(key=lambda j: abs((monitor[j].get("total_mib") or 0) - d.get("total_mib", 0)))
            pairs[i] = cands[0]
            free.remove(cands[0])
    elif monitor and len(monitor) == len(devices):
        pairs = {i: i for i in range(len(devices))}

    for i, d in enumerate(devices):
        j = pairs.get(i)
        if j is None:
            d["used_mib"] = max(d["total_mib"] - d.get("free_mib", 0), 0)
            d["vram_live"] = False
            continue
        m = monitor[j]
        used = m["used_mib"]
        ded_total = m.get("total_mib") or 0
        if (m.get("shared_mib") is not None and ded_total
                and d.get("total_mib", 0) > ded_total * 1.5):
            used = (m.get("dedicated_mib") or 0) + (m.get("shared_mib") or 0)
        d["used_mib"] = used
        d["dedicated_mib"] = m.get("dedicated_mib")
        d["shared_mib"] = m.get("shared_mib")
        d["raw_id"] = m.get("raw_id")
        d["vram_live"] = True
    return devices


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
