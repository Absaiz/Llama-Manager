# -*- coding: utf-8 -*-
"""api/tools.py — /api/estimate (VRAM + capacidades)"""
import re
from pathlib import Path

from flask import Blueprint, jsonify, request

from config import KV_BYTES, LOGS_DIR, MODELS_DIR
from gguf_meta import get_cached_meta, get_cached_full, infer_capabilities, split_layer_bytes, read_gguf_meta
from instances import engine_exe, find_draft_sidecars, get_engine, list_devices
from system_monitor import SYS, attach_live_usage

GIB = 1024 ** 3
VRAM_SAFETY_MARGIN_GB = 0.8   # margen que dejamos libre por GPU (contexto de escritorio, fragmentación)
GPU_FIXED_OVERHEAD_GB  = 0.6  # colchón extra fijo por GPU en repartos multi-GPU (buffer de cómputo/contexto no perfectamente proporcional al -ts)
RAM_SAFETY_MARGIN_GB  = 2.0   # margen que dejamos libre en RAM del sistema

tools_bp = Blueprint("tools", __name__)


def _get_calibration(model_path: Path):
    """Busca calibración KV en logs anteriores para el mismo modelo, del
    motor actualmente activo. Vulkan y ROCm reservan buffers de cómputo de
    tamaños distintos (rocBLAS/hipBLASLt necesitan workspace propio que
    Vulkan no usa) — mezclar calibraciones entre motores subestimaría la
    VRAM real al cambiar de uno a otro, así que solo se acepta un log si su
    CMD registrado corresponde al exe del motor activo."""
    exe_str = str(engine_exe(get_engine()))
    try:
        logs = sorted(LOGS_DIR.glob("port*.log"),
                      key=lambda p: p.stat().st_mtime, reverse=True)[:40]
    except Exception:
        return None
    for lf in logs:
        try:
            text = lf.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        first = text.split("\n", 1)[0]
        if str(model_path) not in first or exe_str not in first:
            continue
        mctx = re.search(r"\s-c\s+(\d+)", first)
        if not mctx:
            continue
        kv_mib = 0.0
        for pat in (r"KV self size\s*=\s*([\d.]+)\s*MiB",
                    r"llama_kv_cache[^\n]*?size\s*=\s*([\d.]+)\s*MiB"):
            vals = re.findall(pat, text)
            if vals:
                kv_mib = sum(float(x) for x in vals)
                break
        if kv_mib <= 0:
            continue
        u_ctk  = re.search(r"-ctk\s+(\S+)", first)
        u_ctv  = re.search(r"-ctv\s+(\S+)", first)
        comp   = sum(float(x) for x in re.findall(
            r"compute buffer size\s*=\s*([\d.]+)\s*MiB", text))
        return {
            "ctx": int(mctx.group(1)), "kv_mib": kv_mib,
            "k": u_ctk.group(1) if u_ctk else "f16",
            "v": u_ctv.group(1) if u_ctv else "f16",
            "comp_mib": comp,
        }
    return None


def _has_embedded_mtp(meta) -> bool:
    """¿El GGUF trae capas MTP propias (nextn)? Qwen3.x-MTP las guarda como
    <arch>.nextn_predict_layers > 0."""
    if not meta:
        return False
    arch = meta.get("general.architecture", "")
    try:
        return int(meta.get(f"{arch}.nextn_predict_layers") or 0) > 0
    except (TypeError, ValueError):
        return False


@tools_bp.route("/api/estimate")
def api_estimate():
    path_str = request.args.get("model", "")
    ctx  = int(request.args.get("ctx", 32768))
    ctk  = request.args.get("ctk", "f16")
    ctv  = request.args.get("ctv", "f16")
    ngl  = int(request.args.get("ngl", 999))
    p    = Path(path_str)
    if not p.exists():
        return jsonify({"error": "modelo no encontrado"})

    # Tamaño en disco (soporta multi-shard)
    mp = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", p.name)
    if mp:
        weights = sum(q.stat().st_size for q in p.parent.glob(
            p.name.replace(f"-{mp.group(1)}-of-", "-*-of-")
        ))
        if weights == 0:
            weights = p.stat().st_size
    else:
        weights = p.stat().st_size

    ncpumoe = int(request.args.get("ncpumoe", 0) or 0)
    meta, tensors = get_cached_full(p)
    model_max_ctx = None
    capabilities  = infer_capabilities(meta)
    kv = None
    n_layer = None
    is_moe = False
    n_experts = None

    if meta:
        arch      = meta.get("general.architecture", "")
        n_layer   = meta.get(f"{arch}.block_count")
        n_embd    = meta.get(f"{arch}.embedding_length")
        n_head    = meta.get(f"{arch}.attention.head_count")
        n_head_kv = meta.get(f"{arch}.attention.head_count_kv") or n_head
        model_max_ctx = meta.get(f"{arch}.context_length")
        n_experts = meta.get(f"{arch}.expert_count")

        if n_layer and n_embd and n_head:
            k_len = meta.get(f"{arch}.attention.key_length") or n_embd // n_head
            v_len = meta.get(f"{arch}.attention.value_length") or n_embd // n_head
            # Arquitecturas híbridas (atención + SSM/Mamba, p.ej. Qwen3.5-MoE
            # y derivados como Ornith) solo tienen KV cache que crece con el
            # contexto en 1 de cada `full_attention_interval` capas — el
            # resto son capas SSM con estado de tamaño fijo. Sin esta clave,
            # asumimos atención completa en todas las capas (comportamiento
            # anterior, correcto para modelos densos/MoE "normales").
            fai = meta.get(f"{arch}.full_attention_interval") or 1
            kv_layers = max(1, round(n_layer / fai))
            kv = ctx * n_head_kv * (k_len * KV_BYTES.get(ctk, 2.0)
                                    + v_len * KV_BYTES.get(ctv, 2.0)) * kv_layers

    # ── Reparto real por tensores (si pudimos parsearlos) ───────────────────
    # gpu_frac aparte para "fixed" (atención/embeddings/norms, sigue ngl) y
    # para "moe" (expertos routed, sigue ngl PERO además puede sacarse a CPU
    # con --n-cpu-moe independientemente de ngl). Si no hay tensores (fallo
    # de parseo) caemos al cálculo antiguo por nº de capas.
    tl_layer, per_layer, global_bytes, is_moe = split_layer_bytes(tensors) if tensors else (0, {}, 0, False)

    if is_moe and tl_layer:
        total_fixed = sum(r["fixed"] for r in per_layer.values()) + global_bytes
        total_moe   = sum(r["moe"] for r in per_layer.values())
        layers_on_gpu = min(ngl, tl_layer)
        # ncpumoe saca los N *primeros* bloques de expertos a CPU (orden que usa llama.cpp)
        moe_layers_cpu = min(max(ncpumoe, 0), tl_layer)
        moe_bytes_cpu = sum(
            r["moe"] for idx, r in per_layer.items() if idx < moe_layers_cpu
        )
        moe_bytes_gpu = total_moe - moe_bytes_cpu
        # Si además ngl<n_layer, las capas fuera de ngl (fixed+moe) también caen a CPU
        if layers_on_gpu < tl_layer:
            fixed_bytes_gpu = sum(r["fixed"] for idx, r in per_layer.items() if idx < layers_on_gpu) + global_bytes
            moe_bytes_gpu = sum(
                r["moe"] for idx, r in per_layer.items()
                if idx < layers_on_gpu and idx >= moe_layers_cpu
            )
        else:
            fixed_bytes_gpu = total_fixed
        weights_gpu = fixed_bytes_gpu + moe_bytes_gpu
        weights_total = total_fixed + total_moe
        gpu_frac = (weights_gpu / weights_total) if weights_total else 1.0
    else:
        gpu_frac = 1.0
        if n_layer and ngl < n_layer:
            gpu_frac = ngl / n_layer
        weights_gpu = None  # se calcula abajo con "weights" (bytes en disco)

    calibrated = False
    calib = _get_calibration(p)
    if calib:
        f_old = KV_BYTES.get(calib["k"], 2.0) + KV_BYTES.get(calib["v"], 2.0)
        f_new = KV_BYTES.get(ctk, 2.0) + KV_BYTES.get(ctv, 2.0)
        kv    = calib["kv_mib"] * (1024**2) * (ctx / calib["ctx"]) * (f_new / f_old)
        calibrated = True

    # La fracción de KV cache en GPU sigue las capas de ATENCIÓN (ngl), no el
    # reparto de expertos: --n-cpu-moe deja la atención/KV de todas las capas
    # en GPU aunque saque expertos a CPU, así que no hay que penalizarla igual.
    attn_frac = min(ngl, tl_layer) / tl_layer if (is_moe and tl_layer) else gpu_frac

    if weights_gpu is None:  # fallback sin tensores parseados: reparto por nº de capas
        weights_gpu = weights * gpu_frac
    kv_gpu = (kv or 0) * attn_frac
    if calibrated and calib["comp_mib"] > 0:
        buffers = calib["comp_mib"] * (1024**2)
    else:
        buffers = (0.5 + (ctx / 131072) * 0.7) * (1024**3)
    total_gpu = weights_gpu + kv_gpu + buffers
    weights_cpu = (weights_total - weights_gpu) if is_moe else (weights * (1 - gpu_frac))

    gb = lambda x: round(x / (1024**3), 2)
    return jsonify({
        "weights_gb":       gb(weights_gpu),
        "kv_gb":            gb(kv_gpu) if kv else None,
        "buffers_gb":       gb(buffers),
        "total_gb":         gb(total_gpu),
        "n_layer":          n_layer or tl_layer or None,
        "is_moe":           is_moe,
        "n_experts":        n_experts,
        "ncpumoe_used":     ncpumoe if is_moe else None,
        "kv_known":         kv is not None,
        "calibrated":       calibrated,
        "cpu_gb":           gb(max(weights_cpu, 0)) if is_moe else gb((weights + (kv or 0)) * (1 - gpu_frac)),
        "model_max_ctx":    model_max_ctx,
        "capabilities":     capabilities,
        "weights_gpu_gb":   gb(weights_gpu),
        "buffers_gb_base":  gb(buffers),
        "kv_per_tok_bytes": round(kv / ctx, 4) if (kv and ctx) else None,
        "gpu_frac":         gpu_frac,
        "arch":             (meta or {}).get("general.architecture"),
        "has_mtp":          _has_embedded_mtp(meta),
        "drafts":           find_draft_sidecars(str(p)),
    })


@tools_bp.route("/api/autosplit")
def api_autosplit():
    """
    Calcula un reparto recomendado de capas/expertos entre las GPUs
    seleccionadas (1 o 2) y la RAM del sistema. La VRAM libre se lee vía
    `llama-server --list-devices` (Vulkan) — la misma runtime y el mismo
    espacio de índices (Vulkan0/Vulkan1) que usará el proceso real al
    arrancar — en vez de rocm-smi/WMI, que en iGPU/APU con memoria
    compartida puede reportar solo la partición dedicada fija y subestimar
    lo que Vulkan tiene realmente disponible.

    Sigue siendo una estimación de tamaño de tensores + memoria libre en el
    momento de la consulta — no sustituye una prueba real de arranque.
    """
    path_str  = request.args.get("model", "")
    ctx       = int(request.args.get("ctx", 32768))
    ctk       = request.args.get("ctk", "f16")
    ctv       = request.args.get("ctv", "f16")
    device_ids = [d for d in (request.args.get("devices", "") or "").split(",") if d]
    p = Path(path_str)
    if not p.exists():
        return jsonify({"error": "modelo no encontrado"})

    weights = p.stat().st_size
    meta, tensors = get_cached_full(p)
    if not meta or not tensors:
        return jsonify({"error": "no se pudieron leer los tensores del modelo (GGUF no parseable)"})

    arch      = meta.get("general.architecture", "")
    n_embd    = meta.get(f"{arch}.embedding_length")
    n_head    = meta.get(f"{arch}.attention.head_count")
    n_head_kv = meta.get(f"{arch}.attention.head_count_kv") or n_head
    tl_layer, per_layer, global_bytes, is_moe = split_layer_bytes(tensors)
    n_layer = tl_layer or meta.get(f"{arch}.block_count") or 0
    if not n_layer:
        return jsonify({"error": "no se detectaron capas (blk.N.*) en el modelo"})

    kv_per_layer = 0
    if n_embd and n_head:
        k_len = meta.get(f"{arch}.attention.key_length") or n_embd // n_head
        v_len = meta.get(f"{arch}.attention.value_length") or n_embd // n_head
        fai = meta.get(f"{arch}.full_attention_interval") or 1
        kv_layers = max(1, round(n_layer / fai))
        kv_per_layer = ctx * n_head_kv * (k_len * KV_BYTES.get(ctk, 2.0) + v_len * KV_BYTES.get(ctv, 2.0))
    kv_total = kv_per_layer * kv_layers if (n_embd and n_head) else 0
    buffers  = (0.5 + (ctx / 131072) * 0.7) * GIB

    # ── VRAM/RAM libres reales, con margen de seguridad ─────────────────────
    # El TOTAL lo sacamos de --list-devices (Vulkan) — coincide con lo que
    # reporta el Administrador de tareas como memoria de GPU total.
    # El USADO lo sacamos de system_monitor (WMI dedicada+compartida, o
    # rocm-smi) en vez del "free" que reporta el propio Vulkan: en pruebas
    # reales, Vulkan devolvía casi toda la memoria como libre mientras el
    # Administrador de tareas mostraba la GPU casi llena — usar el "free"
    # de Vulkan directamente producía presupuestos muy optimistas.
    all_devices, dev_err = list_devices()
    if dev_err or not all_devices:
        return jsonify({"error": f"no se pudieron listar dispositivos Vulkan: {dev_err or 'sin resultados'}"})

    # Total de Vulkan (fiable) + usado de WMI (fiable, ve otros procesos —
    # Vulkan aislado no). Ver nota igual en /api/state.
    attach_live_usage(all_devices)
    gpu_budgets = []  # [(device_id, free_bytes)]
    for i, d in enumerate(all_devices):
        if device_ids and d["id"] not in device_ids:
            continue
        free_mib = max(d["total_mib"] - d["used_mib"], 0)
        free_b = max(free_mib * 1024 * 1024 - VRAM_SAFETY_MARGIN_GB * GIB, 0)
        gpu_budgets.append((d["id"], free_b))
    total_gpu_budget = sum(b for _, b in gpu_budgets)

    ram_free_mib = max(SYS.get("ram_total", 0) - SYS.get("ram_used", 0), 0)
    ram_budget = max(ram_free_mib * 1024 * 1024 - RAM_SAFETY_MARGIN_GB * GIB, 0)

    if not gpu_budgets:
        return jsonify({"error": "no hay lecturas de GPU disponibles (revisa system_monitor / rocm-smi)"})

    result = {
        "n_layer": n_layer, "is_moe": is_moe,
        "gpu_budgets_gb": [(d, round(b / GIB, 2)) for d, b in gpu_budgets],
        "ram_budget_gb": round(ram_budget / GIB, 2),
    }

    if is_moe:
        total_fixed = sum(r["fixed"] for r in per_layer.values()) + global_bytes
        total_moe   = sum(r["moe"] for r in per_layer.values())
        avg_moe = total_moe / n_layer if n_layer else 0

        gpu_for_weights = total_gpu_budget - kv_total - buffers
        if gpu_for_weights < total_fixed:
            result.update({
                "fits": False,
                "recommended_ngl": 999,
                "recommended_ncpumoe": n_layer,
                "recommended_tensor_split": None,
                "weights_gpu_gb": round(total_fixed / GIB, 2),
                "weights_cpu_gb": round(total_moe / GIB, 2),
                "kv_gb": round(kv_total / GIB, 2),
                "reason": (f"Ni siquiera la parte fija (atención/embeddings, "
                           f"{round(total_fixed/GIB,1)} GB) más KV ({round(kv_total/GIB,1)} GB) "
                           f"y buffers ({round(buffers/GIB,1)} GB) caben en "
                           f"{round(total_gpu_budget/GIB,1)} GB de VRAM libre. "
                           f"Baja ctx, usa una cuantización más pequeña, o revisa si el "
                           f"reparto de VRAM/RAM en BIOS deja tan poco a la GPU."),
            })
            return jsonify(result)

        remaining_for_moe = gpu_for_weights - total_fixed
        n_moe_layers_gpu = int(remaining_for_moe // avg_moe) if avg_moe > 0 else n_layer
        n_moe_layers_gpu = max(0, min(n_layer, n_moe_layers_gpu))
        ncpumoe_reco = n_layer - n_moe_layers_gpu
        moe_bytes_cpu = ncpumoe_reco * avg_moe
        weights_gpu = total_fixed + (total_moe - moe_bytes_cpu)

        ts = None
        per_gpu_short = None
        if len(gpu_budgets) >= 2:
            tot = sum(b for _, b in gpu_budgets) or 1
            ts_ratios = [b / tot for _, b in gpu_budgets]
            # La comprobación de arriba solo mira la SUMA de VRAM libre entre
            # las dos GPUs — pero con -ts, cada tarjeta recibe su porción
            # proporcional y tiene que caber en SU PROPIO presupuesto, no en
            # el conjunto. Dos GPUs de tamaño distinto (p.ej. 16GB + 24GB) no
            # comparten memoria entre sí; que sobre en una no libera nada en
            # la otra. Si algún dispositivo no llega con este ncpumoe, subo
            # el nº de capas offloadeadas hasta que quepa en TODAS a la vez.
            for _ in range(n_layer + 1):
                per_gpu_short = None
                need_total = weights_gpu + kv_total + buffers
                for (did, budget), ratio in zip(gpu_budgets, ts_ratios):
                    # + colchón fijo por dispositivo: el buffer de cómputo/
                    # contexto de Vulkan por GPU no escala perfectamente con
                    # el ratio de -ts — en pruebas reales, un reparto que
                    # "cabía" por ~180MB en el cálculo proporcional puro
                    # reventó igualmente por OutOfDeviceMemory.
                    need_i = need_total * ratio + GPU_FIXED_OVERHEAD_GB * GIB
                    if need_i > budget:
                        per_gpu_short = (did, need_i - budget)
                        break
                if per_gpu_short is None or ncpumoe_reco >= n_layer:
                    break
                ncpumoe_reco += 1
                moe_bytes_cpu = ncpumoe_reco * avg_moe
                weights_gpu = total_fixed + (total_moe - moe_bytes_cpu)
            ts = ",".join(f"{round(r, 3)}" for r in ts_ratios)

        fits_ram = moe_bytes_cpu <= ram_budget
        fits_per_gpu = per_gpu_short is None

        result.update({
            "recommended_ngl": 999,
            "recommended_ncpumoe": ncpumoe_reco,
            "recommended_tensor_split": ts,
            "weights_gpu_gb": round(weights_gpu / GIB, 2),
            "weights_cpu_gb": round(moe_bytes_cpu / GIB, 2),
            "kv_gb": round(kv_total / GIB, 2),
            "fits": bool(fits_ram and fits_per_gpu),
            "reason": (
                f"Ajustado subiendo ncpumoe porque {per_gpu_short[0]} por sí sola no llega "
                f"(le faltarían ~{round(per_gpu_short[1]/GIB,2)} GB con el reparto proporcional) "
                f"aunque la suma total de VRAM libre sí alcanzara."
                if per_gpu_short and ncpumoe_reco < n_layer else
                f"Ni con todos los expertos offloadeados cabe en alguna GPU individualmente "
                f"— revisa el reparto -ts o baja ctx."
                if per_gpu_short else
                (None if fits_ram else (
                    f"El reparto cabe en VRAM pero los {round(moe_bytes_cpu/GIB,1)} GB de "
                    f"expertos que caerían a RAM superan tu presupuesto de RAM libre "
                    f"({round(ram_budget/GIB,1)} GB). Sube ncpumoe manualmente igualmente "
                    f"si aceptas usar swap, o baja ctx para liberar más VRAM."
                ))
            ),
        })
    else:
        # Modelo denso: reparto clásico por nº de capas completas
        avg_layer = (weights - global_bytes) / n_layer if n_layer else weights
        gpu_for_weights = total_gpu_budget - kv_total - buffers
        n_gpu_layers = int(gpu_for_weights // avg_layer) if avg_layer > 0 else 0
        n_gpu_layers = max(0, min(n_layer, n_gpu_layers))
        weights_gpu = global_bytes + n_gpu_layers * avg_layer
        cpu_bytes = weights - weights_gpu
        fits_ram = cpu_bytes <= ram_budget

        ts = None
        if len(gpu_budgets) >= 2:
            tot = sum(b for _, b in gpu_budgets) or 1
            ts = ",".join(f"{round(b/tot, 3)}" for _, b in gpu_budgets)

        result.update({
            "recommended_ngl": n_gpu_layers,
            "recommended_ncpumoe": None,
            "recommended_tensor_split": ts,
            "weights_gpu_gb": round(weights_gpu / GIB, 2),
            "weights_cpu_gb": round(max(cpu_bytes, 0) / GIB, 2),
            "kv_gb": round(kv_total / GIB, 2),
            "fits": bool(n_gpu_layers >= n_layer or fits_ram),
            "reason": None if (n_gpu_layers >= n_layer or fits_ram) else (
                "Modelo denso: las capas que no caben en VRAM tendrían que ir a RAM "
                "(mmap, sin --n-cpu-moe porque no es MoE) — será notablemente más lento."
            ),
        })

    return jsonify(result)


def _embd_info(path: Path):
    """Lee architecture + embedding_length de un GGUF (modelo o mmproj)."""
    meta = get_cached_meta(path)
    if not meta:
        return None, None
    arch = meta.get("general.architecture", "")
    embd = meta.get(f"{arch}.embedding_length")
    return arch, embd


@tools_bp.route("/api/mmproj_check")
def api_mmproj_check():
    """
    Compara embedding_length entre el modelo de texto y los mmproj
    disponibles, para detectar antes de arrancar el típico error
    "mismatch between text model (n_embd=X) and mmproj (n_embd=Y)" —
    normalmente causado por dejar el mmproj de un perfil/modelo anterior al
    cambiar de modelo. También sugiere el mmproj de la misma carpeta que el
    modelo, si existe.
    """
    model_str = request.args.get("model", "")
    current_str = request.args.get("current", "")
    mp = Path(model_str)
    if not mp.exists():
        return jsonify({"error": "modelo no encontrado"})

    model_arch, model_embd = _embd_info(mp)

    candidates = []
    try:
        for c in sorted(mp.parent.glob("mmproj*.gguf")):
            c_arch, c_embd = _embd_info(c)
            # Si no pudimos leer el embd del mmproj (claves de metadatos
            # distintas a las del modelo de texto, p.ej. arquitectura "clip"),
            # NO lo marcamos como incompatible — sería un falso positivo.
            # Lo dejamos en null: "no verificado", no "mal".
            compatible = None if c_embd is None else (model_embd is not None and c_embd == model_embd)
            candidates.append({
                "path": str(c), "name": c.name,
                "embd": c_embd,
                "compatible": compatible,
            })
    except Exception:
        pass

    current_embd = None
    current_compatible = None
    if current_str:
        cp = Path(current_str)
        if cp.exists():
            _, current_embd = _embd_info(cp)
            if current_embd is not None:
                current_compatible = (model_embd is not None and current_embd == model_embd)
            # si current_embd es None, current_compatible se queda en None (desconocido)

    # Para autorrellenar/sugerir preferimos uno confirmado compatible; si no
    # hay ninguno confirmado pero hay exactamente un candidato en la carpeta
    # del modelo (caso típico: cada modelo trae su propio mmproj), lo
    # sugerimos igualmente aunque no hayamos podido verificar el n_embd —
    # el nombre-en-la-misma-carpeta ya es una señal razonable.
    suggestion = next((c["path"] for c in candidates if c["compatible"] is True), None)
    if not suggestion and len(candidates) == 1 and candidates[0]["compatible"] is None:
        suggestion = candidates[0]["path"]

    return jsonify({
        "model_arch": model_arch,
        "model_embd": model_embd,
        "candidates": candidates,
        "current_embd": current_embd,
        "current_compatible": current_compatible,
        "suggested_mmproj": suggestion,
    })


@tools_bp.route("/api/meta_debug")
def api_meta_debug():
    """
    Vuelca TODOS los metadatos KV crudos de un GGUF (sin filtrar por wanted()),
    para depurar cosas como el patrón de sliding-window por capa sin tener que
    rebuscar en los logs de arranque. Uso: /api/meta_debug?model=<ruta>
    """
    model_str = request.args.get("model", "")
    p = Path(model_str)
    if not p.exists():
        return jsonify({"error": "modelo no encontrado"})
    try:
        meta_raw = read_gguf_meta(p, want_all=True)
    except Exception as e:
        return jsonify({"error": f"fallo al parsear el GGUF: {e}"})
    if not meta_raw:
        return jsonify({"error": "no se pudo parsear el GGUF"})
    # Resaltamos las claves que probablemente importan para KV/atención
    interesting = {k: v for k, v in meta_raw.items()
                   if any(t in k.lower() for t in
                          ("sliding", "window", "layer_types", "swa", "kv",
                           "block_count", "head_count", "embedding_length"))}
    return jsonify({"all_keys": sorted(meta_raw.keys()), "meta": meta_raw, "interesting": interesting})
