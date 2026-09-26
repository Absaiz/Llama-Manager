# -*- coding: utf-8 -*-
"""
gguf_meta.py — Lectura de metadatos GGUF e inferencia de capacidades del modelo.
Lee: context_length, block_count, attention heads, chat_template, capabilities, tags.
También lee la lista de tensores (nombre, dims, tipo) para calcular tamaños
reales por capa y separar expertos MoE "routed" (offloadeables a CPU vía
--n-cpu-moe) de todo lo que debe quedarse siempre en GPU (atención, embeddings,
experto compartido, norms).
"""
import re
import struct

META_CACHE = {}
TENSOR_CACHE = {}

# Tabla de tamaños GGML: id -> (block_size_elementos, bytes_por_bloque)
# Fuente: ggml.c (tabla type_traits), estable desde hace tiempo en llama.cpp.
GGML_TYPE_SIZES = {
    0: (1, 4),      # F32
    1: (1, 2),      # F16
    2: (32, 18),    # Q4_0
    3: (32, 20),    # Q4_1
    6: (32, 22),    # Q5_0
    7: (32, 24),    # Q5_1
    8: (32, 34),    # Q8_0
    9: (32, 36),    # Q8_1
    10: (256, 84),  # Q2_K
    11: (256, 110), # Q3_K
    12: (256, 144), # Q4_K
    13: (256, 176), # Q5_K
    14: (256, 210), # Q6_K
    15: (256, 292), # Q8_K
    16: (256, 66),  # IQ2_XXS
    17: (256, 74),  # IQ2_XS
    18: (256, 98),  # IQ3_XXS
    19: (256, 50),  # IQ1_S
    20: (32, 18),   # IQ4_NL
    21: (256, 110), # IQ3_S
    22: (256, 82),  # IQ2_S
    23: (256, 136), # IQ4_XS
    24: (1, 1),     # I8
    25: (1, 2),     # I16
    26: (1, 4),     # I32
    27: (1, 8),     # I64
    28: (1, 8),     # F64
    29: (256, 56),  # IQ1_M
    30: (1, 2),     # BF16
    39: (256, 74),  # MXFP4 (bloques 32, pero se ajusta abajo si hiciera falta)
}


def _tensor_bytes(dims, ggml_type):
    """Tamaño en bytes de un tensor dado sus dimensiones y su tipo GGML."""
    n_elem = 1
    for d in dims:
        n_elem *= d
    block, tsize = GGML_TYPE_SIZES.get(ggml_type, (1, 4))  # fallback F32
    return (n_elem // block) * tsize if n_elem % block == 0 else (n_elem / block) * tsize


def read_gguf_meta(path, want_tensors=False, want_all=False):
    """
    Lee los metadatos relevantes de un fichero GGUF sin cargarlo en memoria.
    Si want_tensors=True, además parsea la sección de tensor-info (justo
    después de los KV) y devuelve (meta, tensors) en vez de solo meta.
    tensors es una lista de dicts: {name, dims, ggml_type, bytes}
    Si want_all=True, ignora el filtro wanted() y devuelve TODAS las claves
    KV del fichero (más lento/pesado, pero necesario para depurar campos que
    aún no están en la lista blanca, p.ej. sliding_window/kv_unified).
    """
    SIZES = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    FMTS  = {0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<B",10:"<Q",11:"<q",12:"<d"}

    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            return (None, []) if want_tensors else None
        f.read(4)   # version
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        def rstr():
            ln = struct.unpack("<Q", f.read(8))[0]
            return f.read(ln).decode("utf-8", "replace")

        meta = {}

        def wanted(key):
            return (
                key == "general.architecture"
                or key == "general.name"
                or key == "general.tags"
                or key == "general.capabilities"
                or key == "tokenizer.chat_template"
                or key.endswith(".block_count")
                or key.endswith(".context_length")
                or key.endswith(".attention.head_count")
                or key.endswith(".attention.head_count_kv")
                or key.endswith(".embedding_length")
                or key.endswith(".attention.key_length")
                or key.endswith(".attention.value_length")
                or key.endswith(".expert_count")
                or key.endswith(".expert_used_count")
                or key.endswith(".full_attention_interval")
            )

        try:
            for _ in range(n_kv):
                key = rstr()
                t   = struct.unpack("<I", f.read(4))[0]
                w   = True if want_all else wanted(key)
                if t == 9:  # array
                    et = struct.unpack("<I", f.read(4))[0]
                    n  = struct.unpack("<Q", f.read(8))[0]
                    if et == 8:  # array de strings
                        vals = []
                        for _ in range(n):
                            ln = struct.unpack("<Q", f.read(8))[0]
                            s  = f.read(ln).decode("utf-8", "replace")
                            if w:
                                vals.append(s)
                        if w and vals:
                            meta[key] = vals
                    elif et in SIZES:
                        if w and 0 < n < 100000:
                            vals = [struct.unpack(FMTS[et], f.read(SIZES[et]))[0] for _ in range(n)]
                            meta[key] = vals if want_all else max(vals)
                        else:
                            f.seek(SIZES[et] * n, 1)
                    else:
                        break
                elif t == 8:
                    ln = struct.unpack("<Q", f.read(8))[0]
                    if w:
                        meta[key] = f.read(ln).decode("utf-8", "replace")
                    else:
                        f.seek(ln, 1)
                else:
                    v = struct.unpack(FMTS[t], f.read(SIZES[t]))[0]
                    if w:
                        meta[key] = v
        except Exception:
            if want_tensors:
                return meta, []
            return meta

        if not want_tensors:
            return meta

        # ── Sección de tensor-info: nombre, nº dims, dims, tipo ggml, offset ──
        tensors = []
        try:
            GGUF_ALIGNMENT = int(meta.get("general.alignment", 32) or 32)
            for _ in range(n_tensors):
                name = rstr()
                n_dims = struct.unpack("<I", f.read(4))[0]
                dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
                ggml_type = struct.unpack("<I", f.read(4))[0]
                struct.unpack("<Q", f.read(8))[0]  # offset (no lo necesitamos)
                tensors.append({
                    "name": name, "dims": dims, "ggml_type": ggml_type,
                    "bytes": _tensor_bytes(dims, ggml_type),
                })
            # No hace falta seguir hasta el padding/datos: ya tenemos lo que hace falta.
        except Exception:
            tensors = tensors or []

        return meta, tensors


def split_layer_bytes(tensors):
    """
    Agrupa los tensores por capa (blk.N.*) y separa, por capa:
      - moe_bytes: tensores de expertos routed (offloadeables con --n-cpu-moe)
      - fixed_bytes: todo lo demás de esa capa (atención, norms, experto compartido)
    Tensores fuera de bloques (token_embd, output, output_norm...) van en
    "global_bytes" — siempre residen igual, no dependen de ngl/ncpumoe.
    Devuelve: (n_layer_detectado, {layer_idx: {"moe":B, "fixed":B}}, global_bytes, is_moe)
    """
    per_layer = {}
    global_bytes = 0
    is_moe = False
    blk_re = re.compile(r"^blk\.(\d+)\.(.+)$")
    for t in tensors:
        m = blk_re.match(t["name"])
        if not m:
            global_bytes += t["bytes"]
            continue
        idx = int(m.group(1))
        sub = m.group(2)
        rec = per_layer.setdefault(idx, {"moe": 0, "fixed": 0})
        # Convención de llama.cpp para expertos routed en MoE
        if "_exps" in sub or ".exps" in sub:
            rec["moe"] += t["bytes"]
            is_moe = True
        else:
            rec["fixed"] += t["bytes"]
    n_layer = (max(per_layer.keys()) + 1) if per_layer else 0
    return n_layer, per_layer, global_bytes, is_moe


def infer_capabilities(meta):
    """
    Infiere Vision / Tools / Reasoning desde los metadatos GGUF.
    Fuentes en orden de fiabilidad:
      1. general.capabilities
      2. general.tags
      3. tokenizer.chat_template (más fiable)
      4. Nombre del modelo (heurístico)
    """
    if not meta:
        return []

    caps = set()

    # 1. general.capabilities
    raw_caps = meta.get("general.capabilities", "")
    if isinstance(raw_caps, list):
        for c in raw_caps:
            cl = c.lower()
            if "vision" in cl or "image" in cl: caps.add("vision")
            if "tool" in cl or "function" in cl: caps.add("tools")
            if "reason" in cl or "think" in cl:  caps.add("reasoning")
    elif isinstance(raw_caps, str) and raw_caps:
        cl = raw_caps.lower()
        if "vision" in cl or "image" in cl: caps.add("vision")
        if "tool" in cl or "function" in cl: caps.add("tools")
        if "reason" in cl or "think" in cl:  caps.add("reasoning")

    # 2. general.tags
    tags = meta.get("general.tags", "")
    tag_str = " ".join(tags).lower() if isinstance(tags, list) else str(tags).lower()
    if "vision" in tag_str or "image" in tag_str or "vl" in tag_str: caps.add("vision")
    if "tool" in tag_str or "function" in tag_str:                    caps.add("tools")
    if "reason" in tag_str or "thinking" in tag_str:                  caps.add("reasoning")

    # 3. tokenizer.chat_template
    tmpl = meta.get("tokenizer.chat_template", "")
    if isinstance(tmpl, str) and tmpl:
        tl = tmpl.lower()
        if any(x in tl for x in ["<image>", "image_url", "<img", "pixel_values",
                                   "image_pad", "vision", "<|image_pad|>", "image_token"]):
            caps.add("vision")
        if any(x in tl for x in ["tool_call", "function_call", "<tool_call>",
                                   "tool_use", "<tool>", "available_tools",
                                   "✿function✿", "tool_calls"]):
            caps.add("tools")
        if any(x in tl for x in ["<think>", "<thinking>", "reasoning_content",
                                   "enable_thinking", "budget_tokens"]):
            caps.add("reasoning")

    # 4. Nombre heurístico
    name = str(meta.get("general.name", "")).lower()
    arch = str(meta.get("general.architecture", "")).lower()
    if any(x in name for x in ["-vl", "vision", "pixtral", "llava", "qwen2-vl",
                                 "qwen2.5-vl", "minicpm-v", "internvl", "cogvlm"]):
        caps.add("vision")
    if "qwen" in arch and "vl" in name:
        caps.add("vision")

    return sorted(caps)


def get_cached_meta(path):
    """Devuelve los metadatos cacheados o los lee del disco (sin tensores)."""
    path_str = str(path)
    if path_str not in META_CACHE:
        try:
            META_CACHE[path_str] = read_gguf_meta(path)
        except Exception:
            META_CACHE[path_str] = None
    return META_CACHE[path_str]


def _shard_family(path):
    """
    Si el fichero es una de varias partes ('-NNNNN-of-MMMMM.gguf'), devuelve
    TODOS los shards hermanos ordenados por número (así llama.cpp los carga:
    se indica la parte 1 y el resto van solos). Un GGUF simple → [path].

    IMPORTANTE: en un set sharded, el header de CADA shard solo declara sus
    PROPIOS tensores (p.ej. la parte 00001/00003 de Qwen3.8 Flash-Next trae
    solo metadatos y 0 tensores; las 402+822 restantes viven en las partes
    2 y 3). Leer únicamente el fichero seleccionado daría "GGUF no parseable"
    o un reparto MoE erróneo — hay que concatenar los tensores de todas las
    partes y tomar la metadatos del shard que las traiga.
    """
    from pathlib import Path as _P
    p = path if isinstance(path, _P) else _P(str(path))
    mp = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", p.name)
    if not mp:
        return [p]
    pattern = p.name.replace(f"-{mp.group(1)}-of-", "-*-of-")
    siblings = sorted(p.parent.glob(pattern), key=lambda s: s.name)
    return siblings or [p]


def get_cached_full(path):
    """
    Devuelve (meta, tensors) cacheados o los lee del disco.
    Si el modelo está en varios shards, concatena los tensores de TODAS las
    partes y usa los metadatos del shard que los tenga (véase _shard_family).
    Usa un caché aparte de get_cached_meta porque incluye la lista de
    tensores (más caro de parsear, pero necesario para el reparto MoE).
    Se invalida por tamaño+mtime de TODOS los shards, no solo por ruta, para
    que un modelo re-descargado con el mismo nombre no devuelva datos obsoletos.
    """
    path_str = str(path)
    shards = _shard_family(path)
    try:
        cache_key = tuple((str(s), s.stat().st_size, s.stat().st_mtime) for s in shards)
    except Exception:
        cache_key = (path_str, None, None)
    if TENSOR_CACHE.get("_key") != cache_key or path_str not in TENSOR_CACHE:
        meta, tensors = None, []
        for s in shards:
            try:
                m_i, t_i = read_gguf_meta(s, want_tensors=True)
            except Exception:
                m_i, t_i = None, []
            if (m_i and "general.architecture" in m_i
                    and not (meta and "general.architecture" in meta)):
                meta = m_i
            elif meta is None:
                meta = m_i or {}
            tensors.extend(t_i)
        TENSOR_CACHE.clear()
        TENSOR_CACHE["_key"] = cache_key
        TENSOR_CACHE[path_str] = (meta, tensors)
    return TENSOR_CACHE[path_str]
