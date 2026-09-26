# -*- coding: utf-8 -*-
"""
api/agent.py — Bucle agente con tool use real + persistencia de resultados.
Tools disponibles:
  - web_search(query)          → ddgs real
  - write_file(path, content)  → escribe en LOGS_DIR
  - calculator(expression)     → eval seguro

Persistencia automática al finalizar cada test:
  - logs/agent_results.jsonl   → un JSON por línea, un test por línea
  - logs/agent_bench.csv       → CSV acumulativo para comparar modelos
"""
import csv
import json
import math
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from flask import Blueprint, Response, jsonify, request

from config import LOGS_DIR
from instances import INSTANCES

agent_bp = Blueprint("agent", __name__)

AGENT_RESULTS_FILE = LOGS_DIR / "agent_results.jsonl"
AGENT_BENCH_CSV    = LOGS_DIR / "agent_bench.csv"

# ── Tools reales ──────────────────────────────────────────────────────────────

def _tool_web_search(query: str) -> str:
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=3)
        if not results:
            return "Sin resultados para: " + query
        lines = []
        for r in results:
            lines.append(f"[{r.get('title','')}] {r.get('href','')}\n{r.get('body','')}")
        return "\n\n".join(lines)
    except ImportError:
        return "ERROR: ddgs no instalado. Ejecuta: pip install ddgs"
    except Exception as e:
        return f"ERROR en web_search: {e}"


def _tool_write_file(path: str, content: str) -> str:
    safe_name = Path(path).name
    if not safe_name or ".." in safe_name:
        return "ERROR: nombre de fichero no válido"
    target = LOGS_DIR / safe_name
    try:
        target.write_text(content, encoding="utf-8")
        return f"OK: fichero guardado en {target}"
    except Exception as e:
        return f"ERROR al escribir fichero: {e}"


def _tool_calculator(expression: str) -> str:
    if not re.match(r'^[\d\s\+\-\*\/\.\(\)\%\^eE]+$', expression):
        return "ERROR: expresión no permitida"
    try:
        expr   = expression.replace("^", "**")
        result = eval(expr, {"__builtins__": {}}, {
            "abs": abs, "round": round, "min": min, "max": max,
            "sqrt": math.sqrt, "pi": math.pi, "e": math.e,
        })
        return str(result)
    except Exception as e:
        return f"ERROR en calculator: {e}"


TOOLS = {
    "web_search": _tool_web_search,
    "write_file": _tool_write_file,
    "calculator": _tool_calculator,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Busca información actual en internet usando DuckDuckGo.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Términos de búsqueda"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Escribe contenido en un fichero de texto en el servidor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string", "description": "Nombre del fichero (ej: resultado.txt)"},
                    "content": {"type": "string", "description": "Contenido a escribir"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evalúa expresiones matemáticas. Usa para conversiones y cálculos.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Expresión matemática (ej: 62450 * 0.92)"}},
                "required": ["expression"],
            },
        },
    },
]

AGENT_SYSTEM = """Eres un agente que ejecuta tareas usando herramientas.
REGLAS:
- Usa SOLO las herramientas disponibles: web_search, write_file, calculator.
- No inventes herramientas que no existen.
- Completa la tarea paso a paso usando las herramientas.
- Cuando termines, responde con un resumen de lo que hiciste."""

AGENT_TASK = """Tarea: Busca el precio actual del Bitcoin en USD, conviértelo a EUR \
usando la tasa de cambio actual (búscala también), y guarda el resultado \
en un fichero llamado bitcoin_resultado.txt con el formato:
  Bitcoin: X USD = Y EUR
  Tasa USD/EUR: Z
  Fecha: [fecha actual]"""


# ── Llamada al modelo con timings ─────────────────────────────────────────────

def _call_model(port: int, messages: list) -> tuple[dict, dict]:
    """
    Llama al modelo. Devuelve (response_dict, timings_dict).
    timings_dict puede estar vacío si el modelo no los devuelve.
    """
    payload = {
        "model": "local",
        "messages": messages,
        "tools": TOOLS_SCHEMA,
        "tool_choice": "auto",
        "stream": False,
        "max_tokens": 1024,
    }
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=data, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read().decode("utf-8", "replace"))
            timings = resp.get("timings") or {}
            return resp, timings
    except Exception as e:
        return {"error": str(e)}, {}


def _execute_tool_calls(tool_calls: list) -> list:
    results = []
    for tc in tool_calls:
        fn_name    = tc.get("function", {}).get("name", "")
        fn_args_raw = tc.get("function", {}).get("arguments", "{}")
        tc_id      = tc.get("id", "call_0")
        try:
            fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
        except Exception:
            fn_args = {}
        if fn_name in TOOLS:
            try:
                result = TOOLS[fn_name](**fn_args)
            except Exception as e:
                result = f"ERROR ejecutando {fn_name}: {e}"
        else:
            result = f"ERROR: herramienta '{fn_name}' no existe. Disponibles: {list(TOOLS.keys())}"
        results.append({"role": "tool", "tool_call_id": tc_id, "content": result})
    return results


# ── Persistencia ──────────────────────────────────────────────────────────────

def _save_result(record: dict):
    """
    Guarda el resultado del test en:
      - agent_results.jsonl  (un JSON por línea)
      - agent_bench.csv      (CSV acumulativo)
    """
    # JSONL — registro completo
    try:
        with open(AGENT_RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[agent] Error guardando JSONL: {e}")

    # CSV — resumen para comparar modelos
    csv_exists = AGENT_BENCH_CSV.exists()
    CSV_FIELDS = [
        "timestamp", "model", "port", "task_short",
        "success", "turns", "elapsed_s",
        "tool_calls_count", "invented_tools_count",
        "prompt_tps_avg", "gen_tps_avg",
        "total_prompt_tokens", "total_gen_tokens",
        "tool_sequence",
    ]
    try:
        with open(AGENT_BENCH_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if not csv_exists:
                writer.writeheader()
            writer.writerow({
                "timestamp":             record.get("timestamp", ""),
                "model":                 record.get("model", ""),
                "port":                  record.get("port", ""),
                "task_short":            record.get("task", "")[:60],
                "success":               record.get("success", False),
                "turns":                 record.get("turns", 0),
                "elapsed_s":             record.get("elapsed_s", 0),
                "tool_calls_count":      len(record.get("tool_calls", [])),
                "invented_tools_count":  len(record.get("invented_tools", [])),
                "prompt_tps_avg":        round(record.get("prompt_tps_avg", 0), 1),
                "gen_tps_avg":           round(record.get("gen_tps_avg", 0), 1),
                "total_prompt_tokens":   record.get("total_prompt_tokens", 0),
                "total_gen_tokens":      record.get("total_gen_tokens", 0),
                "tool_sequence":         " → ".join(record.get("tool_calls", [])),
            })
    except Exception as e:
        print(f"[agent] Error guardando CSV: {e}")


# ── Bucle agente ──────────────────────────────────────────────────────────────

def run_agent_test(port: int, task: str = None, max_turns: int = 10):
    """Ejecuta el bucle agente completo. Genera eventos SSE con el progreso."""
    task = task or AGENT_TASK

    # Nombre del modelo desde INSTANCES
    inst  = INSTANCES.get(port)
    model = (inst["alias"] or inst["model_name"]) if inst else f":{port}"

    messages       = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user",   "content": task},
    ]
    t_start        = time.time()
    turns          = 0
    tool_calls_log = []
    invented_tools = []
    success        = False
    final_msg      = ""

    # Acumuladores de timings por turno
    all_prompt_tps = []
    all_gen_tps    = []
    total_prompt_tokens = 0
    total_gen_tokens    = 0

    # Ficheros de salida generados por write_file
    written_files  = []

    def event(data: dict) -> bytes:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode()

    yield event({"type": "start", "task": task, "max_turns": max_turns, "model": model})

    while turns < max_turns:
        turns += 1
        t_turn = time.time()
        yield event({"type": "turn", "turn": turns, "msg": f"Turno {turns} — llamando al modelo..."})

        response, timings = _call_model(port, messages)

        if "error" in response:
            yield event({"type": "error", "msg": f"Error del modelo: {response['error']}"})
            break

        # Recoger timings de este turno
        if timings:
            pp = timings.get("prompt_per_second", 0)
            gp = timings.get("predicted_per_second", 0)
            pn = timings.get("prompt_n", 0)
            gn = timings.get("predicted_n", 0)
            if pp: all_prompt_tps.append(pp)
            if gp: all_gen_tps.append(gp)
            total_prompt_tokens += pn
            total_gen_tokens    += gn
            yield event({
                "type":       "timings",
                "turn":       turns,
                "prompt_tps": round(pp, 1),
                "gen_tps":    round(gp, 1),
                "prompt_n":   pn,
                "gen_n":      gn,
                "turn_s":     round(time.time() - t_turn, 1),
            })

        choice  = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        reason  = choice.get("finish_reason", "")
        messages.append(message)

        tc_list = message.get("tool_calls") or []

        if tc_list:
            for tc in tc_list:
                fn_name = tc.get("function", {}).get("name", "")
                tool_calls_log.append(fn_name)
                if fn_name not in TOOLS:
                    invented_tools.append(fn_name)
                    yield event({"type": "warning", "msg": f"⚠ Herramienta inventada: '{fn_name}'"})

            for tc in tc_list:
                fn_name = tc.get("function", {}).get("name", "")
                fn_args = tc.get("function", {}).get("arguments", "{}")
                yield event({"type": "tool_call", "tool": fn_name, "args": fn_args})
                # Registrar ficheros escritos
                if fn_name == "write_file":
                    try:
                        args = json.loads(fn_args) if isinstance(fn_args, str) else fn_args
                        written_files.append(args.get("path", ""))
                    except Exception:
                        pass

            tool_results = _execute_tool_calls(tc_list)
            for tr in tool_results:
                yield event({"type": "tool_result", "tool_call_id": tr["tool_call_id"],
                             "result": tr["content"][:500]})
            messages.extend(tool_results)

        elif reason == "stop" or not tc_list:
            final_msg = message.get("content", "")
            success   = bool(final_msg and not invented_tools)
            elapsed   = round(time.time() - t_start, 1)

            # Leer contenido de ficheros escritos para el registro
            written_contents = {}
            for wf in written_files:
                safe = Path(wf).name
                fp   = LOGS_DIR / safe
                if fp.exists():
                    try:
                        written_contents[safe] = fp.read_text(encoding="utf-8")
                    except Exception:
                        pass

            # Calcular promedios de timings
            prompt_tps_avg = round(sum(all_prompt_tps) / len(all_prompt_tps), 1) if all_prompt_tps else 0
            gen_tps_avg    = round(sum(all_gen_tps)    / len(all_gen_tps),    1) if all_gen_tps    else 0

            # Construir registro completo
            record = {
                "timestamp":            time.strftime("%Y-%m-%d %H:%M:%S"),
                "model":                model,
                "port":                 port,
                "task":                 task,
                "success":              success,
                "turns":                turns,
                "elapsed_s":            elapsed,
                "tool_calls":           tool_calls_log,
                "invented_tools":       invented_tools,
                "prompt_tps_avg":       prompt_tps_avg,
                "gen_tps_avg":          gen_tps_avg,
                "total_prompt_tokens":  total_prompt_tokens,
                "total_gen_tokens":     total_gen_tokens,
                "written_files":        written_files,
                "written_contents":     written_contents,
                "final_message":        final_msg,
            }
            _save_result(record)

            yield event({
                "type":                "done",
                "success":             success,
                "turns":               turns,
                "elapsed_s":           elapsed,
                "tool_calls":          tool_calls_log,
                "invented_tools":      invented_tools,
                "prompt_tps_avg":      prompt_tps_avg,
                "gen_tps_avg":         gen_tps_avg,
                "total_prompt_tokens": total_prompt_tokens,
                "total_gen_tokens":    total_gen_tokens,
                "written_files":       written_files,
                "final_message":       final_msg,
                "saved_to":            str(AGENT_RESULTS_FILE),
            })
            return

    # Max turns alcanzado — guardar igualmente
    elapsed = round(time.time() - t_start, 1)
    prompt_tps_avg = round(sum(all_prompt_tps) / len(all_prompt_tps), 1) if all_prompt_tps else 0
    gen_tps_avg    = round(sum(all_gen_tps)    / len(all_gen_tps),    1) if all_gen_tps    else 0
    record = {
        "timestamp":           time.strftime("%Y-%m-%d %H:%M:%S"),
        "model":               model,
        "port":                port,
        "task":                task,
        "success":             False,
        "turns":               turns,
        "elapsed_s":           elapsed,
        "tool_calls":          tool_calls_log,
        "invented_tools":      invented_tools,
        "prompt_tps_avg":      prompt_tps_avg,
        "gen_tps_avg":         gen_tps_avg,
        "total_prompt_tokens": total_prompt_tokens,
        "total_gen_tokens":    total_gen_tokens,
        "written_files":       written_files,
        "written_contents":    {},
        "final_message":       "⚠ Límite de turnos alcanzado.",
    }
    _save_result(record)

    yield event({
        "type":           "done",
        "success":        False,
        "turns":          turns,
        "elapsed_s":      elapsed,
        "tool_calls":     tool_calls_log,
        "invented_tools": invented_tools,
        "prompt_tps_avg": prompt_tps_avg,
        "gen_tps_avg":    gen_tps_avg,
        "final_message":  "⚠ Límite de turnos alcanzado sin completar la tarea.",
        "saved_to":       str(AGENT_RESULTS_FILE),
    })


# ── Endpoints ─────────────────────────────────────────────────────────────────

@agent_bp.route("/api/agent_test", methods=["POST"])
def api_agent_test():
    body      = request.get_json(force=True)
    port      = int(body.get("port", 1234))
    task      = body.get("task", None)
    max_turns = int(body.get("max_turns", 10))
    return Response(
        run_agent_test(port, task, max_turns),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@agent_bp.route("/api/agent_tools")
def api_agent_tools():
    return jsonify({
        "tools":        [t["function"]["name"] for t in TOOLS_SCHEMA],
        "schemas":      TOOLS_SCHEMA,
        "default_task": AGENT_TASK,
    })


@agent_bp.route("/api/agent_results")
def api_agent_results():
    """Devuelve todos los resultados guardados en agent_results.jsonl"""
    if not AGENT_RESULTS_FILE.exists():
        return jsonify({"results": []})
    results = []
    try:
        for line in AGENT_RESULTS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except Exception:
                    pass
    except Exception as e:
        return jsonify({"error": str(e)})
    return jsonify({"results": results})
