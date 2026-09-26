# LLAMA MANAGER

Panel web para lanzar y gestionar instancias de `llama-server` (llama.cpp) en Windows y Linux:
varias GPUs, perfiles, proxy OpenAI/Anthropic estable en `:8080`, reparto en red (RPC),
MTP / speculative decoding, estimación de VRAM y pruebas de agentes.

## Arrancar (Windows)

1. Descarga el repo (botón **Code → Download ZIP**) o cópialo a una carpeta, p. ej. `C:\ToolsIA\llama_manager`.
2. Doble clic en **`LlamaManager.bat`**.
   - La primera vez prepara `python\` (Python embebido + Flask) y descarga llama.cpp (Vulkan). No instala nada en el sistema.
   - Cada vez que arranca **busca actualizaciones en este repo y las aplica** (`updater.py`).
3. Panel: http://localhost:8080 · Proxy OpenAI: `http://<IP>:8080/v1`
   · Claude Code: `ANTHROPIC_BASE_URL=http://<IP>:8080`, `ANTHROPIC_AUTH_TOKEN=local`

Opciones en `LlamaManager.bat`: `LLAMA_MANAGER_PORT`, `LLAMA_MANAGER_MODELS` (por defecto
`%USERPROFILE%\.lmstudio\models`), `LLAMA_MANAGER_NO_UPDATE=1`.

## Versiones — `versions.py`

| Variable | Qué fija |
|---|---|
| `APP_VERSION` | versión del manager |
| `UPDATE_REPO`, `UPDATE_BRANCH` | de dónde se auto-actualiza |
| `LLAMA_CPP_BUILD` | build de llama.cpp de **todos** los equipos (RPC exige el mismo) |
| `ROCM_RUNTIME` | runtime del zip ROCm de GitHub (hoy `10.0`) |
| `PYTHON_VERSION`, `PYTHON_STANDALONE_TAG` | Python embebido |

### Auto-actualización

Al arrancar, `updater.py` compara el último commit de `UPDATE_BRANCH` con `.installed_commit`.
Si hay uno nuevo, descarga ese commit y sobrescribe los ficheros de la app. **Nunca toca**
`presets.json` (tus perfiles), `logs\`, `python\`, `llama-*\` ni `github_token.txt`, y no borra
ficheros locales. Si el nuevo `versions.py` pide otro `LLAMA_CPP_BUILD`, actualiza también los
motores instalados. Sin internet o con cualquier error, arranca con lo que hay.

Para publicar una versión: sube los cambios a `main` (sube `APP_VERSION` para que se vea en el panel).
Repo privado: pon un token de GitHub de solo lectura en `github_token.txt`.
Si la carpeta es un clon `git`, el updater no hace nada (usa `git pull`).

### Actualizar llama.cpp a mano

Para las instancias, cambia `LLAMA_CPP_BUILD` y ejecuta `update_llama.bat`
(`update_llama.bat vulkan rocm` para motores concretos).

## Motores

- `vulkan` → `llama-vulkan\` (recomendado en RDNA3)
- `rocm` → `llama-rocm\` (build oficial ROCm, mismo build)
- `rocm-lms` → backend ROCm de LM Studio, autodetectado si existe

El manager lee `llama-server --help` de cada binario y adapta los flags: al backend viejo de
LM Studio le pasa `--no-mmap/--mlock`, al nuevo `--load-mode`. Presets y "Flags extra" antiguos
se traducen solos (queda una `NOTA:` en el log de la instancia).

Cambios de llama.cpp ya absorbidos (b10723 → b11193): `--mmap/--no-mmap/--mlock/-dio` →
`--load-mode`; `--webui` → `--ui`; `enable_thinking` → `--reasoning`; `reasoning_effort` →
`--reasoning-effort`; `preserve_reasoning` activado por defecto; `--draft-max` → `--spec-draft-n-max`.

## Qwen4 (arquitectura `qwen4exp` = Qwen3.8-Flash-Next)

Botón **⚡ Aplicar ajustes Qwen4** (Lanzar modelo → Flags avanzados):
PLE en CPU (`-ot per_layer_token_embd=CPU`), `--lazy-mode auto`, MTP en sidecar
(`mtp-*.gguf` junto al modelo → `-md` + `--spec-type draft-mtp`, n-max 3), `--backend-sampling`,
`--cache-ram 20000 --ctx-checkpoints 32`, temp 1.0 · top-p 0.95 · top-k 20 · min-p 0, KV q8_0 + FA.

## Portable sin conexión

`build_portable.bat` genera `..\LlamaManager-Portable\` + `.zip` con Python, Flask y el motor ya
dentro (`--engines vulkan rocm`, `--with-presets`). Se copia tal cual a otro equipo.
