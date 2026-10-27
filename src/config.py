"""
config.py — Configuración central del Benchmark de Embeddings Payworks.

Responsabilidades:
  * Cargar `.env` (si existe) desde la raíz del repo.
  * Forzar que TODA caché de Hugging Face viva en ./models_cache (ignorado por git).
  * Definir MODELOS_CONFIG (con prefijos doc/query) y CHUNK_PERFILES.
  * Utilidades compartidas: conexión PG, resolución de CLI, upsert de CSVs, logging.

IMPORTANTE: este módulo debe importarse ANTES que `transformers` /
`sentence_transformers` para que HF_HOME se aplique correctamente.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # No generar __pycache__ (git status limpio en el Workbench)

import logging  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Iterable, Sequence  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

# =============================================================================
#  Rutas base y .env
# =============================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=False)


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = _env_str(name, "")
    if value == "":
        return default
    return value.lower() in {"1", "true", "yes", "y", "si", "sí", "on"}


def _resolve_path(value: str, default: str) -> Path:
    """Resuelve rutas relativas contra la raíz del repo (no contra el CWD)."""
    path = Path(value or default).expanduser()
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


# =============================================================================
#  Caché de Hugging Face  (debe fijarse antes de importar transformers)
# =============================================================================
HF_HOME = _resolve_path(_env_str("HF_HOME"), "models_cache")
HF_HOME.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_HOME)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

HF_TOKEN = _env_str("HF_TOKEN") or None
USE_HF_TOKENIZER = _env_bool("USE_HF_TOKENIZER", True)

# =============================================================================
#  Rutas del proyecto
# =============================================================================
DOC_PATH = _resolve_path(_env_str("DOC_PATH"), "docs/Payworks_Master_RAG.md")
GOLDEN_DATASET_PATH = _resolve_path(_env_str("GOLDEN_DATASET_PATH"), "docs/golden_dataset.json")
RESULTS_DIR = _resolve_path(_env_str("RESULTS_DIR"), "results")
SQL_INIT_PATH = ROOT_DIR / "sql" / "init_pgvector.sql"

INGEST_METRICS_CSV = "ingest_metrics.csv"
EVAL_SUMMARY_CSV = "eval_resumen.csv"
EVAL_DETAIL_CSV = "eval_detalle.csv"
REPORT_MD = "reporte_benchmark.md"

# =============================================================================
#  Inferencia
# =============================================================================
VALID_MODES = ("local_cpu", "openai_endpoint")
INFERENCE_MODE = _env_str("INFERENCE_MODE", "local_cpu").lower()
ACTIVE_MODEL = _env_str("ACTIVE_MODEL", "bge-m3")
EMBED_BATCH_SIZE = _env_int("EMBED_BATCH_SIZE", 16)
TORCH_NUM_THREADS = _env_int("TORCH_NUM_THREADS", 0)

ENDPOINT_TIMEOUT_S = _env_int("EMBEDDING_TIMEOUT_S", 120)
ENDPOINT_MAX_RETRIES = _env_int("EMBEDDING_MAX_RETRIES", 5)
ENDPOINT_TRUNCATE_PROMPT_TOKENS = _env_int("EMBEDDING_TRUNCATE_PROMPT_TOKENS", 0)

# =============================================================================
#  Evaluación
# =============================================================================
VALID_SEARCH_MODES = ("exact", "hnsw")
SEARCH_MODE = _env_str("SEARCH_MODE", "exact").lower()
HNSW_EF_SEARCH = _env_int("HNSW_EF_SEARCH", 100)
TOP_K = 5  # Necesario para Δ coseno (Top1 − Top5)

LOG_LEVEL = _env_str("LOG_LEVEL", "INFO").upper()


# =============================================================================
#  Modelos de embeddings candidatos (solo CPU)
# =============================================================================
@dataclass(frozen=True)
class ModeloConfig:
    id_corto: str
    hf_id: str
    dim: int
    tabla: str
    doc_prefix: str  # plantilla: placeholders {texto} y {titulo}
    query_prefix: str  # plantilla: placeholder {pregunta}
    max_seq_length: int  # ventana de tokens efectiva (para medir truncamiento)
    rol: str
    mrl: bool = False  # Matryoshka: permite truncar dimensiones + renormalizar L2
    trust_remote_code: bool = False


MODELOS_CONFIG: dict[str, ModeloConfig] = {
    "bge-m3": ModeloConfig(
        id_corto="bge-m3",
        hf_id="BAAI/bge-m3",
        dim=1024,
        tabla="payworks_rag_1024",
        doc_prefix="{texto}",
        query_prefix="{pregunta}",
        max_seq_length=8192,
        rol="Candidato #1 (Ventana 8192)",
    ),
    "multilingual-e5-large": ModeloConfig(
        id_corto="multilingual-e5-large",
        hf_id="intfloat/multilingual-e5-large",
        dim=1024,
        tabla="payworks_rag_1024",
        doc_prefix="passage: {texto}",
        query_prefix="query: {pregunta}",
        max_seq_length=512,
        rol="Candidato #2 (Ventana corta: 512)",
    ),
    "embeddinggemma-300m": ModeloConfig(
        id_corto="embeddinggemma-300m",
        hf_id="google/embeddinggemma-300m",
        dim=768,
        tabla="payworks_rag_768",
        doc_prefix="title: {titulo} | text: {texto}",
        query_prefix="task: search result | query: {pregunta}",
        max_seq_length=2048,
        rol="Candidato #3 (Alta eficiencia en CPU)",
        mrl=True,
    ),
    "granite-278m-multilingual": ModeloConfig(
        id_corto="granite-278m-multilingual",
        hf_id="ibm-granite/granite-embedding-278m-multilingual",
        dim=768,
        tabla="payworks_rag_768",
        doc_prefix="{texto}",
        query_prefix="{pregunta}",
        max_seq_length=512,
        rol="Candidato #4 (Nativo ecosistema Red Hat/IBM)",
    ),
    "all-MiniLM-L6-v2": ModeloConfig(
        id_corto="all-MiniLM-L6-v2",
        hf_id="sentence-transformers/all-MiniLM-L6-v2",
        dim=384,
        tabla="payworks_rag_384",
        doc_prefix="{texto}",
        query_prefix="{pregunta}",
        max_seq_length=256,
        rol="Línea base negativa (Inglés)",
    ),
}

TABLAS_POR_DIM: dict[int, str] = {m.dim: m.tabla for m in MODELOS_CONFIG.values()}


# =============================================================================
#  Perfiles de chunking (tamaños en CARACTERES)
# =============================================================================
@dataclass(frozen=True)
class ChunkPerfil:
    nombre: str
    chunk_size: int
    chunk_overlap: int
    descripcion: str


CHUNK_PERFILES: dict[str, ChunkPerfil] = {
    "Perfil_A_600": ChunkPerfil("Perfil_A_600", 600, 100, "Granular / Glosario"),
    "Perfil_B_1500": ChunkPerfil("Perfil_B_1500", 1500, 250, "Balanceado / Estándar (≈512 tokens)"),
    "Perfil_C_3000": ChunkPerfil("Perfil_C_3000", 3000, 400, "Contextual amplio (expone truncamiento)"),
}


def experimento_id(modelo_id: str, perfil_nombre: str) -> str:
    """Formato canónico `{modelo}__{perfil}`."""
    return f"{modelo_id}__{perfil_nombre}"


# =============================================================================
#  Endpoint OpenAI-compatible (vLLM / TEI)  — un modelo levantado a la vez
# =============================================================================
def _env_key(id_corto: str) -> str:
    return re.sub(r"[^A-Z0-9]", "_", id_corto.upper())


def _normalizar_base_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if url.endswith("/embeddings"):
        url = url[: -len("/embeddings")]
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def endpoint_settings(modelo: ModeloConfig) -> dict:
    """Resuelve URL / API key / served name con override por modelo.

    Prioridad: EMBEDDING_BASE_URL__<ID> > EMBEDDING_BASE_URL (idem para el resto).
    """
    key = _env_key(modelo.id_corto)
    base_url = _env_str(f"EMBEDDING_BASE_URL__{key}") or _env_str("EMBEDDING_BASE_URL")
    explicit_served_name = (
        _env_str(f"EMBEDDING_SERVED_MODEL_NAME__{key}")
        or _env_str("EMBEDDING_SERVED_MODEL_NAME")
    )
    return {
        "base_url": _normalizar_base_url(base_url) if base_url else "",
        "api_key": _env_str(f"EMBEDDING_API_KEY__{key}") or _env_str("EMBEDDING_API_KEY"),
        "served_model_name": explicit_served_name or modelo.hf_id,
        "is_served_name_explicit": bool(explicit_served_name),
    }


def endpoint_verify_ssl() -> bool | str:
    value = _env_str("EMBEDDING_VERIFY_SSL", "true")
    if value.lower() in {"0", "false", "no", "off"}:
        return False
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    return value  # ruta a CA bundle


# =============================================================================
#  PostgreSQL
# =============================================================================
def pg_connect_kwargs() -> dict:
    """kwargs para `psycopg.connect(**pg_connect_kwargs())`."""
    common = {"application_name": "payworks-embeddings-benchmark", "connect_timeout": 10}
    database_url = _env_str("DATABASE_URL")
    if database_url:
        return {"conninfo": database_url, **common}
    return {
        "host": _env_str("PGHOST", "localhost"),
        "port": _env_int("PGPORT", 5432),
        "dbname": _env_str("PGDATABASE", "payworks"),
        "user": _env_str("PGUSER", "postgres"),
        "password": _env_str("PGPASSWORD", ""),
        "sslmode": _env_str("PGSSLMODE", "prefer"),
        **common,
    }


# =============================================================================
#  Utilidades CLI compartidas
# =============================================================================
def resolver_modelos(valores: Sequence[str] | None, mode: str) -> list[ModeloConfig]:
    """Traduce `--modelo` a ModeloConfig. En openai_endpoint solo se permite 1 modelo."""
    seleccion = list(valores) if valores else [ACTIVE_MODEL]
    if any(v.lower() == "all" for v in seleccion):
        seleccion = list(MODELOS_CONFIG)
    desconocidos = [v for v in seleccion if v not in MODELOS_CONFIG]
    if desconocidos:
        raise SystemExit(
            f"Modelo(s) desconocido(s): {desconocidos}. Opciones: {list(MODELOS_CONFIG)} | all"
        )
    seleccion = list(dict.fromkeys(seleccion))  # dedup preservando orden
    if mode == "openai_endpoint" and len(seleccion) > 1:
        raise SystemExit(
            "En modo openai_endpoint solo hay UN modelo levantado a la vez: "
            "ejecuta un --modelo por corrida (el que esté desplegado)."
        )
    return [MODELOS_CONFIG[v] for v in seleccion]


def resolver_perfiles(valores: Sequence[str] | None) -> list[ChunkPerfil]:
    if not valores or any(v.lower() == "all" for v in valores):
        return list(CHUNK_PERFILES.values())
    desconocidos = [v for v in valores if v not in CHUNK_PERFILES]
    if desconocidos:
        raise SystemExit(
            f"Perfil(es) desconocido(s): {desconocidos}. Opciones: {list(CHUNK_PERFILES)} | all"
        )
    return [CHUNK_PERFILES[v] for v in dict.fromkeys(valores)]


def validar_mode(mode: str) -> str:
    mode = mode.lower()
    if mode not in VALID_MODES:
        raise SystemExit(f"INFERENCE_MODE inválido: {mode!r}. Opciones: {VALID_MODES}")
    return mode


def upsert_csv(path: Path, rows: Iterable[dict], key_cols: Sequence[str]) -> Path:
    """Inserta/reemplaza filas en un CSV acumulativo usando `key_cols` como llave."""
    import pandas as pd

    nuevo = pd.DataFrame(list(rows))
    if nuevo.empty:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previo = pd.read_csv(path)
        if all(c in previo.columns for c in key_cols):
            llaves = set(map(tuple, nuevo[list(key_cols)].astype(str).values.tolist()))
            mask = previo[list(key_cols)].astype(str).apply(tuple, axis=1).isin(llaves)
            previo = previo[~mask]
        nuevo = pd.concat([previo, nuevo], ignore_index=True)
    nuevo.to_csv(path, index=False, encoding="utf-8")
    return path


def setup_logging() -> None:
    """Logging solo a stdout (sin archivos → no ensucia el árbol de git)."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    for ruidoso in ("httpx", "urllib3", "sentence_transformers", "transformers"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
