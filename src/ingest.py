"""
ingest.py — Chunking → embeddings (CPU) → pgvector, con medición de tiempos.

Idempotente: antes de insertar un experimento ejecuta
    DELETE FROM <tabla> WHERE experimento_id = %s
dentro de la MISMA transacción que el INSERT.

Uso (solo UN modelo levantado a la vez en openai_endpoint):
    python src/ingest.py --modelo bge-m3                    # 3 perfiles
    python src/ingest.py --modelo bge-m3 --perfil Perfil_B_1500
    python src/ingest.py --modelo all --mode local_cpu      # todos, secuencialmente en CPU

Salida: results/ingest_metrics.csv (acumulativo, upsert por experimento_id)
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import logging  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import config  # noqa: E402  (debe ir antes de cualquier import de HF)
import psycopg  # noqa: E402
from chunker import Chunk, chunk_documento, leer_documento, sha256_corto  # noqa: E402
from config import ChunkPerfil, ModeloConfig  # noqa: E402
from embeddings_client import EmbeddingsClient  # noqa: E402
from pgvector.psycopg import register_vector  # noqa: E402
from psycopg import sql  # noqa: E402
from psycopg.types.json import Jsonb  # noqa: E402

log = logging.getLogger("ingest")

INSERT_SQL = """
INSERT INTO {tabla} (
    experimento_id, modelo, chunk_perfil, seccion_h1, seccion_h2, seccion_h3,
    contenido, tokens_aprox, metadata, embedding
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def validar_tabla(conn: psycopg.Connection, modelo: ModeloConfig) -> None:
    if conn.execute("SELECT to_regclass(%s)", (modelo.tabla,)).fetchone()[0] is None:
        raise SystemExit(f"La tabla {modelo.tabla} no existe. Ejecuta primero: python src/db_setup.py")
    dim = conn.execute(
        "SELECT atttypmod FROM pg_attribute WHERE attrelid = %s::regclass AND attname = 'embedding'",
        (modelo.tabla,),
    ).fetchone()[0]
    if dim != modelo.dim:
        raise SystemExit(f"{modelo.tabla}.embedding es vector({dim}) pero {modelo.id_corto} produce {modelo.dim}.")


def ingestar_experimento(
    conn: psycopg.Connection,
    client: EmbeddingsClient,
    perfil: ChunkPerfil,
    chunks: list[Chunk],
    doc_meta: dict,
) -> dict:
    modelo = client.modelo
    exp_id = config.experimento_id(modelo.id_corto, perfil.nombre)
    textos = [client.format_document(c.texto_contextualizado, c.titulo_jerarquico) for c in chunks]
    tokens = client.count_tokens(textos)
    truncados = [t > modelo.max_seq_length for t in tokens]

    t0 = time.perf_counter()
    vectores = client.embed_texts(textos, show_progress=True, desc=exp_id)
    ms_embed = (time.perf_counter() - t0) * 1000

    filas = []
    for chunk, vec, n_tok, trunc in zip(chunks, vectores, tokens, truncados):
        metadata = {
            "chunk_index": chunk.index,
            "titulo_jerarquico": chunk.titulo_jerarquico,
            "chars": len(chunk.texto_contextualizado),
            "chunk_size": perfil.chunk_size,
            "chunk_overlap": perfil.chunk_overlap,
            "tokens_modelo": n_tok,
            "max_seq_length": modelo.max_seq_length,
            "truncado": trunc,
            "token_counter": client.token_counter,
            "hf_id": modelo.hf_id,
            "inference_mode": client.mode,
            "doc_prefix": modelo.doc_prefix,
            **doc_meta,
        }
        filas.append((
            exp_id, modelo.id_corto, perfil.nombre, chunk.h1, chunk.h2, chunk.h3,
            chunk.texto_contextualizado, n_tok, Jsonb(metadata), vec,
        ))

    t1 = time.perf_counter()
    insert = sql.SQL(INSERT_SQL).format(tabla=sql.Identifier(modelo.tabla))
    delete = sql.SQL("DELETE FROM {tabla} WHERE experimento_id = %s").format(tabla=sql.Identifier(modelo.tabla))
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(delete, (exp_id,))
        borrados = cur.rowcount
        cur.executemany(insert, filas)
    ms_insert = (time.perf_counter() - t1) * 1000

    largos = [len(c.texto_contextualizado) for c in chunks]
    metricas = {
        "experimento_id": exp_id,
        "modelo": modelo.id_corto,
        "hf_id": modelo.hf_id,
        "chunk_perfil": perfil.nombre,
        "dim": modelo.dim,
        "tabla": modelo.tabla,
        "inference_mode": client.mode,
        "n_chunks": len(chunks),
        "chars_promedio": round(statistics.mean(largos), 1),
        "chars_max": max(largos),
        "tokens_promedio": round(statistics.mean(tokens), 1),
        "tokens_max": max(tokens),
        "max_seq_length": modelo.max_seq_length,
        "n_truncados": sum(truncados),
        "pct_truncados": round(100 * sum(truncados) / len(chunks), 2),
        "token_counter": client.token_counter,
        "ms_embed_total": round(ms_embed, 1),
        "ms_por_chunk": round(ms_embed / len(chunks), 2),
        "ms_insert_total": round(ms_insert, 1),
        "filas_reemplazadas": borrados,
        "doc_sha256": doc_meta["doc_sha256"],
        "fecha_ingesta": datetime.now().isoformat(timespec="seconds"),
    }
    log.info(
        "✔ %s | %d chunks | %.2f ms/chunk | %.1f%% truncados (>%d tok) | %d filas previas reemplazadas",
        exp_id, len(chunks), metricas["ms_por_chunk"], metricas["pct_truncados"],
        modelo.max_seq_length, borrados,
    )
    return metricas


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingesta del benchmark Payworks en pgvector")
    parser.add_argument("--modelo", nargs="*", default=None,
                        help=f"IDs cortos o 'all' (default: ACTIVE_MODEL={config.ACTIVE_MODEL})")
    parser.add_argument("--perfil", nargs="*", default=None, help="Perfiles o 'all' (default: all)")
    parser.add_argument("--mode", choices=config.VALID_MODES, default=None, help="Override de INFERENCE_MODE")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--doc", default=None, help="Ruta al markdown fuente (default: DOC_PATH)")
    parser.add_argument("--skip-model-check", action="store_true",
                        help="No verificar que el endpoint sirva el modelo solicitado")
    args = parser.parse_args()
    config.setup_logging()

    mode = config.validar_mode(args.mode or config.INFERENCE_MODE)
    modelos = config.resolver_modelos(args.modelo, mode)
    perfiles = config.resolver_perfiles(args.perfil)

    markdown = leer_documento(args.doc)
    doc_meta = {"doc_source": Path(args.doc or config.DOC_PATH).name, "doc_sha256": sha256_corto(markdown)}
    chunks_por_perfil = {p.nombre: chunk_documento(markdown, p) for p in perfiles}
    for nombre, chunks in chunks_por_perfil.items():
        log.info("Chunking %s → %d chunks", nombre, len(chunks))

    metrics_path = config.RESULTS_DIR / config.INGEST_METRICS_CSV
    with psycopg.connect(**config.pg_connect_kwargs(), autocommit=True) as conn:
        try:
            register_vector(conn)
        except psycopg.ProgrammingError as exc:
            raise SystemExit(f"pgvector no disponible ({exc}). Ejecuta: python src/db_setup.py")

        for modelo in modelos:
            validar_tabla(conn, modelo)
            log.info("══════ Modelo %s (%s) | modo=%s ══════", modelo.id_corto, modelo.hf_id, mode)
            client = EmbeddingsClient(modelo, mode=mode, batch_size=args.batch_size)
            try:
                if not args.skip_model_check:
                    client.check_endpoint()
                client.warmup()
                for perfil in perfiles:
                    metricas = ingestar_experimento(conn, client, perfil, chunks_por_perfil[perfil.nombre], doc_meta)
                    metricas["load_seconds"] = round(client.load_seconds, 1)
                    config.upsert_csv(metrics_path, [metricas], ["experimento_id"])
            finally:
                client.close()

    log.info("Métricas de ingesta → %s", metrics_path)


if __name__ == "__main__":
    main()
