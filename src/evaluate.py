"""
evaluate.py — Corre el Golden Dataset contra pgvector y genera el reporte comparativo.

Métricas por experimento_id ({modelo}__{perfil}):
  * Hit Rate@1, Hit Rate@3 (Recall@3), Hit Rate@5
  * MRR@3  (1.0 / 0.5 / 0.333 según la posición del primer chunk relevante)
  * Δ Coseno = promedio(sim Top1 − sim Top5)  → separación respuesta vs. ruido
  * Latencia CPU: ms/query (embedding) y ms/query (búsqueda pgvector)
  * Se cruza con results/ingest_metrics.csv: ms/chunk de ingesta y % truncados

Uso (solo UN modelo levantado a la vez en openai_endpoint):
    python src/evaluate.py --modelo bge-m3
    python src/evaluate.py --modelo all --mode local_cpu
    python src/evaluate.py --report-only        # regenera el reporte con todo lo acumulado

Salidas (results/, ignorado por git):
    eval_detalle.csv · eval_resumen.csv · reporte_benchmark.md
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import config  # noqa: E402  (debe ir antes de cualquier import de HF)
import numpy as np  # noqa: E402
import psycopg  # noqa: E402
from config import ModeloConfig  # noqa: E402
from pgvector.psycopg import register_vector  # noqa: E402
from psycopg import sql  # noqa: E402

log = logging.getLogger("evaluate")

CRITERIOS = ("seccion", "keywords", "seccion_o_keywords", "seccion_y_keywords")
_RE_CONTEXTO = re.compile(r"^\[[^\n]*\]\n")  # línea "[h1 > h2 > h3]" inyectada

SEARCH_SQL = """
SELECT id, seccion_h1, seccion_h2, seccion_h3, contenido,
       1 - (embedding <=> %(q)s) AS similitud
FROM {tabla}
WHERE experimento_id = %(exp)s
ORDER BY embedding <=> %(q)s
LIMIT %(k)s
"""


# =============================================================================
#  Golden Dataset y relevancia
# =============================================================================
def normalizar(texto: str) -> str:
    """minúsculas, sin acentos, espacios colapsados (matching robusto en español)."""
    texto = unicodedata.normalize("NFKD", texto or "")
    texto = "".join(ch for ch in texto if not unicodedata.combining(ch))
    return " ".join(texto.lower().split())


def cargar_golden_dataset(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    preguntas = data["preguntas"] if isinstance(data, dict) else data
    for p in preguntas:
        faltan = {"id", "categoria", "pregunta", "ground_truth"} - set(p)
        if faltan:
            raise SystemExit(f"Pregunta {p.get('id', '?')} sin campos {faltan} en {path}")
        gt = p["ground_truth"]
        gt.setdefault("secciones", [])
        gt.setdefault("keywords", [])
        gt.setdefault("criterio", "seccion_o_keywords")
        gt.setdefault("min_keywords", 1)
        if gt["criterio"] not in CRITERIOS:
            raise SystemExit(f"{p['id']}: criterio inválido {gt['criterio']!r}. Opciones: {CRITERIOS}")
        if not gt["secciones"] and not gt["keywords"]:
            raise SystemExit(f"{p['id']}: ground_truth sin 'secciones' ni 'keywords'")
    return preguntas


def es_relevante(resultado: dict, gt: dict) -> bool:
    """Un chunk es relevante según el criterio de su pregunta.

    * seccion  → algún término de `secciones` aparece en seccion_h2 / seccion_h3
    * keywords → al menos `min_keywords` fragmentos de `keywords` aparecen en el cuerpo
    """
    jerarquia = normalizar(f"{resultado['seccion_h2'] or ''} > {resultado['seccion_h3'] or ''}")
    cuerpo = normalizar(_RE_CONTEXTO.sub("", resultado["contenido"] or "", count=1))

    sec_hit = any(normalizar(s) in jerarquia for s in gt["secciones"])
    n_kw = sum(normalizar(k) in cuerpo for k in gt["keywords"])
    kw_hit = bool(gt["keywords"]) and n_kw >= min(int(gt["min_keywords"]), len(gt["keywords"]))

    return {
        "seccion": sec_hit,
        "keywords": kw_hit,
        "seccion_o_keywords": sec_hit or kw_hit,
        "seccion_y_keywords": sec_hit and kw_hit,
    }[gt["criterio"]]


# =============================================================================
#  Búsqueda en pgvector
# =============================================================================
def pgvector_version(conn: psycopg.Connection) -> tuple[int, ...]:
    fila = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    return tuple(int(x) for x in re.findall(r"\d+", fila[0])) if fila else (0,)


def buscar(conn, modelo: ModeloConfig, exp_id: str, q_vec: np.ndarray, k: int,
           search_mode: str, iterative_ok: bool) -> tuple[list[dict], float]:
    """Top-k por distancia coseno. exact = KNN exacto (sin ANN); hnsw = índice HNSW."""
    query = sql.SQL(SEARCH_SQL).format(tabla=sql.Identifier(modelo.tabla))
    t0 = time.perf_counter()
    with conn.transaction(), conn.cursor() as cur:
        if search_mode == "exact":
            cur.execute("SET LOCAL enable_indexscan = off")  # fuerza escaneo exacto
        else:
            cur.execute(sql.SQL("SET LOCAL hnsw.ef_search = {}").format(sql.Literal(config.HNSW_EF_SEARCH)))
            if iterative_ok:  # pgvector ≥ 0.8: evita perder resultados al filtrar por experimento_id
                cur.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        cur.execute(query, {"q": q_vec, "exp": exp_id, "k": k})
        columnas = [d.name for d in cur.description]
        filas = [dict(zip(columnas, r)) for r in cur.fetchall()]
    return filas, (time.perf_counter() - t0) * 1000


def contar_chunks(conn, modelo: ModeloConfig, exp_id: str) -> int:
    q = sql.SQL("SELECT count(*) FROM {tabla} WHERE experimento_id = %s").format(tabla=sql.Identifier(modelo.tabla))
    return conn.execute(q, (exp_id,)).fetchone()[0]


# =============================================================================
#  Evaluación
# =============================================================================
def evaluar_pregunta(resultados: list[dict], gt: dict) -> dict:
    relevancias = [es_relevante(r, gt) for r in resultados]
    rank = next((i + 1 for i, rel in enumerate(relevancias) if rel), None)
    sims = [float(r["similitud"]) for r in resultados]
    return {
        "rank_primer_relevante": rank,
        "hit_at_1": int(rank == 1),
        "hit_at_3": int(rank is not None and rank <= 3),
        "hit_at_5": int(rank is not None and rank <= 5),
        "rr_at_3": round(1.0 / rank, 4) if rank and rank <= 3 else 0.0,
        "sim_top1": round(sims[0], 4) if sims else None,
        "sim_top5": round(sims[min(4, len(sims) - 1)], 4) if sims else None,
        "delta_coseno": round(sims[0] - sims[min(4, len(sims) - 1)], 4) if sims else None,
        "top_secciones": " | ".join(
            f"{i + 1}:{'✔' if rel else '✘'} {r['seccion_h3'] or r['seccion_h2'] or r['seccion_h1'] or '-'}"
            for i, (r, rel) in enumerate(zip(resultados, relevancias))
        ),
        "top_ids": ",".join(str(r["id"]) for r in resultados),
    }


def evaluar_modelo(conn, modelo: ModeloConfig, perfiles, preguntas, mode: str, search_mode: str,
                   skip_check: bool, iterative_ok: bool) -> tuple[list[dict], list[dict]]:
    from embeddings_client import EmbeddingsClient  # import diferido (evita cargar torch en --report-only)

    experimentos = []
    for perfil in perfiles:
        exp_id = config.experimento_id(modelo.id_corto, perfil.nombre)
        n = contar_chunks(conn, modelo, exp_id)
        if n == 0:
            log.warning("Sin datos para %s en %s → ejecuta ingest.py primero. Se omite.", exp_id, modelo.tabla)
            continue
        experimentos.append((perfil, exp_id, n))
    if not experimentos:
        return [], []

    detalle, resumen = [], []
    with EmbeddingsClient(modelo, mode=mode) as client:
        if not skip_check:
            client.check_endpoint()
        client.warmup()

        # La query no depende del perfil: se vectoriza una vez por modelo.
        q_vecs, q_ms = [], []
        for p in preguntas:
            t0 = time.perf_counter()
            q_vecs.append(client.embed_query(p["pregunta"]))
            q_ms.append((time.perf_counter() - t0) * 1000)
        ms_query_embed = statistics.mean(q_ms)
        log.info("%s: %.1f ms/query (embedding, %d preguntas)", modelo.id_corto, ms_query_embed, len(preguntas))

        for perfil, exp_id, n_chunks in experimentos:
            filas_exp = []
            for p, q_vec, ms_emb in zip(preguntas, q_vecs, q_ms):
                resultados, ms_busq = buscar(conn, modelo, exp_id, q_vec, config.TOP_K, search_mode, iterative_ok)
                fila = {
                    "experimento_id": exp_id,
                    "modelo": modelo.id_corto,
                    "chunk_perfil": perfil.nombre,
                    "pregunta_id": p["id"],
                    "categoria": p["categoria"],
                    "pregunta": p["pregunta"],
                    **evaluar_pregunta(resultados, p["ground_truth"]),
                    "ms_query_embed": round(ms_emb, 2),
                    "ms_query_busqueda": round(ms_busq, 2),
                }
                filas_exp.append(fila)
            detalle.extend(filas_exp)

            def prom(campo: str) -> float:
                valores = [f[campo] for f in filas_exp if f[campo] is not None]
                return round(statistics.mean(valores), 4) if valores else 0.0

            fila_resumen = {
                "experimento_id": exp_id,
                "modelo": modelo.id_corto,
                "hf_id": modelo.hf_id,
                "chunk_perfil": perfil.nombre,
                "dim": modelo.dim,
                "inference_mode": mode,
                "search_mode": search_mode,
                "n_preguntas": len(filas_exp),
                "n_chunks_db": n_chunks,
                "hit_rate_at_1": prom("hit_at_1"),
                "hit_rate_at_3": prom("hit_at_3"),
                "hit_rate_at_5": prom("hit_at_5"),
                "mrr_at_3": prom("rr_at_3"),
                "delta_coseno": prom("delta_coseno"),
                "sim_top1_promedio": prom("sim_top1"),
                "ms_query_embed": round(ms_query_embed, 2),
                "ms_query_busqueda": prom("ms_query_busqueda"),
                "fecha_eval": datetime.now().isoformat(timespec="seconds"),
            }
            resumen.append(fila_resumen)
            log.info(
                "✔ %-45s Hit@1=%.3f Hit@3=%.3f MRR@3=%.3f ΔCos=%.4f",
                exp_id, fila_resumen["hit_rate_at_1"], fila_resumen["hit_rate_at_3"],
                fila_resumen["mrr_at_3"], fila_resumen["delta_coseno"],
            )
    return detalle, resumen


# =============================================================================
#  Reporte ejecutivo (Markdown)
# =============================================================================
def _fmt(valor, dec: int = 3) -> str:
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "—"
    if isinstance(valor, (float, np.floating)):
        return f"{valor:.{dec}f}"
    return str(valor)


def _md_table(headers: list[str], filas: list[list]) -> str:
    lineas = ["| " + " | ".join(headers) + " |", "|" + "|".join([" :--- "] + [" ---: "] * (len(headers) - 1)) + "|"]
    lineas += ["| " + " | ".join(str(c) for c in f) + " |" for f in filas]
    return "\n".join(lineas)


def generar_reporte(results_dir: Path) -> Path | None:
    import pandas as pd

    resumen_path = results_dir / config.EVAL_SUMMARY_CSV
    if not resumen_path.exists():
        log.error("No existe %s: ejecuta primero una evaluación.", resumen_path)
        return None
    df = pd.read_csv(resumen_path)
    ingest_path = results_dir / config.INGEST_METRICS_CSV
    if ingest_path.exists():
        ing = pd.read_csv(ingest_path)
        cols = ["experimento_id", "ms_por_chunk", "chars_promedio", "tokens_promedio",
                "tokens_max", "pct_truncados", "token_counter"]
        df = df.merge(ing[[c for c in cols if c in ing.columns]], on="experimento_id", how="left")
    for c in ("ms_por_chunk", "chars_promedio", "pct_truncados", "tokens_max"):
        if c not in df.columns:
            df[c] = np.nan

    df = df.sort_values(
        ["mrr_at_3", "hit_rate_at_3", "hit_rate_at_1", "delta_coseno", "ms_query_embed"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

    tabla_principal = _md_table(
        ["#", "Experimento", "Modelo", "Perfil", "MRR@3", "Hit@3", "Hit@1", "ΔCos (T1−T5)",
         "ms/chunk ingesta", "ms/query embed", "ms/query búsqueda", "Chunks", "Chars prom.", "% truncados"],
        [[i + 1, f"`{r.experimento_id}`", r.modelo, r.chunk_perfil, _fmt(r.mrr_at_3), _fmt(r.hit_rate_at_3),
          _fmt(r.hit_rate_at_1), _fmt(r.delta_coseno, 4), _fmt(r.ms_por_chunk, 1), _fmt(r.ms_query_embed, 1),
          _fmt(r.ms_query_busqueda, 1), r.n_chunks_db, _fmt(r.chars_promedio, 0), _fmt(r.pct_truncados, 1)]
         for i, r in df.iterrows()],
    )

    mejores = df.loc[df.groupby("modelo")["mrr_at_3"].idxmax()].sort_values("mrr_at_3", ascending=False)
    tabla_mejores = _md_table(
        ["Modelo", "Mejor perfil", "MRR@3", "Hit@3", "ΔCos", "ms/query embed"],
        [[r.modelo, r.chunk_perfil, _fmt(r.mrr_at_3), _fmt(r.hit_rate_at_3), _fmt(r.delta_coseno, 4),
          _fmt(r.ms_query_embed, 1)] for r in mejores.itertuples()],
    )

    perfiles = df.groupby("chunk_perfil")[["mrr_at_3", "hit_rate_at_3", "delta_coseno"]].mean().sort_index()
    tabla_perfiles = _md_table(
        ["Perfil", "MRR@3 prom.", "Hit@3 prom.", "ΔCos prom."],
        [[idx, _fmt(r.mrr_at_3), _fmt(r.hit_rate_at_3), _fmt(r.delta_coseno, 4)] for idx, r in perfiles.iterrows()],
    )

    tabla_categorias = "_Sin detalle por pregunta disponible._"
    detalle_path = results_dir / config.EVAL_DETAIL_CSV
    if detalle_path.exists():
        det = pd.read_csv(detalle_path)
        det = det[det["experimento_id"].isin(df["experimento_id"])]
        piv = det.pivot_table(index="experimento_id", columns="categoria", values="rr_at_3", aggfunc="mean")
        piv = piv.reindex(df["experimento_id"])
        tabla_categorias = _md_table(
            ["Experimento"] + [f"MRR@3 · {c}" for c in piv.columns],
            [[f"`{idx}`"] + [_fmt(v) for v in fila] for idx, fila in piv.iterrows()],
        )

    # ---- Hallazgos automáticos
    top = df.iloc[0]
    hallazgos = [
        f"- **Ganador:** `{top.experimento_id}` con MRR@3 = {_fmt(top.mrr_at_3)} y Hit@3 = {_fmt(top.hit_rate_at_3)}.",
    ]
    base = df[df["modelo"] == "all-MiniLM-L6-v2"]
    if not base.empty:
        gap = top.mrr_at_3 - base["mrr_at_3"].max()
        hallazgos.append(f"- **Línea base negativa (MiniLM, inglés):** mejor MRR@3 = {_fmt(base['mrr_at_3'].max())} "
                         f"→ brecha de {_fmt(gap)} vs. el ganador.")
    trunc = df[df["pct_truncados"].fillna(0) > 0]
    if not trunc.empty:
        lista = ", ".join(f"`{r.experimento_id}` ({_fmt(r.pct_truncados, 1)}%)" for r in trunc.itertuples())
        hallazgos.append(f"- **Truncamiento detectado** (chunks que exceden la ventana del modelo): {lista}.")
    df["_eficiencia"] = df["mrr_at_3"] / df["ms_query_embed"].clip(lower=1e-6)
    efi = df.sort_values("_eficiencia", ascending=False).iloc[0]
    hallazgos.append(f"- **Mejor calidad/latencia (MRR@3 ÷ ms/query):** `{efi.experimento_id}` "
                     f"({_fmt(efi.mrr_at_3)} MRR@3 a {_fmt(efi.ms_query_embed, 1)} ms/query).")
    pendientes = len(config.MODELOS_CONFIG) * len(config.CHUNK_PERFILES) - len(df)
    if pendientes > 0:
        hallazgos.append(f"- **Cobertura:** {len(df)}/15 combinaciones evaluadas ({pendientes} pendientes).")

    modos = ", ".join(sorted(df["inference_mode"].dropna().unique()))
    busquedas = ", ".join(sorted(df["search_mode"].dropna().unique()))
    reporte = f"""# Reporte Ejecutivo — Benchmark de Embeddings y Chunking (Payworks)

_Generado: {datetime.now().isoformat(timespec="seconds")} · Experimentos: {len(df)} · Preguntas Golden: {int(df["n_preguntas"].max())} · Inferencia: {modos} · Búsqueda: {busquedas} · Top-K: {config.TOP_K}_

## Hallazgos clave

{chr(10).join(hallazgos)}

## Ranking general (ordenado por MRR@3 y Hit Rate@3)

{tabla_principal}

## Mejor configuración por modelo

{tabla_mejores}

## Efecto del perfil de chunking (promedio entre modelos)

{tabla_perfiles}

## MRR@3 por categoría del Golden Dataset

{tabla_categorias}

## Metodología

- **Relevancia:** un chunk es relevante si cumple el `criterio` de su pregunta en `docs/golden_dataset.json` (coincidencia normalizada —sin acentos/mayúsculas— de `secciones` en `seccion_h2`/`seccion_h3` y/o de `keywords` en el cuerpo).
- **MRR@3:** 1/rank del primer chunk relevante dentro del Top-3 (0 si no aparece).
- **ΔCos (T1−T5):** similitud coseno del Top-1 menos la del Top-5; mayor = mejor separación respuesta/ruido. Comparable solo dentro del mismo modelo (cada espacio vectorial tiene su propia distribución de similitudes).
- **Latencias:** medidas tras *warm-up*; ms/query embed incluye red en modo `openai_endpoint`. ms/chunk proviene de `ingest_metrics.csv`.
- **% truncados:** chunks cuyo conteo de tokens (prefijo incluido) excede `max_seq_length` del modelo.
- **Búsqueda `exact`:** KNN exacto (sin índice ANN) para aislar la calidad del modelo del error de aproximación de HNSW.
"""
    out = results_dir / config.REPORT_MD
    out.write_text(reporte, encoding="utf-8")
    log.info("Reporte ejecutivo → %s", out)
    return out


# =============================================================================
#  CLI
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación del benchmark Payworks contra pgvector")
    parser.add_argument("--modelo", nargs="*", default=None,
                        help=f"IDs cortos o 'all' (default: ACTIVE_MODEL={config.ACTIVE_MODEL})")
    parser.add_argument("--perfil", nargs="*", default=None, help="Perfiles o 'all' (default: all)")
    parser.add_argument("--mode", choices=config.VALID_MODES, default=None, help="Override de INFERENCE_MODE")
    parser.add_argument("--search-mode", choices=config.VALID_SEARCH_MODES, default=None,
                        help=f"exact | hnsw (default: SEARCH_MODE={config.SEARCH_MODE})")
    parser.add_argument("--golden", default=None, help="Ruta al golden dataset (default: GOLDEN_DATASET_PATH)")
    parser.add_argument("--skip-model-check", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Solo regenerar el reporte desde results/")
    args = parser.parse_args()
    config.setup_logging()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        sys.exit(0 if generar_reporte(config.RESULTS_DIR) else 1)

    mode = config.validar_mode(args.mode or config.INFERENCE_MODE)
    search_mode = (args.search_mode or config.SEARCH_MODE).lower()
    if search_mode not in config.VALID_SEARCH_MODES:
        raise SystemExit(f"SEARCH_MODE inválido: {search_mode}")
    modelos = config.resolver_modelos(args.modelo, mode)
    perfiles = config.resolver_perfiles(args.perfil)
    golden_path = Path(args.golden) if args.golden else config.GOLDEN_DATASET_PATH
    preguntas = cargar_golden_dataset(golden_path)
    log.info("Golden Dataset: %d preguntas (%s)", len(preguntas), golden_path.name)

    with psycopg.connect(**config.pg_connect_kwargs(), autocommit=True) as conn:
        register_vector(conn)
        iterative_ok = pgvector_version(conn) >= (0, 8)
        for modelo in modelos:
            log.info("══════ Evaluando %s | modo=%s | búsqueda=%s ══════", modelo.id_corto, mode, search_mode)
            detalle, resumen = evaluar_modelo(conn, modelo, perfiles, preguntas, mode, search_mode,
                                              args.skip_model_check, iterative_ok)
            # Se persiste por modelo: los resultados se acumulan entre corridas (un modelo levantado a la vez)
            config.upsert_csv(config.RESULTS_DIR / config.EVAL_DETAIL_CSV, detalle, ["experimento_id", "pregunta_id"])
            config.upsert_csv(config.RESULTS_DIR / config.EVAL_SUMMARY_CSV, resumen, ["experimento_id"])

    generar_reporte(config.RESULTS_DIR)


if __name__ == "__main__":
    main()
