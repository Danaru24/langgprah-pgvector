"""
db_setup.py — Aplica sql/init_pgvector.sql en PostgreSQL y verifica el resultado.

Uso:
    python src/db_setup.py            # crea extensión, tablas e índices (idempotente)
    python src/db_setup.py --check    # solo verifica (no ejecuta DDL)
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import logging  # noqa: E402

import config  # noqa: E402
import psycopg  # noqa: E402

log = logging.getLogger("db_setup")


def aplicar_ddl(conn: psycopg.Connection) -> None:
    ddl = config.SQL_INIT_PATH.read_text(encoding="utf-8")
    log.info("Aplicando %s ...", config.SQL_INIT_PATH.relative_to(config.ROOT_DIR))
    conn.execute(ddl)  # sin parámetros → psycopg permite múltiples sentencias
    log.info("DDL aplicado correctamente.")


def verificar(conn: psycopg.Connection) -> bool:
    ok = True
    fila = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    if not fila:
        log.error("La extensión 'vector' NO está instalada.")
        return False
    log.info("pgvector versión %s", fila[0])

    for dim, tabla in sorted(config.TABLAS_POR_DIM.items(), reverse=True):
        existe = conn.execute("SELECT to_regclass(%s)", (tabla,)).fetchone()[0]
        if not existe:
            log.error("Tabla %s NO existe.", tabla)
            ok = False
            continue
        dim_real = conn.execute(
            "SELECT atttypmod FROM pg_attribute WHERE attrelid = %s::regclass AND attname = 'embedding'",
            (tabla,),
        ).fetchone()[0]
        indices = [
            r[0]
            for r in conn.execute(
                "SELECT indexname FROM pg_indexes WHERE tablename = %s ORDER BY indexname", (tabla,)
            ).fetchall()
        ]
        experimentos = conn.execute(
            f"SELECT experimento_id, count(*) FROM {tabla} GROUP BY 1 ORDER BY 1"  # noqa: S608 (whitelist)
        ).fetchall()
        estado = "OK" if dim_real == dim else f"DIM INCORRECTA ({dim_real})"
        ok &= dim_real == dim
        log.info("%-20s vector(%s) %s | índices: %s", tabla, dim, estado, ", ".join(indices))
        for exp, n in experimentos:
            log.info("    · %-45s %5d chunks", exp, n)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Inicializa pgvector para el benchmark Payworks")
    parser.add_argument("--check", action="store_true", help="Solo verificar, no aplicar DDL")
    args = parser.parse_args()
    config.setup_logging()

    kwargs = config.pg_connect_kwargs()
    destino = kwargs.get("host") or "DATABASE_URL"
    log.info("Conectando a PostgreSQL (%s) ...", destino)
    try:
        with psycopg.connect(**kwargs, autocommit=True) as conn:
            if not args.check:
                aplicar_ddl(conn)
            if not verificar(conn):
                sys.exit(1)
    except psycopg.errors.InsufficientPrivilege as exc:
        log.error(
            "Privilegios insuficientes (%s). 'CREATE EXTENSION vector' requiere un rol con permisos: "
            "pide al DBA que la habilite y re-ejecuta.", exc,
        )
        sys.exit(1)
    except psycopg.OperationalError as exc:
        log.error("No se pudo conectar a PostgreSQL: %s", exc)
        sys.exit(1)
    log.info("Infraestructura pgvector lista.")


if __name__ == "__main__":
    main()
