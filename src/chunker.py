"""
chunker.py — Chunking híbrido de 2 etapas + inyección de contexto jerárquico.

  Etapa 1: MarkdownHeaderTextSplitter  (#→h1, ##→h2, ###→h3; encabezados a metadatos)
  Etapa 2: RecursiveCharacterTextSplitter (respeta saltos de línea y oraciones)
  Contexto: cada chunk se vectoriza como  "[h1 > h2 > h3]\\n{contenido}"

Nota metodológica: como la Etapa 1 corta primero por sección, `chunk_size` actúa
como COTA SUPERIOR. Si una sección mide menos que `chunk_size`, los perfiles
B/C producirán el mismo chunk para esa sección (visible en "chars promedio").

Uso CLI (inspección sin BD ni modelos):
    python src/chunker.py                 # estadísticas de los 3 perfiles
    python src/chunker.py --show 3        # + primeros 3 chunks de cada perfil
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import statistics  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import config  # noqa: E402  (fija HF_HOME, carga .env)
from config import CHUNK_PERFILES, ChunkPerfil  # noqa: E402
from langchain_text_splitters import (  # noqa: E402
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]
SEPARATORS = ["\n### ", "\n## ", "\n\n", "\n", ". ", " "]
MIN_CHUNK_CHARS = 30  # descarta fragmentos residuales (separadores, líneas sueltas)

PLANTILLA_DOC = """# Payworks — Documentación Técnica Maestra

> PLANTILLA: reemplace este archivo por la documentación oficial (~20,000 caracteres).

## Conceptos y Glosario

### Término de ejemplo

Definición del término.
"""


@dataclass(frozen=True)
class Chunk:
    index: int
    perfil: str
    h1: str
    h2: str
    h3: str
    texto: str

    @property
    def titulo_jerarquico(self) -> str:
        return " > ".join(p for p in (self.h1, self.h2, self.h3) if p)

    @property
    def texto_contextualizado(self) -> str:
        """Texto que se vectoriza: ruta jerárquica + contenido."""
        titulo = self.titulo_jerarquico
        return f"[{titulo}]\n{self.texto}" if titulo else self.texto


def leer_documento(path: Path | str | None = None) -> str:
    ruta = Path(path) if path else config.DOC_PATH
    if not ruta.exists():
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(PLANTILLA_DOC, encoding="utf-8")
        raise SystemExit(
            f"No existía {ruta}. Se creó una plantilla: reemplázala con el documento real y re-ejecuta."
        )
    return ruta.read_text(encoding="utf-8")


def sha256_corto(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:12]


def split_por_encabezados(markdown: str):
    """Etapa 1: secciones con metadatos h1/h2/h3 (encabezados removidos del cuerpo)."""
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=True,
    )
    return splitter.split_text(markdown)


def _crear_splitter_recursivo(perfil: ChunkPerfil) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=perfil.chunk_size,
        chunk_overlap=perfil.chunk_overlap,
        separators=SEPARATORS,
        keep_separator="end",  # el ". " queda al final de la oración, no al inicio del siguiente chunk
        length_function=len,
        strip_whitespace=True,
    )


def chunk_documento(markdown: str, perfil: ChunkPerfil) -> list[Chunk]:
    """Aplica las 2 etapas y devuelve chunks con su jerarquía."""
    secciones = split_por_encabezados(markdown)
    recursivo = _crear_splitter_recursivo(perfil)

    chunks: list[Chunk] = []
    for seccion in secciones:
        meta = seccion.metadata or {}
        # MarkdownHeaderTextSplitter une párrafos con "  \n"; se restauran como "\n\n"
        # para que el separador de párrafo de la Etapa 2 sea efectivo.
        contenido = seccion.page_content.replace("  \n", "\n\n")
        for pieza in recursivo.split_text(contenido):
            pieza = pieza.strip()
            if len("".join(pieza.split())) < MIN_CHUNK_CHARS:
                continue
            chunks.append(
                Chunk(
                    index=len(chunks),
                    perfil=perfil.nombre,
                    h1=str(meta.get("h1", "")).strip(),
                    h2=str(meta.get("h2", "")).strip(),
                    h3=str(meta.get("h3", "")).strip(),
                    texto=pieza,
                )
            )
    return chunks


def estadisticas(chunks: list[Chunk]) -> dict:
    largos = [len(c.texto_contextualizado) for c in chunks] or [0]
    return {
        "n_chunks": len(chunks),
        "chars_promedio": round(statistics.mean(largos), 1),
        "chars_min": min(largos),
        "chars_max": max(largos),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspección del chunking híbrido")
    parser.add_argument("--doc", default=None, help="Ruta al markdown (default: DOC_PATH)")
    parser.add_argument("--perfil", nargs="*", default=None, help="Perfiles (default: all)")
    parser.add_argument("--show", type=int, default=0, help="Muestra los primeros N chunks")
    args = parser.parse_args()

    markdown = leer_documento(args.doc)
    print(f"Documento: {args.doc or config.DOC_PATH} | {len(markdown):,} chars | sha256={sha256_corto(markdown)}")
    for perfil in config.resolver_perfiles(args.perfil):
        chunks = chunk_documento(markdown, perfil)
        st = estadisticas(chunks)
        print(
            f"\n=== {perfil.nombre} (size={perfil.chunk_size}, overlap={perfil.chunk_overlap}) → "
            f"{st['n_chunks']} chunks | chars prom={st['chars_promedio']} "
            f"min={st['chars_min']} max={st['chars_max']}"
        )
        for c in chunks[: args.show]:
            print(f"\n--- chunk #{c.index} ({len(c.texto_contextualizado)} chars) ---")
            print(c.texto_contextualizado)


if __name__ == "__main__":
    main()

__all__ = ["Chunk", "chunk_documento", "leer_documento", "estadisticas", "CHUNK_PERFILES"]
