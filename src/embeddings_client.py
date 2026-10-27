"""
embeddings_client.py — Cliente unificado de embeddings (solo CPU).

Modos:
  * local_cpu        → SentenceTransformers en CPU (pesos en ./models_cache).
  * openai_endpoint  → POST {base_url}/embeddings (vLLM / HF TEI en OpenShift AI).
                       Verifica que el modelo levantado sea el solicitado (solo uno a la vez).

Post-procesado común (garantiza comparabilidad entre modos):
  * Truncado MRL a `dim` (solo modelos Matryoshka, p.ej. EmbeddingGemma) + renormalización L2.
  * Normalización L2 de todos los vectores (coseno == producto punto).

Los prefijos doc/query se aplican MANUALMENTE (no se usan los prompts internos de
SentenceTransformers) para que ambos modos envíen exactamente el mismo texto.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import gc  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from typing import Sequence  # noqa: E402

import numpy as np  # noqa: E402
import requests  # noqa: E402
from requests.adapters import HTTPAdapter  # noqa: E402
from tqdm import tqdm  # noqa: E402
from urllib3.util.retry import Retry  # noqa: E402

import config  # noqa: E402
from config import ModeloConfig  # noqa: E402

log = logging.getLogger("embeddings")


class EmbeddingsClient:
    """Interfaz única: format_* / embed_texts / embed_query / count_tokens."""

    def __init__(self, modelo: ModeloConfig, mode: str | None = None, batch_size: int | None = None):
        self.modelo = modelo
        self.mode = config.validar_mode(mode or config.INFERENCE_MODE)
        self.batch_size = max(1, batch_size or config.EMBED_BATCH_SIZE)
        self.load_seconds = 0.0
        self.token_counter = "heuristica_chars_div_4"

        self._st_model = None
        self._tokenizer = None
        self._tokenizer_intentado = False
        self._session: requests.Session | None = None
        self._endpoint: dict = {}

        if self.mode == "local_cpu":
            self._init_local()
        else:
            self._init_endpoint()

    # ------------------------------------------------------------------ init
    def _init_local(self) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        if config.TORCH_NUM_THREADS > 0:
            torch.set_num_threads(config.TORCH_NUM_THREADS)
        log.info(
            "Cargando %s en CPU (threads=%s, cache=%s)",
            self.modelo.hf_id, torch.get_num_threads(), config.HF_HOME,
        )
        t0 = time.perf_counter()
        self._st_model = SentenceTransformer(
            self.modelo.hf_id,
            device="cpu",
            trust_remote_code=self.modelo.trust_remote_code,
            token=config.HF_TOKEN,
        )
        self._st_model.max_seq_length = self.modelo.max_seq_length
        self.load_seconds = time.perf_counter() - t0
        self._tokenizer = self._st_model.tokenizer
        self._tokenizer_intentado = True
        self.token_counter = "hf_tokenizer"
        self._silenciar_transformers()
        log.info("Modelo cargado en %.1fs (max_seq_length=%s)", self.load_seconds, self.modelo.max_seq_length)

    def _init_endpoint(self) -> None:
        self._endpoint = config.endpoint_settings(self.modelo)
        if not self._endpoint["base_url"]:
            raise SystemExit(
                "Modo openai_endpoint sin URL: define EMBEDDING_BASE_URL "
                f"o EMBEDDING_BASE_URL__{config._env_key(self.modelo.id_corto)} en .env"
            )
        retry = Retry(
            total=config.ENDPOINT_MAX_RETRIES,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "POST"}),
            raise_on_status=False,
        )
        session = requests.Session()
        session.mount("http://", HTTPAdapter(max_retries=retry))
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.headers.update({"Content-Type": "application/json"})
        if self._endpoint["api_key"]:
            session.headers["Authorization"] = f"Bearer {self._endpoint['api_key']}"
        session.verify = config.endpoint_verify_ssl()
        if session.verify is False:
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        self._session = session
        log.info(
            "Endpoint %s | served_model_name=%s",
            self._endpoint["base_url"], self._endpoint["served_model_name"],
        )

    @staticmethod
    def _silenciar_transformers() -> None:
        try:
            from transformers.utils import logging as hf_logging

            hf_logging.set_verbosity_error()  # oculta "Token indices sequence length is longer..."
        except Exception:  # pragma: no cover
            pass

    # ------------------------------------------------------------ formatting
    def format_document(self, texto: str, titulo: str = "") -> str:
        return self.modelo.doc_prefix.format(texto=texto, titulo=titulo or "none")

    def format_query(self, pregunta: str) -> str:
        return self.modelo.query_prefix.format(pregunta=pregunta)

    # ------------------------------------------------------------- endpoint
    @property
    def served_model_name(self) -> str:
        return self._endpoint.get("served_model_name", self.modelo.hf_id)

    def check_endpoint(self) -> list[str]:
        """Verifica que el endpoint sirva el modelo solicitado (solo uno levantado a la vez)."""
        if self.mode != "openai_endpoint":
            return []
        base = self._endpoint["base_url"]
        timeout = min(config.ENDPOINT_TIMEOUT_S, 30)
        ids: list[str] = []
        try:  # vLLM / OpenAI
            r = self._session.get(f"{base}/models", timeout=timeout)
            if r.ok:
                ids = [str(m.get("id", "")) for m in r.json().get("data", []) if m.get("id")]
        except (requests.RequestException, ValueError):
            pass
        if not ids:  # HF TEI
            try:
                r = self._session.get(f"{base[: -len('/v1')]}/info", timeout=timeout)
                if r.ok and r.json().get("model_id"):
                    ids = [str(r.json()["model_id"])]
            except (requests.RequestException, ValueError):
                pass
        if not ids:
            log.warning("No se pudo verificar el modelo servido en %s; se continúa.", base)
            return []

        esperado = self.served_model_name.lower()
        cola_hf = self.modelo.hf_id.split("/")[-1].lower()
        corto = self.modelo.id_corto.lower()

        matching_id = None
        for i in ids:
            i_low = i.lower()
            if i_low == esperado:
                matching_id = i
                break
            if cola_hf in i_low or i_low in (corto, cola_hf):
                matching_id = i

        if not matching_id:
            raise SystemExit(
                f"El endpoint {base} sirve {ids}, pero se solicitó --modelo {self.modelo.id_corto} "
                f"({self.served_model_name}). Solo hay un modelo levantado a la vez: despliega el modelo "
                "correcto, ajusta EMBEDDING_SERVED_MODEL_NAME o usa --skip-model-check."
            )

        # Si el usuario no especificó un nombre explícito en .env, tomamos como referencia
        # el nombre exacto devuelto por /v1/models (p.ej. 'redhataibge-m3') para evitar HTTP 404.
        if not self._endpoint.get("is_served_name_explicit") and self.served_model_name != matching_id:
            log.info(
                "Alineando served_model_name con el nombre retornado por /v1/models: '%s' (en lugar de '%s')",
                matching_id,
                self.served_model_name,
            )
            self._endpoint["served_model_name"] = matching_id

        log.info("Endpoint verificado: sirve %s (modelo activo en peticiones: %s)", ids, self.served_model_name)
        return ids

    def _post_embeddings(self, textos: Sequence[str]) -> np.ndarray:
        payload: dict = {
            "model": self.served_model_name,
            "input": list(textos),
            "encoding_format": "float",
        }
        if config.ENDPOINT_TRUNCATE_PROMPT_TOKENS > 0:
            payload["truncate_prompt_tokens"] = config.ENDPOINT_TRUNCATE_PROMPT_TOKENS
        url = f"{self._endpoint['base_url']}/embeddings"
        r = self._session.post(url, json=payload, timeout=config.ENDPOINT_TIMEOUT_S)
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code} en {url}: {r.text[:500]}")
        data = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
        if len(data) != len(textos):
            raise RuntimeError(f"El endpoint devolvió {len(data)} vectores para {len(textos)} textos")
        return np.asarray([d["embedding"] for d in data], dtype=np.float32)

    # ------------------------------------------------------------ embedding
    def _embed_batch(self, textos: Sequence[str]) -> np.ndarray:
        if self.mode == "local_cpu":
            return self._st_model.encode(
                list(textos),
                batch_size=len(textos),
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
                prompt="",  # desactiva default prompts: los prefijos se aplican manualmente
            ).astype(np.float32)
        return self._post_embeddings(textos)

    def _postprocesar(self, vectores: np.ndarray) -> np.ndarray:
        dim_obtenida = vectores.shape[1]
        if dim_obtenida > self.modelo.dim:
            if not self.modelo.mrl:
                raise RuntimeError(
                    f"{self.modelo.id_corto} devolvió dim={dim_obtenida} (esperada {self.modelo.dim}) "
                    "y no es un modelo Matryoshka: ¿está levantado el modelo correcto?"
                )
            vectores = vectores[:, : self.modelo.dim]  # truncado MRL
        elif dim_obtenida < self.modelo.dim:
            raise RuntimeError(
                f"{self.modelo.id_corto} devolvió dim={dim_obtenida} < {self.modelo.dim}: modelo incorrecto."
            )
        normas = np.linalg.norm(vectores, axis=1, keepdims=True)
        return (vectores / np.clip(normas, 1e-12, None)).astype(np.float32)

    def embed_texts(self, textos: Sequence[str], show_progress: bool = False, desc: str = "embeddings") -> np.ndarray:
        """Vectoriza textos YA formateados (con prefijo). Devuelve (n, dim) normalizado L2."""
        if not textos:
            return np.zeros((0, self.modelo.dim), dtype=np.float32)
        partes: list[np.ndarray] = []
        with tqdm(total=len(textos), desc=desc, unit="chunk", disable=not show_progress,
                  dynamic_ncols=True, file=sys.stdout) as barra:
            for i in range(0, len(textos), self.batch_size):
                lote = textos[i: i + self.batch_size]
                partes.append(self._embed_batch(lote))
                barra.update(len(lote))
        return self._postprocesar(np.vstack(partes))

    def embed_query(self, pregunta: str) -> np.ndarray:
        return self.embed_texts([self.format_query(pregunta)])[0]

    def warmup(self) -> None:
        """Excluye de las latencias la carga perezosa / primera llamada (JIT, conexiones)."""
        self.embed_texts([self.format_query("calentamiento del modelo")])
        self.embed_texts([self.format_document("calentamiento del modelo", "warmup")])

    # --------------------------------------------------------------- tokens
    def _get_tokenizer(self):
        if self._tokenizer_intentado:
            return self._tokenizer
        self._tokenizer_intentado = True
        if not config.USE_HF_TOKENIZER:
            return None
        try:
            from transformers import AutoTokenizer

            self._silenciar_transformers()
            self._tokenizer = AutoTokenizer.from_pretrained(self.modelo.hf_id, token=config.HF_TOKEN)
            self.token_counter = "hf_tokenizer"
        except Exception as exc:  # gated / sin red / sin transformers
            log.warning("Tokenizer de %s no disponible (%s); se usa chars/4.", self.modelo.hf_id, exc)
            self._tokenizer = None
        return self._tokenizer

    def count_tokens(self, textos: Sequence[str]) -> list[int]:
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return [max(1, math.ceil(len(t) / 4)) for t in textos]
        ids = tokenizer(list(textos), add_special_tokens=True, truncation=False)["input_ids"]
        return [len(x) for x in ids]

    # -------------------------------------------------------------- cleanup
    def close(self) -> None:
        self._st_model = None
        self._tokenizer = None
        if self._session is not None:
            self._session.close()
            self._session = None
        gc.collect()

    def __enter__(self) -> "EmbeddingsClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
