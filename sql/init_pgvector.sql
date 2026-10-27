-- =============================================================================
--  init_pgvector.sql — DDL del benchmark de embeddings Payworks
--  Idempotente: puede re-ejecutarse sin efectos secundarios.
--  Una tabla por dimensión; los experimentos se distinguen por experimento_id
--  con formato  {modelo}__{perfil}   (p.ej. bge-m3__Perfil_B_1500)
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- -----------------------------------------------------------------------------
--  1024 dimensiones: bge-m3, multilingual-e5-large
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS payworks_rag_1024 (
    id              BIGSERIAL PRIMARY KEY,
    experimento_id  VARCHAR(100) NOT NULL,
    modelo          VARCHAR(100) NOT NULL,
    chunk_perfil    VARCHAR(50)  NOT NULL,
    seccion_h1      TEXT,
    seccion_h2      TEXT,
    seccion_h3      TEXT,
    contenido       TEXT         NOT NULL,
    tokens_aprox    INT,
    metadata        JSONB        DEFAULT '{}'::jsonb,
    embedding       vector(1024) NOT NULL,
    created_at      TIMESTAMPTZ  DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_payworks_rag_1024_embedding_hnsw
    ON payworks_rag_1024 USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_payworks_rag_1024_experimento
    ON payworks_rag_1024 USING btree (experimento_id);

-- -----------------------------------------------------------------------------
--  768 dimensiones: embeddinggemma-300m, granite-278m-multilingual
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS payworks_rag_768 (
    id              BIGSERIAL PRIMARY KEY,
    experimento_id  VARCHAR(100) NOT NULL,
    modelo          VARCHAR(100) NOT NULL,
    chunk_perfil    VARCHAR(50)  NOT NULL,
    seccion_h1      TEXT,
    seccion_h2      TEXT,
    seccion_h3      TEXT,
    contenido       TEXT         NOT NULL,
    tokens_aprox    INT,
    metadata        JSONB        DEFAULT '{}'::jsonb,
    embedding       vector(768)  NOT NULL,
    created_at      TIMESTAMPTZ  DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_payworks_rag_768_embedding_hnsw
    ON payworks_rag_768 USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_payworks_rag_768_experimento
    ON payworks_rag_768 USING btree (experimento_id);

-- -----------------------------------------------------------------------------
--  384 dimensiones: all-MiniLM-L6-v2 (línea base negativa)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS payworks_rag_384 (
    id              BIGSERIAL PRIMARY KEY,
    experimento_id  VARCHAR(100) NOT NULL,
    modelo          VARCHAR(100) NOT NULL,
    chunk_perfil    VARCHAR(50)  NOT NULL,
    seccion_h1      TEXT,
    seccion_h2      TEXT,
    seccion_h3      TEXT,
    contenido       TEXT         NOT NULL,
    tokens_aprox    INT,
    metadata        JSONB        DEFAULT '{}'::jsonb,
    embedding       vector(384)  NOT NULL,
    created_at      TIMESTAMPTZ  DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_payworks_rag_384_embedding_hnsw
    ON payworks_rag_384 USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_payworks_rag_384_experimento
    ON payworks_rag_384 USING btree (experimento_id);
