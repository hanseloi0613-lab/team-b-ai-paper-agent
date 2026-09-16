-- ============================================================
-- TEAM B
-- Core-100 pgvector schema
--
-- Selected dimension:
--     256
--
-- Basis:
--     Core-100 retrieval evaluation
--
--     SVD-256
--       Precision@6 = 0.716667
--       Recall@6    = 0.132822
--       nDCG@6      = 0.744323
--       MRR         = 0.907262
--       Hit@1       = 0.860000
--
-- IMPORTANT
--
-- Current corpus:
--     100 documents
--     3113 chunks
--
-- For this small corpus we intentionally do NOT add HNSW yet.
--
-- Exact cosine search over 3113 vectors is fast enough,
-- easier to validate, and avoids approximate-search noise.
--
-- HNSW will be added after corpus scaling if necessary.
-- ============================================================


-- ============================================================
-- 1. pgvector extension
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;


-- ============================================================
-- 2. Add vector column
-- ============================================================

ALTER TABLE public.core_chunks
ADD COLUMN IF NOT EXISTS embedding vector(256);


-- ============================================================
-- 3. Embedding metadata
-- ============================================================

ALTER TABLE public.core_chunks
ADD COLUMN IF NOT EXISTS embedding_version VARCHAR(100);


ALTER TABLE public.core_chunks
ADD COLUMN IF NOT EXISTS embedded_at TIMESTAMPTZ;


-- ============================================================
-- 4. Supporting index
--
-- This is NOT the vector similarity index.
--
-- It is only for filtering/checking embedding versions.
-- ============================================================

CREATE INDEX IF NOT EXISTS
idx_core_chunks_embedding_version
ON public.core_chunks (embedding_version);


-- ============================================================
-- 5. Comments
-- ============================================================

COMMENT ON COLUMN public.core_chunks.embedding IS
'TEAM B SVD dense retrieval vector. Core-100 RAG v1 dimension = 256.';


COMMENT ON COLUMN public.core_chunks.embedding_version IS
'Version of TF-IDF/SVD pipeline used to create the embedding.';


COMMENT ON COLUMN public.core_chunks.embedded_at IS
'Timestamp when the embedding was stored.';


-- ============================================================
-- 6. Schema verification
-- ============================================================

SELECT
    column_name,
    data_type,
    udt_name
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'core_chunks'
  AND column_name IN (
      'embedding',
      'embedding_version',
      'embedded_at'
  )
ORDER BY column_name;


-- ============================================================
-- 7. pgvector dimension verification
-- ============================================================

SELECT
    a.attname AS column_name,
    format_type(
        a.atttypid,
        a.atttypmod
    ) AS column_type
FROM pg_attribute a
JOIN pg_class c
  ON c.oid = a.attrelid
JOIN pg_namespace n
  ON n.oid = c.relnamespace
WHERE n.nspname = 'public'
  AND c.relname = 'core_chunks'
  AND a.attname = 'embedding'
  AND a.attnum > 0
  AND NOT a.attisdropped;


-- ============================================================
-- 8. Current state
-- ============================================================

SELECT
    COUNT(*) AS total_chunks,
    COUNT(embedding) AS embedded_chunks,
    COUNT(*) FILTER (
        WHERE embedding IS NULL
    ) AS missing_embeddings
FROM public.core_chunks;