-- ============================================================
-- TEAM B - Core RAG Chunk Schema
-- Version: core100_section_v1
-- ============================================================
--
-- Purpose
--
-- public.core_documents
--          ↓
-- section-aware chunking
--          ↓
-- public.core_chunks
--
--
-- IMPORTANT
--
-- 1. 지금은 embedding/vector 컬럼을 만들지 않는다.
-- 2. TF-IDF → TruncatedSVD 실험 후
--    실제 vector dimension이 확정되면
--    별도의 migration에서 pgvector 컬럼을 추가한다.
--
-- 3. core_documents는 원본 document 단위
--    core_chunks는 retrieval 단위
--
-- Relationship:
--
-- core_documents 1 : N core_chunks
--
-- ============================================================


-- ============================================================
-- 1. Core Chunks Table
-- ============================================================

CREATE TABLE IF NOT EXISTS public.core_chunks (

    -- --------------------------------------------------------
    -- Primary Key
    -- --------------------------------------------------------

    id BIGSERIAL PRIMARY KEY,


    -- --------------------------------------------------------
    -- Parent Document
    -- --------------------------------------------------------

    document_id BIGINT NOT NULL,


    -- --------------------------------------------------------
    -- Chunk Position
    -- --------------------------------------------------------

    chunk_index INTEGER NOT NULL,

    section_index INTEGER NOT NULL,


    -- --------------------------------------------------------
    -- Section Metadata
    -- --------------------------------------------------------

    section_heading TEXT,


    -- --------------------------------------------------------
    -- Actual Chunk Body
    --
    -- 최종 생성기에 evidence로 넘길 본문
    -- --------------------------------------------------------

    chunk_text TEXT NOT NULL,


    -- --------------------------------------------------------
    -- Retrieval Text
    --
    -- TF-IDF / SVD에 실제 입력할 검색용 텍스트
    --
    -- 구성:
    --
    -- title
    -- +
    -- section heading
    -- +
    -- chunk body
    -- --------------------------------------------------------

    retrieval_text TEXT NOT NULL,


    -- --------------------------------------------------------
    -- Statistics
    -- --------------------------------------------------------

    word_count INTEGER NOT NULL,

    token_estimate INTEGER NOT NULL,

    char_count INTEGER NOT NULL,


    -- --------------------------------------------------------
    -- Integrity / Duplicate Detection
    -- --------------------------------------------------------

    content_hash VARCHAR(64) NOT NULL,


    -- --------------------------------------------------------
    -- Pipeline Version
    -- --------------------------------------------------------

    chunker_version VARCHAR(50) NOT NULL,


    -- --------------------------------------------------------
    -- Timestamp
    -- --------------------------------------------------------

    created_at TIMESTAMPTZ NOT NULL
        DEFAULT NOW(),


    -- ========================================================
    -- Foreign Key
    -- ========================================================

    CONSTRAINT fk_core_chunks_document
        FOREIGN KEY (document_id)
        REFERENCES public.core_documents(id)
        ON DELETE CASCADE,


    -- ========================================================
    -- Unique Constraint
    --
    -- 한 document 안에서 chunk_index는 중복 불가
    -- ========================================================

    CONSTRAINT uq_core_chunks_document_index
        UNIQUE (
            document_id,
            chunk_index
        ),


    -- ========================================================
    -- Check Constraints
    -- ========================================================

    CONSTRAINT chk_core_chunks_chunk_index
        CHECK (
            chunk_index >= 0
        ),

    CONSTRAINT chk_core_chunks_section_index
        CHECK (
            section_index >= 0
        ),

    CONSTRAINT chk_core_chunks_word_count
        CHECK (
            word_count > 0
        ),

    CONSTRAINT chk_core_chunks_token_estimate
        CHECK (
            token_estimate > 0
        ),

    CONSTRAINT chk_core_chunks_char_count
        CHECK (
            char_count > 0
        ),

    CONSTRAINT chk_core_chunks_chunk_text
        CHECK (
            length(
                trim(chunk_text)
            ) > 0
        ),

    CONSTRAINT chk_core_chunks_retrieval_text
        CHECK (
            length(
                trim(retrieval_text)
            ) > 0
        ),

    CONSTRAINT chk_core_chunks_content_hash
        CHECK (
            length(
                trim(content_hash)
            ) = 64
        )
);


-- ============================================================
-- 2. Indexes
-- ============================================================


-- ------------------------------------------------------------
-- Document → Chunks 조회
-- ------------------------------------------------------------

CREATE INDEX IF NOT EXISTS
    idx_core_chunks_document_id
ON public.core_chunks (
    document_id
);


-- ------------------------------------------------------------
-- Chunker Version 조회
-- ------------------------------------------------------------

CREATE INDEX IF NOT EXISTS
    idx_core_chunks_chunker_version
ON public.core_chunks (
    chunker_version
);


-- ------------------------------------------------------------
-- Duplicate / Hash lookup
-- ------------------------------------------------------------

CREATE INDEX IF NOT EXISTS
    idx_core_chunks_content_hash
ON public.core_chunks (
    content_hash
);


-- ------------------------------------------------------------
-- Section lookup
-- ------------------------------------------------------------

CREATE INDEX IF NOT EXISTS
    idx_core_chunks_section_heading
ON public.core_chunks (
    section_heading
);


-- ------------------------------------------------------------
-- Document + section
-- ------------------------------------------------------------

CREATE INDEX IF NOT EXISTS
    idx_core_chunks_document_section
ON public.core_chunks (
    document_id,
    section_index
);


-- ============================================================
-- 3. Comments
-- ============================================================

COMMENT ON TABLE public.core_chunks IS
'Section-aware RAG chunks derived from public.core_documents.';


COMMENT ON COLUMN public.core_chunks.document_id IS
'Foreign key referencing public.core_documents.id.';


COMMENT ON COLUMN public.core_chunks.chunk_index IS
'Sequential chunk index within one document, starting at 0.';


COMMENT ON COLUMN public.core_chunks.section_index IS
'Sequential section index within one document.';


COMMENT ON COLUMN public.core_chunks.section_heading IS
'Original or normalized scientific document section heading.';


COMMENT ON COLUMN public.core_chunks.chunk_text IS
'Chunk body used as grounded evidence for generation.';


COMMENT ON COLUMN public.core_chunks.retrieval_text IS
'Search text used for TF-IDF/SVD retrieval: title + section heading + chunk body.';


COMMENT ON COLUMN public.core_chunks.word_count IS
'Whitespace-based approximate word count of chunk_text.';


COMMENT ON COLUMN public.core_chunks.token_estimate IS
'Approximate English token count. Not tokenizer-exact.';


COMMENT ON COLUMN public.core_chunks.char_count IS
'Character count of chunk_text.';


COMMENT ON COLUMN public.core_chunks.content_hash IS
'SHA-256 hash of chunk_text used for duplicate/integrity checks.';


COMMENT ON COLUMN public.core_chunks.chunker_version IS
'Version identifier of the chunking pipeline.';


-- ============================================================
-- 4. Verification
-- ============================================================

SELECT
    table_schema,
    table_name
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name = 'core_chunks';


-- ============================================================
-- 5. Column Verification
-- ============================================================

SELECT
    ordinal_position,
    column_name,
    data_type,
    udt_name,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'core_chunks'
ORDER BY ordinal_position;


-- ============================================================
-- 6. Constraint Verification
-- ============================================================

SELECT
    tc.constraint_name,
    tc.constraint_type
FROM information_schema.table_constraints tc
WHERE tc.table_schema = 'public'
  AND tc.table_name = 'core_chunks'
ORDER BY
    tc.constraint_type,
    tc.constraint_name;


-- ============================================================
-- 7. Index Verification
-- ============================================================

SELECT
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
  AND tablename = 'core_chunks'
ORDER BY indexname;


-- ============================================================
-- 8. Initial Row Verification
-- ============================================================

SELECT
    COUNT(*) AS core_chunks_count
FROM public.core_chunks;