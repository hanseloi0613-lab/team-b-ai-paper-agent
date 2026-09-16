-- ============================================================
-- TEAM B
-- Core Corpus Schema
-- PostgreSQL / AWS RDS
-- ============================================================


-- ============================================================
-- 1. Core Documents
-- ============================================================

CREATE TABLE IF NOT EXISTS public.core_documents (

    id BIGSERIAL PRIMARY KEY,

    -- --------------------------------------------------------
    -- Source identity
    -- --------------------------------------------------------

    source VARCHAR(50) NOT NULL,

    source_id VARCHAR(255) NOT NULL,

    -- --------------------------------------------------------
    -- Basic metadata
    -- --------------------------------------------------------

    title TEXT NOT NULL,

    abstract TEXT,

    authors JSONB NOT NULL
        DEFAULT '[]'::jsonb,

    categories JSONB NOT NULL
        DEFAULT '[]'::jsonb,

    published_at DATE,

    document_type VARCHAR(100),

    doi TEXT,

    url TEXT,

    pdf_url TEXT,

    -- --------------------------------------------------------
    -- Project classification
    -- --------------------------------------------------------

    topic_axis VARCHAR(50) NOT NULL,

    language VARCHAR(10) NOT NULL
        DEFAULT 'en',

    -- --------------------------------------------------------
    -- Content
    -- --------------------------------------------------------

    raw_content TEXT,

    clean_content TEXT,

    -- --------------------------------------------------------
    -- Dedup / normalization
    -- --------------------------------------------------------

    content_hash VARCHAR(64),

    normalization_version VARCHAR(50),

    char_count INTEGER,

    parse_status VARCHAR(50),

    -- --------------------------------------------------------
    -- Provenance / pipeline information
    -- --------------------------------------------------------

    metadata JSONB NOT NULL
        DEFAULT '{}'::jsonb,

    -- --------------------------------------------------------
    -- Timestamps
    -- --------------------------------------------------------

    created_at TIMESTAMPTZ NOT NULL
        DEFAULT NOW(),

    updated_at TIMESTAMPTZ NOT NULL
        DEFAULT NOW(),

    -- --------------------------------------------------------
    -- Constraints
    -- --------------------------------------------------------

    CONSTRAINT uq_core_documents_source_source_id
        UNIQUE (
            source,
            source_id
        ),

    CONSTRAINT uq_core_documents_content_hash
        UNIQUE (
            content_hash
        ),

    CONSTRAINT chk_core_documents_language
        CHECK (
            language = 'en'
        ),

    CONSTRAINT chk_core_documents_topic_axis
        CHECK (
            topic_axis IN (
                'rover_autonomy',
                'onboard_ai',
                'satellite_autonomy'
            )
        )
);


-- ============================================================
-- 2. Indexes
-- ============================================================

CREATE INDEX IF NOT EXISTS
    idx_core_documents_source
ON public.core_documents (
    source
);


CREATE INDEX IF NOT EXISTS
    idx_core_documents_topic_axis
ON public.core_documents (
    topic_axis
);


CREATE INDEX IF NOT EXISTS
    idx_core_documents_published_at
ON public.core_documents (
    published_at
);


CREATE INDEX IF NOT EXISTS
    idx_core_documents_parse_status
ON public.core_documents (
    parse_status
);


-- ============================================================
-- 3. Basic verification
-- ============================================================

SELECT
    table_schema,
    table_name
FROM information_schema.tables
WHERE
    table_schema = 'public'
    AND table_name = 'core_documents';