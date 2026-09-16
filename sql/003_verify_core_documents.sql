-- ============================================================
-- TEAM B
-- Core Pilot 30-paper Formal QA
--
-- Expected:
--
-- arxiv 15
-- ntrs  15
--
-- onboard_ai          10
-- rover_autonomy      10
-- satellite_autonomy  10
--
-- Total               30
-- ============================================================


-- ============================================================
-- 1. TOTAL ROW COUNT
-- ============================================================

SELECT
    COUNT(*) AS total_documents
FROM public.core_documents;


-- EXPECTED:
-- 30


-- ============================================================
-- 2. SOURCE DISTRIBUTION
-- ============================================================

SELECT
    source,
    COUNT(*) AS document_count
FROM public.core_documents
GROUP BY source
ORDER BY source;


-- EXPECTED:
--
-- arxiv   15
-- ntrs    15


-- ============================================================
-- 3. TOPIC AXIS DISTRIBUTION
-- ============================================================

SELECT
    topic_axis,
    COUNT(*) AS document_count
FROM public.core_documents
GROUP BY topic_axis
ORDER BY topic_axis;


-- EXPECTED:
--
-- onboard_ai           10
-- rover_autonomy       10
-- satellite_autonomy   10


-- ============================================================
-- 4. SOURCE × TOPIC AXIS MATRIX
-- ============================================================

SELECT
    source,
    topic_axis,
    COUNT(*) AS document_count
FROM public.core_documents
GROUP BY
    source,
    topic_axis
ORDER BY
    source,
    topic_axis;


-- EXPECTED:
--
-- arxiv | onboard_ai           | 5
-- arxiv | rover_autonomy       | 5
-- arxiv | satellite_autonomy   | 5
-- ntrs  | onboard_ai           | 5
-- ntrs  | rover_autonomy       | 5
-- ntrs  | satellite_autonomy   | 5


-- ============================================================
-- 5. ALL 30 DOCUMENTS
-- ============================================================

SELECT
    id,
    source,
    source_id,
    topic_axis,
    title,
    published_at,
    document_type,
    language,
    char_count,
    parse_status
FROM public.core_documents
ORDER BY
    source,
    topic_axis,
    id;


-- 사람이 실제 제목까지 훑는다.
--
-- 체크:
-- 1. 이상한 주제 문서가 없는가?
-- 2. axis가 잘못 붙은 문서가 없는가?
-- 3. title이 비정상적으로 깨지지 않았는가?


-- ============================================================
-- 6. REQUIRED CONTENT NULL / EMPTY CHECK
-- ============================================================

SELECT
    id,
    source,
    source_id,
    title,

    raw_content IS NULL
        AS raw_is_null,

    clean_content IS NULL
        AS clean_is_null,

    LENGTH(
        COALESCE(
            raw_content,
            ''
        )
    ) AS raw_chars,

    LENGTH(
        COALESCE(
            clean_content,
            ''
        )
    ) AS clean_chars

FROM public.core_documents

WHERE
    raw_content IS NULL
    OR clean_content IS NULL
    OR LENGTH(
        TRIM(
            COALESCE(
                raw_content,
                ''
            )
        )
    ) = 0
    OR LENGTH(
        TRIM(
            COALESCE(
                clean_content,
                ''
            )
        )
    ) = 0

ORDER BY id;


-- EXPECTED:
-- 0 rows


-- ============================================================
-- 7. RAW / CLEAN LENGTH QA
-- ============================================================

SELECT
    id,
    source,
    source_id,
    topic_axis,
    title,

    LENGTH(
        raw_content
    ) AS raw_chars,

    LENGTH(
        clean_content
    ) AS clean_chars,

    ROUND(
        (
            1
            -
            LENGTH(
                clean_content
            )::numeric
            /
            NULLIF(
                LENGTH(
                    raw_content
                ),
                0
            )
        )
        * 100,
        2
    ) AS reduction_percent

FROM public.core_documents

ORDER BY
    source,
    topic_axis,
    id;


-- 검토:
--
-- clean이 0 또는 지나치게 작은 문서가 없는지 확인.
--
-- reduction_percent가 문서마다 다른 것은 정상.
-- Cleaner가 무조건 많이 지우는 것이 목적이 아니다.


-- ============================================================
-- 8. SUSPICIOUSLY SHORT CLEAN DOCUMENTS
-- ============================================================

SELECT
    id,
    source,
    source_id,
    topic_axis,
    title,
    char_count,

    LENGTH(
        clean_content
    ) AS actual_clean_chars

FROM public.core_documents

WHERE
    LENGTH(
        clean_content
    ) < 5000

ORDER BY
    LENGTH(
        clean_content
    );


-- EXPECTED:
-- 0 rows
--
-- 현재 quality threshold:
-- clean_content >= 5000 characters


-- ============================================================
-- 9. DB char_count CONSISTENCY
-- ============================================================

SELECT
    id,
    source,
    source_id,
    title,
    char_count,
    LENGTH(
        clean_content
    ) AS actual_clean_chars

FROM public.core_documents

WHERE
    char_count
    <>
    LENGTH(
        clean_content
    )

ORDER BY id;


-- EXPECTED:
-- 0 rows
--
-- DB char_count와 실제 clean_content 길이가
-- 정확히 일치해야 함.


-- ============================================================
-- 10. PARSE STATUS
-- ============================================================

SELECT
    parse_status,
    COUNT(*) AS document_count
FROM public.core_documents
GROUP BY parse_status
ORDER BY parse_status;


-- EXPECTED:
--
-- success | 30


-- ============================================================
-- 11. NON-SUCCESS DOCUMENTS
-- ============================================================

SELECT
    id,
    source,
    source_id,
    title,
    parse_status

FROM public.core_documents

WHERE
    parse_status IS DISTINCT FROM 'success'

ORDER BY id;


-- EXPECTED:
-- 0 rows


-- ============================================================
-- 12. LANGUAGE
-- ============================================================

SELECT
    language,
    COUNT(*) AS document_count

FROM public.core_documents

GROUP BY language
ORDER BY language;


-- EXPECTED:
--
-- en | 30


-- ============================================================
-- 13. NON-ENGLISH ROWS
-- ============================================================

SELECT
    id,
    source,
    source_id,
    title,
    language

FROM public.core_documents

WHERE
    language IS DISTINCT FROM 'en'

ORDER BY id;


-- EXPECTED:
-- 0 rows


-- ============================================================
-- 14. SOURCE + SOURCE_ID DUPLICATES
-- ============================================================

SELECT
    source,
    source_id,
    COUNT(*) AS duplicate_count

FROM public.core_documents

GROUP BY
    source,
    source_id

HAVING
    COUNT(*) > 1

ORDER BY
    duplicate_count DESC,
    source,
    source_id;


-- EXPECTED:
-- 0 rows


-- ============================================================
-- 15. CONTENT HASH DUPLICATES
-- ============================================================

SELECT
    content_hash,
    COUNT(*) AS duplicate_count,

    STRING_AGG(
        source
        || ':'
        || source_id,
        ' | '
        ORDER BY
            source,
            source_id
    ) AS documents

FROM public.core_documents

WHERE
    content_hash IS NOT NULL

GROUP BY
    content_hash

HAVING
    COUNT(*) > 1

ORDER BY
    duplicate_count DESC;


-- EXPECTED:
-- 0 rows


-- ============================================================
-- 16. MISSING HASH
-- ============================================================

SELECT
    id,
    source,
    source_id,
    title

FROM public.core_documents

WHERE
    content_hash IS NULL
    OR TRIM(
        content_hash
    ) = ''

ORDER BY id;


-- EXPECTED:
-- 0 rows


-- ============================================================
-- 17. HASH LENGTH CHECK
-- ============================================================

SELECT
    id,
    source,
    source_id,
    title,
    content_hash,
    LENGTH(
        content_hash
    ) AS hash_length

FROM public.core_documents

WHERE
    content_hash IS NOT NULL
    AND LENGTH(
        content_hash
    ) <> 64

ORDER BY id;


-- EXPECTED:
-- 0 rows
--
-- SHA-256 hex digest = 64 characters


-- ============================================================
-- 18. NORMALIZATION VERSION
-- ============================================================

SELECT
    normalization_version,
    COUNT(*) AS document_count

FROM public.core_documents

GROUP BY normalization_version
ORDER BY normalization_version;


-- EXPECTED:
--
-- v1 | 30


-- ============================================================
-- 19. MISSING IMPORTANT METADATA
-- ============================================================

SELECT
    id,
    source,
    source_id,
    title,

    CASE
        WHEN title IS NULL
             OR TRIM(
                 title
             ) = ''
        THEN 'missing_title'

        WHEN topic_axis IS NULL
             OR TRIM(
                 topic_axis
             ) = ''
        THEN 'missing_topic_axis'

        WHEN url IS NULL
             OR TRIM(
                 url
             ) = ''
        THEN 'missing_url'

        WHEN metadata IS NULL
        THEN 'missing_metadata'

        ELSE 'ok'
    END AS problem

FROM public.core_documents

WHERE
    title IS NULL
    OR TRIM(
        title
    ) = ''
    OR topic_axis IS NULL
    OR TRIM(
        topic_axis
    ) = ''
    OR url IS NULL
    OR TRIM(
        url
    ) = ''
    OR metadata IS NULL

ORDER BY id;


-- EXPECTED:
-- 0 rows


-- ============================================================
-- 20. QUALITY PASS FROM METADATA
-- ============================================================
--
-- arXiv/NTRS loader 모두 metadata 안에
-- quality provenance를 저장한 경우 검증.
--
-- metadata 구조가 source별로 조금 다를 수 있으므로
-- 먼저 실제 값을 확인한다.
-- ============================================================

SELECT
    id,
    source,
    source_id,
    title,

    metadata -> 'quality'
        AS quality_metadata

FROM public.core_documents

ORDER BY id;


-- 사람이 확인:
-- quality.passed = true


-- ============================================================
-- 21. QUALITY FAIL CHECK
-- ============================================================

SELECT
    id,
    source,
    source_id,
    title,

    metadata
        -> 'quality'
        ->> 'passed'
        AS quality_passed

FROM public.core_documents

WHERE
    metadata ? 'quality'
    AND (
        metadata
            -> 'quality'
            ->> 'passed'
    ) IS DISTINCT FROM 'true'

ORDER BY id;


-- EXPECTED:
-- 0 rows


-- ============================================================
-- 22. CLEAN CONTENT PREVIEW
-- ============================================================

SELECT
    id,
    source,
    source_id,
    topic_axis,
    title,

    LEFT(
        clean_content,
        800
    ) AS clean_preview

FROM public.core_documents

ORDER BY
    source,
    topic_axis,
    id;


-- 반드시 사람이 몇 편 직접 읽는다.
--
-- 확인:
-- 제목
-- Abstract
-- Introduction
-- 정상 영어 문장
-- 이상한 HTML/XML markup 없음


-- ============================================================
-- 23. RAW CONTENT PREVIEW
-- ============================================================

SELECT
    id,
    source,
    source_id,
    topic_axis,
    title,

    LEFT(
        raw_content,
        800
    ) AS raw_preview

FROM public.core_documents

ORDER BY
    source,
    topic_axis,
    id;


-- clean과 raw가 둘 다 실제 내용을 갖고 있는지 비교.


-- ============================================================
-- 24. SOURCE-SPECIFIC AVERAGE SIZE
-- ============================================================

SELECT
    source,

    COUNT(*) AS documents,

    ROUND(
        AVG(
            LENGTH(
                raw_content
            )
        )
    ) AS avg_raw_chars,

    ROUND(
        AVG(
            LENGTH(
                clean_content
            )
        )
    ) AS avg_clean_chars,

    MIN(
        LENGTH(
            clean_content
        )
    ) AS min_clean_chars,

    MAX(
        LENGTH(
            clean_content
        )
    ) AS max_clean_chars

FROM public.core_documents

GROUP BY source
ORDER BY source;


-- arXiv와 NTRS의 평균 길이가 달라도 정상.
-- source format 자체가 다르기 때문.


-- ============================================================
-- 25. AXIS-SPECIFIC SIZE
-- ============================================================

SELECT
    topic_axis,

    COUNT(*) AS documents,

    ROUND(
        AVG(
            LENGTH(
                clean_content
            )
        )
    ) AS avg_clean_chars,

    MIN(
        LENGTH(
            clean_content
        )
    ) AS min_clean_chars,

    MAX(
        LENGTH(
            clean_content
        )
    ) AS max_clean_chars

FROM public.core_documents

GROUP BY topic_axis
ORDER BY topic_axis;


-- ============================================================
-- 26. FINAL PASS / FAIL SUMMARY
-- ============================================================

SELECT

    (
        SELECT COUNT(*)
        FROM public.core_documents
    ) AS total_documents,

    (
        SELECT COUNT(*)
        FROM public.core_documents
        WHERE source = 'arxiv'
    ) AS arxiv_documents,

    (
        SELECT COUNT(*)
        FROM public.core_documents
        WHERE source = 'ntrs'
    ) AS ntrs_documents,

    (
        SELECT COUNT(*)
        FROM public.core_documents
        WHERE topic_axis = 'rover_autonomy'
    ) AS rover_documents,

    (
        SELECT COUNT(*)
        FROM public.core_documents
        WHERE topic_axis = 'onboard_ai'
    ) AS onboard_documents,

    (
        SELECT COUNT(*)
        FROM public.core_documents
        WHERE topic_axis = 'satellite_autonomy'
    ) AS satellite_documents,

    (
        SELECT COUNT(*)
        FROM public.core_documents
        WHERE parse_status = 'success'
    ) AS successful_documents,

    (
        SELECT COUNT(*)
        FROM public.core_documents
        WHERE raw_content IS NULL
           OR clean_content IS NULL
           OR LENGTH(
               TRIM(
                   COALESCE(
                       raw_content,
                       ''
                   )
               )
           ) = 0
           OR LENGTH(
               TRIM(
                   COALESCE(
                       clean_content,
                       ''
                   )
               )
           ) = 0
    ) AS empty_content_documents,

    (
        SELECT COUNT(*)
        FROM (
            SELECT
                content_hash
            FROM public.core_documents
            WHERE content_hash IS NOT NULL
            GROUP BY content_hash
            HAVING COUNT(*) > 1
        ) AS hash_duplicates
    ) AS duplicate_hash_groups;