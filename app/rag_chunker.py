import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import psycopg

from app.config import PROJECT_ROOT, settings


# ============================================================
# TEAM B - Core-100 Section-Aware RAG Chunker
# Version: core100_section_v3
# ============================================================
#
# INPUT
#
#   public.core_documents
#       100 documents
#
# OUTPUT
#
#   public.core_chunks
#
# Pipeline
#
#   core_documents
#        ↓
#   section detection
#        ↓
#   paragraph splitting
#        ↓
#   sentence-aware splitting
#        ↓
#   adaptive-overlap chunk packing
#        ↓
#   exact-duplicate removal (within document)
#        ↓
#   retrieval_text
#        ↓
#   core_chunks
#
#
# FIXES FROM V2
#
# 1. Adaptive overlap
#    - overlap + next unit가 MAX_WORDS를 넘지 않음
#
# 2. Hard MAX_WORDS invariant
#    - 모든 chunk <= 420 words
#
# 3. Exact duplicate removal
#    - 같은 document 안에서 동일 SHA-256 chunk는 1개만 보존
#
# 4. chunk_index 재연속화
#    - duplicate skip 후에도 0,1,2,3... 유지
#
# 5. duplicate QA 강화
#    - within-document duplicate는 Gate 실패
#
#
# IMPORTANT
#
# - 아직 TF-IDF 하지 않음
# - 아직 SVD 하지 않음
# - 아직 pgvector 하지 않음
# - 아직 embedding 만들지 않음
#
# ============================================================


CHUNKER_VERSION = "core100_section_v3"

EXPECTED_DOCUMENTS = 100


# ============================================================
# Chunk Parameters
# ============================================================

TARGET_WORDS = 360
MAX_WORDS = 420
OVERLAP_WORDS = 50
MIN_SECTION_WORDS = 20


# ============================================================
# Database Tables
# ============================================================

DOCUMENT_TABLE = "public.core_documents"
CHUNK_TABLE = "public.core_chunks"


# ============================================================
# Report
# ============================================================

REPORT_FILE = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "core100_chunking_report.json"
)


# ============================================================
# Data Classes
# ============================================================

@dataclass
class Section:
    index: int
    heading: str
    text: str


@dataclass
class Chunk:
    document_id: int
    chunk_index: int
    section_index: int
    section_heading: str

    chunk_text: str
    retrieval_text: str

    word_count: int
    token_estimate: int
    char_count: int

    content_hash: str


# ============================================================
# JSON
# ============================================================

def _save_json(
    path: Path,
    payload: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


# ============================================================
# Basic Text Helpers
# ============================================================

def _normalize_text(
    text: str,
) -> str:

    text = str(
        text or ""
    )

    text = (
        text
        .replace(
            "\r\n",
            "\n",
        )
        .replace(
            "\r",
            "\n",
        )
        .replace(
            "\xa0",
            " ",
        )
        .replace(
            "\x00",
            "",
        )
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def _normalize_inline_space(
    text: str,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        str(
            text or ""
        ),
    ).strip()


def _word_count(
    text: str,
) -> int:

    return len(
        re.findall(
            r"\S+",
            text,
        )
    )


def _token_estimate(
    text: str,
) -> int:
    """
    Rough estimate for English scientific prose.

    tokenizer-exact 값은 아님.

    약:
        1 English word ≈ 1.33 tokens
    """

    words = (
        _word_count(
            text
        )
    )

    return math.ceil(
        words * 1.33
    )


def _sha256(
    text: str,
) -> str:

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# Heading Detection
# ============================================================

HEADING_PATTERN = re.compile(
    r"^(#{1,6})\s+(.+?)\s*$"
)


# ============================================================
# Section Parser
# ============================================================

def _parse_sections(
    clean_content: str,
) -> list[Section]:

    clean_content = (
        _normalize_text(
            clean_content
        )
    )

    if not clean_content:
        return []

    lines = (
        clean_content.splitlines()
    )

    sections: list[Section] = []

    current_heading = "Document Body"
    current_lines: list[str] = []

    section_index = 0

    def flush() -> None:

        nonlocal section_index
        nonlocal current_lines

        body = (
            "\n".join(
                current_lines
            )
            .strip()
        )

        current_lines = []

        if not body:
            return

        heading = (
            current_heading.strip()
            or "Document Body"
        )

        normalized_heading = (
            heading
            .strip()
            .lower()
        )

        # Cleaner에서 생성된 title-only section은 제외.
        # title은 이미 core_documents.title에 존재함.
        if normalized_heading in {
            "title",
            "document title",
        }:
            return

        sections.append(
            Section(
                index=section_index,
                heading=heading,
                text=body,
            )
        )

        section_index += 1

    for raw_line in lines:

        stripped = (
            raw_line.strip()
        )

        heading_match = (
            HEADING_PATTERN.match(
                stripped
            )
        )

        if heading_match:

            flush()

            current_heading = (
                heading_match
                .group(2)
                .strip()
            )

            continue

        current_lines.append(
            raw_line
        )

    flush()

    return sections


# ============================================================
# Paragraph Split
# ============================================================

def _split_paragraphs(
    text: str,
) -> list[str]:

    text = (
        _normalize_text(
            text
        )
    )

    if not text:
        return []

    paragraphs: list[str] = []

    blocks = re.split(
        r"\n\s*\n",
        text,
    )

    for block in blocks:

        block = (
            _normalize_inline_space(
                block
            )
        )

        if not block:
            continue

        paragraphs.append(
            block
        )

    return paragraphs


# ============================================================
# Sentence Split
# ============================================================

SENTENCE_SPLIT_PATTERN = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9\[])"
)


def _split_sentences(
    text: str,
) -> list[str]:

    text = (
        _normalize_inline_space(
            text
        )
    )

    if not text:
        return []

    return [
        sentence.strip()
        for sentence
        in SENTENCE_SPLIT_PATTERN.split(
            text
        )
        if sentence.strip()
    ]


# ============================================================
# Hard Word Split
# ============================================================

def _hard_word_split(
    text: str,
) -> list[str]:

    words = (
        text.split()
    )

    if not words:
        return []

    pieces: list[str] = []

    start = 0

    while start < len(words):

        end = min(
            start + MAX_WORDS,
            len(words),
        )

        piece = (
            " ".join(
                words[
                    start:end
                ]
            )
            .strip()
        )

        if piece:

            pieces.append(
                piece
            )

        if end >= len(words):
            break

        # overlap은 여기서 넣지 않는다.
        # 최종 packing 단계에서 한 번만 처리.
        start = end

    return pieces


# ============================================================
# Large Paragraph Split
# ============================================================

def _split_large_paragraph(
    paragraph: str,
) -> list[str]:

    paragraph = (
        _normalize_inline_space(
            paragraph
        )
    )

    if not paragraph:
        return []

    paragraph_words = (
        _word_count(
            paragraph
        )
    )

    if paragraph_words <= MAX_WORDS:

        return [
            paragraph
        ]

    sentences = (
        _split_sentences(
            paragraph
        )
    )

    # Sentence splitter가 사실상 실패한 경우
    if len(sentences) <= 1:

        return _hard_word_split(
            paragraph
        )

    pieces: list[str] = []

    current_sentences: list[str] = []
    current_words = 0

    for sentence in sentences:

        sentence = (
            _normalize_inline_space(
                sentence
            )
        )

        if not sentence:
            continue

        sentence_words = (
            _word_count(
                sentence
            )
        )

        # 한 문장 자체가 MAX_WORDS보다 긴 경우
        if sentence_words > MAX_WORDS:

            if current_sentences:

                current_text = (
                    " ".join(
                        current_sentences
                    )
                    .strip()
                )

                if current_text:

                    pieces.append(
                        current_text
                    )

                current_sentences = []
                current_words = 0

            pieces.extend(
                _hard_word_split(
                    sentence
                )
            )

            continue

        if (
            current_sentences
            and (
                current_words
                + sentence_words
                > MAX_WORDS
            )
        ):

            current_text = (
                " ".join(
                    current_sentences
                )
                .strip()
            )

            if current_text:

                pieces.append(
                    current_text
                )

            current_sentences = []
            current_words = 0

        current_sentences.append(
            sentence
        )

        current_words += (
            sentence_words
        )

    if current_sentences:

        current_text = (
            " ".join(
                current_sentences
            )
            .strip()
        )

        if current_text:

            pieces.append(
                current_text
            )

    # Defensive gate
    for piece in pieces:

        piece_words = (
            _word_count(
                piece
            )
        )

        if piece_words > MAX_WORDS:

            raise RuntimeError(
                "Large paragraph splitter generated "
                "oversized piece.\n"
                f"Words : {piece_words}\n"
                f"Max   : {MAX_WORDS}"
            )

    return pieces


# ============================================================
# Tail Words
# ============================================================

def _tail_words(
    text: str,
    count: int,
) -> str:

    if count <= 0:
        return ""

    words = (
        text.split()
    )

    if not words:
        return ""

    if len(words) <= count:

        return " ".join(
            words
        )

    return " ".join(
        words[
            -count:
        ]
    )


# ============================================================
# Chunk One Section
# ============================================================

def _chunk_section(
    section: Section,
) -> list[str]:

    paragraphs = (
        _split_paragraphs(
            section.text
        )
    )

    # ========================================================
    # Paragraph -> safe units
    # ========================================================

    units: list[str] = []

    for paragraph in paragraphs:

        pieces = (
            _split_large_paragraph(
                paragraph
            )
        )

        for piece in pieces:

            piece = (
                _normalize_inline_space(
                    piece
                )
            )

            if not piece:
                continue

            piece_words = (
                _word_count(
                    piece
                )
            )

            if piece_words > MAX_WORDS:

                raise RuntimeError(
                    "Oversized unit escaped splitter.\n"
                    f"Section : {section.heading}\n"
                    f"Words   : {piece_words}\n"
                    f"Max     : {MAX_WORDS}"
                )

            units.append(
                piece
            )

    if not units:
        return []

    # ========================================================
    # Packing
    # ========================================================

    chunks: list[str] = []

    current_parts: list[str] = []
    current_words = 0

    # 직전에 확정된 chunk.
    # 다음 unit을 보기 전에는 overlap을 넣지 않는다.
    pending_overlap_source = ""

    # ========================================================
    # Flush
    # ========================================================

    def flush_current() -> None:

        nonlocal current_parts
        nonlocal current_words
        nonlocal pending_overlap_source

        chunk_text = (
            " ".join(
                current_parts
            )
            .strip()
        )

        current_parts = []
        current_words = 0

        if not chunk_text:
            return

        words = (
            _word_count(
                chunk_text
            )
        )

        if words > MAX_WORDS:

            raise RuntimeError(
                "Internal chunk overflow.\n"
                f"Section : {section.heading}\n"
                f"Words   : {words}\n"
                f"Max     : {MAX_WORDS}"
            )

        chunks.append(
            chunk_text
        )

        pending_overlap_source = (
            chunk_text
        )

    # ========================================================
    # Adaptive Overlap
    # ========================================================

    def seed_overlap(
        next_unit_words: int,
    ) -> None:

        nonlocal current_parts
        nonlocal current_words
        nonlocal pending_overlap_source

        if not pending_overlap_source:
            return

        # overlap + next unit <= MAX_WORDS
        remaining_capacity = (
            MAX_WORDS
            - next_unit_words
        )

        overlap_budget = min(
            OVERLAP_WORDS,
            max(
                0,
                remaining_capacity,
            ),
        )

        if overlap_budget > 0:

            overlap = (
                _tail_words(
                    pending_overlap_source,
                    overlap_budget,
                )
            )

            overlap = (
                _normalize_inline_space(
                    overlap
                )
            )

            if overlap:

                current_parts.append(
                    overlap
                )

                current_words += (
                    _word_count(
                        overlap
                    )
                )

        pending_overlap_source = ""

    # ========================================================
    # Process Units
    # ========================================================

    for unit in units:

        unit = (
            _normalize_inline_space(
                unit
            )
        )

        if not unit:
            continue

        unit_words = (
            _word_count(
                unit
            )
        )

        if unit_words > MAX_WORDS:

            raise RuntimeError(
                "Unsafe unit detected during packing.\n"
                f"Section : {section.heading}\n"
                f"Words   : {unit_words}"
            )

        # 새로운 chunk 시작
        if not current_parts:

            seed_overlap(
                unit_words
            )

        # 현재 chunk에 넣으면 MAX 초과
        if (
            current_parts
            and (
                current_words
                + unit_words
                > MAX_WORDS
            )
        ):

            flush_current()

            seed_overlap(
                unit_words
            )

        # Adaptive overlap 이후에도 초과한다면
        # overlap 제거 후 unit만 사용.
        if (
            current_words
            + unit_words
            > MAX_WORDS
        ):

            current_parts = []
            current_words = 0
            pending_overlap_source = ""

        current_parts.append(
            unit
        )

        current_words += (
            unit_words
        )

        # TARGET 도달
        if current_words >= TARGET_WORDS:

            flush_current()

    # ========================================================
    # Final Tail
    # ========================================================

    if current_parts:

        tail = (
            " ".join(
                current_parts
            )
            .strip()
        )

        if tail:

            tail_words = (
                _word_count(
                    tail
                )
            )

            if tail_words > MAX_WORDS:

                raise RuntimeError(
                    "Final tail overflow.\n"
                    f"Section : {section.heading}\n"
                    f"Words   : {tail_words}\n"
                    f"Max     : {MAX_WORDS}"
                )

            # 직전 chunk의 tail과 완전히 동일한
            # overlap-only chunk라면 추가하지 않는다.
            if (
                chunks
                and tail_words <= OVERLAP_WORDS
                and tail
                == _tail_words(
                    chunks[-1],
                    tail_words,
                )
            ):
                pass

            else:

                chunks.append(
                    tail
                )

    # ========================================================
    # Final Invariant
    # ========================================================

    for (
        index,
        chunk_text,
    ) in enumerate(
        chunks
    ):

        count = (
            _word_count(
                chunk_text
            )
        )

        if count <= 0:

            raise RuntimeError(
                "Zero-word chunk generated.\n"
                f"Section : {section.heading}\n"
                f"Index   : {index}"
            )

        if count > MAX_WORDS:

            raise RuntimeError(
                "Chunk max-word invariant violated.\n"
                f"Section : {section.heading}\n"
                f"Index   : {index}\n"
                f"Words   : {count}\n"
                f"Max     : {MAX_WORDS}"
            )

    return chunks


# ============================================================
# Build One Document
# ============================================================

def _build_document_chunks(
    document: dict,
) -> tuple[
    list[Chunk],
    int,
]:
    """
    Returns:
        chunks
        deduplicated_count

    Exact duplicate policy:
        같은 document 안에서 동일 chunk_text SHA-256이
        이미 생성되었다면 후속 chunk를 버린다.

    이유:
        동일 evidence가 Top-K에서 두 자리를 차지하는 것을 방지.
    """

    document_id = int(
        document[
            "id"
        ]
    )

    title = str(
        document.get(
            "title"
        )
        or ""
    ).strip()

    clean_content = str(
        document.get(
            "clean_content"
        )
        or ""
    ).strip()

    if not clean_content:

        raise RuntimeError(
            f"Document {document_id}: "
            "clean_content is empty."
        )

    sections = (
        _parse_sections(
            clean_content
        )
    )

    if not sections:

        raise RuntimeError(
            f"Document {document_id}: "
            "no sections parsed."
        )

    chunks: list[Chunk] = []

    # 같은 document 내부 exact duplicate 확인
    seen_hashes: set[str] = set()

    deduplicated_count = 0

    # duplicate skip 후에도 chunk_index는 연속이어야 함
    next_chunk_index = 0

    for section in sections:

        section_words = (
            _word_count(
                section.text
            )
        )

        if section_words < MIN_SECTION_WORDS:
            continue

        section_chunks = (
            _chunk_section(
                section
            )
        )

        for body in section_chunks:

            body = (
                _normalize_inline_space(
                    body
                )
            )

            if not body:
                continue

            words = (
                _word_count(
                    body
                )
            )

            if words <= 0:
                continue

            if words > MAX_WORDS:

                raise RuntimeError(
                    f"Document {document_id}: "
                    f"chunk exceeds MAX_WORDS.\n"
                    f"Section: {section.heading}\n"
                    f"Words  : {words}"
                )

            # =================================================
            # Exact Duplicate Gate
            # =================================================

            content_hash = (
                _sha256(
                    body
                )
            )

            if content_hash in seen_hashes:

                deduplicated_count += 1

                print(
                    "    [DEDUP] "
                    f"document_id={document_id} | "
                    f"section={section.heading!r} | "
                    f"hash={content_hash[:12]}..."
                )

                continue

            seen_hashes.add(
                content_hash
            )

            # =================================================
            # Retrieval Text
            # =================================================

            retrieval_parts: list[str] = []

            if title:

                retrieval_parts.append(
                    title
                )

            if section.heading:

                retrieval_parts.append(
                    section.heading
                )

            retrieval_parts.append(
                body
            )

            retrieval_text = (
                "\n".join(
                    retrieval_parts
                )
                .strip()
            )

            chunks.append(
                Chunk(
                    document_id=document_id,

                    chunk_index=(
                        next_chunk_index
                    ),

                    section_index=(
                        section.index
                    ),

                    section_heading=(
                        section.heading
                    ),

                    chunk_text=(
                        body
                    ),

                    retrieval_text=(
                        retrieval_text
                    ),

                    word_count=(
                        words
                    ),

                    token_estimate=(
                        _token_estimate(
                            body
                        )
                    ),

                    char_count=(
                        len(
                            body
                        )
                    ),

                    content_hash=(
                        content_hash
                    ),
                )
            )

            next_chunk_index += 1

    if not chunks:

        raise RuntimeError(
            f"Document {document_id}: "
            "generated zero chunks."
        )

    # ========================================================
    # Document-local duplicate invariant
    # ========================================================

    hashes = [
        chunk.content_hash
        for chunk in chunks
    ]

    if (
        len(hashes)
        != len(
            set(
                hashes
            )
        )
    ):

        raise RuntimeError(
            f"Document {document_id}: "
            "exact duplicate remained after dedup."
        )

    # ========================================================
    # chunk_index invariant
    # ========================================================

    expected_indexes = list(
        range(
            len(
                chunks
            )
        )
    )

    actual_indexes = [
        chunk.chunk_index
        for chunk in chunks
    ]

    if actual_indexes != expected_indexes:

        raise RuntimeError(
            f"Document {document_id}: "
            "chunk_index sequence is not continuous."
        )

    return (
        chunks,
        deduplicated_count,
    )


# ============================================================
# Chunk Schema Validation
# ============================================================

REQUIRED_CHUNK_COLUMNS = {
    "id",
    "document_id",
    "chunk_index",
    "section_index",
    "section_heading",
    "chunk_text",
    "retrieval_text",
    "word_count",
    "token_estimate",
    "char_count",
    "content_hash",
    "chunker_version",
    "created_at",
}


def _validate_chunk_schema(
    conn: psycopg.Connection,
) -> None:

    sql = """
    SELECT
        column_name
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'core_chunks'
    """

    with conn.cursor() as cur:

        cur.execute(
            sql
        )

        rows = (
            cur.fetchall()
        )

    columns = {
        str(
            row[0]
        )
        for row in rows
    }

    if not columns:

        raise RuntimeError(
            "public.core_chunks does not exist.\n"
            "Run sql/004_core_chunks.sql first."
        )

    missing = (
        REQUIRED_CHUNK_COLUMNS
        - columns
    )

    if missing:

        raise RuntimeError(
            "core_chunks schema mismatch.\n"
            "Missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )


# ============================================================
# Load Core Documents
# ============================================================

def _load_documents(
    conn: psycopg.Connection,
) -> list[dict]:

    sql = f"""
    SELECT
        id,
        source,
        source_id,
        title,
        topic_axis,
        clean_content,
        parse_status
    FROM {DOCUMENT_TABLE}
    WHERE parse_status = 'success'
      AND clean_content IS NOT NULL
      AND length(trim(clean_content)) > 0
    ORDER BY id
    """

    with conn.cursor() as cur:

        cur.execute(
            sql
        )

        rows = (
            cur.fetchall()
        )

    documents: list[dict] = []

    for row in rows:

        (
            document_id,
            source,
            source_id,
            title,
            topic_axis,
            clean_content,
            parse_status,
        ) = row

        documents.append(
            {
                "id": (
                    int(
                        document_id
                    )
                ),

                "source": (
                    str(
                        source
                    )
                ),

                "source_id": (
                    str(
                        source_id
                    )
                ),

                "title": (
                    str(
                        title or ""
                    )
                ),

                "topic_axis": (
                    str(
                        topic_axis or ""
                    )
                ),

                "clean_content": (
                    str(
                        clean_content
                    )
                ),

                "parse_status": (
                    str(
                        parse_status
                    )
                ),
            }
        )

    if len(
        documents
    ) != EXPECTED_DOCUMENTS:

        raise RuntimeError(
            f"Expected {EXPECTED_DOCUMENTS} "
            f"Core documents, "
            f"found {len(documents)}."
        )

    return documents


# ============================================================
# Local Chunk QA
# ============================================================

def _validate_local_chunks(
    all_chunks: list[Chunk],
) -> dict:

    if not all_chunks:

        raise RuntimeError(
            "No chunks generated."
        )

    document_ids = {
        chunk.document_id
        for chunk in all_chunks
    }

    if len(
        document_ids
    ) != EXPECTED_DOCUMENTS:

        raise RuntimeError(
            "Local chunk coverage failed.\n"
            f"Expected documents: "
            f"{EXPECTED_DOCUMENTS}\n"
            f"Chunked documents : "
            f"{len(document_ids)}"
        )

    empty_chunks = []
    oversized_chunks = []

    # --------------------------------------------------------
    # Global hash groups
    # --------------------------------------------------------

    global_hashes: dict[
        str,
        list[Chunk],
    ] = {}

    # --------------------------------------------------------
    # Document-local hashes
    # --------------------------------------------------------

    per_document_hashes: dict[
        int,
        set[str],
    ] = {}

    duplicate_within_document = []

    for chunk in all_chunks:

        if not chunk.chunk_text.strip():

            empty_chunks.append(
                (
                    chunk.document_id,
                    chunk.chunk_index,
                )
            )

        if chunk.word_count > MAX_WORDS:

            oversized_chunks.append(
                (
                    chunk.document_id,
                    chunk.chunk_index,
                    chunk.word_count,
                )
            )

        global_hashes.setdefault(
            chunk.content_hash,
            [],
        ).append(
            chunk
        )

        doc_hashes = (
            per_document_hashes
            .setdefault(
                chunk.document_id,
                set(),
            )
        )

        if chunk.content_hash in doc_hashes:

            duplicate_within_document.append(
                (
                    chunk.document_id,
                    chunk.chunk_index,
                    chunk.content_hash,
                )
            )

        doc_hashes.add(
            chunk.content_hash
        )

    if empty_chunks:

        raise RuntimeError(
            f"Empty local chunks detected: "
            f"{len(empty_chunks)}"
        )

    if oversized_chunks:

        raise RuntimeError(
            "Oversized chunks detected.\n"
            f"Count  : "
            f"{len(oversized_chunks)}\n"
            f"Sample : "
            f"{oversized_chunks[:10]}"
        )

    if duplicate_within_document:

        raise RuntimeError(
            "Exact duplicate chunks remain "
            "inside same document.\n"
            f"Count  : "
            f"{len(duplicate_within_document)}\n"
            f"Sample : "
            f"{duplicate_within_document[:10]}"
        )

    global_duplicate_groups = {
        digest: chunks
        for (
            digest,
            chunks,
        ) in global_hashes.items()
        if len(chunks) > 1
    }

    return {
        "covered_documents": (
            len(
                document_ids
            )
        ),

        "chunk_count": (
            len(
                all_chunks
            )
        ),

        "empty_chunks": (
            len(
                empty_chunks
            )
        ),

        "oversized_chunks": (
            len(
                oversized_chunks
            )
        ),

        "within_document_duplicate_groups": (
            0
        ),

        # 다른 문서끼리 동일 chunk가 생길 가능성은
        # 별도 관찰값으로만 기록한다.
        "global_duplicate_hash_groups": (
            len(
                global_duplicate_groups
            )
        ),
    }


# ============================================================
# Insert Chunk
# ============================================================

def _insert_chunk(
    conn: psycopg.Connection,
    chunk: Chunk,
) -> None:

    sql = f"""
    INSERT INTO {CHUNK_TABLE} (
        document_id,
        chunk_index,
        section_index,
        section_heading,
        chunk_text,
        retrieval_text,
        word_count,
        token_estimate,
        char_count,
        content_hash,
        chunker_version
    )
    VALUES (
        %s,
        %s,
        %s,
        %s,
        %s,
        %s,
        %s,
        %s,
        %s,
        %s,
        %s
    )
    """

    values = (
        chunk.document_id,
        chunk.chunk_index,
        chunk.section_index,
        chunk.section_heading,
        chunk.chunk_text,
        chunk.retrieval_text,
        chunk.word_count,
        chunk.token_estimate,
        chunk.char_count,
        chunk.content_hash,
        CHUNKER_VERSION,
    )

    with conn.cursor() as cur:

        cur.execute(
            sql,
            values,
        )


# ============================================================
# DB Verification
# ============================================================

def _verify_database(
    conn: psycopg.Connection,
) -> dict:

    with conn.cursor() as cur:

        # ====================================================
        # Total chunks
        # ====================================================

        cur.execute(
            f"""
            SELECT COUNT(*)
            FROM {CHUNK_TABLE}
            WHERE chunker_version = %s
            """,
            (
                CHUNKER_VERSION,
            ),
        )

        total_chunks = int(
            cur.fetchone()[0]
        )

        # ====================================================
        # Covered documents
        # ====================================================

        cur.execute(
            f"""
            SELECT
                COUNT(
                    DISTINCT document_id
                )
            FROM {CHUNK_TABLE}
            WHERE chunker_version = %s
            """,
            (
                CHUNKER_VERSION,
            ),
        )

        covered_documents = int(
            cur.fetchone()[0]
        )

        # ====================================================
        # Empty chunks
        # ====================================================

        cur.execute(
            f"""
            SELECT COUNT(*)
            FROM {CHUNK_TABLE}
            WHERE chunker_version = %s
              AND (
                    chunk_text IS NULL
                    OR length(trim(chunk_text)) = 0
                  )
            """,
            (
                CHUNKER_VERSION,
            ),
        )

        empty_chunks = int(
            cur.fetchone()[0]
        )

        # ====================================================
        # Empty retrieval text
        # ====================================================

        cur.execute(
            f"""
            SELECT COUNT(*)
            FROM {CHUNK_TABLE}
            WHERE chunker_version = %s
              AND (
                    retrieval_text IS NULL
                    OR length(trim(retrieval_text)) = 0
                  )
            """,
            (
                CHUNKER_VERSION,
            ),
        )

        empty_retrieval_text = int(
            cur.fetchone()[0]
        )

        # ====================================================
        # Stats
        # ====================================================

        cur.execute(
            f"""
            SELECT
                MIN(word_count),
                MAX(word_count),
                AVG(word_count),

                MIN(token_estimate),
                MAX(token_estimate),
                AVG(token_estimate),

                MIN(char_count),
                MAX(char_count),
                AVG(char_count)

            FROM {CHUNK_TABLE}
            WHERE chunker_version = %s
            """,
            (
                CHUNKER_VERSION,
            ),
        )

        (
            min_words,
            max_words,
            avg_words,

            min_tokens,
            max_tokens,
            avg_tokens,

            min_chars,
            max_chars,
            avg_chars,
        ) = (
            cur.fetchone()
        )

        # ====================================================
        # Source Counts
        # ====================================================

        cur.execute(
            f"""
            SELECT
                d.source,
                COUNT(*)

            FROM {CHUNK_TABLE} c

            JOIN {DOCUMENT_TABLE} d
              ON d.id = c.document_id

            WHERE c.chunker_version = %s

            GROUP BY d.source

            ORDER BY d.source
            """,
            (
                CHUNKER_VERSION,
            ),
        )

        source_counts = {
            str(
                source
            ): int(
                count
            )
            for (
                source,
                count,
            )
            in cur.fetchall()
        }

        # ====================================================
        # Axis Counts
        # ====================================================

        cur.execute(
            f"""
            SELECT
                d.topic_axis,
                COUNT(*)

            FROM {CHUNK_TABLE} c

            JOIN {DOCUMENT_TABLE} d
              ON d.id = c.document_id

            WHERE c.chunker_version = %s

            GROUP BY d.topic_axis

            ORDER BY d.topic_axis
            """,
            (
                CHUNKER_VERSION,
            ),
        )

        axis_counts = {
            str(
                topic_axis
            ): int(
                count
            )
            for (
                topic_axis,
                count,
            )
            in cur.fetchall()
        }

        # ====================================================
        # Chunks per document
        # ====================================================

        cur.execute(
            f"""
            SELECT
                MIN(chunk_count),
                MAX(chunk_count),
                AVG(chunk_count)

            FROM (
                SELECT
                    document_id,
                    COUNT(*) AS chunk_count

                FROM {CHUNK_TABLE}

                WHERE chunker_version = %s

                GROUP BY document_id
            ) AS x
            """,
            (
                CHUNKER_VERSION,
            ),
        )

        (
            min_chunks_per_doc,
            max_chunks_per_doc,
            avg_chunks_per_doc,
        ) = (
            cur.fetchone()
        )

        # ====================================================
        # Global duplicate hash groups
        # ====================================================

        cur.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT
                    content_hash
                FROM {CHUNK_TABLE}
                WHERE chunker_version = %s
                GROUP BY content_hash
                HAVING COUNT(*) > 1
            ) duplicated
            """,
            (
                CHUNKER_VERSION,
            ),
        )

        global_duplicate_hash_groups = int(
            cur.fetchone()[0]
        )

        # ====================================================
        # Same-document exact duplicate groups
        # ====================================================

        cur.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT
                    document_id,
                    content_hash
                FROM {CHUNK_TABLE}
                WHERE chunker_version = %s
                GROUP BY
                    document_id,
                    content_hash
                HAVING COUNT(*) > 1
            ) duplicated
            """,
            (
                CHUNKER_VERSION,
            ),
        )

        within_document_duplicate_groups = int(
            cur.fetchone()[0]
        )

        # ====================================================
        # Oversized
        # ====================================================

        cur.execute(
            f"""
            SELECT COUNT(*)
            FROM {CHUNK_TABLE}
            WHERE chunker_version = %s
              AND word_count > %s
            """,
            (
                CHUNKER_VERSION,
                MAX_WORDS,
            ),
        )

        oversized_chunks = int(
            cur.fetchone()[0]
        )

        # ====================================================
        # Broken chunk_index sequence
        #
        # document별:
        #
        # min(chunk_index) = 0
        # max(chunk_index) = count(*) - 1
        # ====================================================

        cur.execute(
            f"""
            SELECT COUNT(*)

            FROM (
                SELECT
                    document_id,
                    MIN(chunk_index) AS min_index,
                    MAX(chunk_index) AS max_index,
                    COUNT(*) AS chunk_count

                FROM {CHUNK_TABLE}

                WHERE chunker_version = %s

                GROUP BY document_id
            ) x

            WHERE min_index <> 0
               OR max_index <> chunk_count - 1
            """,
            (
                CHUNKER_VERSION,
            ),
        )

        broken_index_documents = int(
            cur.fetchone()[0]
        )

    return {
        "total_chunks": (
            total_chunks
        ),

        "covered_documents": (
            covered_documents
        ),

        "empty_chunks": (
            empty_chunks
        ),

        "empty_retrieval_text": (
            empty_retrieval_text
        ),

        "oversized_chunks": (
            oversized_chunks
        ),

        "within_document_duplicate_groups": (
            within_document_duplicate_groups
        ),

        "global_duplicate_hash_groups": (
            global_duplicate_hash_groups
        ),

        "broken_index_documents": (
            broken_index_documents
        ),

        "word_stats": {
            "min": (
                int(
                    min_words or 0
                )
            ),

            "max": (
                int(
                    max_words or 0
                )
            ),

            "avg": (
                round(
                    float(
                        avg_words or 0
                    ),
                    2,
                )
            ),
        },

        "token_estimate_stats": {
            "min": (
                int(
                    min_tokens or 0
                )
            ),

            "max": (
                int(
                    max_tokens or 0
                )
            ),

            "avg": (
                round(
                    float(
                        avg_tokens or 0
                    ),
                    2,
                )
            ),
        },

        "char_stats": {
            "min": (
                int(
                    min_chars or 0
                )
            ),

            "max": (
                int(
                    max_chars or 0
                )
            ),

            "avg": (
                round(
                    float(
                        avg_chars or 0
                    ),
                    2,
                )
            ),
        },

        "chunks_per_document": {
            "min": (
                int(
                    min_chunks_per_doc or 0
                )
            ),

            "max": (
                int(
                    max_chunks_per_doc or 0
                )
            ),

            "avg": (
                round(
                    float(
                        avg_chunks_per_doc or 0
                    ),
                    2,
                )
            ),
        },

        "source_counts": (
            source_counts
        ),

        "axis_counts": (
            axis_counts
        ),
    }


# ============================================================
# Print Summary
# ============================================================

def _print_summary(
    verification: dict,
    total_deduplicated: int,
) -> None:

    print()
    print("=" * 78)

    print(
        "CORE-100 CHUNKING COMPLETED"
    )

    print("=" * 78)

    print(
        f"Chunker version   : "
        f"{CHUNKER_VERSION}"
    )

    print(
        f"Documents covered : "
        f"{verification['covered_documents']}"
    )

    print(
        f"Total chunks      : "
        f"{verification['total_chunks']}"
    )

    print(
        f"Deduplicated      : "
        f"{total_deduplicated}"
    )

    print(
        f"Empty chunks      : "
        f"{verification['empty_chunks']}"
    )

    print(
        f"Empty retrieval   : "
        f"{verification['empty_retrieval_text']}"
    )

    print(
        f"Oversized chunks  : "
        f"{verification['oversized_chunks']}"
    )

    print(
        f"Same-doc dupes    : "
        f"{verification['within_document_duplicate_groups']}"
    )

    print(
        f"Global dupes      : "
        f"{verification['global_duplicate_hash_groups']}"
    )

    print(
        f"Broken indexes    : "
        f"{verification['broken_index_documents']}"
    )

    print()

    print(
        "Chunks per document:"
    )

    print(
        f"  min : "
        f"{verification['chunks_per_document']['min']}"
    )

    print(
        f"  max : "
        f"{verification['chunks_per_document']['max']}"
    )

    print(
        f"  avg : "
        f"{verification['chunks_per_document']['avg']}"
    )

    print()

    print(
        "Word count:"
    )

    print(
        f"  min : "
        f"{verification['word_stats']['min']}"
    )

    print(
        f"  max : "
        f"{verification['word_stats']['max']}"
    )

    print(
        f"  avg : "
        f"{verification['word_stats']['avg']}"
    )

    print()

    print(
        "Estimated tokens:"
    )

    print(
        f"  min : "
        f"{verification['token_estimate_stats']['min']}"
    )

    print(
        f"  max : "
        f"{verification['token_estimate_stats']['max']}"
    )

    print(
        f"  avg : "
        f"{verification['token_estimate_stats']['avg']}"
    )

    print()

    print(
        "Character count:"
    )

    print(
        f"  min : "
        f"{verification['char_stats']['min']}"
    )

    print(
        f"  max : "
        f"{verification['char_stats']['max']}"
    )

    print(
        f"  avg : "
        f"{verification['char_stats']['avg']}"
    )

    print()

    print(
        "Chunks by source:"
    )

    for (
        source,
        count,
    ) in (
        verification[
            "source_counts"
        ].items()
    ):

        print(
            f"  {source:10} : "
            f"{count}"
        )

    print()

    print(
        "Chunks by axis:"
    )

    for (
        topic_axis,
        count,
    ) in (
        verification[
            "axis_counts"
        ].items()
    ):

        print(
            f"  {topic_axis:22} : "
            f"{count}"
        )

    print()

    print(
        "Report:"
    )

    print(
        f"  {REPORT_FILE}"
    )

    print("=" * 78)


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 78)

    print(
        "TEAM B - Core-100 "
        "Section-Aware RAG Chunker"
    )

    print("=" * 78)

    print(
        f"Chunker version : "
        f"{CHUNKER_VERSION}"
    )

    print(
        f"Target words    : "
        f"{TARGET_WORDS}"
    )

    print(
        f"Max words       : "
        f"{MAX_WORDS}"
    )

    print(
        f"Overlap words   : "
        f"{OVERLAP_WORDS}"
    )

    print(
        f"Min section     : "
        f"{MIN_SECTION_WORDS}"
    )

    print(
        f"Expected docs   : "
        f"{EXPECTED_DOCUMENTS}"
    )

    print()

    print(
        "[DB] Connecting to AWS RDS..."
    )

    try:

        with psycopg.connect(
            settings.dsn
        ) as conn:

            print(
                "[DB] Connected."
            )

            # =================================================
            # Schema
            # =================================================

            _validate_chunk_schema(
                conn
            )

            print(
                "[DB] core_chunks schema OK."
            )

            # =================================================
            # Documents
            # =================================================

            documents = (
                _load_documents(
                    conn
                )
            )

            print(
                f"[DB] Core documents: "
                f"{len(documents)}"
            )

            # =================================================
            # Build Chunks
            # =================================================

            all_chunks: list[Chunk] = []

            document_stats: list[dict] = []

            total_deduplicated = 0

            print()
            print(
                "[LOCAL] Building chunks..."
            )
            print()

            for (
                index,
                document,
            ) in enumerate(
                documents,
                start=1,
            ):

                (
                    chunks,
                    deduplicated_count,
                ) = (
                    _build_document_chunks(
                        document
                    )
                )

                all_chunks.extend(
                    chunks
                )

                total_deduplicated += (
                    deduplicated_count
                )

                document_stats.append(
                    {
                        "document_id": (
                            document[
                                "id"
                            ]
                        ),

                        "source": (
                            document[
                                "source"
                            ]
                        ),

                        "source_id": (
                            document[
                                "source_id"
                            ]
                        ),

                        "topic_axis": (
                            document[
                                "topic_axis"
                            ]
                        ),

                        "title": (
                            document[
                                "title"
                            ]
                        ),

                        "chunk_count": (
                            len(
                                chunks
                            )
                        ),

                        "deduplicated_count": (
                            deduplicated_count
                        ),
                    }
                )

                dedup_suffix = ""

                if deduplicated_count:

                    dedup_suffix = (
                        f" | dedup={deduplicated_count}"
                    )

                print(
                    f"[{index:03d}/"
                    f"{EXPECTED_DOCUMENTS}] "
                    f"{document['source']:6} | "
                    f"{document['topic_axis']:20} | "
                    f"{document['source_id']:15} | "
                    f"{len(chunks):3} chunks"
                    f"{dedup_suffix}"
                )

            # =================================================
            # Local QA
            # =================================================

            local_qa = (
                _validate_local_chunks(
                    all_chunks
                )
            )

            print()
            print(
                "[LOCAL] Chunk generation PASS."
            )

            print(
                f"[LOCAL] Documents       : "
                f"{local_qa['covered_documents']}"
            )

            print(
                f"[LOCAL] Chunks          : "
                f"{local_qa['chunk_count']}"
            )

            print(
                f"[LOCAL] Deduplicated    : "
                f"{total_deduplicated}"
            )

            print(
                f"[LOCAL] Empty           : "
                f"{local_qa['empty_chunks']}"
            )

            print(
                f"[LOCAL] Oversized       : "
                f"{local_qa['oversized_chunks']}"
            )

            print(
                f"[LOCAL] Same-doc dupes  : "
                f"{local_qa['within_document_duplicate_groups']}"
            )

            print(
                f"[LOCAL] Global dupes    : "
                f"{local_qa['global_duplicate_hash_groups']}"
            )

            # =================================================
            # Transactional Rebuild
            # =================================================

            print()
            print(
                "[DB] Rebuilding core_chunks..."
            )

            with conn.cursor() as cur:

                cur.execute(
                    f"""
                    DELETE FROM {CHUNK_TABLE}
                    """
                )

            print(
                "[DB] Existing chunks cleared "
                "inside transaction."
            )

            # =================================================
            # Insert
            # =================================================

            total_to_insert = (
                len(
                    all_chunks
                )
            )

            for (
                index,
                chunk,
            ) in enumerate(
                all_chunks,
                start=1,
            ):

                _insert_chunk(
                    conn,
                    chunk,
                )

                if (
                    index % 100 == 0
                    or index
                    == total_to_insert
                ):

                    print(
                        f"[DB] Inserted "
                        f"{index}/"
                        f"{total_to_insert}"
                    )

            # =================================================
            # Verify Before Commit
            # =================================================

            verification = (
                _verify_database(
                    conn
                )
            )

            # =================================================
            # Gate 1 - Coverage
            # =================================================

            if (
                verification[
                    "covered_documents"
                ]
                != EXPECTED_DOCUMENTS
            ):

                raise RuntimeError(
                    "Chunk document coverage failed.\n"
                    f"Expected: "
                    f"{EXPECTED_DOCUMENTS}\n"
                    f"Actual  : "
                    f"{verification['covered_documents']}"
                )

            # =================================================
            # Gate 2 - Count
            # =================================================

            if (
                verification[
                    "total_chunks"
                ]
                != len(
                    all_chunks
                )
            ):

                raise RuntimeError(
                    "Chunk count mismatch.\n"
                    f"Local : "
                    f"{len(all_chunks)}\n"
                    f"DB    : "
                    f"{verification['total_chunks']}"
                )

            # =================================================
            # Gate 3 - Empty
            # =================================================

            if (
                verification[
                    "empty_chunks"
                ]
                != 0
            ):

                raise RuntimeError(
                    "Empty chunks detected."
                )

            # =================================================
            # Gate 4 - Empty Retrieval
            # =================================================

            if (
                verification[
                    "empty_retrieval_text"
                ]
                != 0
            ):

                raise RuntimeError(
                    "Empty retrieval_text detected."
                )

            # =================================================
            # Gate 5 - Oversized
            # =================================================

            if (
                verification[
                    "oversized_chunks"
                ]
                != 0
            ):

                raise RuntimeError(
                    "Oversized chunks detected."
                )

            # =================================================
            # Gate 6 - Same-document Duplicate
            # =================================================

            if (
                verification[
                    "within_document_duplicate_groups"
                ]
                != 0
            ):

                raise RuntimeError(
                    "Same-document exact duplicate "
                    "chunks detected."
                )

            # =================================================
            # Gate 7 - Continuous chunk_index
            # =================================================

            if (
                verification[
                    "broken_index_documents"
                ]
                != 0
            ):

                raise RuntimeError(
                    "Broken chunk_index sequence detected.\n"
                    f"Documents: "
                    f"{verification['broken_index_documents']}"
                )

            # =================================================
            # Current Core-100 strict global duplicate gate
            #
            # 지금 Core-100에서는 global duplicate도 0을 목표로 한다.
            # 향후 corpus 확장 후 다른 문서 간 동일 boilerplate가
            # 자연스럽게 존재하면 정책을 다시 분리 가능.
            # =================================================

            if (
                verification[
                    "global_duplicate_hash_groups"
                ]
                != 0
            ):

                raise RuntimeError(
                    "Global exact duplicate chunk groups "
                    "still remain.\n"
                    f"Groups: "
                    f"{verification['global_duplicate_hash_groups']}"
                )

            # =================================================
            # Commit
            # =================================================

            conn.commit()

            # =================================================
            # Report
            # =================================================

            report = {
                "chunker_version": (
                    CHUNKER_VERSION
                ),

                "parameters": {
                    "target_words": (
                        TARGET_WORDS
                    ),

                    "max_words": (
                        MAX_WORDS
                    ),

                    "overlap_words": (
                        OVERLAP_WORDS
                    ),

                    "min_section_words": (
                        MIN_SECTION_WORDS
                    ),
                },

                "document_count": (
                    len(
                        documents
                    )
                ),

                "chunk_count": (
                    len(
                        all_chunks
                    )
                ),

                "deduplicated_count": (
                    total_deduplicated
                ),

                "local_qa": (
                    local_qa
                ),

                "verification": (
                    verification
                ),

                "documents": (
                    document_stats
                ),
            }

            _save_json(
                REPORT_FILE,
                report,
            )

            # =================================================
            # Summary
            # =================================================

            _print_summary(
                verification,
                total_deduplicated,
            )

            print()
            print(
                "[PASS] Core-100 chunking "
                "and exact-dedup completed."
            )

            print()

            print(
                "CORE-100 CHUNK CORPUS IS READY TO FREEZE."
            )

            print()

            print(
                "NEXT:"
            )

            print(
                "1. Fit TF-IDF on retrieval_text"
            )

            print(
                "2. Compare TruncatedSVD "
                "128 / 256 / 384"
            )

            print(
                "3. Select dimension"
            )

            print(
                "4. L2 normalize"
            )

            print(
                "5. Add pgvector column"
            )

            print(
                "6. Store vectors"
            )

            print(
                "7. Build Top-K retrieval"
            )

            print("=" * 78)

    except Exception as exc:

        print()
        print("=" * 78)

        print(
            "CORE-100 CHUNKING FAILED"
        )

        print("=" * 78)

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print()

        print(
            "Transaction was not committed."
        )

        print(
            "Existing committed DB state "
            "should remain preserved."
        )

        print()

        print(
            "Do NOT start TF-IDF/SVD yet."
        )

        print("=" * 78)

        raise


if __name__ == "__main__":
    main()