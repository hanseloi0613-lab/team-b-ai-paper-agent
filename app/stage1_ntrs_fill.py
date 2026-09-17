from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
from typing import Any

import httpx

from app.config import PROJECT_ROOT
from app.collectors.ntrs_collector import (
    HEADERS,
    acceptable_document_type,
    build_metadata,
    download_pdf,
    get_pdf_filenames,
    onboard_relevance,
    search_citations,
)
from app.parsers.pdf_parser import parse_pdf_file
from app.cleaner import clean_document


# ============================================================
# TEAM B
# NASA NTRS -> Stage1 5K production filler
# ============================================================
#
# 기존:
#   4,030 clean documents
#
# 목표:
#   onboard_ai 697 -> 1,667
#   TOTAL       4,030 -> 5,000
#
# 기존 corpus를 삭제하거나 재작성하지 않는다.
# 새로운 NTRS 문서만 append한다.
# ============================================================


VERSION = "stage1_ntrs_fill_v1"

TOTAL_TARGET = 5000
ONBOARD_TARGET = 1667


MAIN_ROOT = (
    PROJECT_ROOT
    / "data"
    / "stage1_scale"
    / "stage1_scale_5k_v1"
)

MAIN_CORPUS = (
    MAIN_ROOT
    / "documents.jsonl"
)

REPORT_FILE = (
    MAIN_ROOT
    / "ntrs_fill_report.json"
)

REJECT_FILE = (
    MAIN_ROOT
    / "ntrs_rejected.jsonl"
)

NTRS_ROOT = (
    PROJECT_ROOT
    / "data"
    / "stage1_scale"
    / VERSION
)

WORK_ROOT = (
    NTRS_ROOT
    / "work"
)

CACHE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "ntrs"
    / VERSION
)


# ============================================================
# Production Queries
# ============================================================

QUERIES = (
    "spacecraft autonomy",
    "onboard autonomy",
    "autonomous spacecraft",
    "distributed spacecraft autonomy",
    "spacecraft onboard autonomy",
    "mission autonomy spacecraft",
    "autonomous mission planning",
    "spacecraft mission planning",
    "spacecraft planning execution",
    "onboard decision making",
    "spacecraft decision making",
    "deep space autonomy",
    "spacecraft artificial intelligence",
    "onboard artificial intelligence spacecraft",
    "spacecraft machine learning",
    "onboard machine learning spacecraft",
    "autonomous spacecraft operations",
    "spacecraft resource management",
    "spacecraft scheduling",
    "autonomous guidance navigation spacecraft",
    "satellite autonomous operations",
    "satellite onboard autonomy",
)


# 강한 직접 관련 표현.
STRONG_PHRASES = (
    "spacecraft autonomy",
    "onboard autonomy",
    "on-board autonomy",
    "autonomous spacecraft",
    "mission autonomy",
    "autonomous mission",
    "spacecraft planning",
    "spacecraft decision",
    "onboard decision",
    "on-board decision",
    "deep space autonomy",
    "distributed spacecraft autonomy",
    "spacecraft artificial intelligence",
    "onboard artificial intelligence",
    "spacecraft machine learning",
    "onboard machine learning",
    "autonomous spacecraft operations",
)


# ============================================================
# Helpers
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def safe_string(
    value: Any,
) -> str:

    if value is None:
        return ""

    return (
        str(value)
        .encode(
            "utf-8",
            errors="replace",
        )
        .decode(
            "utf-8",
            errors="replace",
        )
    )


def sanitize(
    value: Any,
) -> Any:

    if isinstance(
        value,
        str,
    ):
        return safe_string(
            value
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            safe_string(key): sanitize(item)
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        list,
    ):
        return [
            sanitize(item)
            for item in value
        ]

    if isinstance(
        value,
        tuple,
    ):
        return [
            sanitize(item)
            for item in value
        ]

    return value


def normalize(
    value: Any,
) -> str:

    return " ".join(
        safe_string(
            value
        ).split()
    )


def title_key(
    value: Any,
) -> str:

    text = normalize(
        value
    ).lower()

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return " ".join(
        text.split()
    )


def append_jsonl(
    path: Path,
    row: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    line = json.dumps(
        sanitize(
            row
        ),
        ensure_ascii=False,
        default=str,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            line
            + "\n"
        )
        handle.flush()


def write_json(
    path: Path,
    payload: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            sanitize(
                payload
            ),
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


# ============================================================
# Main Corpus State
# ============================================================

def load_main_state():

    if not MAIN_CORPUS.exists():
        raise FileNotFoundError(
            f"Main corpus missing: "
            f"{MAIN_CORPUS}"
        )

    counts = Counter()

    seen_sources: set[str] = set()
    seen_hashes: set[str] = set()
    seen_titles: set[str] = set()

    valid_lines = 0

    with MAIN_CORPUS.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:

        for line_number, line in enumerate(
            handle,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(
                    line
                )

            except json.JSONDecodeError:
                print(
                    f"[WARNING] invalid JSONL "
                    f"line={line_number}",
                    flush=True,
                )
                continue

            valid_lines += 1

            axis = normalize(
                row.get(
                    "topic_axis"
                )
            )

            if axis:
                counts[
                    axis
                ] += 1

            source = normalize(
                row.get(
                    "source"
                )
            ).lower()

            source_id = normalize(
                row.get(
                    "source_id"
                )
            )

            if (
                source
                and source_id
            ):
                seen_sources.add(
                    f"{source}:{source_id}"
                )

            content_hash = normalize(
                row.get(
                    "content_hash"
                )
            )

            if content_hash:
                seen_hashes.add(
                    content_hash
                )

            key = title_key(
                row.get(
                    "title"
                )
            )

            if key:
                seen_titles.add(
                    key
                )

    return (
        valid_lines,
        counts,
        seen_sources,
        seen_hashes,
        seen_titles,
    )


# ============================================================
# Persistent Rejection State
# ============================================================

def load_rejected_ids() -> set[str]:

    result: set[str] = set()

    if not REJECT_FILE.exists():
        return result

    with REJECT_FILE.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:

        for line in handle:

            try:
                row = json.loads(
                    line
                )

            except json.JSONDecodeError:
                continue

            citation_id = normalize(
                row.get(
                    "source_id"
                )
            )

            if citation_id:
                result.add(
                    citation_id
                )

    return result


def reject_document(
    citation_id: str,
    reason: str,
    rejected_ids: set[str],
) -> None:

    rejected_ids.add(
        citation_id
    )

    append_jsonl(
        REJECT_FILE,
        {
            "source": "ntrs",
            "source_id": citation_id,
            "rejected_at": utc_now(),
            "reason": reason,
        },
    )


# ============================================================
# Search Cache
# ============================================================

def query_cache_path(
    query: str,
    offset: int,
    size: int,
) -> Path:

    query_hash = hashlib.sha256(
        query.encode(
            "utf-8"
        )
    ).hexdigest()[:16]

    return (
        CACHE_ROOT
        / "search"
        / (
            f"{query_hash}"
            f"_offset_{offset:05d}"
            f"_size_{size}.json"
        )
    )


def search_cached(
    client: httpx.Client,
    *,
    query: str,
    offset: int,
    size: int,
):

    path = query_cache_path(
        query,
        offset,
        size,
    )

    if path.exists():

        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            print(
                f"[CACHE SEARCH] "
                f"q={query!r} "
                f"offset={offset} "
                f"received="
                f"{len(payload['results'])}",
                flush=True,
            )

            return (
                payload[
                    "results"
                ],
                int(
                    payload.get(
                        "total",
                        0,
                    )
                ),
                False,
            )

        except Exception:
            pass

    (
        results,
        total,
    ) = search_citations(
        client,
        query=query,
        size=size,
        offset=offset,
    )

    write_json(
        path,
        {
            "query": query,
            "offset": offset,
            "size": size,
            "total": total,
            "results": results,
        },
    )

    return (
        results,
        total,
        True,
    )


# ============================================================
# Downloads Cache
# ============================================================

def downloads_cache_path(
    citation_id: str,
) -> Path:

    return (
        CACHE_ROOT
        / "downloads"
        / f"{citation_id}.json"
    )


def get_downloads_cached(
    client: httpx.Client,
    citation_id: str,
):

    path = downloads_cache_path(
        citation_id
    )

    if path.exists():

        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            return (
                payload.get(
                    "pdf_names",
                    [],
                ),
                False,
            )

        except Exception:
            pass

    names = get_pdf_filenames(
        client,
        citation_id,
    )

    write_json(
        path,
        {
            "source_id": citation_id,
            "pdf_names": names,
        },
    )

    return (
        names,
        True,
    )


# ============================================================
# Production Relevance Gate
# ============================================================

def citation_text(
    citation: dict,
) -> str:

    parts = [
        citation.get(
            "title"
        ),
        citation.get(
            "abstract"
        ),
    ]

    keywords = (
        citation.get(
            "keywords"
        )
        or []
    )

    categories = (
        citation.get(
            "subjectCategories"
        )
        or []
    )

    if isinstance(
        keywords,
        list,
    ):
        parts.extend(
            keywords
        )

    if isinstance(
        categories,
        list,
    ):
        parts.extend(
            categories
        )

    return " ".join(
        normalize(
            item
        )
        for item
        in parts
        if item is not None
    ).lower()


def production_relevance(
    citation: dict,
    min_score: int,
):

    (
        base_pass,
        score,
    ) = onboard_relevance(
        citation
    )

    if not base_pass:
        return (
            False,
            score,
            "missing_space_or_intelligence_anchor",
        )

    text = citation_text(
        citation
    )

    strong = any(
        phrase in text
        for phrase
        in STRONG_PHRASES
    )

    if (
        score < min_score
        and not strong
    ):
        return (
            False,
            score,
            f"relevance_score<{min_score}",
        )

    return (
        True,
        score,
        "passed",
    )


# ============================================================
# PDF Preference
# ============================================================

def sort_pdf_names(
    names: list[str],
    citation_id: str,
) -> list[str]:

    def rank(
        name: str,
    ):

        lower = name.lower()

        bad = any(
            token in lower
            for token
            in (
                "presentation",
                "poster",
                "slides",
                "briefing",
            )
        )

        id_match = (
            citation_id
            in name
        )

        return (
            1 if bad else 0,
            0 if id_match else 1,
            len(name),
        )

    return sorted(
        names,
        key=rank,
    )


# ============================================================
# Process Candidate
# ============================================================

def process_candidate(
    client: httpx.Client,
    *,
    citation: dict,
    citation_id: str,
    relevance_score: int,
    pdf_names: list[str],
    request_delay: float,
    max_words: int,
    max_sections: int,
    keep_work: bool,
):

    paper_dir = (
        WORK_ROOT
        / citation_id
    )

    paper_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = (
        paper_dir
        / "metadata.json"
    )

    parsed_path = (
        paper_dir
        / "parsed_document.json"
    )

    pdf_path = (
        paper_dir
        / "paper.pdf"
    )

    failures: list[str] = []

    ranked_names = sort_pdf_names(
        pdf_names,
        citation_id,
    )

    # 한 citation에 첨부 PDF가 여러 개인 경우
    # 상위 3개까지만 시험.
    for filename in ranked_names[:3]:

        try:
            pdf_path.unlink(
                missing_ok=True
            )

            parsed_path.unlink(
                missing_ok=True
            )

        except OSError:
            pass

        try:
            pdf_url = download_pdf(
                client,
                citation_id=(
                    citation_id
                ),
                filename=filename,
                destination=pdf_path,
            )

            time.sleep(
                request_delay
            )

        except Exception as exc:
            failures.append(
                f"download:{filename}:{exc}"
            )
            continue

        metadata = build_metadata(
            citation,
            pdf_url=pdf_url,
            relevance_score=(
                relevance_score
            ),
        )

        write_json(
            metadata_path,
            metadata,
        )

        try:
            parsed = parse_pdf_file(
                pdf_path
            )

            write_json(
                parsed_path,
                {
                    "title": (
                        parsed.get(
                            "title",
                            "",
                        )
                    ),
                    "abstract": (
                        parsed.get(
                            "abstract",
                            "",
                        )
                    ),
                    "sections": (
                        parsed.get(
                            "sections",
                            [],
                        )
                    ),
                    "stats": (
                        parsed.get(
                            "stats",
                            {},
                        )
                    ),
                },
            )

            cleaned = clean_document(
                parsed_path
            )

        except Exception as exc:
            failures.append(
                f"parse_clean:"
                f"{filename}:{exc}"
            )
            continue

        quality = (
            cleaned.get(
                "quality"
            )
            or {}
        )

        if not quality.get(
            "passed",
            False,
        ):
            failures.append(
                "quality:"
                + ",".join(
                    str(item)
                    for item
                    in (
                        quality.get(
                            "reasons"
                        )
                        or []
                    )
                )
            )
            continue

        word_count = int(
            quality.get(
                "word_count",
                0,
            )
            or 0
        )

        section_count = int(
            quality.get(
                "section_count",
                0,
            )
            or 0
        )

        # 너무 큰 기술보고서 한 편이
        # training distribution을 지배하지 않도록 제한.
        if word_count > max_words:
            failures.append(
                f"word_count>{max_words}"
            )
            continue

        # PDF heading detector가 깨진 문서 / 초대형 보고서 방지.
        if section_count > max_sections:
            failures.append(
                f"section_count>{max_sections}"
            )
            continue

        content_hash = normalize(
            cleaned.get(
                "content_hash"
            )
        )

        if not content_hash:
            failures.append(
                "missing_content_hash"
            )
            continue

        clean_content = safe_string(
            cleaned.get(
                "clean_content"
            )
        )

        if not clean_content:
            failures.append(
                "empty_clean_content"
            )
            continue

        row = {
            "corpus_version": (
                "stage1_scale_5k_v1"
            ),
            "collector_version": (
                VERSION
            ),
            "collected_at": (
                utc_now()
            ),
            **metadata,
            "matched_query_source": (
                "ntrs_production"
            ),
            "download_filename": (
                filename
            ),
            "clean_content": (
                clean_content
            ),
            "content_hash": (
                content_hash
            ),
            "quality": (
                quality
            ),
            "cleaning_stats": (
                cleaned.get(
                    "cleaning_stats",
                    {}
                )
            ),
        }

        if not keep_work:
            try:
                shutil.rmtree(
                    paper_dir
                )
            except OSError:
                pass

        return (
            row,
            None,
        )

    if not keep_work:
        try:
            shutil.rmtree(
                paper_dir
            )
        except OSError:
            pass

    return (
        None,
        " | ".join(
            failures
        )[:2000],
    )


# ============================================================
# Report
# ============================================================

def save_report(
    *,
    counts: Counter,
    total_documents: int,
    start_total: int,
    accepted_this_run: int,
    stats: Counter,
    started_at: float,
):

    report = {
        "version": VERSION,
        "generated_at": (
            utc_now()
        ),
        "status": (
            "PASS"
            if (
                total_documents
                >= TOTAL_TARGET
                and counts.get(
                    "onboard_ai",
                    0,
                )
                >= ONBOARD_TARGET
            )
            else "RUNNING"
        ),
        "target_total": (
            TOTAL_TARGET
        ),
        "target_onboard": (
            ONBOARD_TARGET
        ),
        "documents": (
            total_documents
        ),
        "onboard_ai": (
            counts.get(
                "onboard_ai",
                0,
            )
        ),
        "start_total": (
            start_total
        ),
        "accepted_this_run": (
            accepted_this_run
        ),
        "elapsed_minutes": round(
            (
                time.time()
                - started_at
            )
            / 60.0,
            2,
        ),
        "stats": dict(
            stats
        ),
    }

    write_json(
        REPORT_FILE,
        report,
    )

    return report


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--max-pages-per-query",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--request-delay",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--min-score",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max-words",
        type=int,
        default=40000,
    )

    parser.add_argument(
        "--max-sections",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--keep-work",
        action="store_true",
    )

    args = parser.parse_args()

    CACHE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    WORK_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        total_documents,
        counts,
        seen_sources,
        seen_hashes,
        seen_titles,
    ) = load_main_state()

    rejected_ids = (
        load_rejected_ids()
    )

    session_skip: set[str] = set()

    start_total = (
        total_documents
    )

    started_at = (
        time.time()
    )

    accepted_this_run = 0

    stats = Counter()

    print(
        "=" * 80,
        flush=True,
    )
    print(
        "TEAM B - NTRS PRODUCTION FILL",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )

    print(
        f"Current total      : "
        f"{total_documents:,}/"
        f"{TOTAL_TARGET:,}",
        flush=True,
    )

    print(
        f"Current onboard_ai : "
        f"{counts.get('onboard_ai', 0):,}/"
        f"{ONBOARD_TARGET:,}",
        flush=True,
    )

    print(
        f"Need               : "
        f"{max(0, TOTAL_TARGET-total_documents):,}",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    if (
        total_documents
        >= TOTAL_TARGET
        and counts.get(
            "onboard_ai",
            0,
        )
        >= ONBOARD_TARGET
    ):
        print(
            "Already complete.",
            flush=True,
        )
        return

    timeout = httpx.Timeout(
        connect=30.0,
        read=120.0,
        write=120.0,
        pool=30.0,
    )

    with httpx.Client(
        headers=HEADERS,
        follow_redirects=True,
        timeout=timeout,
    ) as client:

        exhausted_queries: set[str] = set()

        # Round-robin:
        # query별 page 0을 먼저 훑고,
        # 그 다음 page 1 ...
        # 특정 검색어 편향을 줄인다.
        for page_index in range(
            args.max_pages_per_query
        ):

            if (
                total_documents
                >= TOTAL_TARGET
                or counts.get(
                    "onboard_ai",
                    0,
                )
                >= ONBOARD_TARGET
            ):
                break

            active_queries = 0

            for query in QUERIES:

                if (
                    total_documents
                    >= TOTAL_TARGET
                    or counts.get(
                        "onboard_ai",
                        0,
                    )
                    >= ONBOARD_TARGET
                ):
                    break

                if query in exhausted_queries:
                    continue

                active_queries += 1

                offset = (
                    page_index
                    * args.page_size
                )

                try:
                    (
                        citations,
                        total_hits,
                        api_called,
                    ) = search_cached(
                        client,
                        query=query,
                        offset=offset,
                        size=args.page_size,
                    )

                except Exception as exc:
                    stats[
                        "search_errors"
                    ] += 1

                    print(
                        f"[SEARCH ERROR] "
                        f"q={query!r} "
                        f"offset={offset} "
                        f"{exc}",
                        flush=True,
                    )

                    # source 전체를 죽이지 않는다.
                    continue

                if api_called:
                    time.sleep(
                        args.request_delay
                    )

                if not citations:
                    exhausted_queries.add(
                        query
                    )
                    continue

                if (
                    len(
                        citations
                    )
                    < args.page_size
                    or (
                        total_hits > 0
                        and (
                            offset
                            + len(
                                citations
                            )
                        )
                        >= total_hits
                    )
                ):
                    exhausted_queries.add(
                        query
                    )

                for citation in citations:

                    if (
                        total_documents
                        >= TOTAL_TARGET
                        or counts.get(
                            "onboard_ai",
                            0,
                        )
                        >= ONBOARD_TARGET
                    ):
                        break

                    if not isinstance(
                        citation,
                        dict,
                    ):
                        continue

                    stats[
                        "metadata_seen"
                    ] += 1

                    citation_id = normalize(
                        citation.get(
                            "id"
                        )
                    )

                    if not citation_id:
                        stats[
                            "missing_id"
                        ] += 1
                        continue

                    source_key = (
                        f"ntrs:{citation_id}"
                    )

                    if (
                        source_key
                        in seen_sources
                    ):
                        stats[
                            "source_duplicate"
                        ] += 1
                        continue

                    if (
                        citation_id
                        in rejected_ids
                        or citation_id
                        in session_skip
                    ):
                        stats[
                            "previously_rejected"
                        ] += 1
                        continue

                    if not acceptable_document_type(
                        citation
                    ):
                        stats[
                            "type_reject"
                        ] += 1

                        reject_document(
                            citation_id,
                            "document_type",
                            rejected_ids,
                        )

                        continue

                    (
                        relevant,
                        relevance_score,
                        relevance_reason,
                    ) = production_relevance(
                        citation,
                        args.min_score,
                    )

                    if not relevant:
                        stats[
                            "relevance_reject"
                        ] += 1

                        reject_document(
                            citation_id,
                            relevance_reason,
                            rejected_ids,
                        )

                        continue

                    title = normalize(
                        citation.get(
                            "title"
                        )
                    )

                    key = title_key(
                        title
                    )

                    if (
                        key
                        and key
                        in seen_titles
                    ):
                        stats[
                            "title_duplicate"
                        ] += 1

                        reject_document(
                            citation_id,
                            "duplicate_title",
                            rejected_ids,
                        )

                        continue

                    # ========================================
                    # Download metadata
                    # ========================================

                    try:
                        (
                            pdf_names,
                            api_called,
                        ) = get_downloads_cached(
                            client,
                            citation_id,
                        )

                        if api_called:
                            time.sleep(
                                args.request_delay
                            )

                    except Exception as exc:
                        stats[
                            "downloads_api_error"
                        ] += 1

                        # 네트워크 문제는 영구 reject하지 않는다.
                        session_skip.add(
                            citation_id
                        )

                        print(
                            f"[DOWNLOADS ERROR] "
                            f"id={citation_id} "
                            f"{exc}",
                            flush=True,
                        )

                        continue

                    if not pdf_names:
                        stats[
                            "no_pdf"
                        ] += 1

                        reject_document(
                            citation_id,
                            "no_pdf",
                            rejected_ids,
                        )

                        continue

                    print()
                    print(
                        "-" * 80,
                        flush=True,
                    )

                    print(
                        f"[CANDIDATE] "
                        f"id={citation_id} "
                        f"score={relevance_score}",
                        flush=True,
                    )

                    print(
                        f"[TITLE] "
                        f"{title}",
                        flush=True,
                    )

                    # ========================================
                    # PDF -> parser -> cleaner
                    # ========================================

                    try:
                        (
                            row,
                            failure,
                        ) = process_candidate(
                            client,
                            citation=citation,
                            citation_id=(
                                citation_id
                            ),
                            relevance_score=(
                                relevance_score
                            ),
                            pdf_names=(
                                pdf_names
                            ),
                            request_delay=(
                                args.request_delay
                            ),
                            max_words=(
                                args.max_words
                            ),
                            max_sections=(
                                args.max_sections
                            ),
                            keep_work=(
                                args.keep_work
                            ),
                        )

                    except Exception as exc:
                        row = None
                        failure = str(
                            exc
                        )

                    if row is None:
                        stats[
                            "document_reject"
                        ] += 1

                        reject_document(
                            citation_id,
                            failure
                            or "document_reject",
                            rejected_ids,
                        )

                        print(
                            f"[REJECT] "
                            f"id={citation_id} "
                            f"{failure}",
                            flush=True,
                        )

                        continue

                    content_hash = normalize(
                        row.get(
                            "content_hash"
                        )
                    )

                    if (
                        content_hash
                        in seen_hashes
                    ):
                        stats[
                            "content_duplicate"
                        ] += 1

                        reject_document(
                            citation_id,
                            "duplicate_content_hash",
                            rejected_ids,
                        )

                        print(
                            f"[DUPLICATE HASH] "
                            f"{citation_id}",
                            flush=True,
                        )

                        continue

                    # ========================================
                    # ACCEPT
                    # ========================================

                    append_jsonl(
                        MAIN_CORPUS,
                        row,
                    )

                    seen_sources.add(
                        source_key
                    )

                    seen_hashes.add(
                        content_hash
                    )

                    if key:
                        seen_titles.add(
                            key
                        )

                    counts[
                        "onboard_ai"
                    ] += 1

                    total_documents += 1

                    accepted_this_run += 1

                    stats[
                        "accepted"
                    ] += 1

                    quality = (
                        row.get(
                            "quality"
                        )
                        or {}
                    )

                    elapsed = max(
                        0.001,
                        time.time()
                        - started_at,
                    )

                    rate = (
                        accepted_this_run
                        / elapsed
                    )

                    remaining = max(
                        0,
                        TOTAL_TARGET
                        - total_documents,
                    )

                    if rate > 0:
                        eta_minutes = (
                            remaining
                            / rate
                            / 60
                        )
                        eta = (
                            f"{eta_minutes:.1f}m"
                        )
                    else:
                        eta = "?"

                    print(
                        f"[SAVED] "
                        f"onboard_ai="
                        f"{counts['onboard_ai']:,}/"
                        f"{ONBOARD_TARGET:,} | "
                        f"total="
                        f"{total_documents:,}/"
                        f"{TOTAL_TARGET:,} | "
                        f"words="
                        f"{quality.get('word_count')} | "
                        f"sections="
                        f"{quality.get('section_count')} | "
                        f"ETA={eta}",
                        flush=True,
                    )

                    save_report(
                        counts=counts,
                        total_documents=(
                            total_documents
                        ),
                        start_total=(
                            start_total
                        ),
                        accepted_this_run=(
                            accepted_this_run
                        ),
                        stats=stats,
                        started_at=(
                            started_at
                        ),
                    )

            if active_queries == 0:
                break

    report = save_report(
        counts=counts,
        total_documents=(
            total_documents
        ),
        start_total=(
            start_total
        ),
        accepted_this_run=(
            accepted_this_run
        ),
        stats=stats,
        started_at=(
            started_at
        ),
    )

    status = (
        "PASS"
        if (
            total_documents
            >= TOTAL_TARGET
            and counts.get(
                "onboard_ai",
                0,
            )
            >= ONBOARD_TARGET
        )
        else "INCOMPLETE"
    )

    report[
        "status"
    ] = status

    write_json(
        REPORT_FILE,
        report,
    )

    print()
    print(
        "=" * 80,
        flush=True,
    )

    print(
        "NTRS PRODUCTION FILL COMPLETE",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        f"Status       : {status}",
        flush=True,
    )

    print(
        f"Documents    : "
        f"{total_documents:,}/"
        f"{TOTAL_TARGET:,}",
        flush=True,
    )

    print(
        f"onboard_ai   : "
        f"{counts.get('onboard_ai', 0):,}/"
        f"{ONBOARD_TARGET:,}",
        flush=True,
    )

    print(
        f"Added NTRS   : "
        f"{accepted_this_run:,}",
        flush=True,
    )

    print(
        f"Report       : "
        f"{REPORT_FILE}",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    if status != "PASS":
        raise SystemExit(
            2
        )


if __name__ == "__main__":
    main()
