from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

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
from app.parsers.pdf_parser import (
    parse_pdf_file,
)
from app.cleaner import (
    clean_document,
)


# ============================================================
# TEAM B - NTRS 5-document smoke test
# ============================================================

VERSION = "ntrs_smoke_v1"

ROOT = (
    PROJECT_ROOT
    / "data"
    / "stage1_scale"
    / VERSION
)

WORK_ROOT = (
    ROOT
    / "work"
)

OUTPUT = (
    ROOT
    / "accepted.jsonl"
)

REPORT = (
    ROOT
    / "report.json"
)

TARGET = 5


QUERIES = (
    "spacecraft autonomy",
    "onboard autonomy",
    "autonomous spacecraft",
    "autonomous mission planning",
    "spacecraft mission planning",
    "onboard decision making",
    "deep space autonomy",
    "spacecraft artificial intelligence",
)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def safe_json_write(
    path: Path,
    payload,
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
        errors="replace",
    )


def append_jsonl(
    path: Path,
    payload: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    line = json.dumps(
        payload,
        ensure_ascii=False,
        default=str,
    )

    with path.open(
        "a",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        handle.write(
            line
            + "\n"
        )


def main() -> None:

    if ROOT.exists():
        shutil.rmtree(
            ROOT
        )

    WORK_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    accepted = 0
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()

    stats = {
        "metadata_seen": 0,
        "relevance_reject": 0,
        "type_reject": 0,
        "no_pdf": 0,
        "download_reject": 0,
        "parse_reject": 0,
        "quality_reject": 0,
        "duplicate_reject": 0,
        "accepted": 0,
    }

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

        for query in QUERIES:

            if accepted >= TARGET:
                break

            try:
                (
                    results,
                    total,
                ) = search_citations(
                    client,
                    query=query,
                    size=50,
                    offset=0,
                )

            except Exception as exc:
                print(
                    f"[SEARCH FAIL] "
                    f"{query}: {exc}",
                    flush=True,
                )
                continue

            print(
                f"[QUERY] "
                f"{query} "
                f"pool={len(results)} "
                f"total={total}",
                flush=True,
            )

            # API를 몰아치지 않도록 query 간 간격.
            time.sleep(
                1.5
            )

            for citation in results:

                if accepted >= TARGET:
                    break

                if not isinstance(
                    citation,
                    dict,
                ):
                    continue

                stats[
                    "metadata_seen"
                ] += 1

                citation_id = str(
                    citation.get(
                        "id",
                        "",
                    )
                ).strip()

                if (
                    not citation_id
                    or citation_id
                    in seen_ids
                ):
                    continue

                seen_ids.add(
                    citation_id
                )

                if not acceptable_document_type(
                    citation
                ):
                    stats[
                        "type_reject"
                    ] += 1
                    continue

                (
                    relevant,
                    relevance_score,
                ) = onboard_relevance(
                    citation
                )

                if not relevant:
                    stats[
                        "relevance_reject"
                    ] += 1
                    continue

                print()
                print(
                    "=" * 72,
                    flush=True,
                )
                print(
                    f"[CANDIDATE] "
                    f"id={citation_id}",
                    flush=True,
                )
                print(
                    f"title="
                    f"{citation.get('title', '')}",
                    flush=True,
                )
                print(
                    f"relevance="
                    f"{relevance_score}",
                    flush=True,
                )

                # --------------------------------------------
                # Download list
                # --------------------------------------------

                try:
                    pdf_names = (
                        get_pdf_filenames(
                            client,
                            citation_id,
                        )
                    )

                except Exception as exc:
                    stats[
                        "no_pdf"
                    ] += 1

                    print(
                        f"[NO DOWNLOAD LIST] "
                        f"{exc}",
                        flush=True,
                    )
                    continue

                if not pdf_names:
                    stats[
                        "no_pdf"
                    ] += 1

                    print(
                        "[NO PDF]",
                        flush=True,
                    )
                    continue

                filename = (
                    pdf_names[
                        0
                    ]
                )

                print(
                    f"[PDF] {filename}",
                    flush=True,
                )

                paper_dir = (
                    WORK_ROOT
                    / citation_id
                )

                paper_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                pdf_path = (
                    paper_dir
                    / "paper.pdf"
                )

                # --------------------------------------------
                # Download
                # --------------------------------------------

                try:
                    pdf_url = download_pdf(
                        client,
                        citation_id=(
                            citation_id
                        ),
                        filename=filename,
                        destination=pdf_path,
                    )

                except Exception as exc:
                    stats[
                        "download_reject"
                    ] += 1

                    print(
                        f"[DOWNLOAD REJECT] "
                        f"{exc}",
                        flush=True,
                    )
                    continue

                # --------------------------------------------
                # Metadata for existing parser
                # --------------------------------------------

                metadata = build_metadata(
                    citation,
                    pdf_url=pdf_url,
                    relevance_score=(
                        relevance_score
                    ),
                )

                metadata_path = (
                    paper_dir
                    / "metadata.json"
                )

                safe_json_write(
                    metadata_path,
                    metadata,
                )

                # --------------------------------------------
                # Existing TEAM B PDF parser
                # --------------------------------------------

                try:
                    parsed = parse_pdf_file(
                        pdf_path
                    )

                except Exception as exc:
                    stats[
                        "parse_reject"
                    ] += 1

                    print(
                        f"[PARSE REJECT] "
                        f"{exc}",
                        flush=True,
                    )
                    continue

                parsed_path = (
                    paper_dir
                    / "parsed_document.json"
                )

                parsed_payload = {
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
                }

                safe_json_write(
                    parsed_path,
                    parsed_payload,
                )

                # --------------------------------------------
                # Existing TEAM B cleaner / quality gate
                # --------------------------------------------

                try:
                    cleaned = clean_document(
                        parsed_path
                    )

                except Exception as exc:
                    stats[
                        "quality_reject"
                    ] += 1

                    print(
                        f"[CLEAN REJECT] "
                        f"{exc}",
                        flush=True,
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
                    stats[
                        "quality_reject"
                    ] += 1

                    print(
                        f"[QUALITY REJECT] "
                        f"{quality.get('reasons')}",
                        flush=True,
                    )
                    continue

                content_hash = str(
                    cleaned.get(
                        "content_hash",
                        "",
                    )
                ).strip()

                if (
                    not content_hash
                    or content_hash
                    in seen_hashes
                ):
                    stats[
                        "duplicate_reject"
                    ] += 1

                    print(
                        "[DUPLICATE]",
                        flush=True,
                    )
                    continue

                seen_hashes.add(
                    content_hash
                )

                clean_content = str(
                    cleaned.get(
                        "clean_content",
                        "",
                    )
                )

                row = {
                    "corpus_version": (
                        VERSION
                    ),
                    "collected_at": (
                        utc_now()
                    ),
                    **metadata,
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
                            {},
                        )
                    ),
                }

                append_jsonl(
                    OUTPUT,
                    row,
                )

                accepted += 1

                stats[
                    "accepted"
                ] = accepted

                print(
                    f"[PASS] "
                    f"{accepted}/{TARGET} "
                    f"id={citation_id} "
                    f"words="
                    f"{quality.get('word_count')} "
                    f"sections="
                    f"{quality.get('section_count')}",
                    flush=True,
                )

                # NASA API를 빠르게 연속 호출하지 않는다.
                time.sleep(
                    1.5
                )

    report = {
        "version": VERSION,
        "status": (
            "PASS"
            if accepted >= TARGET
            else "INCOMPLETE"
        ),
        "target": TARGET,
        "accepted": accepted,
        "stats": stats,
        "output": str(
            OUTPUT
        ),
    }

    safe_json_write(
        REPORT,
        report,
    )

    print()
    print(
        "=" * 72
    )
    print(
        "NTRS SMOKE TEST COMPLETE"
    )
    print(
        "=" * 72
    )

    print(
        f"Status   : "
        f"{report['status']}"
    )

    print(
        f"Accepted : "
        f"{accepted}/{TARGET}"
    )

    print(
        "Stats    : "
        + json.dumps(
            stats,
            ensure_ascii=False,
        )
    )

    print(
        f"Output   : {OUTPUT}"
    )

    print(
        f"Report   : {REPORT}"
    )

    print(
        "=" * 72
    )

    if accepted < TARGET:
        raise SystemExit(
            2
        )


if __name__ == "__main__":
    main()
