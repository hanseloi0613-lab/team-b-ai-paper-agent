import json
from datetime import datetime
from pathlib import Path

from app.cleaner import clean_pdf
from app.collectors.arxiv_collector import (
    ARXIV_QUERIES,
    download_pdf,
    search_arxiv,
)
from app.config import settings
from app.db import (
    content_hash_exists,
    get_connection,
    insert_core_document,
    source_document_exists,
)


def run_arxiv_pipeline():
    settings.temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    settings.report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stats = {
        "source": "arxiv",
        "started_at": datetime.now().isoformat(),
        "candidates": 0,
        "downloaded": 0,
        "cleaned": 0,
        "saved": 0,
        "duplicates": 0,
        "rejected": 0,
        "failed": 0,
        "per_axis": {},
    }

    with get_connection() as conn:

        for topic_axis in ARXIV_QUERIES:
            print()
            print("=" * 70)
            print(
                f"TOPIC AXIS: {topic_axis}"
            )
            print("=" * 70)

            candidates = search_arxiv(
                topic_axis=topic_axis,
                max_results=(
                    settings
                    .arxiv_candidates_per_axis
                ),
            )

            stats["candidates"] += len(
                candidates
            )

            axis_stats = {
                "candidates": len(
                    candidates
                ),
                "saved": 0,
                "rejected": 0,
                "duplicates": 0,
                "failed": 0,
            }

            for document in candidates:

                if (
                    axis_stats["saved"]
                    >= settings.core_target_per_axis
                ):
                    break

                source_id = document[
                    "source_id"
                ]

                print()
                print(
                    f"[Candidate] {source_id}"
                )

                print(
                    f"[Title] "
                    f"{document['title']}"
                )

                if source_document_exists(
                    conn,
                    document["source"],
                    source_id,
                ):
                    print(
                        "[Skip] Already stored "
                        "source document."
                    )

                    stats["duplicates"] += 1
                    axis_stats[
                        "duplicates"
                    ] += 1

                    continue

                pdf_url = document.get(
                    "pdf_url"
                )

                if not pdf_url:
                    print(
                        "[Reject] No PDF URL."
                    )

                    stats["rejected"] += 1
                    axis_stats[
                        "rejected"
                    ] += 1

                    continue

                safe_id = (
                    source_id
                    .replace("/", "_")
                )

                pdf_path = (
                    settings.temp_dir
                    / "arxiv"
                    / topic_axis
                    / f"{safe_id}.pdf"
                )

                try:
                    download_pdf(
                        pdf_url=pdf_url,
                        output_path=pdf_path,
                    )

                    stats["downloaded"] += 1

                    cleaned = clean_pdf(
                        pdf_path
                    )

                    stats["cleaned"] += 1

                    if not cleaned[
                        "quality_pass"
                    ]:
                        print(
                            "[Reject] "
                            + cleaned[
                                "quality_reason"
                            ]
                        )

                        stats[
                            "rejected"
                        ] += 1

                        axis_stats[
                            "rejected"
                        ] += 1

                        continue

                    if content_hash_exists(
                        conn,
                        cleaned[
                            "content_hash"
                        ],
                    ):
                        print(
                            "[Skip] Duplicate "
                            "content hash."
                        )

                        stats[
                            "duplicates"
                        ] += 1

                        axis_stats[
                            "duplicates"
                        ] += 1

                        continue

                    document.update(
                        {
                            "raw_content": (
                                cleaned[
                                    "raw_content"
                                ]
                            ),
                            "clean_content": (
                                cleaned[
                                    "clean_content"
                                ]
                            ),
                            "content_hash": (
                                cleaned[
                                    "content_hash"
                                ]
                            ),
                            "char_count": (
                                cleaned[
                                    "char_count"
                                ]
                            ),
                            "normalization_version": (
                                settings
                                .normalization_version
                            ),
                            "parse_status": "clean",
                        }
                    )

                    insert_core_document(
                        conn,
                        document,
                    )

                    # 한 문서 저장 성공 단위 commit
                    conn.commit()

                    stats["saved"] += 1
                    axis_stats["saved"] += 1

                    print(
                        "[Saved] "
                        f"{topic_axis}: "
                        f"{axis_stats['saved']}/"
                        f"{settings.core_target_per_axis}"
                    )

                except Exception as exc:
                    conn.rollback()

                    stats["failed"] += 1
                    axis_stats[
                        "failed"
                    ] += 1

                    print(
                        "[ERROR] "
                        f"{source_id}: "
                        f"{exc}"
                    )

                finally:
                    # PDF는 임시 파일
                    if pdf_path.exists():
                        pdf_path.unlink()

            stats["per_axis"][
                topic_axis
            ] = axis_stats

    stats["finished_at"] = (
        datetime.now().isoformat()
    )

    report_name = (
        "arxiv_ingestion_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        + ".json"
    )

    report_path = (
        settings.report_dir
        / report_name
    )

    report_path.write_text(
        json.dumps(
            stats,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("ARXIV INGESTION COMPLETE")
    print("=" * 70)

    print(
        json.dumps(
            stats,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print(
        f"Report saved: {report_path}"
    )