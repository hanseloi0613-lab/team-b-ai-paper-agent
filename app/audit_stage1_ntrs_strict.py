from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

from app.config import PROJECT_ROOT
from app.stage1_ntrs_fill_fast import (
    strict_production_relevance,
)


ROOT = (
    PROJECT_ROOT
    / "data"
    / "stage1_scale"
    / "stage1_scale_5k_v1"
)

CORPUS = (
    ROOT
    / "documents.jsonl"
)

REJECT = (
    ROOT
    / "ntrs_rejected.jsonl"
)

REPORT = (
    ROOT
    / "strict_ntrs_audit.json"
)


def now():
    return datetime.now(
        timezone.utc
    ).isoformat()


def main():

    rows = []

    with CORPUS.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:

        for line in f:

            if line.strip():

                rows.append(
                    json.loads(
                        line
                    )
                )

    kept = []
    removed = []

    for row in rows:

        is_our_ntrs_fill = (
            str(
                row.get(
                    "source",
                    ""
                )
            ).lower()
            == "ntrs"

            and
            row.get(
                "topic_axis"
            )
            == "onboard_ai"

            and
            row.get(
                "collector_version"
            )
            == "stage1_ntrs_fill_v1"
        )

        if not is_our_ntrs_fill:
            kept.append(
                row
            )
            continue

        (
            passed,
            score,
            reason,
        ) = strict_production_relevance(
            row,
            3,
        )

        if passed:
            kept.append(
                row
            )

        else:
            removed.append(
                {
                    "source_id": (
                        row.get(
                            "source_id"
                        )
                    ),
                    "title": (
                        row.get(
                            "title"
                        )
                    ),
                    "score": score,
                    "reason": reason,
                }
            )

    tmp = CORPUS.with_suffix(
        ".jsonl.strict.tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in kept:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

    tmp.replace(
        CORPUS
    )

    if removed:

        with REJECT.open(
            "a",
            encoding="utf-8",
        ) as f:

            for item in removed:

                f.write(
                    json.dumps(
                        {
                            "source": "ntrs",
                            "source_id": (
                                item[
                                    "source_id"
                                ]
                            ),
                            "rejected_at": now(),
                            "reason": (
                                "strict_audit:"
                                + item[
                                    "reason"
                                ]
                            ),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    report = {
        "status": "PASS",
        "before": len(
            rows
        ),
        "after": len(
            kept
        ),
        "removed": len(
            removed
        ),
        "removed_documents": (
            removed
        ),
    }

    REPORT.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "=" * 72
    )

    print(
        "STRICT NTRS AUDIT"
    )

    print(
        "=" * 72
    )

    print(
        f"BEFORE  : {len(rows)}"
    )

    print(
        f"REMOVED : {len(removed)}"
    )

    print(
        f"AFTER   : {len(kept)}"
    )

    print()

    for item in removed:

        print(
            "[REMOVED]",
            item[
                "source_id"
            ],
            "|",
            item[
                "reason"
            ],
        )

        print(
            "          ",
            item[
                "title"
            ],
        )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()
