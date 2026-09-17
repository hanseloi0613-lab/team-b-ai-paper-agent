from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
import json
from pathlib import Path

from app.config import PROJECT_ROOT
from app.stage1_ntrs_fill_fast_v2 import (
    strict_production_relevance_v2,
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

REPORT = (
    ROOT
    / "strict_ntrs_audit_v2.json"
)


def row_key(
    row: dict,
) -> str:

    source = str(
        row.get(
            "source",
            ""
        )
    ).strip().lower()

    source_id = str(
        row.get(
            "source_id",
            ""
        )
    ).strip()

    if (
        source
        and source_id
    ):
        return (
            f"source:"
            f"{source}:"
            f"{source_id}"
        )

    content_hash = str(
        row.get(
            "content_hash",
            ""
        )
    ).strip()

    if content_hash:
        return (
            "hash:"
            + content_hash
        )

    return (
        "fallback:"
        + str(
            hash(
                json.dumps(
                    row,
                    sort_keys=True,
                    default=str,
                )
            )
        )
    )


def load_rows(
    path: Path,
) -> list[dict]:

    output = []

    if not path.exists():
        return output

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:

        for line in handle:

            line = (
                line.strip()
            )

            if not line:
                continue

            try:
                row = json.loads(
                    line
                )

            except json.JSONDecodeError:
                continue

            if isinstance(
                row,
                dict,
            ):
                output.append(
                    row
                )

    return output


def main():

    backups = sorted(
        ROOT.glob(
            "documents.jsonl.bak_strict_*"
        ),
        key=lambda p: (
            p.stat().st_mtime
        ),
    )

    if not backups:
        raise SystemExit(
            "ERROR: strict backup not found"
        )

    backup = backups[
        -1
    ]

    print(
        f"[BACKUP] {backup}"
    )

    backup_rows = load_rows(
        backup
    )

    current_rows = load_rows(
        CORPUS
    )

    merged = OrderedDict()

    # audit 전 원본 먼저
    for row in backup_rows:
        merged[
            row_key(
                row
            )
        ] = row

    # collector 재시작 이후 새 row가 있으면
    # current가 우선.
    for row in current_rows:
        merged[
            row_key(
                row
            )
        ] = row

    kept = []

    removed = []

    restored = 0

    current_keys = {
        row_key(row)
        for row
        in current_rows
    }

    for key, row in (
        merged.items()
    ):

        is_ntrs_fill = (
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

        if not is_ntrs_fill:

            kept.append(
                row
            )

            continue

        (
            passed,
            score,
            reason,
        ) = (
            strict_production_relevance_v2(
                row,
                3,
            )
        )

        if passed:

            kept.append(
                row
            )

            if key not in (
                current_keys
            ):
                restored += 1

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
                    "score": (
                        score
                    ),
                    "reason": (
                        reason
                    ),
                }
            )

    tmp = (
        CORPUS
        .with_suffix(
            ".jsonl.v2.tmp"
        )
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for row in kept:

            handle.write(
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

    report = {
        "generated_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "backup": str(
            backup
        ),
        "backup_rows": len(
            backup_rows
        ),
        "current_rows": len(
            current_rows
        ),
        "merged_rows": len(
            merged
        ),
        "restored": (
            restored
        ),
        "removed": len(
            removed
        ),
        "final_rows": len(
            kept
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
        "=" * 78
    )

    print(
        "STRICT AUDIT V2 REPAIR"
    )

    print(
        "=" * 78
    )

    print(
        f"backup rows : "
        f"{len(backup_rows)}"
    )

    print(
        f"current     : "
        f"{len(current_rows)}"
    )

    print(
        f"merged      : "
        f"{len(merged)}"
    )

    print(
        f"restored    : "
        f"{restored}"
    )

    print(
        f"removed     : "
        f"{len(removed)}"
    )

    print(
        f"FINAL       : "
        f"{len(kept)}"
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
        "=" * 78
    )


if __name__ == "__main__":
    main()
