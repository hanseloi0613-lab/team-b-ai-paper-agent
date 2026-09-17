from __future__ import annotations

import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT
    / "data"
    / "stage1_scale"
    / "stage1_scale_5k_v1"
)

CORPUS = DATA_DIR / "documents.jsonl"

BLACKLIST_TXT = DATA_DIR / "permanent_reject_source_ids.txt"
BLACKLIST_JSON = DATA_DIR / "permanent_reject_source_ids.json"
REPORT_PATH = DATA_DIR / "remove_bad_ntrs_refill_report.json"


# ============================================================
# Current expected corpus state
# ============================================================

EXPECTED_BEFORE_TOTAL = 4878
EXPECTED_REMOVE_COUNT = 25
EXPECTED_AFTER_TOTAL = 4853


TARGET_COUNTS = {
    "rover_autonomy": 1667,
    "onboard_ai": 1667,
    "satellite_autonomy": 1666,
}


EXPECTED_AFTER_COUNTS = {
    "rover_autonomy": 1665,
    "onboard_ai": 1642,
    "satellite_autonomy": 1546,
}


# ============================================================
# The 27 NTRS refill documents were manually reviewed.
#
# KEEP:
#   20020092013
#   Autonomous Reconfigurable Control Allocation (ARCA)
#
#   19950017336
#   experiment scheduling system for ACTS satellite
#
# Everything below is REMOVE.
# ============================================================

CURRENT_BAD_REFILL_IDS = {
    "19660020353",
    "19660005770",
    "20190029601",
    "20140017761",
    "19660011774",
    "20180003230",
    "19650006254",
    "19670025356",
    "19670023301",
    "20160013658",
    "20260001590",
    "19630005428",
    "19660028289",

    # 4865 = KEEP
    # 20020092013

    "20100036494",
    "20040077264",

    # 4868 = KEEP
    # 19950017336

    "19790001954",
    "20140007519",
    "20220013669",
    "19920004856",
    "20000031896",
    "19750011035",
    "19870003154",
    "19950010785",
    "20110002672",
    "20090014009",
}


KEEP_IDS = {
    "20020092013",
    "19950017336",
}


# ============================================================
# Helpers
# ============================================================

def first_value(
    obj: dict,
    names: tuple[str, ...],
    default="",
):
    for name in names:
        value = obj.get(name)

        if value not in (None, "", [], {}):
            return value

    for container_name in (
        "metadata",
        "meta",
        "document",
    ):
        nested = obj.get(container_name)

        if isinstance(nested, dict):
            for name in names:
                value = nested.get(name)

                if value not in (
                    None,
                    "",
                    [],
                    {},
                ):
                    return value

    return default


def get_source_id(obj: dict) -> str:
    value = first_value(
        obj,
        (
            "source_id",
            "external_id",
            "paper_id",
            "arxiv_id",
            "ntrs_id",
        ),
        "",
    )

    if value:
        return str(value).strip()

    value = first_value(
        obj,
        (
            "url",
            "source_url",
            "pdf_url",
            "document_url",
        ),
        "",
    )

    return str(value or "").strip()


def get_title(obj: dict) -> str:
    return str(
        first_value(
            obj,
            (
                "title",
                "paper_title",
                "name",
            ),
            "",
        )
        or ""
    ).strip()


def get_axis(obj: dict) -> str:
    return str(
        first_value(
            obj,
            (
                "topic_axis",
                "axis",
                "domain_axis",
                "target_axis",
            ),
            "",
        )
        or ""
    ).strip()


def get_source(obj: dict) -> str:
    return str(
        first_value(
            obj,
            (
                "source",
                "source_type",
                "provider",
                "dataset",
            ),
            "",
        )
        or ""
    ).strip()


# ============================================================
# Safety checks
# ============================================================

if not CORPUS.exists():
    raise FileNotFoundError(
        f"Corpus not found: {CORPUS}"
    )


raw_lines = CORPUS.read_text(
    encoding="utf-8"
).splitlines()


if len(raw_lines) != EXPECTED_BEFORE_TOTAL:
    raise RuntimeError(
        "SAFETY STOP\n"
        f"Expected {EXPECTED_BEFORE_TOTAL} documents "
        f"before cleanup, but found {len(raw_lines)}.\n"
        "Corpus will NOT be modified."
    )


records: list[tuple[int, dict, str]] = []

for line_no, raw in enumerate(
    raw_lines,
    start=1,
):
    try:
        obj = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(
            f"Invalid JSON at line {line_no}: {exc}"
        ) from exc

    if not isinstance(obj, dict):
        raise RuntimeError(
            f"Line {line_no} is not a JSON object."
        )

    records.append(
        (
            line_no,
            obj,
            raw,
        )
    )


# ============================================================
# Check that the 25 expected documents are really present
# exactly once.
# ============================================================

source_id_lines: dict[str, list[int]] = {}

for line_no, obj, _ in records:
    source_id = get_source_id(obj)

    if source_id:
        source_id_lines.setdefault(
            source_id,
            [],
        ).append(line_no)


missing_bad_ids = sorted(
    source_id
    for source_id in CURRENT_BAD_REFILL_IDS
    if source_id not in source_id_lines
)


duplicated_bad_ids = {
    source_id: source_id_lines[source_id]
    for source_id in CURRENT_BAD_REFILL_IDS
    if len(source_id_lines.get(source_id, [])) > 1
}


if missing_bad_ids:
    raise RuntimeError(
        "SAFETY STOP\n"
        "Some expected bad-refill IDs are not present:\n"
        + "\n".join(missing_bad_ids)
    )


if duplicated_bad_ids:
    raise RuntimeError(
        "SAFETY STOP\n"
        "Some bad-refill IDs occur more than once:\n"
        + json.dumps(
            duplicated_bad_ids,
            ensure_ascii=False,
            indent=2,
        )
    )


# KEEP rows must exist too.
missing_keep_ids = sorted(
    source_id
    for source_id in KEEP_IDS
    if source_id not in source_id_lines
)

if missing_keep_ids:
    raise RuntimeError(
        "SAFETY STOP\n"
        "Expected KEEP documents are missing:\n"
        + "\n".join(missing_keep_ids)
    )


# ============================================================
# Build permanent blacklist
#
# Sources:
# 1. previous blacklist if present
# 2. every previous quarantine file
# 3. current 25 rejected refill docs
# ============================================================

permanent_blacklist: set[str] = set()


# ------------------------------------------------------------
# Existing blacklist
# ------------------------------------------------------------

if BLACKLIST_TXT.exists():
    for line in BLACKLIST_TXT.read_text(
        encoding="utf-8"
    ).splitlines():

        value = line.strip()

        if value:
            permanent_blacklist.add(value)


# ------------------------------------------------------------
# Previous quarantine files
# ------------------------------------------------------------

quarantine_files = sorted(
    DATA_DIR.glob(
        "documents.quarantine_*.jsonl"
    )
)


quarantine_loaded = 0

for quarantine_file in quarantine_files:

    with quarantine_file.open(
        "r",
        encoding="utf-8",
    ) as f:

        for raw in f:

            raw = raw.strip()

            if not raw:
                continue

            try:
                item = json.loads(raw)
            except Exception:
                continue

            source_id = str(
                item.get(
                    "source_id",
                    "",
                )
                or ""
            ).strip()

            # Fallback in case source_id only exists
            # inside nested original record.
            if (
                not source_id
                and isinstance(
                    item.get("record"),
                    dict,
                )
            ):
                source_id = get_source_id(
                    item["record"]
                )

            if source_id:
                permanent_blacklist.add(
                    source_id
                )
                quarantine_loaded += 1


# ------------------------------------------------------------
# Current 25 bad refill documents
# ------------------------------------------------------------

permanent_blacklist.update(
    CURRENT_BAD_REFILL_IDS
)


# KEEP IDs must NEVER accidentally be blacklisted.
for keep_id in KEEP_IDS:
    permanent_blacklist.discard(
        keep_id
    )


# ============================================================
# Prepare removal
# ============================================================

kept_lines: list[str] = []
removed_records: list[dict] = []

before_axis_counts = Counter()
after_axis_counts = Counter()
removed_axis_counts = Counter()


for line_no, obj, raw in records:

    source_id = get_source_id(obj)
    title = get_title(obj)
    axis = get_axis(obj)
    source = get_source(obj)

    before_axis_counts[axis] += 1

    if source_id in CURRENT_BAD_REFILL_IDS:

        removed_axis_counts[axis] += 1

        removed_records.append(
            {
                "original_line": line_no,
                "source_id": source_id,
                "source": source,
                "axis": axis,
                "title": title,
                "reason": "manual_review_bad_ntrs_refill",
                "record": obj,
            }
        )

    else:

        kept_lines.append(raw)
        after_axis_counts[axis] += 1


# ============================================================
# Verify removal count BEFORE touching corpus
# ============================================================

if len(removed_records) != EXPECTED_REMOVE_COUNT:
    raise RuntimeError(
        "SAFETY STOP\n"
        f"Expected to remove {EXPECTED_REMOVE_COUNT}, "
        f"but computed {len(removed_records)}.\n"
        "Corpus will NOT be modified."
    )


if len(kept_lines) != EXPECTED_AFTER_TOTAL:
    raise RuntimeError(
        "SAFETY STOP\n"
        f"Expected {EXPECTED_AFTER_TOTAL} remaining docs, "
        f"but computed {len(kept_lines)}.\n"
        "Corpus will NOT be modified."
    )


actual_after_counts = {
    axis: after_axis_counts.get(axis, 0)
    for axis in TARGET_COUNTS
}


if actual_after_counts != EXPECTED_AFTER_COUNTS:
    raise RuntimeError(
        "SAFETY STOP\n"
        "Axis counts are not what we expected.\n"
        f"Expected: {EXPECTED_AFTER_COUNTS}\n"
        f"Actual:   {actual_after_counts}\n"
        "Corpus will NOT be modified."
    )


# ============================================================
# Backup
# ============================================================

stamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)


backup_path = (
    DATA_DIR
    / f"documents.before_bad_refill_cleanup_{stamp}.jsonl"
)


removed_path = (
    DATA_DIR
    / f"documents.bad_refill_quarantine_{stamp}.jsonl"
)


temp_path = (
    DATA_DIR
    / "documents.jsonl.tmp"
)


shutil.copy2(
    CORPUS,
    backup_path,
)


# ============================================================
# Save removed documents
# ============================================================

with removed_path.open(
    "w",
    encoding="utf-8",
) as f:

    for item in removed_records:

        f.write(
            json.dumps(
                item,
                ensure_ascii=False,
            )
        )

        f.write("\n")


# ============================================================
# Write new corpus atomically
# ============================================================

with temp_path.open(
    "w",
    encoding="utf-8",
) as f:

    for raw in kept_lines:

        f.write(raw)
        f.write("\n")


temp_path.replace(
    CORPUS
)


# ============================================================
# Save permanent blacklist
# ============================================================

sorted_blacklist = sorted(
    permanent_blacklist
)


BLACKLIST_TXT.write_text(
    "\n".join(sorted_blacklist)
    + "\n",
    encoding="utf-8",
)


BLACKLIST_JSON.write_text(
    json.dumps(
        {
            "count": len(sorted_blacklist),
            "source_ids": sorted_blacklist,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# Refill deficits
# ============================================================

deficits = {
    axis: TARGET_COUNTS[axis]
    - actual_after_counts[axis]
    for axis in TARGET_COUNTS
}


if deficits != {
    "rover_autonomy": 2,
    "onboard_ai": 25,
    "satellite_autonomy": 120,
}:
    raise RuntimeError(
        "Unexpected refill deficit after cleanup: "
        f"{deficits}"
    )


# ============================================================
# Final report
# ============================================================

report = {
    "status": "PASS",
    "before_total": EXPECTED_BEFORE_TOTAL,
    "removed_total": len(removed_records),
    "after_total": len(kept_lines),

    "before_axis_counts": {
        axis: before_axis_counts.get(axis, 0)
        for axis in TARGET_COUNTS
    },

    "removed_axis_counts": {
        axis: removed_axis_counts.get(axis, 0)
        for axis in TARGET_COUNTS
    },

    "after_axis_counts": actual_after_counts,

    "refill_deficits": deficits,

    "keep_ids": sorted(
        KEEP_IDS
    ),

    "current_bad_refill_ids": sorted(
        CURRENT_BAD_REFILL_IDS
    ),

    "permanent_blacklist_count": len(
        sorted_blacklist
    ),

    "previous_quarantine_files": [
        str(path)
        for path in quarantine_files
    ],

    "quarantine_entries_loaded": quarantine_loaded,

    "backup": str(
        backup_path
    ),

    "removed_documents": str(
        removed_path
    ),

    "permanent_blacklist_txt": str(
        BLACKLIST_TXT
    ),

    "permanent_blacklist_json": str(
        BLACKLIST_JSON
    ),

    "corpus": str(
        CORPUS
    ),
}


REPORT_PATH.write_text(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# Console
# ============================================================

print("=" * 88)
print(
    "TEAM B - REMOVE BAD NTRS REFILL + "
    "CREATE PERMANENT BLACKLIST"
)
print("=" * 88)

print(
    f"BEFORE               : "
    f"{EXPECTED_BEFORE_TOTAL:,}"
)

print(
    f"REMOVED              : "
    f"{len(removed_records):,}"
)

print(
    f"AFTER                : "
    f"{len(kept_lines):,}"
)

print()

print("REMOVED BY AXIS")

for axis in TARGET_COUNTS:
    print(
        f"  {axis:20s}: "
        f"{removed_axis_counts.get(axis, 0):,}"
    )


print()

print("CURRENT / TARGET / REFILL")

for axis, target in TARGET_COUNTS.items():

    current = actual_after_counts[axis]
    refill = deficits[axis]

    print(
        f"  {axis:20s}: "
        f"{current:,}/{target:,} "
        f"| refill={refill:,}"
    )


print()

print("KEPT FROM LAST 27")

for keep_id in sorted(KEEP_IDS):

    line = source_id_lines[keep_id][0]
    obj = records[line - 1][1]

    print(
        f"  KEEP {keep_id} | "
        f"{get_title(obj)}"
    )


print()

print(
    f"PERMANENT BLACKLIST  : "
    f"{len(sorted_blacklist):,} unique source IDs"
)

print(
    f"BLACKLIST FILE       : "
    f"{BLACKLIST_TXT}"
)

print(
    f"BACKUP               : "
    f"{backup_path}"
)

print(
    f"REMOVED QUARANTINE   : "
    f"{removed_path}"
)

print(
    f"REPORT               : "
    f"{REPORT_PATH}"
)

print("=" * 88)

print(
    "NEXT REFILL TARGET:"
)

print(
    "  rover_autonomy      +2"
)

print(
    "  onboard_ai         +25"
)

print(
    "  satellite_autonomy +120"
)

print(
    "  TOTAL              +147"
)

print("=" * 88)