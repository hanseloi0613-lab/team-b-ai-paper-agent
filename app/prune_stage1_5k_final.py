from __future__ import annotations

import csv
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
HIGH_RISK = DATA_DIR / "final_5k_domain_high_risk.tsv"

EXPECTED_RAW_TOTAL = 5000


# ============================================================
# Explicit duplicate-title removals
# ============================================================

DUPLICATE_OFFDOMAIN_SOURCE_IDS = {
    # WaveDriver
    "2605.18723v1",
    "2509.18643v2",

    # Pioneer Anomaly
    "1001.3686v2",
    "0807.1088v1",

    # Sustainable development / satellite imagery
    "1810.04881v1",
    "2009.05738v1",
}


# ============================================================
# Explicit NTRS removals from manual review
# ============================================================

NTRS_REMOVE_SOURCE_IDS = {
    # historical/general mission planning
    "19660020353",
    "19660005770",

    # spacecraft concept / propulsion rather than AI/autonomy
    "20140017761",

    # Voyager system history / management
    "19660011774",

    # space communications architecture,
    # but no meaningful AI/autonomy component
    "20180003230",

    # historical Ranger lunar mission
    "19650006254",

    # Mars nuclear propulsion mission study
    "19670025356",

    # Venus science objectives / experiments
    "19670023301",

    # generic robotic manipulator path planning,
    # not sufficiently space-autonomy specific
    "20260001590",

    # historical lunar rendezvous study
    "19630005428",

    # historical Mars transportation study
    "19660028289",
}


# ============================================================
# NTRS documents intentionally retained
# ============================================================

NTRS_KEEP_SOURCE_IDS = {
    # fault detection / diagnostic / spacecraft health inference
    "20100027332",

    # reinforcement-learning cognitive engine for space comms
    "20170007960",

    # cognitive/data-driven spacecraft networking
    "20240007261",

    # explicitly autonomous in-space assembly
    "20205010016",

    # distributed/adaptive on-orbit attitude control
    "20260003381",
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

if not HIGH_RISK.exists():
    raise FileNotFoundError(
        f"High-risk TSV not found: {HIGH_RISK}"
    )


raw_lines = CORPUS.read_text(
    encoding="utf-8"
).splitlines()

if len(raw_lines) != EXPECTED_RAW_TOTAL:
    raise RuntimeError(
        "Safety stop: expected raw corpus to contain "
        f"{EXPECTED_RAW_TOTAL} lines, "
        f"but found {len(raw_lines)}."
    )


records: list[tuple[int, dict, str]] = []

for line_no, raw in enumerate(
    raw_lines,
    start=1,
):
    obj = json.loads(raw)

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
# Load high-risk line numbers
# ============================================================

high_risk_lines: set[int] = set()

with HIGH_RISK.open(
    "r",
    encoding="utf-8",
    newline="",
) as f:
    reader = csv.DictReader(
        f,
        delimiter="\t",
    )

    for row in reader:
        value = str(
            row.get("line", "")
        ).strip()

        if not value:
            continue

        high_risk_lines.add(
            int(value)
        )


if len(high_risk_lines) != 132:
    raise RuntimeError(
        "Safety stop: expected exactly 132 "
        "HIGH-RISK documents, "
        f"but TSV contains {len(high_risk_lines)}."
    )


# ============================================================
# Build removal decision
# ============================================================

kept_raw: list[str] = []
removed: list[dict] = []

reason_counter = Counter()
removed_axis_counter = Counter()
kept_axis_counter = Counter()


for line_no, obj, raw in records:

    source_id = get_source_id(obj)
    axis = get_axis(obj)
    title = get_title(obj)
    source = get_source(obj)

    reasons: list[str] = []

    # A. Global high-risk audit
    if line_no in high_risk_lines:
        reasons.append(
            "domain_high_risk"
        )

    # B. Known duplicate-title/off-domain rows
    if (
        source_id
        in DUPLICATE_OFFDOMAIN_SOURCE_IDS
    ):
        reasons.append(
            "duplicate_title_offdomain"
        )

    # C. Manual NTRS removal
    if (
        source_id
        in NTRS_REMOVE_SOURCE_IDS
    ):
        reasons.append(
            "manual_ntrs_offdomain"
        )

    # Explicit KEEP override applies ONLY to
    # manual NTRS review.
    # It cannot rescue something classified
    # as global HIGH-RISK.
    if (
        source_id
        in NTRS_KEEP_SOURCE_IDS
        and "domain_high_risk" not in reasons
        and "duplicate_title_offdomain"
        not in reasons
    ):
        reasons = [
            r
            for r in reasons
            if r != "manual_ntrs_offdomain"
        ]

    if reasons:
        removed_axis_counter[axis] += 1

        for reason in reasons:
            reason_counter[reason] += 1

        removed.append(
            {
                "original_line": line_no,
                "source_id": source_id,
                "source": source,
                "axis": axis,
                "title": title,
                "reasons": reasons,
                "record": obj,
            }
        )

    else:
        kept_axis_counter[axis] += 1
        kept_raw.append(raw)


# ============================================================
# Backup BEFORE modifying
# ============================================================

stamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

backup_path = (
    DATA_DIR
    / f"documents.pre_final_prune_{stamp}.jsonl"
)

quarantine_path = (
    DATA_DIR
    / f"documents.quarantine_{stamp}.jsonl"
)

report_path = (
    DATA_DIR
    / "final_prune_report.json"
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
# Write quarantine
# ============================================================

with quarantine_path.open(
    "w",
    encoding="utf-8",
) as f:
    for item in removed:
        f.write(
            json.dumps(
                item,
                ensure_ascii=False,
            )
        )
        f.write("\n")


# ============================================================
# Atomic corpus rewrite
# ============================================================

with temp_path.open(
    "w",
    encoding="utf-8",
) as f:
    for raw in kept_raw:
        f.write(raw)
        f.write("\n")


temp_path.replace(CORPUS)


# ============================================================
# Compute deficits
# ============================================================

TARGET = {
    "rover_autonomy": 1667,
    "onboard_ai": 1667,
    "satellite_autonomy": 1666,
}

current_counts = {
    axis: kept_axis_counter.get(axis, 0)
    for axis in TARGET
}

deficits = {
    axis: TARGET[axis] - current_counts[axis]
    for axis in TARGET
}


# ============================================================
# Report
# ============================================================

report = {
    "status": "PRUNED",
    "raw_total": EXPECTED_RAW_TOTAL,
    "removed_total": len(removed),
    "remaining_total": len(kept_raw),
    "removed_by_axis": dict(
        removed_axis_counter
    ),
    "remaining_by_axis": current_counts,
    "deficits_to_refill": deficits,
    "removal_reason_counts": dict(
        reason_counter
    ),
    "high_risk_tsv_count": len(
        high_risk_lines
    ),
    "backup": str(backup_path),
    "quarantine": str(
        quarantine_path
    ),
    "corpus": str(CORPUS),
}

report_path.write_text(
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

print("=" * 84)
print(
    "TEAM B - STAGE1 5K FINAL DOMAIN PRUNE"
)
print("=" * 84)

print(
    f"RAW TOTAL          : "
    f"{EXPECTED_RAW_TOTAL:,}"
)

print(
    f"REMOVED            : "
    f"{len(removed):,}"
)

print(
    f"REMAINING          : "
    f"{len(kept_raw):,}"
)

print()
print("REMOVED BY AXIS")

for axis in TARGET:
    print(
        f"  {axis:20s}: "
        f"{removed_axis_counter.get(axis, 0):,}"
    )

print()
print("CURRENT / TARGET / DEFICIT")

for axis, target in TARGET.items():
    current = current_counts[axis]
    deficit = deficits[axis]

    print(
        f"  {axis:20s}: "
        f"{current:,}/{target:,} "
        f"| refill={deficit:,}"
    )

print()
print("REMOVAL REASONS")

for reason, count in sorted(
    reason_counter.items()
):
    print(
        f"  {reason:30s}: "
        f"{count:,}"
    )

print()
print(
    f"BACKUP      : {backup_path}"
)
print(
    f"QUARANTINE  : {quarantine_path}"
)
print(
    f"REPORT      : {report_path}"
)

print("=" * 84)
print(
    "NEXT: REFILL ONLY THE REPORTED "
    "AXIS DEFICITS."
)
print("=" * 84)