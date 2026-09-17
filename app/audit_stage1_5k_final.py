from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "stage1_scale"
    / "stage1_scale_5k_v1"
    / "documents.jsonl"
)

OUT_DIR = CORPUS_PATH.parent
REPORT_PATH = OUT_DIR / "final_5k_audit_report.json"
SUSPECT_PATH = OUT_DIR / "final_5k_suspect_ntrs.tsv"

EXPECTED_TOTAL = 5000
EXPECTED_AXIS = {
    "rover_autonomy": 1667,
    "onboard_ai": 1667,
    "satellite_autonomy": 1666,
}


SPACE_MARKERS = (
    "spacecraft",
    "satellite",
    "space mission",
    "space system",
    "space systems",
    "spaceflight",
    "space station",
    "orbital",
    "orbit",
    "lunar",
    "moon",
    "mars",
    "planetary",
    "deep space",
    "cubesat",
    "space robotics",
    "space robot",
    "rover",
    "gateway",
    "international space station",
    "swift mission",
    "eo-1",
    "starling",
)

AUTONOMY_MARKERS = (
    "autonomy",
    "autonomous",
    "onboard",
    "on-board",
    "artificial intelligence",
    "machine learning",
    "reinforcement learning",
    "decision making",
    "decision-making",
    "mission planning",
    "planning",
    "scheduling",
    "navigation",
    "guidance",
    "command execution",
    "fault management",
    "health management",
    "edge computing",
    "intelligent system",
)

AVIATION_MARKERS = (
    "unmanned aircraft",
    "uas",
    "urban air mobility",
    "advanced air mobility",
    "air traffic",
    "national airspace",
    "aircraft",
    "aeronautics",
    "aviation",
)


def first_value(obj: dict, names: tuple[str, ...], default=None):
    for name in names:
        value = obj.get(name)
        if value not in (None, "", [], {}):
            return value

    for container_name in ("metadata", "meta", "document"):
        nested = obj.get(container_name)
        if isinstance(nested, dict):
            for name in names:
                value = nested.get(name)
                if value not in (None, "", [], {}):
                    return value

    return default


def get_title(obj: dict) -> str:
    value = first_value(obj, ("title", "paper_title", "name"), "")
    return str(value or "").strip()


def get_content(obj: dict) -> str:
    value = first_value(
        obj,
        (
            "clean_content",
            "content",
            "full_text",
            "text",
            "body",
            "document_text",
        ),
        "",
    )

    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = (
                    item.get("text")
                    or item.get("content")
                    or item.get("body")
                    or ""
                )
                if text:
                    parts.append(str(text))
        return "\n".join(parts)

    return str(value or "")


def get_axis(obj: dict) -> str:
    value = first_value(
        obj,
        ("topic_axis", "axis", "domain_axis", "target_axis"),
        "",
    )
    return str(value or "").strip()


def get_source(obj: dict) -> str:
    value = first_value(
        obj,
        ("source", "source_type", "provider", "dataset"),
        "",
    )
    return str(value or "").strip()


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

    # URL을 마지막 fallback으로 사용.
    value = first_value(
        obj,
        ("url", "source_url", "pdf_url", "document_url"),
        "",
    )

    return str(value or "").strip()


def get_collector_version(obj: dict) -> str:
    value = first_value(
        obj,
        ("collector_version", "collector", "collection_version"),
        "",
    )
    return str(value or "").strip()


def get_categories(obj: dict) -> str:
    value = first_value(
        obj,
        ("categories", "subjectCategories", "subject_categories"),
        "",
    )

    if isinstance(value, list):
        return " ".join(str(x) for x in value)

    return str(value or "")


def get_abstract(obj: dict) -> str:
    value = first_value(
        obj,
        ("abstract", "summary", "description"),
        "",
    )
    return str(value or "")


def normalize_content(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def normalize_title(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9가-힣]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def contains_any(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def is_ntrs(obj: dict) -> bool:
    combined = " ".join(
        [
            get_source(obj),
            get_collector_version(obj),
            str(first_value(
                obj,
                ("url", "source_url", "pdf_url", "document_url"),
                "",
            )),
        ]
    ).lower()

    return (
        "ntrs" in combined
        or "nasa technical reports" in combined
        or "stage1_ntrs" in combined
    )


def ntrs_review_reason(obj: dict) -> list[str]:
    """
    자동 삭제 기준이 아니라 '사람이 한 번 더 볼 후보'를 찾는 audit.
    false positive가 있을 수 있으므로 이 단계에서는 corpus를 수정하지 않는다.
    """
    title = get_title(obj)
    abstract = get_abstract(obj)
    categories = get_categories(obj)

    # 제목/초록/카테고리를 중심으로 판정.
    # 본문 전체의 우연한 단어 하나 때문에 통과하는 것을 막기 위함.
    review_text = f"{title}\n{abstract}\n{categories}"

    has_space = contains_any(review_text, SPACE_MARKERS)
    has_autonomy = contains_any(review_text, AUTONOMY_MARKERS)
    has_aviation = contains_any(review_text, AVIATION_MARKERS)

    reasons = []

    if not has_space:
        reasons.append("no_clear_space_anchor")

    if not has_autonomy:
        reasons.append("no_clear_autonomy_anchor")

    if has_aviation and not has_space:
        reasons.append("aviation_without_space_anchor")

    return reasons


records = []
invalid_json = []

with CORPUS_PATH.open("r", encoding="utf-8") as f:
    for line_no, raw in enumerate(f, start=1):
        raw = raw.strip()

        if not raw:
            invalid_json.append(
                {
                    "line": line_no,
                    "reason": "empty_line",
                }
            )
            continue

        try:
            obj = json.loads(raw)
        except Exception as exc:
            invalid_json.append(
                {
                    "line": line_no,
                    "reason": f"json_error:{type(exc).__name__}:{exc}",
                }
            )
            continue

        if not isinstance(obj, dict):
            invalid_json.append(
                {
                    "line": line_no,
                    "reason": "json_not_object",
                }
            )
            continue

        records.append((line_no, obj))


axis_counter = Counter()
source_counter = Counter()

empty_title = []
empty_content = []
word_counts = []

source_id_map = defaultdict(list)
content_hash_map = defaultdict(list)
title_map = defaultdict(list)

suspects = []

for line_no, obj in records:
    title = get_title(obj)
    content = get_content(obj)
    axis = get_axis(obj)
    source = get_source(obj)
    source_id = get_source_id(obj)

    axis_counter[axis or "<missing>"] += 1
    source_counter[source or "<missing>"] += 1

    if not title:
        empty_title.append(line_no)

    if not content.strip():
        empty_content.append(line_no)
    else:
        word_counts.append(len(content.split()))

        content_hash = sha256_text(normalize_content(content))
        content_hash_map[content_hash].append(line_no)

    if source_id:
        source_id_map[source_id.lower()].append(line_no)

    if title:
        title_map[normalize_title(title)].append(line_no)

    if is_ntrs(obj):
        reasons = ntrs_review_reason(obj)

        if reasons:
            suspects.append(
                {
                    "line": line_no,
                    "title": title,
                    "source_id": source_id,
                    "axis": axis,
                    "reasons": reasons,
                }
            )


duplicate_source_ids = {
    key: lines
    for key, lines in source_id_map.items()
    if len(lines) > 1
}

duplicate_content = {
    key: lines
    for key, lines in content_hash_map.items()
    if len(lines) > 1
}

duplicate_titles = {
    key: lines
    for key, lines in title_map.items()
    if key and len(lines) > 1
}


def percentile(values: list[int], p: float):
    if not values:
        return None

    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * p))
    return ordered[max(0, min(index, len(ordered) - 1))]


axis_exact = dict(axis_counter) == EXPECTED_AXIS

hard_pass = (
    len(records) == EXPECTED_TOTAL
    and len(invalid_json) == 0
    and len(empty_content) == 0
    and len(duplicate_source_ids) == 0
    and len(duplicate_content) == 0
    and axis_exact
)

if not hard_pass:
    status = "FAIL"
elif duplicate_titles or suspects:
    status = "REVIEW_REQUIRED"
else:
    status = "PASS"


report = {
    "status": status,
    "corpus": str(CORPUS_PATH),
    "expected_total": EXPECTED_TOTAL,
    "valid_json_records": len(records),
    "invalid_json_count": len(invalid_json),
    "invalid_json_examples": invalid_json[:20],
    "axis_counts": dict(axis_counter),
    "expected_axis_counts": EXPECTED_AXIS,
    "axis_exact_match": axis_exact,
    "source_counts": dict(source_counter),
    "empty_title_count": len(empty_title),
    "empty_title_lines": empty_title[:50],
    "empty_content_count": len(empty_content),
    "empty_content_lines": empty_content[:50],
    "duplicate_source_id_groups": len(duplicate_source_ids),
    "duplicate_source_id_examples": dict(
        list(duplicate_source_ids.items())[:20]
    ),
    "duplicate_content_groups": len(duplicate_content),
    "duplicate_content_examples": dict(
        list(duplicate_content.items())[:20]
    ),
    "duplicate_title_groups": len(duplicate_titles),
    "duplicate_title_examples": dict(
        list(duplicate_titles.items())[:20]
    ),
    "ntrs_manual_review_count": len(suspects),
    "word_count_stats": {
        "min": min(word_counts) if word_counts else None,
        "median": median(word_counts) if word_counts else None,
        "p95": percentile(word_counts, 0.95),
        "max": max(word_counts) if word_counts else None,
    },
}

REPORT_PATH.write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

with SUSPECT_PATH.open("w", encoding="utf-8") as f:
    f.write("line\tsource_id\taxis\treasons\ttitle\n")

    for item in suspects:
        title = item["title"].replace("\t", " ").replace("\n", " ")
        reasons = ",".join(item["reasons"])

        f.write(
            f"{item['line']}\t"
            f"{item['source_id']}\t"
            f"{item['axis']}\t"
            f"{reasons}\t"
            f"{title}\n"
        )


print("=" * 78)
print("TEAM B - STAGE1 5K FINAL AUDIT")
print("=" * 78)
print(f"STATUS                  : {status}")
print(f"VALID JSON              : {len(records):,}/{EXPECTED_TOTAL:,}")
print(f"INVALID JSON            : {len(invalid_json):,}")
print()
print("AXIS COUNTS")
for axis, expected in EXPECTED_AXIS.items():
    actual = axis_counter.get(axis, 0)
    print(f"  {axis:20s}: {actual:,}/{expected:,}")
print()
print(f"EMPTY TITLE             : {len(empty_title):,}")
print(f"EMPTY CONTENT           : {len(empty_content):,}")
print(f"DUP SOURCE-ID GROUPS    : {len(duplicate_source_ids):,}")
print(f"DUP CONTENT GROUPS      : {len(duplicate_content):,}")
print(f"DUP TITLE GROUPS        : {len(duplicate_titles):,}")
print(f"NTRS MANUAL REVIEW      : {len(suspects):,}")
print()
if word_counts:
    print("WORD COUNTS")
    print(f"  min                   : {min(word_counts):,}")
    print(f"  median                : {int(median(word_counts)):,}")
    print(f"  p95                   : {percentile(word_counts, 0.95):,}")
    print(f"  max                   : {max(word_counts):,}")
print()
print(f"REPORT                  : {REPORT_PATH}")
print(f"SUSPECT LIST            : {SUSPECT_PATH}")
print("=" * 78)

if status == "PASS":
    print("RESULT: CLEAN 5K - READY TO FREEZE")
elif status == "REVIEW_REQUIRED":
    print("RESULT: HARD INTEGRITY PASS, BUT MANUAL REVIEW IS REQUIRED")
else:
    print("RESULT: DO NOT FREEZE OR TOKENIZE YET")
