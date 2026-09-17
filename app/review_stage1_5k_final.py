from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CORPUS = (
    ROOT
    / "data"
    / "stage1_scale"
    / "stage1_scale_5k_v1"
    / "documents.jsonl"
)

SUSPECT_TSV = CORPUS.parent / "final_5k_suspect_ntrs.tsv"
OUT = CORPUS.parent / "final_5k_manual_review.txt"


def first_value(obj: dict, names, default=""):
    for name in names:
        value = obj.get(name)
        if value not in (None, "", [], {}):
            return value

    for container in ("metadata", "meta", "document"):
        nested = obj.get(container)
        if isinstance(nested, dict):
            for name in names:
                value = nested.get(name)
                if value not in (None, "", [], {}):
                    return value

    return default


def title_of(obj):
    return str(first_value(obj, ("title", "paper_title", "name"), "")).strip()


def content_of(obj):
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
                text = item.get("text") or item.get("content") or item.get("body") or ""
                if text:
                    parts.append(str(text))
        return "\n".join(parts)

    return str(value or "")


def source_of(obj):
    return str(
        first_value(
            obj,
            ("source", "source_type", "provider", "dataset"),
            "",
        )
    ).strip()


def source_id_of(obj):
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

    return str(
        first_value(
            obj,
            ("url", "source_url", "pdf_url", "document_url"),
            "",
        )
    ).strip()


def axis_of(obj):
    return str(
        first_value(
            obj,
            ("topic_axis", "axis", "domain_axis", "target_axis"),
            "",
        )
    ).strip()


def abstract_of(obj):
    return str(
        first_value(
            obj,
            ("abstract", "summary", "description"),
            "",
        )
    ).strip()


def categories_of(obj):
    value = first_value(
        obj,
        ("categories", "subjectCategories", "subject_categories"),
        "",
    )

    if isinstance(value, list):
        return ", ".join(str(x) for x in value)

    return str(value or "").strip()


def norm_title(title):
    title = title.lower()
    title = re.sub(r"[^a-z0-9가-힣]+", " ", title)
    return re.sub(r"\s+", " ", title).strip()


records = []

with CORPUS.open("r", encoding="utf-8") as f:
    for line_no, raw in enumerate(f, 1):
        obj = json.loads(raw)
        records.append((line_no, obj))


# ------------------------------------------------------------
# Duplicate-title groups
# ------------------------------------------------------------
titles = defaultdict(list)

for line_no, obj in records:
    key = norm_title(title_of(obj))
    if key:
        titles[key].append((line_no, obj))

dup_groups = {
    key: items
    for key, items in titles.items()
    if len(items) > 1
}


# ------------------------------------------------------------
# NTRS suspect line numbers
# ------------------------------------------------------------
suspect_lines = set()

if SUSPECT_TSV.exists():
    rows = SUSPECT_TSV.read_text(encoding="utf-8").splitlines()[1:]

    for row in rows:
        if not row.strip():
            continue

        parts = row.split("\t", 4)
        try:
            suspect_lines.add(int(parts[0]))
        except Exception:
            pass


# ------------------------------------------------------------
# Long-document outliers
# ------------------------------------------------------------
long_docs = []

for line_no, obj in records:
    content = content_of(obj)
    words = len(content.split())

    if words >= 40000:
        long_docs.append((words, line_no, obj))

long_docs.sort(reverse=True)


# ------------------------------------------------------------
# Report
# ------------------------------------------------------------
lines = []

lines.append("=" * 100)
lines.append("TEAM B - STAGE1 5K FINAL MANUAL REVIEW")
lines.append("=" * 100)

lines.append("")
lines.append(
    f"DUPLICATE TITLE GROUPS : {len(dup_groups)}"
)
lines.append(
    f"NTRS SUSPECT DOCS      : {len(suspect_lines)}"
)
lines.append(
    f"LONG DOCS >= 40K WORDS : {len(long_docs)}"
)
lines.append("")


# Duplicate titles
lines.append("=" * 100)
lines.append("1. DUPLICATE TITLE GROUPS")
lines.append("=" * 100)

if not dup_groups:
    lines.append("NONE")
else:
    for group_no, (_, items) in enumerate(dup_groups.items(), 1):
        lines.append("")
        lines.append(f"[DUP GROUP {group_no}]")

        for line_no, obj in items:
            content = content_of(obj)

            lines.append(f"LINE       : {line_no}")
            lines.append(f"TITLE      : {title_of(obj)}")
            lines.append(f"SOURCE     : {source_of(obj)}")
            lines.append(f"SOURCE_ID  : {source_id_of(obj)}")
            lines.append(f"AXIS       : {axis_of(obj)}")
            lines.append(f"WORDS      : {len(content.split()):,}")
            lines.append("-" * 100)


# NTRS suspects
lines.append("")
lines.append("=" * 100)
lines.append("2. NTRS MANUAL REVIEW")
lines.append("=" * 100)

for line_no in sorted(suspect_lines):
    obj = records[line_no - 1][1]

    abstract = re.sub(r"\s+", " ", abstract_of(obj))
    preview = re.sub(r"\s+", " ", content_of(obj))[:700]

    lines.append("")
    lines.append(f"[NTRS SUSPECT LINE {line_no}]")
    lines.append(f"TITLE      : {title_of(obj)}")
    lines.append(f"SOURCE_ID  : {source_id_of(obj)}")
    lines.append(f"AXIS       : {axis_of(obj)}")
    lines.append(f"CATEGORIES : {categories_of(obj)}")
    lines.append(f"WORDS      : {len(content_of(obj).split()):,}")

    if abstract:
        lines.append(f"ABSTRACT   : {abstract[:1000]}")
    else:
        lines.append(f"PREVIEW    : {preview}")

    lines.append("-" * 100)


# Long docs
lines.append("")
lines.append("=" * 100)
lines.append("3. LONG DOCUMENT OUTLIERS >= 40,000 WORDS")
lines.append("=" * 100)

if not long_docs:
    lines.append("NONE")
else:
    for words, line_no, obj in long_docs:
        lines.append("")
        lines.append(f"[LONG DOC LINE {line_no}]")
        lines.append(f"TITLE      : {title_of(obj)}")
        lines.append(f"SOURCE     : {source_of(obj)}")
        lines.append(f"SOURCE_ID  : {source_id_of(obj)}")
        lines.append(f"AXIS       : {axis_of(obj)}")
        lines.append(f"WORDS      : {words:,}")

        abstract = re.sub(r"\s+", " ", abstract_of(obj))
        if abstract:
            lines.append(f"ABSTRACT   : {abstract[:700]}")

        lines.append("-" * 100)


OUT.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print()
print("=" * 100)
print(f"FULL REVIEW FILE: {OUT}")
print("=" * 100)
