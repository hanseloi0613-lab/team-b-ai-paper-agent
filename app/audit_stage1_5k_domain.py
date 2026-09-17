from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CORPUS = (
    ROOT
    / "data"
    / "stage1_scale"
    / "stage1_scale_5k_v1"
    / "documents.jsonl"
)

OUT_DIR = CORPUS.parent

HIGH_PATH = OUT_DIR / "final_5k_domain_high_risk.tsv"
REVIEW_PATH = OUT_DIR / "final_5k_domain_review.tsv"
REPORT_PATH = OUT_DIR / "final_5k_domain_audit.json"


# ============================================================
# Helpers
# ============================================================

def first_value(obj: dict, names: tuple[str, ...], default=""):
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


def to_text(value) -> str:
    if isinstance(value, list):
        return " ".join(str(x) for x in value)

    return str(value or "")


def get_title(obj: dict) -> str:
    return to_text(
        first_value(
            obj,
            ("title", "paper_title", "name"),
        )
    ).strip()


def get_abstract(obj: dict) -> str:
    return to_text(
        first_value(
            obj,
            ("abstract", "summary", "description"),
        )
    ).strip()


def get_categories(obj: dict) -> str:
    return to_text(
        first_value(
            obj,
            (
                "categories",
                "subjectCategories",
                "subject_categories",
            ),
        )
    ).strip()


def get_axis(obj: dict) -> str:
    return to_text(
        first_value(
            obj,
            (
                "topic_axis",
                "axis",
                "domain_axis",
                "target_axis",
            ),
        )
    ).strip()


def get_source(obj: dict) -> str:
    return to_text(
        first_value(
            obj,
            (
                "source",
                "source_type",
                "provider",
                "dataset",
            ),
        )
    ).strip()


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
    )

    if value:
        return str(value).strip()

    return to_text(
        first_value(
            obj,
            (
                "url",
                "source_url",
                "pdf_url",
                "document_url",
            ),
        )
    ).strip()


def has_any(text: str, markers: tuple[str, ...]) -> bool:
    text = text.lower()
    return any(marker in text for marker in markers)


# ============================================================
# Domain vocabulary
# ============================================================

SPACE_MARKERS = (
    "spacecraft",
    "satellite",
    "space mission",
    "space missions",
    "space system",
    "space systems",
    "space exploration",
    "space communication",
    "space communications",
    "space robotics",
    "space robot",
    "space telescope",
    "orbital",
    "orbit",
    "on-orbit",
    "in-space",
    "deep space",
    "cislunar",
    "lunar",
    "moon",
    "mars",
    "martian",
    "planetary",
    "asteroid",
    "comet",
    "cubesat",
    "space station",
    "iss",
    "gateway",
)

AUTONOMY_AI_MARKERS = (
    "autonomous",
    "autonomy",
    "onboard",
    "on-board",
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "reinforcement learning",
    "neural network",
    "cognitive",
    "intelligent",
    "decision making",
    "decision-making",
    "mission planning",
    "automated planning",
    "adaptive planning",
    "replanning",
    "re-planning",
    "scheduling",
    "navigation",
    "guidance",
    "path planning",
    "trajectory planning",
    "obstacle avoidance",
    "fault detection",
    "fault diagnosis",
    "fault isolation",
    "fault management",
    "health management",
    "anomaly detection",
    "adaptive control",
    "robotic",
    "robotics",
    "multi-agent",
    "multiagent",
    "computer vision",
)

ROVER_MARKERS = (
    "rover",
    "planetary rover",
    "mars rover",
    "lunar rover",
    "planetary robot",
    "mobile robot",
    "ground robot",
    "robot navigation",
    "autonomous navigation",
    "terrain navigation",
    "terrain traversal",
    "locomotion",
    "path planning",
    "obstacle avoidance",
)

SATELLITE_AUTONOMY_MARKERS = (
    "autonomous satellite",
    "satellite autonomy",
    "autonomous spacecraft",
    "spacecraft autonomy",
    "onboard ai",
    "onboard artificial intelligence",
    "onboard machine learning",
    "on-board machine learning",
    "onboard processing",
    "onboard decision",
    "onboard planning",
    "onboard navigation",
    "onboard autonomy",
    "fault management",
    "health management",
    "autonomous operations",
)

OBVIOUS_OFFDOMAIN = (
    "agriculture",
    "crop yield",
    "food security",
    "wildfire",
    "wildland fire",
    "gas flaring",
    "petroleum industry",
    "oil field",
    "sustainable development",
    "energy transition",
    "dark matter",
    "cosmic microwave background",
    "cosmology",
    "hubble tension",
    "galaxy formation",
    "galactic",
    "air-sea interaction",
    "ocean mixing",
    "atmospheric convection",
    "marine biology",
    "medical imaging",
    "clinical",
    "tumor",
    "financial market",
)

AVIATION_MARKERS = (
    "urban air mobility",
    "advanced air mobility",
    "national airspace",
    "air traffic",
    "unmanned aircraft",
    "uas",
    "commercial aircraft",
    "civil aviation",
    "aeronautics",
)


# ============================================================
# Load
# ============================================================

rows = []

with CORPUS.open("r", encoding="utf-8") as f:
    for line_no, raw in enumerate(f, start=1):
        obj = json.loads(raw)

        title = get_title(obj)
        abstract = get_abstract(obj)
        categories = get_categories(obj)
        axis = get_axis(obj)

        probe = (
            f"{title}\n"
            f"{abstract}\n"
            f"{categories}"
        ).lower()

        title_lower = title.lower()

        has_space = has_any(probe, SPACE_MARKERS)
        has_ai = has_any(probe, AUTONOMY_AI_MARKERS)
        has_rover = has_any(probe, ROVER_MARKERS)
        has_sat_auto = has_any(
            probe,
            SATELLITE_AUTONOMY_MARKERS,
        )

        has_offdomain = has_any(
            probe,
            OBVIOUS_OFFDOMAIN,
        )

        has_aviation = has_any(
            probe,
            AVIATION_MARKERS,
        )

        risk = 0
        reasons = []

        # ----------------------------------------------------
        # Global obvious contamination
        # ----------------------------------------------------

        if has_offdomain and not has_ai:
            risk += 4
            reasons.append("obvious_offdomain_without_autonomy")

        # ----------------------------------------------------
        # onboard_ai
        # ----------------------------------------------------

        if axis == "onboard_ai":
            if not has_space:
                risk += 2
                reasons.append("no_space_anchor")

            if not has_ai:
                risk += 3
                reasons.append("no_ai_autonomy_anchor")

            if has_aviation and not has_space:
                risk += 3
                reasons.append("aviation_without_space")

        # ----------------------------------------------------
        # satellite_autonomy
        # ----------------------------------------------------

        elif axis == "satellite_autonomy":
            if not has_space:
                risk += 2
                reasons.append("no_satellite_space_anchor")

            if not (has_ai or has_sat_auto):
                risk += 3
                reasons.append("no_satellite_autonomy_anchor")

            if has_offdomain and not has_ai:
                risk += 2
                reasons.append("remote_sensing_domain_only")

        # ----------------------------------------------------
        # rover_autonomy
        # ----------------------------------------------------

        elif axis == "rover_autonomy":
            if not has_rover:
                risk += 2
                reasons.append("no_rover_robot_anchor")

            if not has_ai:
                risk += 2
                reasons.append("no_autonomy_navigation_anchor")

            if has_aviation:
                # aerial robotics can still be useful,
                # so review rather than automatic rejection.
                risk += 1
                reasons.append("aviation_or_aerial_robotics")

        else:
            risk += 5
            reasons.append("unknown_axis")

        rows.append(
            {
                "line": line_no,
                "source_id": get_source_id(obj),
                "source": get_source(obj),
                "axis": axis,
                "risk": risk,
                "reasons": ",".join(reasons),
                "title": re.sub(
                    r"\s+",
                    " ",
                    title,
                ).strip(),
            }
        )


# ============================================================
# Risk groups
# ============================================================

HIGH_THRESHOLD = 5
REVIEW_THRESHOLD = 3

high_risk = [
    row
    for row in rows
    if row["risk"] >= HIGH_THRESHOLD
]

review = [
    row
    for row in rows
    if REVIEW_THRESHOLD <= row["risk"] < HIGH_THRESHOLD
]

high_risk.sort(
    key=lambda x: (
        -x["risk"],
        x["axis"],
        x["line"],
    )
)

review.sort(
    key=lambda x: (
        -x["risk"],
        x["axis"],
        x["line"],
    )
)


# ============================================================
# Save TSV
# ============================================================

def write_tsv(path: Path, data: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        f.write(
            "line\t"
            "source_id\t"
            "source\t"
            "axis\t"
            "risk\t"
            "reasons\t"
            "title\n"
        )

        for row in data:
            f.write(
                f'{row["line"]}\t'
                f'{row["source_id"]}\t'
                f'{row["source"]}\t'
                f'{row["axis"]}\t'
                f'{row["risk"]}\t'
                f'{row["reasons"]}\t'
                f'{row["title"]}\n'
            )


write_tsv(HIGH_PATH, high_risk)
write_tsv(REVIEW_PATH, review)


# ============================================================
# Summary
# ============================================================

axis_total = Counter(
    row["axis"]
    for row in rows
)

high_axis = Counter(
    row["axis"]
    for row in high_risk
)

review_axis = Counter(
    row["axis"]
    for row in review
)

report = {
    "total_documents": len(rows),
    "axis_counts": dict(axis_total),
    "high_risk_count": len(high_risk),
    "review_count": len(review),
    "high_risk_by_axis": dict(high_axis),
    "review_by_axis": dict(review_axis),
    "thresholds": {
        "high_risk": HIGH_THRESHOLD,
        "review": REVIEW_THRESHOLD,
    },
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
# Console output
# ============================================================

print("=" * 100)
print("TEAM B - STAGE1 FULL 5K DOMAIN AUDIT")
print("=" * 100)

print(f"TOTAL DOCUMENTS      : {len(rows):,}")
print(f"HIGH RISK            : {len(high_risk):,}")
print(f"REVIEW               : {len(review):,}")

print()
print("HIGH RISK BY AXIS")

for axis in (
    "rover_autonomy",
    "onboard_ai",
    "satellite_autonomy",
):
    print(
        f"  {axis:20s}: "
        f"{high_axis.get(axis, 0):,}"
    )

print()
print("REVIEW BY AXIS")

for axis in (
    "rover_autonomy",
    "onboard_ai",
    "satellite_autonomy",
):
    print(
        f"  {axis:20s}: "
        f"{review_axis.get(axis, 0):,}"
    )


print()
print("=" * 100)
print("HIGH-RISK CANDIDATES")
print("=" * 100)

for row in high_risk[:100]:
    print(
        f'LINE={row["line"]} | '
        f'AXIS={row["axis"]} | '
        f'RISK={row["risk"]} | '
        f'{row["reasons"]} | '
        f'{row["title"]}'
    )


print()
print("=" * 100)
print("FILES")
print("=" * 100)

print(f"HIGH RISK : {HIGH_PATH}")
print(f"REVIEW    : {REVIEW_PATH}")
print(f"REPORT    : {REPORT_PATH}")

print("=" * 100)