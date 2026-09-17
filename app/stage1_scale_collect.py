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

from app.config import PROJECT_ROOT, settings
from app.collectors.arxiv_collector import search_query_page
from app.parsers.pdf_parser import parse_pdf_file
from app.cleaner import clean_document


# ============================================================
# TEAM B
# Transformer Stage1 Scalable Full-Text Corpus Collector
# ============================================================
#
# RAG Core100과 완전히 분리된 Transformer 학습 corpus.
#
# arXiv metadata
#   -> relevance gate
#   -> PDF download
#   -> PDF parse
#   -> clean_document()
#   -> quality gate
#   -> source/content dedup
#   -> documents.jsonl
#
# 기존 documents.jsonl이 존재하면 삭제하지 않고 RESUME한다.
# ============================================================


VERSION = "stage1_scale_5k_v1"

AXES = (
    "rover_autonomy",
    "onboard_ai",
    "satellite_autonomy",
)


# ============================================================
# Search Queries
# ============================================================

SCALE_QUERIES = {
    "rover_autonomy": [
        {
            "name": "planetary_rover",
            "query": 'all:"planetary rover"',
        },
        {
            "name": "rover_navigation",
            "query": 'all:rover AND all:navigation',
        },
        {
            "name": "rover_autonomy",
            "query": 'all:rover AND all:autonomy',
        },
        {
            "name": "rover_path_planning",
            "query": 'all:rover AND all:"path planning"',
        },
        {
            "name": "robot_terrain_navigation",
            "query": 'cat:cs.RO AND all:terrain AND all:navigation',
        },
        {
            "name": "robot_traversability",
            "query": 'cat:cs.RO AND all:traversability',
        },
        {
            "name": "robot_obstacle",
            "query": 'cat:cs.RO AND all:"obstacle avoidance"',
        },
        {
            "name": "robot_localization",
            "query": 'cat:cs.RO AND all:localization AND all:navigation',
        },
    ],

    # ========================================================
    # ONBOARD AI
    #
    # 기존 query + 확장 query.
    #
    # 목표:
    # spacecraft / satellite / deep-space / space-system이라는
    # "space domain anchor"를 유지하면서,
    #
    # autonomy / planning / decision / scheduling /
    # onboard AI / ML / resource management 등을 확장한다.
    # ========================================================

    "onboard_ai": [
        # ---------- 기존 축 ----------
        {
            "name": "spacecraft_autonomy",
            "query": 'all:spacecraft AND all:autonomy',
        },
        {
            "name": "autonomous_spacecraft",
            "query": 'all:"autonomous spacecraft"',
        },
        {
            "name": "spacecraft_planning",
            "query": 'all:spacecraft AND all:planning',
        },
        {
            "name": "spacecraft_navigation",
            "query": 'all:spacecraft AND all:navigation',
        },
        {
            "name": "deep_space_autonomy",
            "query": 'all:"deep space" AND all:autonomy',
        },
        {
            "name": "onboard_ai",
            "query": (
                'all:spacecraft AND all:onboard '
                'AND all:"artificial intelligence"'
            ),
        },
        {
            "name": "onboard_ml",
            "query": (
                'all:spacecraft AND all:onboard '
                'AND all:"machine learning"'
            ),
        },
        {
            "name": "mission_autonomy",
            "query": 'all:"mission autonomy"',
        },

        # ---------- NEW: onboard / decision ----------
        {
            "name": "onboard_autonomy",
            "query": 'all:"onboard autonomy"',
        },
        {
            "name": "onboard_decision_making",
            "query": (
                'all:onboard AND '
                'all:"decision making" AND '
                '(all:spacecraft OR all:satellite)'
            ),
        },
        {
            "name": "spacecraft_decision_making",
            "query": (
                'all:spacecraft AND '
                'all:"decision making"'
            ),
        },
        {
            "name": "autonomous_decision_spacecraft",
            "query": (
                'all:spacecraft AND '
                'all:"autonomous decision"'
            ),
        },

        # ---------- NEW: planning / scheduling ----------
        {
            "name": "autonomous_mission_planning",
            "query": 'all:"autonomous mission planning"',
        },
        {
            "name": "spacecraft_mission_planning",
            "query": (
                'all:spacecraft AND '
                'all:"mission planning"'
            ),
        },
        {
            "name": "satellite_mission_planning",
            "query": (
                'all:satellite AND '
                'all:"mission planning"'
            ),
        },
        {
            "name": "spacecraft_scheduling",
            "query": (
                'all:spacecraft AND '
                'all:scheduling'
            ),
        },
        {
            "name": "satellite_scheduling",
            "query": (
                'all:satellite AND '
                'all:scheduling'
            ),
        },
        {
            "name": "spacecraft_task_scheduling",
            "query": (
                'all:spacecraft AND '
                'all:"task scheduling"'
            ),
        },

        # ---------- NEW: operations / mission management ----------
        {
            "name": "spacecraft_operations_autonomy",
            "query": (
                'all:"spacecraft operations" AND '
                'all:autonomy'
            ),
        },
        {
            "name": "autonomous_mission_operations",
            "query": 'all:"autonomous mission operations"',
        },
        {
            "name": "space_mission_autonomy",
            "query": (
                'all:"space mission" AND '
                'all:autonomy'
            ),
        },
        {
            "name": "mission_management_spacecraft",
            "query": (
                'all:spacecraft AND '
                'all:"mission management"'
            ),
        },
        {
            "name": "resource_management_spacecraft",
            "query": (
                'all:spacecraft AND '
                'all:"resource management"'
            ),
        },

        # ---------- NEW: onboard computing / AI ----------
        {
            "name": "satellite_onboard_ai",
            "query": (
                'all:satellite AND '
                'all:onboard AND '
                'all:"artificial intelligence"'
            ),
        },
        {
            "name": "satellite_onboard_ml",
            "query": (
                'all:satellite AND '
                'all:onboard AND '
                'all:"machine learning"'
            ),
        },
        {
            "name": "onboard_deep_learning_space",
            "query": (
                'all:onboard AND '
                'all:"deep learning" AND '
                '(all:spacecraft OR all:satellite)'
            ),
        },
        {
            "name": "onboard_reinforcement_learning",
            "query": (
                'all:onboard AND '
                'all:"reinforcement learning" AND '
                '(all:spacecraft OR all:satellite)'
            ),
        },

        # ---------- NEW: space systems ----------
        {
            "name": "space_systems_autonomy",
            "query": (
                'all:"space systems" AND '
                'all:autonomy'
            ),
        },
        {
            "name": "space_system_autonomous",
            "query": (
                'all:"space system" AND '
                'all:autonomous'
            ),
        },
        {
            "name": "spacecraft_ai",
            "query": (
                'all:spacecraft AND '
                'all:"artificial intelligence"'
            ),
        },
        {
            "name": "spacecraft_machine_learning",
            "query": (
                'all:spacecraft AND '
                'all:"machine learning"'
            ),
        },

        # ---------- NEW: control/navigation autonomy ----------
        {
            "name": "autonomous_guidance_spacecraft",
            "query": (
                'all:spacecraft AND '
                'all:autonomous AND '
                'all:guidance'
            ),
        },
        {
            "name": "autonomous_navigation_spacecraft",
            "query": (
                'all:spacecraft AND '
                'all:"autonomous navigation"'
            ),
        },
        {
            "name": "spacecraft_guidance_planning",
            "query": (
                'all:spacecraft AND '
                'all:guidance AND '
                'all:planning'
            ),
        },

        # ---------- NEW: category anchored ----------
        {
            "name": "cs_ai_spacecraft",
            "query": (
                'cat:cs.AI AND '
                'all:spacecraft'
            ),
        },
        {
            "name": "cs_ro_spacecraft",
            "query": (
                'cat:cs.RO AND '
                'all:spacecraft'
            ),
        },
        {
            "name": "eess_spacecraft_autonomy",
            "query": (
                'cat:eess.SY AND '
                'all:spacecraft AND '
                'all:autonomous'
            ),
        },
        {
            "name": "satellite_autonomous_control",
            "query": (
                'all:satellite AND '
                'all:autonomous AND '
                'all:control'
            ),
        },
    ],

    "satellite_autonomy": [
        {
            "name": "satellite_fault_detection",
            "query": 'all:satellite AND all:"fault detection"',
        },
        {
            "name": "spacecraft_fault_detection",
            "query": 'all:spacecraft AND all:"fault detection"',
        },
        {
            "name": "satellite_anomaly",
            "query": 'all:satellite AND all:anomaly',
        },
        {
            "name": "spacecraft_anomaly",
            "query": 'all:spacecraft AND all:anomaly',
        },
        {
            "name": "fault_diagnosis",
            "query": 'all:spacecraft AND all:"fault diagnosis"',
        },
        {
            "name": "health_monitoring",
            "query": 'all:satellite AND all:"health monitoring"',
        },
        {
            "name": "satellite_ml",
            "query": 'all:satellite AND all:"machine learning"',
        },
        {
            "name": "satellite_autonomy",
            "query": 'all:satellite AND all:autonomy',
        },
        {
            "name": "autonomous_operations",
            "query": 'all:spacecraft AND all:"autonomous operations"',
        },
    ],
}


# ============================================================
# Relevance Keywords
# ============================================================

AXIS_KEYWORDS = {
    "rover_autonomy": (
        "rover",
        "planetary",
        "mars",
        "lunar",
        "navigation",
        "path planning",
        "terrain",
        "traversability",
        "obstacle",
        "hazard",
        "localization",
        "robot",
        "autonomous",
        "autonomy",
    ),

    "onboard_ai": (
        "spacecraft",
        "satellite",
        "space mission",
        "space missions",
        "space system",
        "space systems",
        "deep space",
        "interplanetary",
        "orbital",
        "onboard",
        "on-board",
        "autonomy",
        "autonomous",
        "planning",
        "mission planning",
        "decision",
        "decision making",
        "scheduling",
        "task scheduling",
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "reinforcement learning",
        "guidance",
        "navigation",
        "control",
        "mission management",
        "resource management",
        "operations",
    ),

    "satellite_autonomy": (
        "satellite",
        "spacecraft",
        "fault",
        "anomaly",
        "diagnosis",
        "detection",
        "recovery",
        "fdir",
        "health monitoring",
        "fault management",
        "autonomous operations",
        "autonomy",
    ),
}


# onboard_ai는 단순 keyword 1개만으로 통과시키지 않는다.
# 반드시:
#
#   SPACE DOMAIN + AI/AUTONOMY/DECISION/PLANNING
#
# 두 그룹 모두 존재해야 한다.

ONBOARD_SPACE_ANCHORS = (
    "spacecraft",
    "satellite",
    "deep space",
    "space mission",
    "space missions",
    "space system",
    "space systems",
    "interplanetary",
    "orbital",
    "onboard",
    "on-board",
)

ONBOARD_INTELLIGENCE_ANCHORS = (
    "autonomy",
    "autonomous",
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "reinforcement learning",
    "decision",
    "decision making",
    "planning",
    "mission planning",
    "scheduling",
    "task scheduling",
    "mission management",
    "resource management",
    "guidance",
    "navigation",
    "control",
    "onboard",
    "on-board",
)


# ============================================================
# Paths
# ============================================================

ROOT = (
    PROJECT_ROOT
    / "data"
    / "stage1_scale"
    / VERSION
)

OUTPUT_JSONL = (
    ROOT
    / "documents.jsonl"
)

REPORT_FILE = (
    ROOT
    / "collection_report.json"
)

WORK_ROOT = (
    ROOT
    / "work"
)

PERMANENT_REJECT_FILE = (
    ROOT
    / "permanent_reject_source_ids.txt"
)


# ============================================================
# HTTP
# ============================================================

HEADERS = {
    "User-Agent": "TEAM-B-University-Research-Project/1.0",
    "Accept": (
        "application/pdf,"
        "application/octet-stream;q=0.9,"
        "*/*;q=0.8"
    ),
}


# ============================================================
# Helpers
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def sanitize_string(
    value: Any,
) -> str:
    """
    PDF extraction에서 드물게 들어오는 lone surrogate를 제거한다.
    """

    if value is None:
        return ""

    text = str(value)

    return (
        text
        .encode(
            "utf-8",
            errors="replace",
        )
        .decode(
            "utf-8",
            errors="replace",
        )
    )


def sanitize_object(
    value: Any,
) -> Any:

    if isinstance(
        value,
        str,
    ):
        return sanitize_string(
            value
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            sanitize_string(key): sanitize_object(item)
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        list,
    ):
        return [
            sanitize_object(item)
            for item
            in value
        ]

    if isinstance(
        value,
        tuple,
    ):
        return [
            sanitize_object(item)
            for item
            in value
        ]

    return value


def normalize_text(
    value: Any,
) -> str:

    if value is None:
        return ""

    return " ".join(
        sanitize_string(
            value
        ).split()
    )


def safe_filename(
    value: str,
) -> str:

    value = normalize_text(
        value
    )

    return (
        value
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def arxiv_base_id(
    value: str,
) -> str:

    value = normalize_text(
        value
    )

    value = (
        value
        .rstrip("/")
        .split("/")[-1]
    )

    return re.sub(
        r"v\d+$",
        "",
        value,
        flags=re.IGNORECASE,
    )


def document_base_id(
    document: dict,
) -> str:

    candidate = (
        document.get(
            "source_base_id"
        )
        or document.get(
            "source_id"
        )
        or ""
    )

    return arxiv_base_id(
        str(candidate)
    )



def load_permanent_reject_ids() -> set[str]:
    if not PERMANENT_REJECT_FILE.exists():
        return set()

    rejected: set[str] = set()

    with PERMANENT_REJECT_FILE.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            base_id = arxiv_base_id(line)

            if base_id:
                rejected.add(base_id)

    return rejected


def split_targets(
    total_target: int,
) -> dict[str, int]:

    base = (
        total_target
        // len(AXES)
    )

    remainder = (
        total_target
        % len(AXES)
    )

    result: dict[str, int] = {}

    for index, axis in enumerate(
        AXES
    ):
        result[axis] = (
            base
            + (
                1
                if index < remainder
                else 0
            )
        )

    return result


def metadata_text(
    document: dict,
) -> str:

    categories = (
        document.get(
            "categories"
        )
        or []
    )

    if isinstance(
        categories,
        list,
    ):
        category_text = " ".join(
            str(item)
            for item
            in categories
        )
    else:
        category_text = str(
            categories
        )

    return " ".join(
        [
            normalize_text(
                document.get(
                    "title"
                )
            ),
            normalize_text(
                document.get(
                    "abstract"
                )
            ),
            normalize_text(
                category_text
            ),
        ]
    ).lower()


def relevance_score(
    axis: str,
    document: dict,
) -> int:

    text = metadata_text(
        document
    )

    return sum(
        1
        for keyword
        in AXIS_KEYWORDS[
            axis
        ]
        if keyword in text
    )


def passes_relevance_gate(
    axis: str,
    document: dict,
    min_score: int,
) -> tuple[bool, int]:

    text = metadata_text(
        document
    )

    score = relevance_score(
        axis,
        document,
    )

    if score < min_score:
        return False, score

    if (
        axis
        == "onboard_ai"
    ):
        has_space_anchor = any(
            term in text
            for term
            in ONBOARD_SPACE_ANCHORS
        )

        has_intelligence_anchor = any(
            term in text
            for term
            in ONBOARD_INTELLIGENCE_ANCHORS
        )

        if not (
            has_space_anchor
            and has_intelligence_anchor
        ):
            return False, score

    return True, score


def write_json(
    path: Path,
    payload: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_payload = (
        sanitize_object(
            payload
        )
    )

    path.write_text(
        json.dumps(
            clean_payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def append_jsonl(
    path: Path,
    payload: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_payload = (
        sanitize_object(
            payload
        )
    )

    line = json.dumps(
        clean_payload,
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


# ============================================================
# Resume Existing Corpus
# ============================================================

def load_existing() -> tuple[
    Counter,
    set[str],
    set[str],
]:

    counts: Counter = Counter()
    source_ids: set[str] = set()
    content_hashes: set[str] = set()

    if not OUTPUT_JSONL.exists():
        return (
            counts,
            source_ids,
            content_hashes,
        )

    with OUTPUT_JSONL.open(
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
                    f"[RESUME WARNING] "
                    f"bad JSONL line "
                    f"{line_number}"
                )
                continue

            axis = str(
                row.get(
                    "topic_axis",
                    "",
                )
            ).strip()

            if axis in AXES:
                counts[
                    axis
                ] += 1

            base_id = (
                document_base_id(
                    row
                )
            )

            if base_id:
                source_ids.add(
                    base_id
                )

            content_hash = str(
                row.get(
                    "content_hash",
                    "",
                )
                or ""
            ).strip()

            if content_hash:
                content_hashes.add(
                    content_hash
                )

    return (
        counts,
        source_ids,
        content_hashes,
    )


# ============================================================
# PDF Download
# ============================================================

def pdf_candidates(
    document: dict,
) -> list[str]:

    candidates: list[str] = []

    original_pdf_url = normalize_text(
        document.get(
            "pdf_url"
        )
    )

    source_id = normalize_text(
        document.get(
            "source_id"
        )
    )

    base_id = document_base_id(
        document
    )

    if original_pdf_url:
        candidates.append(
            original_pdf_url
        )

    # arXiv의 특정 version URL이 404를 반환하는 경우가 있어
    # base ID PDF URL도 반드시 fallback으로 사용한다.
    if base_id:
        candidates.append(
            f"https://arxiv.org/pdf/{base_id}"
        )

    if source_id:
        candidates.append(
            f"https://arxiv.org/pdf/{source_id}"
        )

    result: list[str] = []

    seen: set[str] = set()

    for url in candidates:
        url = url.strip()

        if (
            not url
            or url in seen
        ):
            continue

        seen.add(
            url
        )

        result.append(
            url
        )

    return result


def download_pdf(
    document: dict,
    destination: Path,
) -> str:

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_path = (
        destination.with_suffix(
            ".pdf.part"
        )
    )

    urls = pdf_candidates(
        document
    )

    if not urls:
        raise RuntimeError(
            "PDF URL unavailable"
        )

    last_error: Exception | None = None

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

        for url in urls:

            for attempt in range(
                1,
                4,
            ):
                try:
                    with client.stream(
                        "GET",
                        url,
                    ) as response:

                        # versioned URL 404:
                        # retry하지 말고 다음 fallback URL로 이동.
                        if (
                            response.status_code
                            == 404
                        ):
                            raise FileNotFoundError(
                                f"404 Not Found: {url}"
                            )

                        response.raise_for_status()

                        with tmp_path.open(
                            "wb"
                        ) as handle:

                            first_bytes = b""

                            for chunk in response.iter_bytes(
                                chunk_size=1024 * 256
                            ):
                                if not chunk:
                                    continue

                                if not first_bytes:
                                    first_bytes = (
                                        chunk[:16]
                                    )

                                handle.write(
                                    chunk
                                )

                    if not tmp_path.exists():
                        raise RuntimeError(
                            "PDF temporary file missing"
                        )

                    if (
                        tmp_path.stat().st_size
                        < 1000
                    ):
                        raise RuntimeError(
                            "Downloaded PDF too small"
                        )

                    with tmp_path.open(
                        "rb"
                    ) as handle:
                        magic = handle.read(
                            5
                        )

                    if magic != b"%PDF-":
                        raise RuntimeError(
                            "Downloaded content is not PDF"
                        )

                    tmp_path.replace(
                        destination
                    )

                    return url

                except FileNotFoundError as exc:
                    last_error = exc

                    print(
                        f"[PDF FALLBACK] "
                        f"{url} -> 404"
                    )

                    break

                except Exception as exc:
                    last_error = exc

                    if attempt < 3:
                        wait_seconds = (
                            2 * attempt
                        )

                        print(
                            f"[DOWNLOAD RETRY] "
                            f"{attempt}/3 "
                            f"wait={wait_seconds}s "
                            f"reason={exc}"
                        )

                        time.sleep(
                            wait_seconds
                        )

                    else:
                        print(
                            f"[DOWNLOAD FAILED URL] "
                            f"{url} "
                            f"reason={exc}"
                        )

    try:
        tmp_path.unlink(
            missing_ok=True
        )
    except OSError:
        pass

    raise RuntimeError(
        "PDF download failed: "
        f"{last_error}"
    )


# ============================================================
# Process One Document
# ============================================================

def process_document(
    *,
    axis: str,
    document: dict,
    relevance: int,
    seen_hashes: set[str],
    keep_work: bool,
) -> dict:

    base_id = (
        document_base_id(
            document
        )
    )

    if not base_id:
        raise RuntimeError(
            "source_base_id unavailable"
        )

    paper_dir = (
        WORK_ROOT
        / axis
        / safe_filename(
            base_id
        )
    )

    paper_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = (
        paper_dir
        / "metadata.json"
    )

    pdf_path = (
        paper_dir
        / "paper.pdf"
    )

    parsed_path = (
        paper_dir
        / "parsed_document.json"
    )

    metadata = {
        **sanitize_object(
            document
        ),
        "topic_axis": axis,
    }

    write_json(
        metadata_path,
        metadata,
    )

    downloaded_url = download_pdf(
        document,
        pdf_path,
    )

    parsed = parse_pdf_file(
        pdf_path
    )

    parsed_payload = {
        "title": sanitize_string(
            parsed.get(
                "title",
                "",
            )
        ),
        "abstract": sanitize_string(
            parsed.get(
                "abstract",
                "",
            )
        ),
        "sections": sanitize_object(
            parsed.get(
                "sections",
                [],
            )
        ),
        "stats": sanitize_object(
            parsed.get(
                "stats",
                {},
            )
        ),
    }

    write_json(
        parsed_path,
        parsed_payload,
    )

    cleaned = clean_document(
        parsed_path
    )

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
        reasons = (
            quality.get(
                "reasons"
            )
            or ["unknown"]
        )

        raise RuntimeError(
            "QUALITY_REJECT: "
            + ", ".join(
                str(item)
                for item
                in reasons
            )
        )

    content_hash = str(
        cleaned.get(
            "content_hash",
            "",
        )
        or ""
    ).strip()

    if not content_hash:
        raise RuntimeError(
            "content_hash missing"
        )

    if content_hash in seen_hashes:
        raise RuntimeError(
            "CONTENT_DUPLICATE"
        )

    clean_content = sanitize_string(
        cleaned.get(
            "clean_content",
            "",
        )
    )

    if not clean_content:
        raise RuntimeError(
            "clean_content empty"
        )

    row = {
        "corpus_version": VERSION,
        "collected_at": utc_now(),

        "source": (
            document.get(
                "source",
                "arxiv",
            )
        ),
        "source_id": (
            document.get(
                "source_id"
            )
        ),
        "source_base_id": base_id,

        "topic_axis": axis,

        "title": (
            cleaned.get(
                "title"
            )
            or document.get(
                "title"
            )
            or ""
        ),

        "abstract": (
            cleaned.get(
                "abstract"
            )
            or document.get(
                "abstract"
            )
            or ""
        ),

        "authors": (
            document.get(
                "authors"
            )
            or []
        ),

        "categories": (
            document.get(
                "categories"
            )
            or []
        ),

        "published_at": (
            document.get(
                "published_at"
            )
        ),

        "document_type": (
            document.get(
                "document_type",
                "arxiv_preprint",
            )
        ),

        "doi": (
            document.get(
                "doi"
            )
        ),

        "url": (
            document.get(
                "url"
            )
        ),

        "pdf_url": (
            downloaded_url
        ),

        "matched_queries": (
            document.get(
                "matched_queries"
            )
            or []
        ),

        "relevance_score": (
            relevance
        ),

        "clean_content": (
            clean_content
        ),

        "content_hash": (
            content_hash
        ),

        "quality": sanitize_object(
            quality
        ),

        "cleaning_stats": sanitize_object(
            cleaned.get(
                "cleaning_stats",
                {},
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

    return sanitize_object(
        row
    )


# ============================================================
# Report
# ============================================================

def build_report(
    *,
    target: int,
    targets: dict[str, int],
    counts: Counter,
    start_time: float,
    run_stats: Counter,
) -> dict:

    total = sum(
        counts.get(
            axis,
            0,
        )
        for axis
        in AXES
    )

    status = (
        "PASS"
        if (
            total >= target
            and all(
                counts.get(
                    axis,
                    0,
                )
                >= targets[
                    axis
                ]
                for axis
                in AXES
            )
        )
        else "INCOMPLETE"
    )

    return {
        "version": VERSION,
        "generated_at": utc_now(),
        "status": status,
        "target": target,
        "documents": total,
        "axis_targets": {
            axis: targets[
                axis
            ]
            for axis
            in AXES
        },
        "axis_counts": {
            axis: counts.get(
                axis,
                0,
            )
            for axis
            in AXES
        },
        "elapsed_minutes": round(
            (
                time.time()
                - start_time
            )
            / 60.0,
            2,
        ),
        "run_stats": dict(
            run_stats
        ),
        "corpus": str(
            OUTPUT_JSONL
        ),
    }


def save_report(
    *,
    target: int,
    targets: dict[str, int],
    counts: Counter,
    start_time: float,
    run_stats: Counter,
) -> dict:

    report = build_report(
        target=target,
        targets=targets,
        counts=counts,
        start_time=start_time,
        run_stats=run_stats,
    )

    write_json(
        REPORT_FILE,
        report,
    )

    return report


def print_progress(
    *,
    axis: str,
    target: int,
    targets: dict[str, int],
    counts: Counter,
    start_total: int,
    start_time: float,
) -> None:

    total = sum(
        counts.get(
            item,
            0,
        )
        for item
        in AXES
    )

    new_documents = max(
        0,
        total
        - start_total,
    )

    elapsed = max(
        0.001,
        time.time()
        - start_time,
    )

    rate = (
        new_documents
        / elapsed
    )

    remaining = max(
        0,
        target
        - total,
    )

    if rate > 0:
        eta_minutes = (
            remaining
            / rate
            / 60.0
        )

        eta_text = (
            f"{eta_minutes:.1f} min"
        )

    else:
        eta_text = "?"

    print(
        f"[SAVED] "
        f"{axis}: "
        f"{counts[axis]:,}/"
        f"{targets[axis]:,} | "
        f"total={total:,}/"
        f"{target:,} | "
        f"{rate:.3f} docs/s | "
        f"ETA={eta_text}",
        flush=True,
    )



# ============================================================
# Resilient arXiv Search
# ============================================================

def resilient_search_query_page(
    *,
    topic_axis: str,
    query_name: str,
    query: str,
    start: int,
    page_size: int,
):

    # arXiv/export.arxiv.org가 일시적으로
    # 429 / 5xx를 반환할 때 같은 query/page를
    # 버리지 않고 기다렸다가 다시 시도한다.
    delays = (
        30,
        60,
        120,
        240,
        300,
    )

    transient_markers = (
        "429",
        "500",
        "502",
        "503",
        "504",
        "service unavailable",
        "rate limit",
        "timed out",
        "timeout",
        "connection reset",
        "connection refused",
        "temporary failure",
    )

    for attempt in range(
        len(delays) + 1
    ):

        try:
            return search_query_page(
                topic_axis=topic_axis,
                query_name=query_name,
                query=query,
                start=start,
                page_size=page_size,
            )

        except Exception as exc:

            message = str(
                exc
            ).lower()

            transient = any(
                marker in message
                for marker
                in transient_markers
            )

            if (
                not transient
                or attempt
                >= len(delays)
            ):
                raise

            wait_seconds = (
                delays[
                    attempt
                ]
            )

            print(
                f"[ARXIV BACKOFF] "
                f"{topic_axis}/"
                f"{query_name}/"
                f"start={start} | "
                f"attempt="
                f"{attempt + 1}/"
                f"{len(delays)} | "
                f"wait="
                f"{wait_seconds}s | "
                f"reason={exc}",
                flush=True,
            )

            time.sleep(
                wait_seconds
            )


# ============================================================
# Collection
# ============================================================

def collect(
    args: argparse.Namespace,
) -> dict:

    ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    WORK_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    targets = split_targets(
        args.target
    )

    (
        counts,
        seen_source_ids,
        seen_hashes,
    ) = load_existing()

    permanent_reject_ids = load_permanent_reject_ids()

    print(
        f"Permanent rejects : "
        f"{len(permanent_reject_ids):,}",
        flush=True,
    )

    start_total = sum(
        counts.get(
            axis,
            0,
        )
        for axis
        in AXES
    )

    start_time = (
        time.time()
    )

    run_stats: Counter = Counter()

    print(
        "=" * 80,
        flush=True,
    )
    print(
        "TEAM B - STAGE1 SCALE COLLECTION",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )

    print(
        f"Resume documents : "
        f"{start_total:,}",
        flush=True,
    )

    for axis in AXES:
        print(
            f"{axis:22s}: "
            f"{counts.get(axis, 0):,}/"
            f"{targets[axis]:,}",
            flush=True,
        )

    print(
        "=" * 80,
        flush=True,
    )

    # ========================================================
    # Axis
    # ========================================================

    for axis in AXES:

        if (
            counts.get(
                axis,
                0,
            )
            >= targets[
                axis
            ]
        ):
            print(
                f"[AXIS COMPLETE] "
                f"{axis}: "
                f"{counts.get(axis, 0):,}/"
                f"{targets[axis]:,}",
                flush=True,
            )
            continue

        print()
        print(
            "#" * 80,
            flush=True,
        )
        print(
            f"[AXIS START] "
            f"{axis}: "
            f"{counts.get(axis, 0):,}/"
            f"{targets[axis]:,}",
            flush=True,
        )
        print(
            "#" * 80,
            flush=True,
        )

        axis_queries = (
            SCALE_QUERIES[
                axis
            ]
        )

        # round-robin:
        # query A page0 -> query B page0 -> ...
        # query A page1 -> query B page1 -> ...
        #
        # 특정 query 하나에 corpus가 편향되는 것을 줄인다.

        exhausted_queries: set[str] = set()

        for page_index in range(
            args.max_pages_per_query
        ):

            if (
                counts.get(
                    axis,
                    0,
                )
                >= targets[
                    axis
                ]
            ):
                break

            active_query_count = 0

            for query_spec in axis_queries:

                if (
                    counts.get(
                        axis,
                        0,
                    )
                    >= targets[
                        axis
                    ]
                ):
                    break

                query_name = str(
                    query_spec[
                        "name"
                    ]
                )

                query = str(
                    query_spec[
                        "query"
                    ]
                )

                if query_name in exhausted_queries:
                    continue

                active_query_count += 1

                start = (
                    page_index
                    * args.page_size
                )

                try:
                    (
                        documents,
                        total_results,
                    ) = resilient_search_query_page(
                        topic_axis=axis,
                        query_name=(
                            f"stage1_{query_name}"
                        ),
                        query=query,
                        start=start,
                        page_size=args.page_size,
                    )

                except Exception as exc:
                    run_stats[
                        "search_errors"
                    ] += 1

                    print(
                        f"[SEARCH ERROR] "
                        f"{axis}/"
                        f"{query_name}/"
                        f"start={start}: "
                        f"{exc}",
                        flush=True,
                    )

                    continue

                if not documents:
                    exhausted_queries.add(
                        query_name
                    )
                    continue

                if (
                    total_results is not None
                    and (
                        start
                        + len(
                            documents
                        )
                    )
                    >= total_results
                ):
                    exhausted_queries.add(
                        query_name
                    )

                elif (
                    len(
                        documents
                    )
                    < args.page_size
                ):
                    exhausted_queries.add(
                        query_name
                    )

                # ============================================
                # Document
                # ============================================

                for document in documents:

                    if (
                        counts.get(
                            axis,
                            0,
                        )
                        >= targets[
                            axis
                        ]
                    ):
                        break

                    run_stats[
                        "metadata_seen"
                    ] += 1

                    base_id = (
                        document_base_id(
                            document
                        )
                    )

                    if not base_id:
                        run_stats[
                            "missing_source_id"
                        ] += 1
                        continue

                    if (
                        base_id
                        in permanent_reject_ids
                    ):
                        run_stats[
                            "permanent_rejects"
                        ] += 1

                        seen_source_ids.add(
                            base_id
                        )

                        continue

                    if (
                        base_id
                        in seen_source_ids
                    ):
                        run_stats[
                            "source_duplicates"
                        ] += 1
                        continue

                    (
                        passes,
                        score,
                    ) = passes_relevance_gate(
                        axis,
                        document,
                        args.min_score,
                    )

                    if not passes:
                        run_stats[
                            "relevance_rejects"
                        ] += 1
                        continue

                    try:
                        row = process_document(
                            axis=axis,
                            document=document,
                            relevance=score,
                            seen_hashes=seen_hashes,
                            keep_work=args.keep_work,
                        )

                    except Exception as exc:
                        run_stats[
                            "document_rejects"
                        ] += 1

                        print(
                            f"[REJECT] "
                            f"{base_id} "
                            f"{exc}",
                            flush=True,
                        )

                        # rejected paper는 다음 query에서
                        # 계속 다시 시도하지 않도록 source seen 처리.
                        seen_source_ids.add(
                            base_id
                        )

                        continue

                    append_jsonl(
                        OUTPUT_JSONL,
                        row,
                    )

                    seen_source_ids.add(
                        base_id
                    )

                    seen_hashes.add(
                        str(
                            row[
                                "content_hash"
                            ]
                        )
                    )

                    counts[
                        axis
                    ] += 1

                    run_stats[
                        "saved"
                    ] += 1

                    print_progress(
                        axis=axis,
                        target=args.target,
                        targets=targets,
                        counts=counts,
                        start_total=start_total,
                        start_time=start_time,
                    )

                    if (
                        run_stats[
                            "saved"
                        ]
                        % 25
                        == 0
                    ):
                        save_report(
                            target=args.target,
                            targets=targets,
                            counts=counts,
                            start_time=start_time,
                            run_stats=run_stats,
                        )

                    if (
                        args.document_delay
                        > 0
                    ):
                        time.sleep(
                            args.document_delay
                        )

            if (
                active_query_count
                == 0
            ):
                break

            if (
                len(
                    exhausted_queries
                )
                == len(
                    axis_queries
                )
            ):
                break

    report = save_report(
        target=args.target,
        targets=targets,
        counts=counts,
        start_time=start_time,
        run_stats=run_stats,
    )

    print()
    print(
        "=" * 80,
        flush=True,
    )
    print(
        "STAGE1 SCALE COLLECTION COMPLETE",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )

    print(
        f"Status       : "
        f"{report['status']}",
        flush=True,
    )

    print(
        f"Documents    : "
        f"{report['documents']:,} / "
        f"{args.target:,}",
        flush=True,
    )

    for axis in AXES:
        print(
            f"{axis:22s} "
            f"{counts.get(axis, 0):,} / "
            f"{targets[axis]:,}",
            flush=True,
        )

    print(
        f"Elapsed      : "
        f"{report['elapsed_minutes']:.2f} min",
        flush=True,
    )

    print(
        f"Corpus       : "
        f"{OUTPUT_JSONL}",
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

    return report


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "TEAM B Stage1 scalable "
            "full-text corpus collector"
        )
    )

    parser.add_argument(
        "--target",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--page-size",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--max-pages-per-query",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--min-score",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--document-delay",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--keep-work",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    if args.target <= 0:
        raise ValueError(
            "--target must be > 0"
        )

    if args.page_size <= 0:
        raise ValueError(
            "--page-size must be > 0"
        )

    if (
        args.max_pages_per_query
        <= 0
    ):
        raise ValueError(
            "--max-pages-per-query "
            "must be > 0"
        )

    report = collect(
        args
    )

    if (
        report[
            "status"
        ]
        != "PASS"
    ):
        raise SystemExit(
            2
        )


if __name__ == "__main__":
    main()
