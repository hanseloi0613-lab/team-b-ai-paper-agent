import json
import re
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT


# ============================================================
# TEAM B - NASA NTRS Core-100 Human QA Selector
# ============================================================
#
# 목적:
#
# ntrs_collector.py가 만든 Human Review Pool에서
#
#   rover_autonomy       12
#   onboard_ai           12
#   satellite_autonomy   11
#
# 총 35편을 Human QA 기준으로 확정한다.
#
#
# 현재 DB:
#
#   arXiv : 50
#   NTRS  : 15
#   TOTAL : 65
#
#
# 이번 선정 완료 후:
#
#   NTRS 신규 selected = 35
#
# 다음 단계:
#
#   selected 35
#       ↓
#   Resolver
#       ↓
#   Parser
#       ↓
#   Cleaner
#       ↓
#   Loader
#       ↓
#   AWS RDS 65 → 100
#
#
# 중요:
#
# 이 스크립트는:
#
#   - 다운로드하지 않는다.
#   - DB에 insert하지 않는다.
#   - Resolver를 실행하지 않는다.
#
# 오직 Human QA selection manifest를 만든다.
# ============================================================


SELECTION_VERSION = "core100_v1"


# ============================================================
# Expected Counts
# ============================================================

EXPECTED_COUNTS = {
    "rover_autonomy": 12,
    "onboard_ai": 12,
    "satellite_autonomy": 11,
}


EXPECTED_TOTAL = sum(
    EXPECTED_COUNTS.values()
)


# ============================================================
# Paths
# ============================================================

CACHE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "ntrs"
    / "core100_v1"
)


COLLECTION_REPORT_FILE = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "ntrs_core100_collection_report.json"
)


SELECTION_DIR = (
    PROJECT_ROOT
    / "data"
    / "selections"
)


OUTPUT_FILE = (
    SELECTION_DIR
    / "ntrs_core100_selected.json"
)


# ============================================================
# Human QA Selection
# ============================================================
#
# 선정 기준:
#
# 1. 각 axis와 직접 연결될 것
# 2. full-text candidate가 있을 것
# 3. 단순히 키워드만 포함한 주변 논문은 피할 것
# 4. 같은 내용/같은 제목의 중복 문서는 피할 것
# 5. 역사적 foundational paper + 최근 연구를 적절히 섞을 것
# 6. 세 axis 사이 동일 source_id 중복 금지
#
# ============================================================

SELECTIONS = {

    # ========================================================
    # 1. Rover Autonomy
    # ========================================================
    #
    # 중심:
    #
    # - autonomous navigation
    # - obstacle / hazard avoidance
    # - localization / visual odometry
    # - path planning
    # - communication delay
    # - autonomous science / decision making
    #
    # ========================================================

    "rover_autonomy": [

        {
            "source_id": "20250009477",
            "reason": (
                "JPL rover에 autonomous navigation, "
                "real-time trajectory generation, "
                "obstacle avoidance, SLAM을 직접 구현한 "
                "현대적인 rover autonomy 자료."
            ),
        },

        {
            "source_id": "20040087098",
            "reason": (
                "통신 제약 하에서 planetary rover가 "
                "자율적으로 운용되기 위한 "
                "planning under uncertainty를 직접 다룸."
            ),
        },

        {
            "source_id": "20160011500",
            "reason": (
                "GPS가 없는 planetary surface에서 "
                "rover localization과 orientation estimation을 "
                "다루는 navigation 핵심 자료."
            ),
        },

        {
            "source_id": "20120016473",
            "reason": (
                "visual odometry와 localization, "
                "obstacle avoidance/SLAM과 연결되는 "
                "rover navigation 핵심 기술."
            ),
        },

        {
            "source_id": "20120016849",
            "reason": (
                "LIDAR 기반 safeguarded lunar rover "
                "tele-operation/navigation 자료로 "
                "통신 지연과 autonomous navigation의 "
                "경계 문제를 다루기에 적합."
            ),
        },

        {
            "source_id": "19950005122",
            "reason": (
                "Mars micro-rover의 autonomous dead reckoning, "
                "hazard detection, sensing/control을 다루는 "
                "직접적인 rover autonomy 자료."
            ),
        },

        {
            "source_id": "19890010498",
            "reason": (
                "Mars rover를 포함한 mobile robot의 "
                "path planning을 genetic algorithm으로 다루는 "
                "경로계획 foundational 자료."
            ),
        },

        {
            "source_id": "20040010821",
            "reason": (
                "Mars rover의 onboard science evaluation과 "
                "human-directed planning 감소를 다루는 "
                "autonomous science 핵심 자료."
            ),
        },

        {
            "source_id": "20000097574",
            "reason": (
                "장거리 Mars rover 운용에서 "
                "scientist interaction의 한계를 줄이기 위한 "
                "autonomous science 분석을 직접 다룸."
            ),
        },

        {
            "source_id": "20000085880",
            "reason": (
                "Mars Sample Return 맥락에서 "
                "autonomous science decision making을 "
                "직접 다루는 자료."
            ),
        },

        {
            "source_id": "20240011392",
            "reason": (
                "Earth-Moon communication latency가 "
                "lunar rover teleoperation 안전성에 미치는 영향을 "
                "다뤄 autonomy 필요성을 설명하기 적합."
            ),
        },

        {
            "source_id": "20260004612",
            "reason": (
                "LiDAR를 사용하는 autonomous surface rover와 "
                "charging service capability를 다루는 "
                "최신 surface autonomy 사례."
            ),
        },
    ],


    # ========================================================
    # 2. Onboard AI / Deep-Space Autonomy
    # ========================================================
    #
    # 중심:
    #
    # - communication delay
    # - onboard decision making
    # - planning / scheduling
    # - onboard executive
    # - autonomous navigation
    # - autonomous control
    # - AI / RL
    #
    # ========================================================

    "onboard_ai": [

        {
            "source_id": "20240006994",
            "reason": (
                "Starling CubeSat swarm에서 "
                "autonomous onboard decision-making, "
                "relative navigation, maneuver planning을 "
                "실제 flight result로 다룸."
            ),
        },

        {
            "source_id": "20260005696",
            "reason": (
                "NASA Distributed Spacecraft Autonomy가 "
                "다중 spacecraft에서 완전 자율 분산 운용을 "
                "flight experiment로 검증한 최신 사례."
            ),
        },

        {
            "source_id": "20260004947",
            "reason": (
                "real-time autonomous guidance, navigation, "
                "control을 spacecraft onboard에서 수행하는 "
                "autoNGC flight-test 자료."
            ),
        },

        {
            "source_id": "20260001759",
            "reason": (
                "CAPSTONE을 autonomous SmallSat technology "
                "testbed로 다루며 cislunar/deep-space "
                "autonomy와 직접 연결됨."
            ),
        },

        {
            "source_id": "20190029005",
            "reason": (
                "automated planning/scheduling, "
                "fault diagnostics 등 autonomous technology를 "
                "실제 flight software 환경으로 전환하는 문제를 다룸."
            ),
        },

        {
            "source_id": "20250002440",
            "reason": (
                "OSIRIS-APEX의 Apophis operations에서 "
                "greater spacecraft autonomy를 검토하는 "
                "실제 deep-space mission 사례."
            ),
        },

        {
            "source_id": "20250000634",
            "reason": (
                "reinforcement learning을 "
                "spacecraft autonomous navigation과 "
                "environment characterization에 적용."
            ),
        },

        {
            "source_id": "19950017299",
            "reason": (
                "transmission delay 문제를 배경으로 "
                "onboard expert system, decision making, "
                "mission planning executive를 직접 다룸."
            ),
        },

        {
            "source_id": "19990052840",
            "reason": (
                "Deep Space One Remote Agent 계열의 "
                "AI system이 human mission operator 업무를 "
                "spacecraft onboard에서 자율 수행하는 핵심 사례."
            ),
        },

        {
            "source_id": "20020060783",
            "reason": (
                "Remote Agent 계열 autonomous spacecraft "
                "executive의 real-time execution, dynamic recovery, "
                "resource constraint 대응 구조를 다룸."
            ),
        },

        {
            "source_id": "20180005241",
            "reason": (
                "deep-space human exploration에서 "
                "ground communication 의존을 줄이기 위한 "
                "autonomous spacecraft power control 사례."
            ),
        },

        {
            "source_id": "20160012788",
            "reason": (
                "communication delay가 큰 deep-space 환경에서 "
                "autonomous rendezvous/docking navigation과 "
                "visual odometry를 다룸."
            ),
        },
    ],


    # ========================================================
    # 3. Satellite / Space-System Autonomous Operations
    # ========================================================
    #
    # 중심:
    #
    # - monitoring
    # - fault detection
    # - fault isolation
    # - diagnosis
    # - recovery
    # - FDIR
    # - health management
    # - anomaly detection
    #
    # ========================================================

    "satellite_autonomy": [

        {
            "source_id": "20140010520",
            "reason": (
                "deep-space habitat의 communication latency 하에서 "
                "autonomous fault management와 "
                "fault consequence analysis를 직접 다룸."
            ),
        },

        {
            "source_id": "20160001200",
            "reason": (
                "Orion GN&C의 fault management system "
                "verification을 다뤄 실제 spacecraft FDIR "
                "engineering 사례로 적합."
            ),
        },

        {
            "source_id": "20190003966",
            "reason": (
                "complex space system의 health state를 위한 "
                "data-driven monitoring을 다루며 "
                "modern anomaly/health monitoring과 연결됨."
            ),
        },

        {
            "source_id": "20080033120",
            "reason": (
                "reliable integrated system health management에서 "
                "sensor selection과 data validation을 다뤄 "
                "autonomous health management에 직접적."
            ),
        },

        {
            "source_id": "20060006365",
            "reason": (
                "Livingstone 2 model-based diagnosis를 "
                "EO-1 satellite에 onboard 배치한 "
                "실제 autonomous diagnosis 사례."
            ),
        },

        {
            "source_id": "20050010075",
            "reason": (
                "EO-1 satellite의 Livingstone 2 "
                "model-based diagnostic engine과 "
                "onboard planning integration을 다룸."
            ),
        },

        {
            "source_id": "19950011120",
            "reason": (
                "NASA Goddard의 satellite monitoring, "
                "fault detection, fault isolation expert system을 "
                "직접 다룸."
            ),
        },

        {
            "source_id": "19940019472",
            "reason": (
                "satellite telemetry monitoring과 "
                "problem/failure detection을 graphical expert system으로 "
                "지원하는 operational monitoring 자료."
            ),
        },

        {
            "source_id": "19950020971",
            "reason": (
                "spacecraft integrated vehicle health management에서 "
                "fuzzy-logic 기반 autonomous fault diagnosis를 다룸."
            ),
        },

        {
            "source_id": "19930013066",
            "reason": (
                "autonomous spacecraft의 onboard fault management를 "
                "mission phase와 environment까지 고려해 다루는 "
                "직접적인 fault-management 자료."
            ),
        },

        {
            "source_id": "20250005402",
            "reason": (
                "space power system에서 model-driven + data-driven "
                "anomaly detection을 결합하는 최신 monitoring 사례."
            ),
        },
    ],
}


# ============================================================
# JSON Helpers
# ============================================================

def _load_json(
    path: Path,
) -> dict:

    if not path.exists():

        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def _save_json(
    path: Path,
    payload: dict,
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
    )


# ============================================================
# Normalize Title
# ============================================================

def _normalize_title(
    title: str,
) -> str:

    value = str(
        title
        or ""
    ).lower()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


# ============================================================
# Review Pool
# ============================================================

def _review_pool_path(
    topic_axis: str,
) -> Path:

    return (
        CACHE_ROOT
        / topic_axis
        / "_review_pool.json"
    )


def _load_review_pool(
    topic_axis: str,
) -> list[dict]:

    path = (
        _review_pool_path(
            topic_axis
        )
    )

    payload = (
        _load_json(
            path
        )
    )

    documents = (
        payload.get(
            "documents"
        )
    )

    if not isinstance(
        documents,
        list,
    ):

        raise RuntimeError(
            f"{topic_axis}: "
            "_review_pool.json has no "
            "valid documents list."
        )

    return documents


# ============================================================
# Collection Report
# ============================================================

def _load_existing_ntrs_ids() -> set[str]:

    payload = (
        _load_json(
            COLLECTION_REPORT_FILE
        )
    )

    current_db = (
        payload.get(
            "current_db",
            {},
        )
    )

    if not isinstance(
        current_db,
        dict,
    ):

        raise RuntimeError(
            "Invalid collection report: "
            "current_db missing."
        )

    source_ids = (
        current_db.get(
            "existing_ntrs_ids",
            [],
        )
    )

    if not isinstance(
        source_ids,
        list,
    ):

        raise RuntimeError(
            "Invalid existing_ntrs_ids "
            "in collection report."
        )

    return {
        str(
            source_id
        ).strip()

        for source_id
        in source_ids
    }


# ============================================================
# Build Review Pool Index
# ============================================================

def _build_pool_index(
    topic_axis: str,
    documents: list[dict],
) -> dict[
    str,
    dict,
]:

    result = {}

    for (
        rank,
        document,
    ) in enumerate(
        documents,
        start=1,
    ):

        source_id = str(
            document.get(
                "source_id",
                "",
            )
        ).strip()

        if not source_id:

            raise RuntimeError(
                f"{topic_axis}: "
                "review candidate with no source_id."
            )

        if source_id in result:

            raise RuntimeError(
                f"{topic_axis}: "
                f"duplicate source_id in review pool: "
                f"{source_id}"
            )

        copied = dict(
            document
        )

        copied[
            "_review_rank"
        ] = rank

        result[
            source_id
        ] = copied

    return result


# ============================================================
# Validate Static Selection Counts
# ============================================================

def _validate_selection_counts() -> None:

    total = 0

    for (
        topic_axis,
        expected_count,
    ) in EXPECTED_COUNTS.items():

        selections = (
            SELECTIONS.get(
                topic_axis
            )
        )

        if not isinstance(
            selections,
            list,
        ):

            raise RuntimeError(
                f"{topic_axis}: "
                "selection list missing."
            )

        actual_count = len(
            selections
        )

        if (
            actual_count
            != expected_count
        ):

            raise RuntimeError(
                f"{topic_axis}: "
                f"expected {expected_count}, "
                f"found {actual_count}."
            )

        total += actual_count

    if total != EXPECTED_TOTAL:

        raise RuntimeError(
            f"Expected total "
            f"{EXPECTED_TOTAL}, "
            f"found {total}."
        )


# ============================================================
# Cross-Axis Source ID Validation
# ============================================================

def _validate_no_cross_axis_id_duplicates() -> None:

    owners = {}

    for (
        topic_axis,
        selections,
    ) in SELECTIONS.items():

        for selection in selections:

            source_id = str(
                selection[
                    "source_id"
                ]
            ).strip()

            if source_id in owners:

                raise RuntimeError(
                    "Selected source_id appears "
                    "in multiple axes:\n"
                    f"{source_id}\n"
                    f"{owners[source_id]} "
                    f"<-> {topic_axis}"
                )

            owners[
                source_id
            ] = topic_axis


# ============================================================
# One Candidate Validation
# ============================================================

def _validate_candidate(
    *,
    topic_axis: str,
    candidate: dict,
    existing_ntrs_ids: set[str],
) -> None:

    source_id = str(
        candidate.get(
            "source_id",
            "",
        )
    ).strip()


    if not source_id:

        raise ValueError(
            f"{topic_axis}: "
            "source_id missing."
        )


    # --------------------------------------------------------
    # Existing DB conflict
    # --------------------------------------------------------

    if source_id in existing_ntrs_ids:

        raise RuntimeError(
            f"{topic_axis} / {source_id}: "
            "already exists in AWS RDS."
        )


    # --------------------------------------------------------
    # Axis
    # --------------------------------------------------------

    candidate_axis = str(
        candidate.get(
            "topic_axis",
            "",
        )
    ).strip()

    if (
        candidate_axis
        != topic_axis
    ):

        raise RuntimeError(
            f"{source_id}: "
            f"candidate axis mismatch: "
            f"{candidate_axis} != "
            f"{topic_axis}"
        )


    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    source = str(
        candidate.get(
            "source",
            "",
        )
    ).lower()

    if source != "ntrs":

        raise RuntimeError(
            f"{source_id}: "
            f"unexpected source: "
            f"{source}"
        )


    # --------------------------------------------------------
    # Title / Abstract
    # --------------------------------------------------------

    title = str(
        candidate.get(
            "title",
            "",
        )
    ).strip()

    abstract = str(
        candidate.get(
            "abstract",
            "",
        )
    ).strip()

    if not title:

        raise RuntimeError(
            f"{source_id}: "
            "title missing."
        )

    if not abstract:

        raise RuntimeError(
            f"{source_id}: "
            "abstract missing."
        )


    # --------------------------------------------------------
    # Full Document
    # --------------------------------------------------------

    if candidate.get(
        "only_abstract",
        False,
    ):

        raise RuntimeError(
            f"{source_id}: "
            "abstract-only candidate."
        )


    if not candidate.get(
        "downloads_available",
        False,
    ):

        raise RuntimeError(
            f"{source_id}: "
            "downloadsAvailable=False."
        )


    usable_urls = [

        candidate.get(
            "fulltext_url"
        ),

        candidate.get(
            "original_url"
        ),

        candidate.get(
            "pdf_url"
        ),
    ]


    if not any(
        usable_urls
    ):

        raise RuntimeError(
            f"{source_id}: "
            "no usable full-text URL."
        )


# ============================================================
# Duplicate Title / DOI Validation
# ============================================================

def _validate_selected_content_duplicates(
    selected_records: list[dict],
) -> None:

    seen_titles = {}

    seen_dois = {}


    for record in selected_records:

        source_id = (
            record[
                "source_id"
            ]
        )


        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        normalized_title = (
            _normalize_title(
                record.get(
                    "title",
                    "",
                )
            )
        )


        if normalized_title:

            if normalized_title in seen_titles:

                raise RuntimeError(
                    "Selected papers have "
                    "duplicate normalized titles:\n"
                    f"{seen_titles[normalized_title]} "
                    f"<-> {source_id}\n"
                    f"Title: "
                    f"{record.get('title')}"
                )

            seen_titles[
                normalized_title
            ] = source_id


        # ----------------------------------------------------
        # DOI
        # ----------------------------------------------------

        doi = str(
            record.get(
                "doi"
            )
            or ""
        ).strip().lower()


        if doi:

            if doi in seen_dois:

                raise RuntimeError(
                    "Selected papers have "
                    "duplicate DOI:\n"
                    f"{seen_dois[doi]} "
                    f"<-> {source_id}\n"
                    f"DOI: {doi}"
                )

            seen_dois[
                doi
            ] = source_id


# ============================================================
# Build Selected Records
# ============================================================

def _build_selected_records(
    *,
    existing_ntrs_ids: set[str],
) -> dict[
    str,
    list[dict],
]:

    selected_by_axis = {}


    for topic_axis in EXPECTED_COUNTS:

        review_documents = (
            _load_review_pool(
                topic_axis
            )
        )


        pool_index = (
            _build_pool_index(
                topic_axis,
                review_documents,
            )
        )


        axis_selected = []


        for selection in SELECTIONS[
            topic_axis
        ]:

            source_id = str(
                selection[
                    "source_id"
                ]
            ).strip()


            if source_id not in pool_index:

                raise RuntimeError(
                    f"{topic_axis}: "
                    f"selected NTRS ID "
                    f"{source_id} is not in "
                    "_review_pool.json."
                )


            candidate = dict(
                pool_index[
                    source_id
                ]
            )


            _validate_candidate(

                topic_axis=(
                    topic_axis
                ),

                candidate=(
                    candidate
                ),

                existing_ntrs_ids=(
                    existing_ntrs_ids
                ),
            )


            review_rank = (
                candidate.pop(
                    "_review_rank"
                )
            )


            candidate[
                "selection"
            ] = {

                "selection_version": (
                    SELECTION_VERSION
                ),

                "method": (
                    "human_qa"
                ),

                "review_rank": (
                    review_rank
                ),

                "reason": (
                    selection[
                        "reason"
                    ]
                ),
            }


            axis_selected.append(
                candidate
            )


        selected_by_axis[
            topic_axis
        ] = axis_selected


    return selected_by_axis


# ============================================================
# Flatten
# ============================================================

def _flatten_selected_records(
    selected_by_axis: dict[
        str,
        list[dict],
    ],
) -> list[dict]:

    result = []

    for topic_axis in EXPECTED_COUNTS:

        result.extend(
            selected_by_axis[
                topic_axis
            ]
        )

    return result


# ============================================================
# Build Manifest
# ============================================================

def _build_manifest(
    selected_by_axis: dict[
        str,
        list[dict],
    ],
) -> dict[str, Any]:

    selected_documents = {

        topic_axis: [

            document[
                "source_id"
            ]

            for document
            in selected_by_axis[
                topic_axis
            ]
        ]

        for topic_axis
        in EXPECTED_COUNTS
    }


    selected_counts = {

        topic_axis: len(
            source_ids
        )

        for (
            topic_axis,
            source_ids,
        )
        in selected_documents.items()
    }


    return {

        "selection_version": (
            SELECTION_VERSION
        ),

        "source": (
            "ntrs"
        ),

        "selection_method": (
            "human_qa"
        ),

        "purpose": (
            "Core-100 NTRS expansion "
            "from existing 15 to final 50."
        ),

        "target_new_counts": (
            EXPECTED_COUNTS
        ),

        "selected_counts": (
            selected_counts
        ),

        "selected_total": (
            sum(
                selected_counts.values()
            )
        ),

        # ----------------------------------------------------
        # Downstream canonical selection list
        #
        # Resolver는 우선 이 필드를 사용하면 된다.
        # ----------------------------------------------------

        "selected_documents": (
            selected_documents
        ),

        # ----------------------------------------------------
        # QA / provenance를 위한 full metadata snapshot
        # ----------------------------------------------------

        "selected_records": (
            selected_by_axis
        ),

        "provenance": {

            "collector_version": (
                "core100_v1"
            ),

            "collection_report": (
                str(
                    COLLECTION_REPORT_FILE
                )
            ),

            "review_pool_files": {

                topic_axis: str(
                    _review_pool_path(
                        topic_axis
                    )
                )

                for topic_axis
                in EXPECTED_COUNTS
            },
        },
    }


# ============================================================
# Print Selected Papers
# ============================================================

def _print_selected(
    selected_by_axis: dict[
        str,
        list[dict],
    ],
) -> None:

    print()
    print("=" * 78)

    print(
        "NTRS CORE-100 HUMAN QA SELECTION"
    )

    print("=" * 78)


    for topic_axis in EXPECTED_COUNTS:

        documents = (
            selected_by_axis[
                topic_axis
            ]
        )


        print()
        print(
            f"[{topic_axis}] "
            f"{len(documents)} selected"
        )

        print("-" * 78)


        for (
            index,
            document,
        ) in enumerate(
            documents,
            start=1,
        ):

            selection = (
                document[
                    "selection"
                ]
            )


            print(
                f"{index:02d}. "
                f"{document['source_id']} | "
                f"rank "
                f"{selection['review_rank']:02d}"
            )

            print(
                f"    "
                f"{document['title']}"
            )

            print(
                f"    "
                f"Type: "
                f"{document.get('document_type')}"
            )

            print(
                f"    "
                f"Published: "
                f"{document.get('published_at')}"
            )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 78)

    print(
        "TEAM B - NASA NTRS "
        "Core-100 Human QA Selector"
    )

    print("=" * 78)

    print(
        f"Version          : "
        f"{SELECTION_VERSION}"
    )

    print(
        f"Cache root       : "
        f"{CACHE_ROOT}"
    )

    print(
        f"Collection report: "
        f"{COLLECTION_REPORT_FILE}"
    )

    print(
        f"Output           : "
        f"{OUTPUT_FILE}"
    )


    # ========================================================
    # 1. Static Count Validation
    # ========================================================

    _validate_selection_counts()

    print()
    print(
        "[QA] Selection counts:"
    )


    for (
        topic_axis,
        count,
    ) in EXPECTED_COUNTS.items():

        print(
            f"  "
            f"{topic_axis:22} : "
            f"{count}"
        )


    print(
        f"  "
        f"{'TOTAL':22} : "
        f"{EXPECTED_TOTAL}"
    )


    # ========================================================
    # 2. Cross-axis ID duplicate
    # ========================================================

    _validate_no_cross_axis_id_duplicates()

    print(
        "[QA] Cross-axis source_id duplicates: 0"
    )


    # ========================================================
    # 3. Existing AWS NTRS IDs
    # ========================================================

    existing_ntrs_ids = (
        _load_existing_ntrs_ids()
    )


    print(
        f"[QA] Existing AWS NTRS IDs: "
        f"{len(existing_ntrs_ids)}"
    )


    # ========================================================
    # 4. Build + Validate
    # ========================================================

    selected_by_axis = (
        _build_selected_records(

            existing_ntrs_ids=(
                existing_ntrs_ids
            ),
        )
    )


    all_selected = (
        _flatten_selected_records(
            selected_by_axis
        )
    )


    if (
        len(all_selected)
        != EXPECTED_TOTAL
    ):

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_TOTAL} selected records, "
            f"found "
            f"{len(all_selected)}."
        )


    # ========================================================
    # 5. Content-level duplicate QA
    # ========================================================

    _validate_selected_content_duplicates(
        all_selected
    )


    print(
        "[QA] Duplicate selected titles: 0"
    )

    print(
        "[QA] Duplicate selected DOI: 0"
    )

    print(
        "[QA] Existing DB conflicts: 0"
    )

    print(
        "[QA] Full-text URL gate: "
        f"{len(all_selected)} / "
        f"{EXPECTED_TOTAL}"
    )


    # ========================================================
    # 6. Print
    # ========================================================

    _print_selected(
        selected_by_axis
    )


    # ========================================================
    # 7. Manifest
    # ========================================================

    manifest = (
        _build_manifest(
            selected_by_axis
        )
    )


    if (
        manifest[
            "selected_total"
        ]
        != EXPECTED_TOTAL
    ):

        raise RuntimeError(
            "Manifest total count mismatch."
        )


    _save_json(
        OUTPUT_FILE,
        manifest,
    )


    # ========================================================
    # Final
    # ========================================================

    print()
    print("=" * 78)

    print(
        "NTRS CORE-100 SELECTION COMPLETED"
    )

    print("=" * 78)

    print(
        f"rover_autonomy       : "
        f"{len(selected_by_axis['rover_autonomy'])}"
    )

    print(
        f"onboard_ai           : "
        f"{len(selected_by_axis['onboard_ai'])}"
    )

    print(
        f"satellite_autonomy   : "
        f"{len(selected_by_axis['satellite_autonomy'])}"
    )

    print(
        f"TOTAL                : "
        f"{len(all_selected)}"
    )

    print()

    print(
        f"Manifest:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print()
    print(
        "[PASS] Human QA selection "
        "is frozen."
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Run NTRS Core-100 Resolver "
        "for these 35 documents only."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()