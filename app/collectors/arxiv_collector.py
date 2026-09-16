import hashlib
import json
import time
from pathlib import Path
from typing import Any

import feedparser
import httpx
import psycopg

from app.config import PROJECT_ROOT, settings


# ============================================================
# CORE-100 VERSION
# ============================================================
#
# Pilot:
#   arXiv 15
#
# 이번 목표:
#   신규 arXiv 35편 후보 확보
#
# 최종 arXiv Core 50:
#
# rover_autonomy       17 = 기존 5 + 신규 12
# onboard_ai           16 = 기존 5 + 신규 11
# satellite_autonomy   17 = 기존 5 + 신규 12
#
# 여기서는 아직 최종 35편을 자동 확정하지 않는다.
#
# Collector:
#     충분한 후보 pool 생성
#
# Human QA:
#     그중 실제 관련 논문 선정
#
# Resolver:
#     HTML -> TeX -> PDF
#
# ============================================================

COLLECTOR_VERSION = "core100_v1"


NEW_TARGETS = {
    "rover_autonomy": 12,
    "onboard_ai": 11,
    "satellite_autonomy": 12,
}


# ============================================================
# Pagination Settings
# ============================================================
#
# 각 query를 한 번에 너무 크게 요청하지 않는다.
#
# round 1:
#     모든 query start=0
#
# 후보가 부족하면
#
# round 2:
#     모든 query start=25
#
# 이런 식으로 query 다양성을 먼저 확보한다.
#
# ============================================================

PAGE_SIZE = 25

MAX_PAGES_PER_QUERY = 4

# 최종 필요량의 4배 정도를 후보 pool로 확보.
POOL_MULTIPLIER = 4

# 사람이 콘솔에서 볼 개수.
CONSOLE_PREVIEW_LIMIT = 25

# 별도 review JSON에는 목표의 3배를 남긴다.
REVIEW_POOL_MULTIPLIER = 3


# ============================================================
# arXiv Multi-Query Configuration
# ============================================================

ARXIV_QUERIES = {

    # ========================================================
    # 1. Rover Autonomy
    # ========================================================

    "rover_autonomy": [

        {
            "name": "autonomous_navigation",

            "query": (
                'all:rover AND '
                'all:"autonomous navigation"'
            ),
        },

        {
            "name": "planetary_path_planning",

            "query": (
                'all:"planetary rover" AND '
                'all:"path planning"'
            ),
        },

        {
            "name": "lunar_rover_autonomy",

            "query": (
                'all:"lunar rover" AND '
                'all:autonomous'
            ),
        },

        {
            "name": "mars_rover_autonomy",

            "query": (
                'all:"Mars rover" AND '
                '('
                'all:autonomous OR '
                'all:"path planning" OR '
                'all:navigation'
                ')'
            ),
        },
    ],

    # ========================================================
    # 2. Onboard AI
    # ========================================================

    "onboard_ai": [

        {
            "name": "onboard_autonomy",

            "query": (
                'all:spacecraft AND '
                'all:"onboard autonomy"'
            ),
        },

        {
            "name": "deep_space_autonomy",

            "query": (
                'all:"deep space" AND '
                'all:autonomy'
            ),
        },

        {
            "name": "autonomous_spacecraft",

            "query": (
                'all:"autonomous spacecraft" AND '
                '('
                'all:planning OR '
                'all:navigation OR '
                'all:decision'
                ')'
            ),
        },

        {
            "name": "onboard_ai",

            "query": (
                'all:spacecraft AND '
                'all:"onboard AI"'
            ),
        },

        {
            "name": "onboard_decision",

            "query": (
                'all:spacecraft AND '
                'all:"autonomous decision"'
            ),
        },
    ],

    # ========================================================
    # 3. Satellite / Spacecraft Autonomy
    # ========================================================

    "satellite_autonomy": [

        {
            "name": "satellite_fault_detection",

            "query": (
                'all:satellite AND '
                'all:"fault detection"'
            ),
        },

        {
            "name": "spacecraft_fault_detection",

            "query": (
                'all:spacecraft AND '
                'all:"fault detection"'
            ),
        },

        {
            "name": "fault_diagnosis",

            "query": (
                '('
                'all:satellite OR '
                'all:spacecraft'
                ') AND '
                'all:"fault diagnosis"'
            ),
        },

        {
            "name": "fault_recovery",

            "query": (
                'all:spacecraft AND '
                'all:"fault recovery"'
            ),
        },

        {
            "name": "autonomous_operations",

            "query": (
                '('
                'all:satellite OR '
                'all:spacecraft'
                ') AND '
                'all:"autonomous operations"'
            ),
        },

        {
            "name": "onboard_fault",

            "query": (
                '('
                'all:satellite OR '
                'all:spacecraft'
                ') AND '
                'all:"onboard fault"'
            ),
        },
    ],
}


# ============================================================
# HTTP
# ============================================================

HEADERS = {
    "User-Agent": (
        "TEAM-B-University-Research-Project/1.0"
    )
}


# ============================================================
# Paths
# ============================================================
#
# 기존 Pilot cache는 절대 덮어쓰지 않는다.
#
# 기존:
#
# data/cache/arxiv/
#     rover_autonomy/
#     ...
#
# 신규:
#
# data/cache/arxiv/core100_v1/
#     rover_autonomy/
#     ...
#
# ============================================================

CACHE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "arxiv"
    / COLLECTOR_VERSION
)

REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "reports"
)

FINAL_REPORT = (
    REPORT_DIR
    / "arxiv_core100_collection_report.json"
)


# ============================================================
# Text Helpers
# ============================================================

def _normalize_space(
    text: str | None,
) -> str:

    if not text:
        return ""

    return " ".join(
        str(
            text
        ).split()
    )


# ============================================================
# arXiv ID Helpers
# ============================================================

def _get_arxiv_id(
    entry_id: str,
) -> str:

    return (
        entry_id
        .rstrip("/")
        .split("/")[-1]
    )


def _get_base_arxiv_id(
    source_id: str,
) -> str:
    """
    2401.11371v3
        ↓
    2401.11371
    """

    if "v" not in source_id:

        return source_id

    base, version = (
        source_id.rsplit(
            "v",
            1,
        )
    )

    if version.isdigit():

        return base

    return source_id


def _get_version_number(
    source_id: str,
) -> int:

    if "v" not in source_id:

        return 0

    _, version = (
        source_id.rsplit(
            "v",
            1,
        )
    )

    if version.isdigit():

        return int(
            version
        )

    return 0


# ============================================================
# PDF URL
# ============================================================

def _get_pdf_url(
    entry,
) -> str | None:

    for link in getattr(
        entry,
        "links",
        [],
    ):

        if (
            getattr(
                link,
                "type",
                "",
            )
            == "application/pdf"
        ):

            return link.href

        if (
            getattr(
                link,
                "title",
                "",
            )
            == "pdf"
        ):

            return link.href

    return None


# ============================================================
# Resolver Candidate URLs
# ============================================================

def _build_content_urls(
    source_id: str,
) -> dict:

    base_id = (
        _get_base_arxiv_id(
            source_id
        )
    )

    return {
        "html_candidate_url": (
            f"https://arxiv.org/html/"
            f"{base_id}"
        ),

        "source_candidate_url": (
            f"https://arxiv.org/src/"
            f"{base_id}"
        ),

        "pdf_candidate_url": (
            f"https://arxiv.org/pdf/"
            f"{source_id}"
        ),
    }


# ============================================================
# Query Hash
# ============================================================

def _query_hash(
    query: str,
) -> str:

    return hashlib.sha256(
        query.encode(
            "utf-8"
        )
    ).hexdigest()[:12]


# ============================================================
# Page Cache
# ============================================================

def _page_cache_path(
    *,
    topic_axis: str,
    query_name: str,
    query: str,
    start: int,
    page_size: int,
) -> Path:

    query_id = (
        _query_hash(
            query
        )
    )

    directory = (
        CACHE_ROOT
        / topic_axis
        / query_name
    )

    filename = (
        f"{query_id}"
        f"_start_{start:05d}"
        f"_size_{page_size}.json"
    )

    return (
        directory
        / filename
    )


# ============================================================
# Load Page Cache
# ============================================================

def _load_page_cache(
    *,
    topic_axis: str,
    query_name: str,
    query: str,
    start: int,
    page_size: int,
) -> tuple[
    list[dict],
    int | None,
] | None:

    path = (
        _page_cache_path(
            topic_axis=topic_axis,
            query_name=query_name,
            query=query,
            start=start,
            page_size=page_size,
        )
    )

    if not path.exists():

        return None

    try:

        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except (
        json.JSONDecodeError,
        OSError,
    ):

        print(
            f"[CACHE] Invalid cache ignored: "
            f"{path}"
        )

        return None

    if (
        payload.get(
            "collector_version"
        )
        != COLLECTOR_VERSION
    ):

        return None

    if (
        payload.get(
            "query"
        )
        != query
    ):

        return None

    if (
        payload.get(
            "start"
        )
        != start
    ):

        return None

    if (
        payload.get(
            "page_size"
        )
        != page_size
    ):

        return None

    documents = payload.get(
        "documents",
        [],
    )

    if not isinstance(
        documents,
        list,
    ):

        return None

    total_results = (
        payload.get(
            "total_results"
        )
    )

    print(
        f"[CACHE] "
        f"{topic_axis}/"
        f"{query_name}/"
        f"start={start}: "
        f"{len(documents)}"
    )

    return (
        documents,
        total_results,
    )


# ============================================================
# Save Page Cache
# ============================================================

def _save_page_cache(
    *,
    topic_axis: str,
    query_name: str,
    query: str,
    start: int,
    page_size: int,
    total_results: int | None,
    documents: list[dict],
) -> None:

    path = (
        _page_cache_path(
            topic_axis=topic_axis,
            query_name=query_name,
            query=query,
            start=start,
            page_size=page_size,
        )
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "collector_version": (
            COLLECTOR_VERSION
        ),

        "topic_axis": (
            topic_axis
        ),

        "query_name": (
            query_name
        ),

        "query": (
            query
        ),

        "start": (
            start
        ),

        "page_size": (
            page_size
        ),

        "total_results": (
            total_results
        ),

        "document_count": (
            len(
                documents
            )
        ),

        "documents": (
            documents
        ),
    }

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        f"[CACHE] Saved: "
        f"{path}"
    )


# ============================================================
# arXiv HTTP
# ============================================================

def _get_arxiv(
    url: str,
    *,
    params: dict | None = None,
) -> httpx.Response:

    response = httpx.get(
        url,
        params=params,
        timeout=settings.request_timeout,
        follow_redirects=True,
        headers=HEADERS,
    )

    if (
        response.status_code
        != 429
    ):

        response.raise_for_status()

        return response

    # ========================================================
    # 429
    # ========================================================

    print()
    print(
        "[arXiv] HTTP 429 rate limit detected."
    )

    retry_after = (
        response.headers.get(
            "Retry-After"
        )
    )

    if not (
        retry_after
        and retry_after.isdigit()
    ):

        raise RuntimeError(
            "arXiv returned HTTP 429 "
            "without Retry-After. "
            "Automatic retries stopped."
        )

    wait_seconds = int(
        retry_after
    )

    print(
        f"[arXiv] Retry-After: "
        f"{wait_seconds} seconds"
    )

    time.sleep(
        wait_seconds
    )

    response = httpx.get(
        url,
        params=params,
        timeout=settings.request_timeout,
        follow_redirects=True,
        headers=HEADERS,
    )

    if (
        response.status_code
        == 429
    ):

        raise RuntimeError(
            "arXiv rate limit is still active "
            "after Retry-After."
        )

    response.raise_for_status()

    return response


# ============================================================
# Atom Total Results
# ============================================================

def _extract_total_results(
    feed,
) -> int | None:

    value = None

    try:

        value = (
            feed.feed.get(
                "opensearch_totalresults"
            )
        )

    except Exception:

        value = None

    if value is None:

        return None

    try:

        return int(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return None


# ============================================================
# Parse Atom Entry
# ============================================================

def _parse_entry(
    entry,
    *,
    topic_axis: str,
    query_name: str,
    query: str,
    query_rank: int,
) -> dict:

    source_id = (
        _get_arxiv_id(
            entry.id
        )
    )

    source_base_id = (
        _get_base_arxiv_id(
            source_id
        )
    )

    authors = [
        author.name
        for author
        in getattr(
            entry,
            "authors",
            [],
        )
    ]

    categories = [
        tag.term
        for tag
        in getattr(
            entry,
            "tags",
            [],
        )
    ]

    primary_category = None

    primary = getattr(
        entry,
        "arxiv_primary_category",
        None,
    )

    if primary is not None:

        primary_category = getattr(
            primary,
            "term",
            None,
        )

    content_urls = (
        _build_content_urls(
            source_id
        )
    )

    published = (
        getattr(
            entry,
            "published",
            "",
        )
        or ""
    )

    return {
        "source": "arxiv",

        "source_id": (
            source_id
        ),

        "source_base_id": (
            source_base_id
        ),

        "title": (
            _normalize_space(
                getattr(
                    entry,
                    "title",
                    "",
                )
            )
        ),

        "abstract": (
            _normalize_space(
                getattr(
                    entry,
                    "summary",
                    "",
                )
            )
        ),

        "authors": (
            authors
        ),

        "categories": (
            categories
        ),

        "published_at": (
            published[:10]
            if published
            else None
        ),

        "document_type": (
            "arxiv_preprint"
        ),

        "doi": (
            getattr(
                entry,
                "arxiv_doi",
                None,
            )
        ),

        "url": (
            entry.id
        ),

        "pdf_url": (
            _get_pdf_url(
                entry
            )
        ),

        "html_candidate_url": (
            content_urls[
                "html_candidate_url"
            ]
        ),

        "source_candidate_url": (
            content_urls[
                "source_candidate_url"
            ]
        ),

        "pdf_candidate_url": (
            content_urls[
                "pdf_candidate_url"
            ]
        ),

        "topic_axis": (
            topic_axis
        ),

        "language": "en",

        "matched_queries": [
            query_name
        ],

        "metadata": {
            "updated_at": (
                getattr(
                    entry,
                    "updated",
                    None,
                )
            ),

            "primary_category": (
                primary_category
            ),

            "first_query_name": (
                query_name
            ),

            "first_query": (
                query
            ),

            "query_ranks": {
                query_name: (
                    query_rank
                )
            },
        },
    }


# ============================================================
# One Query Page
# ============================================================

def search_query_page(
    *,
    topic_axis: str,
    query_name: str,
    query: str,
    start: int,
    page_size: int,
) -> tuple[
    list[dict],
    int | None,
]:

    cached = (
        _load_page_cache(
            topic_axis=topic_axis,
            query_name=query_name,
            query=query,
            start=start,
            page_size=page_size,
        )
    )

    if cached is not None:

        return cached

    print()
    print("-" * 70)

    print(
        f"[arXiv] Axis   : "
        f"{topic_axis}"
    )

    print(
        f"[arXiv] Query  : "
        f"{query_name}"
    )

    print(
        f"[arXiv] Start  : "
        f"{start}"
    )

    print(
        f"[arXiv] Size   : "
        f"{page_size}"
    )

    print(
        f"[arXiv] Search : "
        f"{query}"
    )

    print("-" * 70)

    params = {
        "search_query": (
            query
        ),

        "start": (
            start
        ),

        "max_results": (
            page_size
        ),

        "sortBy": (
            "relevance"
        ),

        "sortOrder": (
            "descending"
        ),
    }

    response = (
        _get_arxiv(
            settings.arxiv_api_url,
            params=params,
        )
    )

    feed = (
        feedparser.parse(
            response.text
        )
    )

    if getattr(
        feed,
        "bozo",
        False,
    ):

        print(
            "[arXiv] Warning: "
            "feedparser detected "
            "a parsing issue."
        )

    total_results = (
        _extract_total_results(
            feed
        )
    )

    documents = []

    for offset, entry in enumerate(
        feed.entries,
        start=1,
    ):

        global_rank = (
            start
            + offset
        )

        document = (
            _parse_entry(
                entry,
                topic_axis=topic_axis,
                query_name=query_name,
                query=query,
                query_rank=global_rank,
            )
        )

        documents.append(
            document
        )

    print(
        f"[arXiv] Received : "
        f"{len(documents)}"
    )

    print(
        f"[arXiv] Total    : "
        f"{total_results}"
    )

    _save_page_cache(
        topic_axis=topic_axis,
        query_name=query_name,
        query=query,
        start=start,
        page_size=page_size,
        total_results=total_results,
        documents=documents,
    )

    # API 요청 뒤에만 sleep.
    time.sleep(
        settings.request_delay
    )

    return (
        documents,
        total_results,
    )


# ============================================================
# Merge / Dedup
# ============================================================

def _merge_documents(
    all_documents: list[dict],
) -> list[dict]:

    merged: dict[
        str,
        dict
    ] = {}

    for document in all_documents:

        dedup_key = (
            document[
                "source_base_id"
            ]
        )

        if (
            dedup_key
            not in merged
        ):

            merged[
                dedup_key
            ] = document

            continue

        existing = (
            merged[
                dedup_key
            ]
        )

        # ====================================================
        # 여러 query 이름 누적
        # ====================================================

        for query_name in document.get(
            "matched_queries",
            [],
        ):

            if (
                query_name
                not in existing[
                    "matched_queries"
                ]
            ):

                existing[
                    "matched_queries"
                ].append(
                    query_name
                )

        # ====================================================
        # query rank 누적
        # ====================================================

        existing_metadata = (
            existing.setdefault(
                "metadata",
                {},
            )
        )

        existing_ranks = (
            existing_metadata.setdefault(
                "query_ranks",
                {},
            )
        )

        new_metadata = (
            document.get(
                "metadata",
                {},
            )
        )

        new_ranks = (
            new_metadata.get(
                "query_ranks",
                {},
            )
            if isinstance(
                new_metadata,
                dict,
            )
            else {}
        )

        for (
            query_name,
            rank,
        ) in new_ranks.items():

            previous_rank = (
                existing_ranks.get(
                    query_name
                )
            )

            if (
                previous_rank is None
                or rank < previous_rank
            ):

                existing_ranks[
                    query_name
                ] = rank

        # ====================================================
        # 더 최신 version이면 metadata 본체는 최신으로
        # ====================================================

        existing_version = (
            _get_version_number(
                existing[
                    "source_id"
                ]
            )
        )

        new_version = (
            _get_version_number(
                document[
                    "source_id"
                ]
            )
        )

        if (
            new_version
            > existing_version
        ):

            preserved_queries = list(
                existing[
                    "matched_queries"
                ]
            )

            preserved_ranks = dict(
                existing_ranks
            )

            replacement = dict(
                document
            )

            replacement[
                "matched_queries"
            ] = (
                preserved_queries
            )

            replacement_metadata = dict(
                replacement.get(
                    "metadata",
                    {},
                )
            )

            replacement_metadata[
                "query_ranks"
            ] = (
                preserved_ranks
            )

            replacement[
                "metadata"
            ] = (
                replacement_metadata
            )

            merged[
                dedup_key
            ] = replacement

    return list(
        merged.values()
    )


# ============================================================
# Existing arXiv Papers From AWS RDS
# ============================================================

def _load_existing_arxiv_ids() -> set[str]:
    """
    DB에 이미 적재된 arXiv 논문을 자동으로 읽는다.

    현재는 Pilot 15편이 나와야 한다.

    향후 50편이 된 뒤 다시 실행해도
    이미 DB에 있는 모든 arXiv 문서를 자동 제외한다.
    """

    print()
    print(
        "[DB] Loading existing "
        "arXiv IDs from AWS RDS..."
    )

    with psycopg.connect(
        settings.dsn
    ) as conn:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                SELECT source_id
                FROM public.core_documents
                WHERE source = 'arxiv'
                """
            )

            rows = (
                cursor.fetchall()
            )

    existing = {
        _get_base_arxiv_id(
            str(
                row[0]
            )
        )
        for row in rows
        if row
        and row[0]
    }

    print(
        f"[DB] Existing arXiv papers: "
        f"{len(existing)}"
    )

    return existing


# ============================================================
# Exclude Existing DB Documents
# ============================================================

def _exclude_existing(
    documents: list[dict],
    existing_ids: set[str],
) -> tuple[
    list[dict],
    int,
]:

    result = []

    excluded = 0

    for document in documents:

        base_id = (
            document[
                "source_base_id"
            ]
        )

        if base_id in existing_ids:

            excluded += 1

            continue

        result.append(
            document
        )

    return (
        result,
        excluded,
    )


# ============================================================
# Candidate Ranking
# ============================================================
#
# 이것은 AI relevance score가 아니다.
#
# 단순한 QA 우선순위:
#
# 1. 여러 query에 동시에 걸린 논문
# 2. query에서 상위 rank였던 논문
#
# 최종 선정은 사람이 title + abstract를 확인한다.
# ============================================================

def _best_query_rank(
    document: dict,
) -> int:

    metadata = (
        document.get(
            "metadata",
            {},
        )
    )

    if not isinstance(
        metadata,
        dict,
    ):

        return 999999

    ranks = (
        metadata.get(
            "query_ranks",
            {},
        )
    )

    if not isinstance(
        ranks,
        dict,
    ):

        return 999999

    valid_ranks = []

    for rank in ranks.values():

        try:

            valid_ranks.append(
                int(
                    rank
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            continue

    if not valid_ranks:

        return 999999

    return min(
        valid_ranks
    )


def _candidate_sort_key(
    document: dict,
):

    matched_count = len(
        document.get(
            "matched_queries",
            [],
        )
    )

    best_rank = (
        _best_query_rank(
            document
        )
    )

    published = str(
        document.get(
            "published_at"
        )
        or ""
    )

    # reverse=True와 함께 사용:
    #
    # matched_count 큰 것 우선
    # best_rank 작은 것 우선 -> -best_rank
    # 날짜는 마지막 보조기준
    return (
        matched_count,
        -best_rank,
        published,
    )


def _rank_candidates(
    documents: list[dict],
) -> list[dict]:

    result = list(
        documents
    )

    result.sort(
        key=_candidate_sort_key,
        reverse=True,
    )

    return result


# ============================================================
# Search Axis With Pagination
# ============================================================

def search_axis_core100(
    *,
    topic_axis: str,
    existing_ids: set[str],
) -> tuple[
    list[dict],
    dict,
]:

    query_specs = (
        ARXIV_QUERIES[
            topic_axis
        ]
    )

    target_new = (
        NEW_TARGETS[
            topic_axis
        ]
    )

    desired_pool = (
        target_new
        * POOL_MULTIPLIER
    )

    print()
    print("=" * 70)

    print(
        f"[AXIS] "
        f"{topic_axis}"
    )

    print(
        f"[AXIS] New target       : "
        f"{target_new}"
    )

    print(
        f"[AXIS] Desired QA pool  : "
        f"{desired_pool}"
    )

    print(
        f"[AXIS] Query count      : "
        f"{len(query_specs)}"
    )

    print("=" * 70)

    all_documents = []

    exhausted_queries = set()

    pages_requested = 0

    # ========================================================
    # Round-robin pagination
    # ========================================================
    #
    # round 0:
    #   모든 query start 0
    #
    # round 1:
    #   모든 query start 25
    #
    # 이런 구조라 특정 query 하나가 후보를 독점하지 않는다.
    # ========================================================

    for page_index in range(
        MAX_PAGES_PER_QUERY
    ):

        start = (
            page_index
            * PAGE_SIZE
        )

        print()
        print(
            f"[ROUND] "
            f"{page_index + 1}/"
            f"{MAX_PAGES_PER_QUERY}"
        )

        for query_spec in query_specs:

            query_name = (
                query_spec[
                    "name"
                ]
            )

            query = (
                query_spec[
                    "query"
                ]
            )

            if (
                query_name
                in exhausted_queries
            ):

                continue

            documents, total_results = (
                search_query_page(
                    topic_axis=topic_axis,
                    query_name=query_name,
                    query=query,
                    start=start,
                    page_size=PAGE_SIZE,
                )
            )

            pages_requested += 1

            all_documents.extend(
                documents
            )

            # ================================================
            # Exhaustion
            # ================================================

            if (
                len(documents)
                < PAGE_SIZE
            ):

                exhausted_queries.add(
                    query_name
                )

                continue

            if (
                total_results
                is not None
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

        # ====================================================
        # Current pool
        # ====================================================

        merged = (
            _merge_documents(
                all_documents
            )
        )

        eligible, excluded = (
            _exclude_existing(
                merged,
                existing_ids,
            )
        )

        print()
        print(
            f"[ROUND] Raw       : "
            f"{len(all_documents)}"
        )

        print(
            f"[ROUND] Unique    : "
            f"{len(merged)}"
        )

        print(
            f"[ROUND] Existing  : "
            f"{excluded}"
        )

        print(
            f"[ROUND] Eligible  : "
            f"{len(eligible)}"
        )

        # 모든 query가 최소 1 page는 실행된 뒤
        # 충분하면 조기 종료.
        if (
            len(
                eligible
            )
            >= desired_pool
        ):

            print(
                "[ROUND] Desired candidate "
                "pool reached."
            )

            break

        if (
            len(exhausted_queries)
            == len(
                query_specs
            )
        ):

            print(
                "[ROUND] All queries exhausted."
            )

            break

    # ========================================================
    # Final Axis Pool
    # ========================================================

    merged = (
        _merge_documents(
            all_documents
        )
    )

    eligible, excluded = (
        _exclude_existing(
            merged,
            existing_ids,
        )
    )

    ranked = (
        _rank_candidates(
            eligible
        )
    )

    stats = {
        "topic_axis": (
            topic_axis
        ),

        "target_new": (
            target_new
        ),

        "desired_pool": (
            desired_pool
        ),

        "raw_candidates": (
            len(
                all_documents
            )
        ),

        "unique_candidates": (
            len(
                merged
            )
        ),

        "excluded_existing": (
            excluded
        ),

        "eligible_candidates": (
            len(
                ranked
            )
        ),

        "pages_requested": (
            pages_requested
        ),

        "queries_exhausted": (
            sorted(
                exhausted_queries
            )
        ),
    }

    print()
    print(
        f"[AXIS] Final eligible: "
        f"{len(ranked)}"
    )

    if (
        len(
            ranked
        )
        < target_new
    ):

        print(
            "[WARNING] Candidate count "
            "is below required target."
        )

    return (
        ranked,
        stats,
    )


# ============================================================
# Save Axis Files
# ============================================================

def _save_axis_outputs(
    *,
    topic_axis: str,
    documents: list[dict],
    stats: dict,
) -> tuple[
    Path,
    Path,
]:

    axis_dir = (
        CACHE_ROOT
        / topic_axis
    )

    axis_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Full eligible pool
    # ========================================================

    merged_path = (
        axis_dir
        / "_merged_candidates.json"
    )

    merged_payload = {
        "collector_version": (
            COLLECTOR_VERSION
        ),

        "topic_axis": (
            topic_axis
        ),

        "stats": (
            stats
        ),

        "document_count": (
            len(
                documents
            )
        ),

        "documents": (
            documents
        ),
    }

    merged_path.write_text(
        json.dumps(
            merged_payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Human review pool
    # ========================================================

    target_new = (
        NEW_TARGETS[
            topic_axis
        ]
    )

    review_count = min(
        len(
            documents
        ),
        max(
            target_new
            * REVIEW_POOL_MULTIPLIER,
            target_new,
        ),
    )

    review_documents = (
        documents[
            :review_count
        ]
    )

    review_path = (
        axis_dir
        / "_review_pool.json"
    )

    review_payload = {
        "collector_version": (
            COLLECTOR_VERSION
        ),

        "topic_axis": (
            topic_axis
        ),

        "target_new": (
            target_new
        ),

        "review_count": (
            review_count
        ),

        "important": (
            "Ordering is only a QA priority "
            "based on multi-query hits and "
            "query rank. Human review is still "
            "required before final selection."
        ),

        "documents": (
            review_documents
        ),
    }

    review_path.write_text(
        json.dumps(
            review_payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        f"[REPORT] Full pool  : "
        f"{merged_path}"
    )

    print(
        f"[REPORT] Review pool: "
        f"{review_path}"
    )

    return (
        merged_path,
        review_path,
    )


# ============================================================
# Console Preview
# ============================================================

def print_candidates(
    topic_axis: str,
    documents: list[dict],
) -> None:

    print()
    print("=" * 70)

    print(
        f"{topic_axis} - "
        f"top review candidates"
    )

    print("=" * 70)

    preview = (
        documents[
            :CONSOLE_PREVIEW_LIMIT
        ]
    )

    for index, document in enumerate(
        preview,
        start=1,
    ):

        abstract = (
            document.get(
                "abstract",
                "",
            )
        )

        if len(
            abstract
        ) > 350:

            abstract = (
                abstract[:350]
                + "..."
            )

        print()

        print(
            f"{index:02d}. "
            f"{document['title']}"
        )

        print(
            f"    ID: "
            f"{document['source_id']}"
        )

        print(
            f"    Published: "
            f"{document['published_at']}"
        )

        print(
            f"    Categories: "
            f"{', '.join(document['categories'])}"
        )

        print(
            f"    Matched queries: "
            f"{', '.join(document['matched_queries'])}"
        )

        print(
            f"    Best query rank: "
            f"{_best_query_rank(document)}"
        )

        print(
            f"    Abstract: "
            f"{abstract}"
        )

        print(
            f"    URL: "
            f"{document['url']}"
        )


# ============================================================
# Cross-Axis Overlap
# ============================================================

def _find_cross_axis_overlaps(
    axis_results: dict[
        str,
        list[dict],
    ],
) -> list[dict]:
    """
    같은 arXiv 논문이 두 axis 후보에 동시에 등장하는지 검사.

    아직 제거하지 않는다.

    이유:
    어느 axis가 더 적합한지는 Human QA에서 결정한다.
    """

    index: dict[
        str,
        dict,
    ] = {}

    for (
        topic_axis,
        documents,
    ) in axis_results.items():

        for document in documents:

            base_id = (
                document[
                    "source_base_id"
                ]
            )

            record = (
                index.setdefault(
                    base_id,
                    {
                        "source_base_id": (
                            base_id
                        ),

                        "title": (
                            document[
                                "title"
                            ]
                        ),

                        "axes": [],
                    },
                )
            )

            if (
                topic_axis
                not in record[
                    "axes"
                ]
            ):

                record[
                    "axes"
                ].append(
                    topic_axis
                )

    overlaps = [
        record
        for record
        in index.values()
        if len(
            record[
                "axes"
            ]
        ) > 1
    ]

    overlaps.sort(
        key=lambda item: (
            item[
                "source_base_id"
            ]
        )
    )

    return overlaps


# ============================================================
# Save Final Report
# ============================================================

def _save_final_report(
    *,
    existing_count: int,
    axis_results: dict[
        str,
        list[dict],
    ],
    axis_stats: dict[
        str,
        dict,
    ],
    overlaps: list[dict],
) -> None:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "collector_version": (
            COLLECTOR_VERSION
        ),

        "existing_arxiv_documents": (
            existing_count
        ),

        "new_targets": (
            NEW_TARGETS
        ),

        "axis_stats": (
            axis_stats
        ),

        "cross_axis_overlap_count": (
            len(
                overlaps
            )
        ),

        "cross_axis_overlaps": (
            overlaps
        ),

        "next_step": (
            "Human QA of _review_pool.json, "
            "then select 12 rover + 11 onboard "
            "+ 12 satellite new papers."
        ),
    }

    _save_json(
        FINAL_REPORT,
        payload,
    )


# ============================================================
# JSON Save Helper
# ============================================================

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
# Main
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "TEAM B - arXiv Core-100 "
        "Candidate Collector"
    )

    print("=" * 70)

    print(
        f"Version          : "
        f"{COLLECTOR_VERSION}"
    )

    print(
        f"Page size        : "
        f"{PAGE_SIZE}"
    )

    print(
        f"Max pages/query  : "
        f"{MAX_PAGES_PER_QUERY}"
    )

    print(
        f"Request delay    : "
        f"{settings.request_delay} sec"
    )

    print(
        f"Cache root       : "
        f"{CACHE_ROOT}"
    )

    print()
    print(
        "New arXiv targets:"
    )

    for (
        topic_axis,
        target,
    ) in NEW_TARGETS.items():

        print(
            f"  {topic_axis:22} "
            f"+{target}"
        )

    # ========================================================
    # Existing DB IDs
    # ========================================================

    try:

        existing_ids = (
            _load_existing_arxiv_ids()
        )

    except Exception as exc:

        print()
        print("=" * 70)

        print(
            "DATABASE CHECK FAILED"
        )

        print("=" * 70)

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print()
        print(
            "Core-100 collection stopped "
            "because existing DB documents "
            "must be excluded safely."
        )

        return

    # ========================================================
    # Collect
    # ========================================================

    axis_results = {}

    axis_stats = {}

    for topic_axis in (
        ARXIV_QUERIES
    ):

        try:

            documents, stats = (
                search_axis_core100(
                    topic_axis=topic_axis,
                    existing_ids=existing_ids,
                )
            )

            axis_results[
                topic_axis
            ] = (
                documents
            )

            axis_stats[
                topic_axis
            ] = (
                stats
            )

            _save_axis_outputs(
                topic_axis=topic_axis,
                documents=documents,
                stats=stats,
            )

            print_candidates(
                topic_axis,
                documents,
            )

        except Exception as exc:

            print()
            print("=" * 70)

            print(
                f"[STOP] "
                f"{topic_axis}"
            )

            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print()
            print(
                "Completed page caches "
                "have been preserved."
            )

            print(
                "Run again later to resume."
            )

            print("=" * 70)

            return

    # ========================================================
    # Cross-axis overlap
    # ========================================================

    overlaps = (
        _find_cross_axis_overlaps(
            axis_results
        )
    )

    # ========================================================
    # Final Report
    # ========================================================

    _save_final_report(
        existing_count=len(
            existing_ids
        ),
        axis_results=axis_results,
        axis_stats=axis_stats,
        overlaps=overlaps,
    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 70)

    print(
        "ARXIV CORE-100 COLLECTION COMPLETED"
    )

    print("=" * 70)

    for (
        topic_axis,
        documents,
    ) in axis_results.items():

        target = (
            NEW_TARGETS[
                topic_axis
            ]
        )

        print(
            f"{topic_axis:22} : "
            f"{len(documents):4} "
            f"eligible "
            f"(need {target})"
        )

    print(
        f"Cross-axis overlaps    : "
        f"{len(overlaps)}"
    )

    print(
        f"Existing DB excluded   : "
        f"{len(existing_ids)} "
        f"arXiv papers"
    )

    print(
        f"Final report           : "
        f"{FINAL_REPORT}"
    )

    print()
    print(
        "NEXT:"
    )

    print(
        "Human QA of each "
        "_review_pool.json"
    )

    print(
        "Select exactly:"
    )

    print(
        "rover_autonomy      +12"
    )

    print(
        "onboard_ai          +11"
    )

    print(
        "satellite_autonomy  +12"
    )

    print()
    print(
        "Do NOT run the resolver "
        "for the whole candidate pool."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()