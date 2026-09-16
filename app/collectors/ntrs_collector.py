import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urljoin

import httpx
import psycopg

from app.config import PROJECT_ROOT, settings


# ============================================================
# TEAM B - NASA NTRS Core-100 Candidate Collector
# ============================================================
#
# 현재 AWS RDS:
#
#   arXiv : 50
#   NTRS  : 15
#   TOTAL : 65
#
# 이번 NTRS 추가 목표:
#
#   rover_autonomy       +12
#   onboard_ai           +12
#   satellite_autonomy   +11
#   ------------------------
#   TOTAL                +35
#
# 최종:
#
#   arXiv : 50
#   NTRS  : 50
#   TOTAL : 100
#
#
# 이 Collector가 하는 것:
#
#   NASA NTRS metadata 검색
#           ↓
#   PUBLIC + Full Document 필터
#           ↓
#   기술/학술 document type 필터
#           ↓
#   query 간 source_id dedup
#           ↓
#   AWS에 이미 있는 NTRS 15편 제외
#           ↓
#   Human QA용 review pool 생성
#
#
# 이 Collector가 하지 않는 것:
#
#   - 최종 35편 자동선정
#   - TXT/PDF 다운로드
#   - Resolver
#   - Parser
#   - Cleaner
#   - DB INSERT
#
# 최종 선정은 Human QA로 한다.
# ============================================================


COLLECTOR_VERSION = "core100_v1"


# ============================================================
# NASA NTRS API
# ============================================================

NTRS_API_ROOT = (
    "https://ntrs.nasa.gov/api"
)

NTRS_SEARCH_URL = (
    f"{NTRS_API_ROOT}/citations/search"
)

NTRS_PUBLIC_ROOT = (
    "https://ntrs.nasa.gov"
)


# ============================================================
# Database
# ============================================================

TABLE_SCHEMA = "public"

TABLE_NAME = "core_documents"


EXPECTED_EXISTING_NTRS_TOTAL = 15


EXPECTED_EXISTING_NTRS_BY_AXIS = {

    "rover_autonomy": 5,

    "onboard_ai": 5,

    "satellite_autonomy": 5,
}


# ============================================================
# Core-100 Targets
# ============================================================

TARGET_NEW_COUNTS = {

    "rover_autonomy": 12,

    "onboard_ai": 12,

    "satellite_autonomy": 11,
}


TARGET_NEW_TOTAL = sum(
    TARGET_NEW_COUNTS.values()
)


# 최종 선정보다 넓은 Human QA pool
REVIEW_POOL_MULTIPLIER = 4


# Query 하나당 최대 후보
RESULTS_PER_QUERY = 50


# 콘솔에는 너무 많이 출력하지 않음
CONSOLE_PREVIEW_LIMIT = 25


# ============================================================
# Paths
# ============================================================

CACHE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "ntrs"
    / COLLECTOR_VERSION
)


REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "reports"
)


REPORT_FILE = (
    REPORT_DIR
    / "ntrs_core100_collection_report.json"
)


# ============================================================
# Search Queries
# ============================================================
#
# 기존 파일럿 query보다 넓힌다.
#
# 하지만 최종 자동선정은 하지 않는다.
#
# Broad Candidate Collection
#       ↓
# Human QA
#
# ============================================================

NTRS_QUERIES = {

    # ========================================================
    # 1. Planetary Rover Autonomy
    # ========================================================

    "rover_autonomy": [

        {
            "name": (
                "planetary_rover_navigation"
            ),
            "query": (
                '"planetary rover" '
                '"autonomous navigation"'
            ),
        },

        {
            "name": (
                "mars_rover_navigation"
            ),
            "query": (
                '"Mars rover" '
                'autonomous navigation'
            ),
        },

        {
            "name": (
                "lunar_rover_navigation"
            ),
            "query": (
                '"lunar rover" '
                'autonomous navigation'
            ),
        },

        {
            "name": (
                "planetary_path_planning"
            ),
            "query": (
                '"planetary rover" '
                '"path planning"'
            ),
        },

        {
            "name": (
                "rover_hazard_avoidance"
            ),
            "query": (
                'rover '
                '"hazard avoidance"'
            ),
        },

        {
            "name": (
                "rover_obstacle_avoidance"
            ),
            "query": (
                'rover '
                '"obstacle avoidance"'
            ),
        },

        {
            "name": (
                "rover_traversability"
            ),
            "query": (
                'rover traversability'
            ),
        },

        {
            "name": (
                "rover_localization_navigation"
            ),
            "query": (
                'rover localization navigation'
            ),
        },

        {
            "name": (
                "planetary_rover_autonomy"
            ),
            "query": (
                '"planetary rover" autonomy'
            ),
        },

        {
            "name": (
                "mars_rover_localization"
            ),
            "query": (
                '"Mars rover" localization'
            ),
        },

        {
            "name": (
                "lunar_rover_navigation_broad"
            ),
            "query": (
                '"lunar rover" navigation'
            ),
        },

        {
            "name": (
                "rover_autonomous_science"
            ),
            "query": (
                'rover '
                '"autonomous science"'
            ),
        },
    ],

    # ========================================================
    # 2. Onboard AI / Deep-Space Autonomy
    # ========================================================

    "onboard_ai": [

        {
            "name": (
                "onboard_autonomy"
            ),
            "query": (
                'spacecraft '
                '"onboard autonomy"'
            ),
        },

        {
            "name": (
                "autonomous_spacecraft"
            ),
            "query": (
                '"autonomous spacecraft"'
            ),
        },

        {
            "name": (
                "deep_space_autonomy"
            ),
            "query": (
                '"deep space" '
                'spacecraft autonomy'
            ),
        },

        {
            "name": (
                "onboard_ai"
            ),
            "query": (
                'spacecraft '
                '"onboard AI"'
            ),
        },

        {
            "name": (
                "spacecraft_ai_autonomy"
            ),
            "query": (
                'spacecraft '
                '"artificial intelligence" '
                'autonomy'
            ),
        },

        {
            "name": (
                "autonomous_decision"
            ),
            "query": (
                'spacecraft '
                '"autonomous decision"'
            ),
        },

        {
            "name": (
                "autonomous_navigation"
            ),
            "query": (
                'spacecraft '
                'autonomous navigation'
            ),
        },

        {
            "name": (
                "deep_space_navigation"
            ),
            "query": (
                '"deep space" '
                'autonomous navigation'
            ),
        },

        {
            "name": (
                "autonomous_planning"
            ),
            "query": (
                'spacecraft '
                'autonomous planning'
            ),
        },

        {
            "name": (
                "mission_autonomy"
            ),
            "query": (
                'spacecraft '
                '"mission autonomy"'
            ),
        },

        {
            "name": (
                "onboard_planning"
            ),
            "query": (
                'spacecraft '
                '"onboard planning"'
            ),
        },

        {
            "name": (
                "autonomous_control"
            ),
            "query": (
                'spacecraft '
                '"autonomous control"'
            ),
        },
    ],

    # ========================================================
    # 3. Satellite / Spacecraft Autonomous Operations
    # ========================================================

    "satellite_autonomy": [

        {
            "name": (
                "satellite_fault_detection"
            ),
            "query": (
                'satellite '
                '"fault detection"'
            ),
        },

        {
            "name": (
                "spacecraft_fault_detection"
            ),
            "query": (
                'spacecraft '
                '"fault detection"'
            ),
        },

        {
            "name": (
                "fault_diagnosis"
            ),
            "query": (
                'spacecraft '
                '"fault diagnosis"'
            ),
        },

        {
            "name": (
                "fault_recovery"
            ),
            "query": (
                'spacecraft '
                '"fault recovery"'
            ),
        },

        {
            "name": (
                "fault_management"
            ),
            "query": (
                'spacecraft '
                '"fault management"'
            ),
        },

        {
            "name": (
                "autonomous_operations"
            ),
            "query": (
                'spacecraft '
                '"autonomous operations"'
            ),
        },

        {
            "name": (
                "satellite_anomaly_detection"
            ),
            "query": (
                'satellite '
                '"anomaly detection"'
            ),
        },

        {
            "name": (
                "spacecraft_anomaly_detection"
            ),
            "query": (
                'spacecraft '
                '"anomaly detection"'
            ),
        },

        {
            "name": (
                "fault_isolation"
            ),
            "query": (
                'spacecraft '
                '"fault isolation"'
            ),
        },

        {
            "name": (
                "health_management"
            ),
            "query": (
                'spacecraft '
                '"health management"'
            ),
        },

        {
            "name": (
                "satellite_autonomous_operations"
            ),
            "query": (
                'satellite '
                '"autonomous operations"'
            ),
        },

        {
            "name": (
                "spacecraft_fdir"
            ),
            "query": (
                'spacecraft FDIR'
            ),
        },
    ],
}


# ============================================================
# Allowed STI Types
# ============================================================
#
# 발표자료, poster, abstract-only 등은
# Core paper corpus에서 제외한다.
#
# ============================================================

ALLOWED_STI_TYPES = {

    "ACCEPTED_MANUSCRIPT",

    "CONFERENCE_PAPER",

    "CONFERENCE_PROCEEDINGS",

    "CONFERENCE_PUBLICATION",

    "CONTRACTOR_OR_GRANTEE_REPORT",

    "CONTRACTOR_REPORT",

    "CONTRIBUTION_TO_LARGER_WORK",

    "PREPRINT",

    "REPRINT",

    "SPECIAL_PUBLICATION",

    "TECHNICAL_MEMORANDUM",

    "TECHNICAL_PUBLICATION",

    "TECHNICAL_TRANSLATION",

    "THESIS_DISSERTATION",

    "WHITE_PAPER",
}


# ============================================================
# Human Review Signals
# ============================================================
#
# 자동 합격/탈락용이 아니다.
#
# Human QA에서 관련 후보를 위로 올리기 위한
# 단순 lexical signal이다.
#
# ============================================================

AXIS_KEYWORDS = {

    "rover_autonomy": [

        "rover",

        "planetary",

        "lunar",

        "mars",

        "navigation",

        "path planning",

        "localization",

        "traversability",

        "terrain",

        "hazard avoidance",

        "obstacle avoidance",

        "mobility",

        "autonomy",

        "autonomous science",
    ],

    "onboard_ai": [

        "spacecraft",

        "deep space",

        "onboard",

        "on-board",

        "autonomy",

        "autonomous",

        "artificial intelligence",

        "machine learning",

        "decision",

        "planning",

        "navigation",

        "guidance",

        "mission autonomy",
    ],

    "satellite_autonomy": [

        "spacecraft",

        "satellite",

        "fault",

        "anomaly",

        "diagnosis",

        "detection",

        "isolation",

        "recovery",

        "fdir",

        "health management",

        "fault management",

        "monitoring",

        "autonomous operations",

        "autonomy",
    ],
}


# ============================================================
# HTTP
# ============================================================

HEADERS = {

    "User-Agent": (
        "TEAM-B-University-Research-Project/1.0"
    ),

    "Accept": (
        "application/json"
    ),
}


# ============================================================
# JSON Helpers
# ============================================================

def _load_json(
    path: Path,
) -> dict:

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
# Text Helpers
# ============================================================

def _normalize_space(
    value: str | None,
) -> str:

    if not value:

        return ""

    return " ".join(
        str(
            value
        ).split()
    )


def _safe_float(
    value,
) -> float | None:

    try:

        if value is None:

            return None

        return float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return None


# ============================================================
# URL Helpers
# ============================================================

def _absolute_ntrs_url(
    value: str | None,
) -> str | None:

    if not value:

        return None

    return urljoin(
        NTRS_PUBLIC_ROOT,
        value,
    )


# ============================================================
# Existing NTRS Documents in AWS
# ============================================================

def _load_existing_ntrs_from_db() -> dict[
    str,
    str,
]:

    sql = f"""
    SELECT
        source_id,
        topic_axis
    FROM {TABLE_SCHEMA}.{TABLE_NAME}
    WHERE source = 'ntrs'
    ORDER BY topic_axis, source_id
    """

    print()
    print(
        "[DB] Loading existing NTRS IDs "
        "from AWS RDS..."
    )

    with psycopg.connect(
        settings.dsn
    ) as conn:

        with conn.cursor() as cur:

            cur.execute(
                sql
            )

            rows = (
                cur.fetchall()
            )

    existing = {

        str(
            source_id
        ): str(
            topic_axis
        )

        for (
            source_id,
            topic_axis,
        )
        in rows
    }

    print(
        f"[DB] Existing NTRS documents: "
        f"{len(existing)}"
    )

    axis_counts = {

        axis: 0

        for axis
        in EXPECTED_EXISTING_NTRS_BY_AXIS
    }

    for topic_axis in existing.values():

        if topic_axis in axis_counts:

            axis_counts[
                topic_axis
            ] += 1

    for (
        topic_axis,
        count,
    ) in axis_counts.items():

        print(
            f"[DB]   "
            f"{topic_axis:22} : "
            f"{count}"
        )

    # ========================================================
    # Core-100 시작 상태 검증
    # ========================================================

    if (
        len(existing)
        != EXPECTED_EXISTING_NTRS_TOTAL
    ):

        raise RuntimeError(
            "Unexpected current NTRS DB state.\n"
            f"Expected "
            f"{EXPECTED_EXISTING_NTRS_TOTAL} "
            f"existing NTRS documents, "
            f"found {len(existing)}."
        )

    for (
        topic_axis,
        expected_count,
    ) in EXPECTED_EXISTING_NTRS_BY_AXIS.items():

        actual_count = (
            axis_counts.get(
                topic_axis,
                0,
            )
        )

        if (
            actual_count
            != expected_count
        ):

            raise RuntimeError(
                "Unexpected NTRS axis count.\n"
                f"{topic_axis}: "
                f"expected {expected_count}, "
                f"found {actual_count}."
            )

    return existing


# ============================================================
# Query Cache
# ============================================================

def _query_hash(
    query: str,
) -> str:

    return hashlib.sha256(
        query.encode(
            "utf-8"
        )
    ).hexdigest()[:12]


def _cache_path(
    topic_axis: str,
    query_name: str,
    query: str,
) -> Path:

    query_dir = (
        CACHE_ROOT
        / topic_axis
    )

    query_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        query_dir
        / (
            f"{query_name}_"
            f"{_query_hash(query)}.json"
        )
    )


def _load_query_cache(
    *,
    topic_axis: str,
    query_name: str,
    query: str,
    max_results: int,
) -> list[dict] | None:

    path = (
        _cache_path(
            topic_axis,
            query_name,
            query,
        )
    )

    if not path.exists():

        return None

    try:

        payload = (
            _load_json(
                path
            )
        )

    except Exception:

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
            "max_results"
        )
        != max_results
    ):

        return None

    documents = (
        payload.get(
            "documents"
        )
    )

    if not isinstance(
        documents,
        list,
    ):

        return None

    print(
        f"[CACHE] Loaded: "
        f"{path}"
    )

    return documents


def _save_query_cache(
    *,
    topic_axis: str,
    query_name: str,
    query: str,
    max_results: int,
    documents: list[dict],
    api_total: int | None,
    rejected_reasons: dict[str, int],
) -> None:

    path = (
        _cache_path(
            topic_axis,
            query_name,
            query,
        )
    )

    payload = {

        "collector_version": (
            COLLECTOR_VERSION
        ),

        "source": (
            "ntrs"
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

        "max_results": (
            max_results
        ),

        "api_total": (
            api_total
        ),

        "rejected_reasons": (
            rejected_reasons
        ),

        "documents": (
            documents
        ),
    }

    _save_json(
        path,
        payload,
    )

    print(
        f"[CACHE] Saved: "
        f"{path}"
    )


# ============================================================
# NASA NTRS HTTP
# ============================================================

def _request_ntrs_search(
    *,
    query: str,
    max_results: int,
) -> dict:

    params = {

        "q": (
            query
        ),

        "disseminated": (
            "DOCUMENT_AND_METADATA"
        ),

        "page.size": min(
            max_results,
            100,
        ),
    }

    # ========================================================
    # GET search
    #
    # q가 URL query parameter로 확실하게 전달되도록 한다.
    # ========================================================

    response = (
        httpx.get(
            NTRS_SEARCH_URL,
            params=params,
            timeout=settings.request_timeout,
            follow_redirects=True,
            headers=HEADERS,
        )
    )

    # ========================================================
    # Rate Limit
    # ========================================================

    if (
        response.status_code
        == 429
    ):

        retry_after = (
            response.headers.get(
                "Retry-After"
            )
        )

        if (
            retry_after
            and retry_after.isdigit()
        ):

            wait_seconds = int(
                retry_after
            )

        else:

            wait_seconds = max(
                settings.request_delay,
                30,
            )

        print(
            f"[NTRS] HTTP 429. "
            f"Waiting "
            f"{wait_seconds} sec..."
        )

        time.sleep(
            wait_seconds
        )

        response = (
            httpx.get(
                NTRS_SEARCH_URL,
                params=params,
                timeout=settings.request_timeout,
                follow_redirects=True,
                headers=HEADERS,
            )
        )

    response.raise_for_status()

    payload = (
        response.json()
    )

    if not isinstance(
        payload,
        dict,
    ):

        raise RuntimeError(
            "Unexpected NASA NTRS response."
        )

    return payload


# ============================================================
# Metadata Helpers
# ============================================================

def _extract_authors(
    item: dict,
) -> list[str]:

    authors: list[str] = []

    affiliations = (
        item.get(
            "authorAffiliations",
            [],
        )
    )

    if not isinstance(
        affiliations,
        list,
    ):

        return authors

    for affiliation in affiliations:

        if not isinstance(
            affiliation,
            dict,
        ):

            continue

        meta = (
            affiliation.get(
                "meta",
                {},
            )
        )

        if not isinstance(
            meta,
            dict,
        ):

            continue

        author = (
            meta.get(
                "author",
                {},
            )
        )

        if not isinstance(
            author,
            dict,
        ):

            continue

        name = (
            _normalize_space(
                author.get(
                    "name"
                )
            )
        )

        if (
            name
            and name not in authors
        ):

            authors.append(
                name
            )

    return authors


def _extract_doi(
    item: dict,
) -> str | None:

    identifiers = (
        item.get(
            "sourceIdentifiers",
            [],
        )
    )

    if isinstance(
        identifiers,
        list,
    ):

        for identifier in identifiers:

            if not isinstance(
                identifier,
                dict,
            ):

                continue

            id_type = str(
                identifier.get(
                    "type",
                    "",
                )
            ).upper()

            if (
                id_type
                == "DOI"
            ):

                value = (
                    identifier.get(
                        "number"
                    )
                    or identifier.get(
                        "value"
                    )
                )

                if value:

                    return str(
                        value
                    )

    publications = (
        item.get(
            "publications",
            [],
        )
    )

    if isinstance(
        publications,
        list,
    ):

        for publication in publications:

            if not isinstance(
                publication,
                dict,
            ):

                continue

            doi = (
                publication.get(
                    "doi"
                )
            )

            if doi:

                return str(
                    doi
                )

    return None


def _extract_published_date(
    item: dict,
) -> str | None:

    publications = (
        item.get(
            "publications",
            [],
        )
    )

    if isinstance(
        publications,
        list,
    ):

        for publication in publications:

            if not isinstance(
                publication,
                dict,
            ):

                continue

            value = (
                publication.get(
                    "publicationDate"
                )
            )

            if value:

                return str(
                    value
                )[:10]

    for key in (
        "distributionDate",
        "submittedDate",
        "created",
    ):

        value = (
            item.get(
                key
            )
        )

        if value:

            return str(
                value
            )[:10]

    return None


# ============================================================
# Download URLs
# ============================================================

def _extract_download_urls(
    item: dict,
) -> dict:

    fulltext_url = None

    pdf_url = None

    original_url = None


    downloads = (
        item.get(
            "downloads",
            [],
        )
    )

    if not isinstance(
        downloads,
        list,
    ):

        downloads = []


    download_summary = []


    for download in downloads:

        if not isinstance(
            download,
            dict,
        ):

            continue

        links = (
            download.get(
                "links",
                {},
            )
        )

        if not isinstance(
            links,
            dict,
        ):

            links = {}

        current_fulltext = (
            _absolute_ntrs_url(
                links.get(
                    "fulltext"
                )
            )
        )

        current_pdf = (
            _absolute_ntrs_url(
                links.get(
                    "pdf"
                )
            )
        )

        current_original = (
            _absolute_ntrs_url(
                links.get(
                    "original"
                )
            )
        )

        if (
            fulltext_url is None
            and current_fulltext
        ):

            fulltext_url = (
                current_fulltext
            )

        if (
            pdf_url is None
            and current_pdf
        ):

            pdf_url = (
                current_pdf
            )

        if (
            original_url is None
            and current_original
        ):

            original_url = (
                current_original
            )

        download_summary.append(
            {

                "name": (
                    download.get(
                        "name"
                    )
                ),

                "type": (
                    download.get(
                        "type"
                    )
                ),

                "mimetype": (
                    download.get(
                        "mimetype"
                    )
                ),

                "draft": (
                    download.get(
                        "draft"
                    )
                ),

                "fulltext_url": (
                    current_fulltext
                ),

                "pdf_url": (
                    current_pdf
                ),

                "original_url": (
                    current_original
                ),
            }
        )

    return {

        "fulltext_url": (
            fulltext_url
        ),

        "pdf_url": (
            pdf_url
        ),

        "original_url": (
            original_url
        ),

        "downloads": (
            download_summary
        ),
    }


# ============================================================
# Candidate Filter
# ============================================================

def _candidate_is_usable(
    item: dict,
) -> tuple[
    bool,
    str,
]:

    source_id = (
        item.get(
            "id"
        )
    )

    if source_id is None:

        return (
            False,
            "no_source_id",
        )

    # ========================================================
    # Public
    # ========================================================

    distribution = str(
        item.get(
            "distribution",
            "",
        )
    ).upper()

    if (
        distribution
        != "PUBLIC"
    ):

        return (
            False,
            "not_public",
        )

    # ========================================================
    # Full document
    # ========================================================

    disseminated = str(
        item.get(
            "disseminated",
            "",
        )
    ).upper()

    if (
        disseminated
        != "DOCUMENT_AND_METADATA"
    ):

        return (
            False,
            "metadata_only",
        )

    # ========================================================
    # Download
    # ========================================================

    if not item.get(
        "downloadsAvailable",
        False,
    ):

        return (
            False,
            "no_download",
        )

    # ========================================================
    # Abstract only
    # ========================================================

    if item.get(
        "onlyAbstract",
        False,
    ):

        return (
            False,
            "abstract_only",
        )

    # ========================================================
    # Document type
    # ========================================================

    sti_type = str(
        item.get(
            "stiType",
            "",
        )
    ).upper()

    if (
        sti_type
        not in ALLOWED_STI_TYPES
    ):

        return (
            False,
            f"sti_type:{sti_type}",
        )

    # ========================================================
    # Title
    # ========================================================

    title = (
        _normalize_space(
            item.get(
                "title"
            )
        )
    )

    if not title:

        return (
            False,
            "no_title",
        )

    # ========================================================
    # Abstract
    # ========================================================

    abstract = (
        _normalize_space(
            item.get(
                "abstract"
            )
        )
    )

    if not abstract:

        return (
            False,
            "no_abstract",
        )

    return (
        True,
        "ok",
    )


# ============================================================
# Parse Candidate
# ============================================================

def _parse_candidate(
    *,
    item: dict,
    topic_axis: str,
    query_name: str,
) -> dict:

    source_id = str(
        item.get(
            "id"
        )
    ).strip()


    downloads = (
        _extract_download_urls(
            item
        )
    )


    categories = (
        item.get(
            "subjectCategories",
            [],
        )
    )

    if not isinstance(
        categories,
        list,
    ):

        categories = [
            str(
                categories
            )
        ]


    center = (
        item.get(
            "center",
            {},
        )
    )

    if not isinstance(
        center,
        dict,
    ):

        center = {}


    record_url = (
        f"https://ntrs.nasa.gov/"
        f"citations/{source_id}"
    )


    api_score = (
        _safe_float(
            item.get(
                "score"
            )
        )
    )


    return {

        "source": (
            "ntrs"
        ),

        "source_id": (
            source_id
        ),

        "title": (
            _normalize_space(
                item.get(
                    "title"
                )
            )
        ),

        "abstract": (
            _normalize_space(
                item.get(
                    "abstract"
                )
            )
        ),

        "authors": (
            _extract_authors(
                item
            )
        ),

        "categories": (
            categories
        ),

        "published_at": (
            _extract_published_date(
                item
            )
        ),

        "document_type": (
            item.get(
                "stiType"
            )
        ),

        "document_type_details": (
            item.get(
                "stiTypeDetails"
            )
        ),

        "doi": (
            _extract_doi(
                item
            )
        ),

        "url": (
            record_url
        ),

        "pdf_url": (
            downloads[
                "pdf_url"
            ]
        ),

        "fulltext_url": (
            downloads[
                "fulltext_url"
            ]
        ),

        "original_url": (
            downloads[
                "original_url"
            ]
        ),

        "topic_axis": (
            topic_axis
        ),

        "language": (
            "en"
        ),

        "matched_queries": [
            query_name
        ],

        "query_scores": {

            query_name: (
                api_score
            ),
        },

        "api_score_max": (
            api_score
        ),

        "distribution": (
            item.get(
                "distribution"
            )
        ),

        "disseminated": (
            item.get(
                "disseminated"
            )
        ),

        "downloads_available": (
            item.get(
                "downloadsAvailable",
                False,
            )
        ),

        "only_abstract": (
            item.get(
                "onlyAbstract",
                False,
            )
        ),

        "downloads": (
            downloads[
                "downloads"
            ]
        ),

        "metadata": {

            "center_code": (
                center.get(
                    "code"
                )
            ),

            "center_name": (
                center.get(
                    "name"
                )
            ),

            "technical_review_type": (
                item.get(
                    "technicalReviewType"
                )
            ),

            "submitted_date": (
                item.get(
                    "submittedDate"
                )
            ),

            "distribution_date": (
                item.get(
                    "distributionDate"
                )
            ),

            "status": (
                item.get(
                    "status"
                )
            ),

            "keywords": (
                item.get(
                    "keywords",
                    [],
                )
            ),

            "other_report_numbers": (
                item.get(
                    "otherReportNumbers",
                    [],
                )
            ),

            "source_identifiers": (
                item.get(
                    "sourceIdentifiers",
                    [],
                )
            ),
        },
    }


# ============================================================
# Search One Query
# ============================================================

def search_single_query(
    *,
    topic_axis: str,
    query_name: str,
    query: str,
    max_results: int,
) -> list[dict]:

    print()
    print("-" * 70)

    print(
        f"[NTRS] Axis  : "
        f"{topic_axis}"
    )

    print(
        f"[NTRS] Query : "
        f"{query_name}"
    )

    print(
        f"[NTRS] Search: "
        f"{query}"
    )

    print("-" * 70)


    # ========================================================
    # Cache
    # ========================================================

    cached = (
        _load_query_cache(

            topic_axis=topic_axis,

            query_name=query_name,

            query=query,

            max_results=max_results,
        )
    )

    if cached is not None:

        print(
            f"[NTRS] Cached candidates: "
            f"{len(cached)}"
        )

        return cached


    # ========================================================
    # API
    # ========================================================

    payload = (
        _request_ntrs_search(

            query=query,

            max_results=max_results,
        )
    )


    stats = (
        payload.get(
            "stats",
            {},
        )
    )

    if not isinstance(
        stats,
        dict,
    ):

        stats = {}


    api_total = (
        stats.get(
            "total"
        )
    )


    raw_results = (
        payload.get(
            "results",
            [],
        )
    )

    if not isinstance(
        raw_results,
        list,
    ):

        raise RuntimeError(
            "NASA NTRS response has no "
            "valid results list."
        )


    print(
        f"[NTRS] API total matches : "
        f"{api_total}"
    )

    print(
        f"[NTRS] Returned page     : "
        f"{len(raw_results)}"
    )


    # ========================================================
    # Sanity Gate
    #
    # 검색어 하나를 넣었는데 NASA 전체 60만+ 결과가
    # 나오는 예전 오류를 즉시 잡는다.
    # ========================================================

    try:

        total_number = int(
            api_total
        )

    except (
        TypeError,
        ValueError,
    ):

        total_number = None


    if (
        total_number is not None
        and total_number > 500_000
    ):

        raise RuntimeError(
            "NTRS query appears to be ignored. "
            f"Focused query returned "
            f"{total_number} total matches."
        )


    # ========================================================
    # Filter
    # ========================================================

    documents: list[dict] = []


    rejected_reasons: dict[
        str,
        int,
    ] = {}


    for item in raw_results:

        if not isinstance(
            item,
            dict,
        ):

            continue


        usable, reason = (
            _candidate_is_usable(
                item
            )
        )

        if not usable:

            rejected_reasons[
                reason
            ] = (
                rejected_reasons.get(
                    reason,
                    0,
                )
                + 1
            )

            continue


        document = (
            _parse_candidate(

                item=item,

                topic_axis=topic_axis,

                query_name=query_name,
            )
        )


        # downloadsAvailable == True인데
        # 실제 usable URL이 하나도 없는 경우 제외

        if not any(
            [

                document.get(
                    "fulltext_url"
                ),

                document.get(
                    "pdf_url"
                ),

                document.get(
                    "original_url"
                ),
            ]
        ):

            rejected_reasons[
                "no_download_url"
            ] = (
                rejected_reasons.get(
                    "no_download_url",
                    0,
                )
                + 1
            )

            continue


        documents.append(
            document
        )


    print(
        f"[NTRS] Usable candidates : "
        f"{len(documents)}"
    )


    if rejected_reasons:

        print(
            f"[NTRS] Rejected          : "
            f"{rejected_reasons}"
        )


    # ========================================================
    # Cache
    # ========================================================

    _save_query_cache(

        topic_axis=topic_axis,

        query_name=query_name,

        query=query,

        max_results=max_results,

        documents=documents,

        api_total=api_total,

        rejected_reasons=(
            rejected_reasons
        ),
    )


    # ========================================================
    # Request interval
    # ========================================================

    time.sleep(
        settings.request_delay
    )


    return documents


# ============================================================
# Merge / Deduplicate
# ============================================================

def _merge_documents(
    documents: list[dict],
) -> list[dict]:

    merged: dict[
        str,
        dict,
    ] = {}


    for document in documents:

        source_id = str(
            document.get(
                "source_id",
                "",
            )
        ).strip()


        if not source_id:

            continue


        if source_id not in merged:

            merged[
                source_id
            ] = document

            continue


        existing = (
            merged[
                source_id
            ]
        )


        # ====================================================
        # Matched queries
        # ====================================================

        existing_queries = (
            existing.setdefault(
                "matched_queries",
                [],
            )
        )


        for query_name in document.get(
            "matched_queries",
            [],
        ):

            if (
                query_name
                not in existing_queries
            ):

                existing_queries.append(
                    query_name
                )


        # ====================================================
        # API scores
        # ====================================================

        existing_query_scores = (
            existing.setdefault(
                "query_scores",
                {},
            )
        )


        for (
            query_name,
            score,
        ) in document.get(
            "query_scores",
            {},
        ).items():

            previous_score = (
                _safe_float(
                    existing_query_scores.get(
                        query_name
                    )
                )
            )

            current_score = (
                _safe_float(
                    score
                )
            )

            if (
                current_score is not None
                and (
                    previous_score is None
                    or current_score
                    > previous_score
                )
            ):

                existing_query_scores[
                    query_name
                ] = current_score


        score_values = [

            score

            for score in (

                _safe_float(
                    value
                )

                for value
                in existing_query_scores.values()
            )

            if score is not None
        ]


        existing[
            "api_score_max"
        ] = (

            max(
                score_values
            )

            if score_values

            else None
        )


    return list(
        merged.values()
    )


# ============================================================
# Exclude Existing DB Documents
# ============================================================

def _exclude_existing_documents(
    documents: list[dict],
    existing_ntrs: dict[
        str,
        str,
    ],
) -> tuple[
    list[dict],
    list[dict],
]:

    eligible = []

    excluded = []


    for document in documents:

        source_id = (
            document[
                "source_id"
            ]
        )


        if source_id in existing_ntrs:

            excluded.append(
                {

                    "source_id": (
                        source_id
                    ),

                    "existing_axis": (
                        existing_ntrs[
                            source_id
                        ]
                    ),

                    "candidate_axis": (
                        document[
                            "topic_axis"
                        ]
                    ),

                    "title": (
                        document[
                            "title"
                        ]
                    ),
                }
            )

            continue


        eligible.append(
            document
        )


    return (
        eligible,
        excluded,
    )


# ============================================================
# Human Review Ranking
# ============================================================

def _keyword_hits(
    topic_axis: str,
    document: dict,
) -> list[str]:

    text = " ".join(
        [

            str(
                document.get(
                    "title",
                    "",
                )
            ),

            str(
                document.get(
                    "abstract",
                    "",
                )
            ),

            " ".join(
                str(
                    value
                )

                for value
                in document.get(
                    "categories",
                    [],
                )
            ),
        ]
    ).lower()


    hits = []


    for keyword in AXIS_KEYWORDS[
        topic_axis
    ]:

        if keyword.lower() in text:

            hits.append(
                keyword
            )


    return hits


def _decorate_for_review(
    topic_axis: str,
    documents: list[dict],
) -> list[dict]:

    decorated = []


    for document in documents:

        item = dict(
            document
        )


        hits = (
            _keyword_hits(
                topic_axis,
                document,
            )
        )


        item[
            "review_signals"
        ] = {

            "axis_keyword_hits": (
                hits
            ),

            "axis_keyword_hit_count": (
                len(hits)
            ),

            "matched_query_count": (
                len(
                    item.get(
                        "matched_queries",
                        [],
                    )
                )
            ),

            "api_score_max": (
                item.get(
                    "api_score_max"
                )
            ),
        }


        decorated.append(
            item
        )


    # ========================================================
    # Human review용 정렬
    #
    # 자동선정 아님.
    # ========================================================

    decorated.sort(

        key=lambda item: (

            item.get(
                "review_signals",
                {},
            ).get(
                "axis_keyword_hit_count",
                0,
            ),

            item.get(
                "review_signals",
                {},
            ).get(
                "matched_query_count",
                0,
            ),

            (
                item.get(
                    "review_signals",
                    {},
                ).get(
                    "api_score_max"
                )
                or 0.0
            ),

            item.get(
                "published_at"
            )
            or "",
        ),

        reverse=True,
    )


    return decorated


def _build_review_pool(
    topic_axis: str,
    documents: list[dict],
) -> list[dict]:

    target_count = (
        TARGET_NEW_COUNTS[
            topic_axis
        ]
    )


    desired_review_count = (
        target_count
        * REVIEW_POOL_MULTIPLIER
    )


    review_count = min(
        len(documents),
        desired_review_count,
    )


    return documents[
        :review_count
    ]


# ============================================================
# Search One Axis
# ============================================================

def search_axis(
    *,
    topic_axis: str,
    existing_ntrs: dict[
        str,
        str,
    ],
) -> dict:

    queries = (
        NTRS_QUERIES[
            topic_axis
        ]
    )


    print()
    print("=" * 70)

    print(
        f"[AXIS] "
        f"{topic_axis}"
    )

    print(
        f"[AXIS] Query count         : "
        f"{len(queries)}"
    )

    print(
        f"[AXIS] New target          : "
        f"{TARGET_NEW_COUNTS[topic_axis]}"
    )

    print("=" * 70)


    raw_documents = []


    for query_spec in queries:

        query_documents = (
            search_single_query(

                topic_axis=topic_axis,

                query_name=(
                    query_spec[
                        "name"
                    ]
                ),

                query=(
                    query_spec[
                        "query"
                    ]
                ),

                max_results=(
                    RESULTS_PER_QUERY
                ),
            )
        )


        raw_documents.extend(
            query_documents
        )


    # ========================================================
    # Dedup
    # ========================================================

    merged = (
        _merge_documents(
            raw_documents
        )
    )


    # ========================================================
    # Existing 15 exclusion
    # ========================================================

    (
        eligible,
        excluded_existing,
    ) = (
        _exclude_existing_documents(
            merged,
            existing_ntrs,
        )
    )


    # ========================================================
    # Human-review signals
    # ========================================================

    ranked = (
        _decorate_for_review(
            topic_axis,
            eligible,
        )
    )


    review_pool = (
        _build_review_pool(
            topic_axis,
            ranked,
        )
    )


    target_count = (
        TARGET_NEW_COUNTS[
            topic_axis
        ]
    )


    desired_review_count = (
        target_count
        * REVIEW_POOL_MULTIPLIER
    )


    print()
    print(
        f"[AXIS] Raw usable           : "
        f"{len(raw_documents)}"
    )

    print(
        f"[AXIS] Unique               : "
        f"{len(merged)}"
    )

    print(
        f"[AXIS] Existing DB excluded : "
        f"{len(excluded_existing)}"
    )

    print(
        f"[AXIS] Eligible new         : "
        f"{len(ranked)}"
    )

    print(
        f"[AXIS] Review pool          : "
        f"{len(review_pool)}"
    )

    print(
        f"[AXIS] Desired review pool  : "
        f"{desired_review_count}"
    )


    # ========================================================
    # Minimum Gate
    # ========================================================

    if (
        len(ranked)
        < target_count
    ):

        raise RuntimeError(
            f"{topic_axis}: "
            "not enough new candidates.\n"
            f"Need at least "
            f"{target_count}, "
            f"found "
            f"{len(ranked)}."
        )


    if (
        len(review_pool)
        < desired_review_count
    ):

        print(
            "[WARN] Review pool is smaller "
            "than the preferred 4x target."
        )

        print(
            "[WARN] Final selection is still "
            "possible if Human QA finds enough "
            "high-quality papers."
        )


    return {

        "raw_usable_count": (
            len(
                raw_documents
            )
        ),

        "unique_count": (
            len(
                merged
            )
        ),

        "excluded_existing": (
            excluded_existing
        ),

        "eligible_documents": (
            ranked
        ),

        "review_pool": (
            review_pool
        ),
    }


# ============================================================
# Save Axis Results
# ============================================================

def _save_axis_results(
    *,
    topic_axis: str,
    result: dict,
) -> dict:

    axis_dir = (
        CACHE_ROOT
        / topic_axis
    )


    axis_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    all_path = (
        axis_dir
        / "_all_candidates.json"
    )


    review_path = (
        axis_dir
        / "_review_pool.json"
    )


    # 이후 Resolver/selection과 호환용
    merged_compat_path = (
        axis_dir
        / "_merged_candidates.json"
    )


    # ========================================================
    # All eligible candidates
    # ========================================================

    all_payload = {

        "collector_version": (
            COLLECTOR_VERSION
        ),

        "source": (
            "ntrs"
        ),

        "topic_axis": (
            topic_axis
        ),

        "target_new_count": (
            TARGET_NEW_COUNTS[
                topic_axis
            ]
        ),

        "existing_db_excluded": (
            len(
                result[
                    "excluded_existing"
                ]
            )
        ),

        "eligible_new_count": (
            len(
                result[
                    "eligible_documents"
                ]
            )
        ),

        "documents": (
            result[
                "eligible_documents"
            ]
        ),
    }


    # ========================================================
    # Human QA Review Pool
    # ========================================================

    review_payload = {

        "collector_version": (
            COLLECTOR_VERSION
        ),

        "source": (
            "ntrs"
        ),

        "topic_axis": (
            topic_axis
        ),

        "target_new_count": (
            TARGET_NEW_COUNTS[
                topic_axis
            ]
        ),

        "review_pool_multiplier": (
            REVIEW_POOL_MULTIPLIER
        ),

        "review_pool_count": (
            len(
                result[
                    "review_pool"
                ]
            )
        ),

        "documents": (
            result[
                "review_pool"
            ]
        ),
    }


    # ========================================================
    # Compatibility file
    # ========================================================

    merged_payload = {

        "collector_version": (
            COLLECTOR_VERSION
        ),

        "source": (
            "ntrs"
        ),

        "topic_axis": (
            topic_axis
        ),

        "unique_candidates": (
            len(
                result[
                    "eligible_documents"
                ]
            )
        ),

        "documents": (
            result[
                "eligible_documents"
            ]
        ),
    }


    _save_json(
        all_path,
        all_payload,
    )


    _save_json(
        review_path,
        review_payload,
    )


    _save_json(
        merged_compat_path,
        merged_payload,
    )


    print(
        f"[REPORT] All candidates : "
        f"{all_path}"
    )

    print(
        f"[REPORT] Review pool    : "
        f"{review_path}"
    )

    print(
        f"[REPORT] Merged compat  : "
        f"{merged_compat_path}"
    )


    return {

        "all_candidates": (
            str(
                all_path
            )
        ),

        "review_pool": (
            str(
                review_path
            )
        ),

        "merged_candidates": (
            str(
                merged_compat_path
            )
        ),
    }


# ============================================================
# Cross-Axis Overlap
# ============================================================

def _find_cross_axis_overlaps(
    axis_results: dict[
        str,
        dict,
    ],
) -> list[dict]:

    ownership: dict[
        str,
        list[
            tuple[
                str,
                dict,
            ]
        ],
    ] = {}


    for (
        topic_axis,
        result,
    ) in axis_results.items():

        for document in result[
            "eligible_documents"
        ]:

            source_id = (
                document[
                    "source_id"
                ]
            )


            ownership.setdefault(
                source_id,
                [],
            ).append(
                (
                    topic_axis,
                    document,
                )
            )


    overlaps = []


    for (
        source_id,
        entries,
    ) in ownership.items():

        axes = [

            topic_axis

            for (
                topic_axis,
                _document,
            )
            in entries
        ]


        if (
            len(
                set(
                    axes
                )
            )
            <= 1
        ):

            continue


        first_document = (
            entries[
                0
            ][
                1
            ]
        )


        overlaps.append(
            {

                "source_id": (
                    source_id
                ),

                "title": (
                    first_document.get(
                        "title"
                    )
                ),

                "axes": (
                    sorted(
                        set(
                            axes
                        )
                    )
                ),
            }
        )


    overlaps.sort(

        key=lambda item: (
            item[
                "source_id"
            ]
        )
    )


    return overlaps


# ============================================================
# Console Preview
# ============================================================

def print_review_candidates(
    *,
    topic_axis: str,
    documents: list[dict],
) -> None:

    print()
    print("=" * 70)

    print(
        f"{topic_axis} - "
        f"Human Review Pool "
        f"({len(documents)})"
    )

    print("=" * 70)


    preview_documents = (
        documents[
            :CONSOLE_PREVIEW_LIMIT
        ]
    )


    for (
        index,
        document,
    ) in enumerate(
        preview_documents,
        start=1,
    ):

        signals = (
            document.get(
                "review_signals",
                {},
            )
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
            f"    Type: "
            f"{document['document_type']}"
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
            f"    Axis keyword hits: "
            f"{signals.get('axis_keyword_hit_count', 0)}"
        )

        print(
            f"    API score max: "
            f"{signals.get('api_score_max')}"
        )


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
                abstract[
                    :350
                ]
                + "..."
            )


        print(
            f"    Abstract: "
            f"{abstract}"
        )

        print(
            f"    Citation: "
            f"{document['url']}"
        )

        print(
            f"    TXT candidate: "
            f"{document['fulltext_url']}"
        )

        print(
            f"    Original candidate: "
            f"{document['original_url']}"
        )

        print(
            f"    PDF candidate: "
            f"{document['pdf_url']}"
        )


    if (
        len(documents)
        > CONSOLE_PREVIEW_LIMIT
    ):

        print()

        print(
            f"... console preview limited "
            f"to "
            f"{CONSOLE_PREVIEW_LIMIT}."
        )

        print(
            "Review the JSON file for "
            "the complete pool."
        )


# ============================================================
# Global Report
# ============================================================

def _save_global_report(
    *,
    existing_ntrs: dict[
        str,
        str,
    ],
    axis_results: dict[
        str,
        dict,
    ],
    axis_paths: dict[
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

        "source": (
            "ntrs"
        ),

        "current_db": {

            "existing_ntrs_count": (
                len(
                    existing_ntrs
                )
            ),

            "existing_ntrs_ids": (
                sorted(
                    existing_ntrs.keys()
                )
            ),
        },

        "target_new_counts": (
            TARGET_NEW_COUNTS
        ),

        "target_new_total": (
            TARGET_NEW_TOTAL
        ),

        "results_per_query": (
            RESULTS_PER_QUERY
        ),

        "review_pool_multiplier": (
            REVIEW_POOL_MULTIPLIER
        ),

        "axis_results": {

            topic_axis: {

                "query_count": (
                    len(
                        NTRS_QUERIES[
                            topic_axis
                        ]
                    )
                ),

                "raw_usable_count": (
                    result[
                        "raw_usable_count"
                    ]
                ),

                "unique_count": (
                    result[
                        "unique_count"
                    ]
                ),

                "existing_db_excluded": (
                    len(
                        result[
                            "excluded_existing"
                        ]
                    )
                ),

                "eligible_new_count": (
                    len(
                        result[
                            "eligible_documents"
                        ]
                    )
                ),

                "review_pool_count": (
                    len(
                        result[
                            "review_pool"
                        ]
                    )
                ),

                "paths": (
                    axis_paths[
                        topic_axis
                    ]
                ),
            }

            for (
                topic_axis,
                result,
            )
            in axis_results.items()
        },

        "cross_axis_overlap_count": (
            len(
                overlaps
            )
        ),

        "cross_axis_overlaps": (
            overlaps
        ),
    }


    _save_json(
        REPORT_FILE,
        payload,
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "TEAM B - NASA NTRS "
        "Core-100 Candidate Collector"
    )

    print("=" * 70)

    print(
        f"Version        : "
        f"{COLLECTOR_VERSION}"
    )

    print(
        f"API            : "
        f"{NTRS_SEARCH_URL}"
    )

    print(
        f"Results/query  : "
        f"{RESULTS_PER_QUERY}"
    )

    print(
        f"Request delay  : "
        f"{settings.request_delay} sec"
    )

    print(
        f"Cache root     : "
        f"{CACHE_ROOT}"
    )


    print()
    print(
        "New NTRS target:"
    )


    for (
        topic_axis,
        count,
    ) in TARGET_NEW_COUNTS.items():

        print(
            f"  "
            f"{topic_axis:22} : "
            f"{count}"
        )


    print(
        f"  "
        f"{'TOTAL':22} : "
        f"{TARGET_NEW_TOTAL}"
    )


    # ========================================================
    # 1. Existing DB exclusion list
    # ========================================================

    existing_ntrs = (
        _load_existing_ntrs_from_db()
    )


    # ========================================================
    # 2. Collection
    # ========================================================

    axis_results = {}

    axis_paths = {}


    for topic_axis in NTRS_QUERIES:

        try:

            result = (
                search_axis(

                    topic_axis=(
                        topic_axis
                    ),

                    existing_ntrs=(
                        existing_ntrs
                    ),
                )
            )


            axis_results[
                topic_axis
            ] = result


            axis_paths[
                topic_axis
            ] = (
                _save_axis_results(

                    topic_axis=(
                        topic_axis
                    ),

                    result=(
                        result
                    ),
                )
            )


            print_review_candidates(

                topic_axis=(
                    topic_axis
                ),

                documents=(
                    result[
                        "review_pool"
                    ]
                ),
            )


        except Exception as exc:

            print()
            print("=" * 70)

            print(
                f"[STOP] Failed axis: "
                f"{topic_axis}"
            )

            print(
                f"[STOP] "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print()

            print(
                "Completed query caches "
                "have been preserved."
            )

            print("=" * 70)

            raise


    # ========================================================
    # 3. Cross-axis overlap
    # ========================================================

    overlaps = (
        _find_cross_axis_overlaps(
            axis_results
        )
    )


    # ========================================================
    # 4. Global report
    # ========================================================

    _save_global_report(

        existing_ntrs=(
            existing_ntrs
        ),

        axis_results=(
            axis_results
        ),

        axis_paths=(
            axis_paths
        ),

        overlaps=(
            overlaps
        ),
    )


    # ========================================================
    # 5. Final Summary
    # ========================================================

    print()
    print("=" * 70)

    print(
        "NTRS CORE-100 CANDIDATE "
        "COLLECTION COMPLETED"
    )

    print("=" * 70)


    for (
        topic_axis,
        result,
    ) in axis_results.items():

        print(
            f"{topic_axis:22} | "
            f"eligible "
            f"{len(result['eligible_documents']):3} | "
            f"review "
            f"{len(result['review_pool']):3} | "
            f"need "
            f"{TARGET_NEW_COUNTS[topic_axis]:2}"
        )


    print("-" * 70)


    print(
        f"Existing NTRS excluded : "
        f"{len(existing_ntrs)}"
    )

    print(
        f"Cross-axis overlaps    : "
        f"{len(overlaps)}"
    )

    print(
        f"Report                 : "
        f"{REPORT_FILE}"
    )


    print()
    print(
        "[PASS] Candidate pools are ready "
        "for Human QA."
    )


    print()
    print(
        "NEXT:"
    )

    print(
        "1. Review each "
        "_review_pool.json"
    )

    print(
        "2. Select:"
    )

    print(
        "   rover_autonomy       12"
    )

    print(
        "   onboard_ai           12"
    )

    print(
        "   satellite_autonomy   11"
    )

    print(
        "3. Do not select the same "
        "NTRS ID for multiple axes."
    )

    print(
        "4. Do NOT download/resolve "
        "full text until selection is frozen."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()