import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urljoin

import httpx

from app.config import PROJECT_ROOT, settings


# ============================================================
# NASA NTRS API
# ============================================================

NTRS_API_ROOT = "https://ntrs.nasa.gov/api"

NTRS_SEARCH_URL = (
    f"{NTRS_API_ROOT}/citations/search"
)

NTRS_PUBLIC_ROOT = (
    "https://ntrs.nasa.gov"
)


# ============================================================
# Cache Version
# ============================================================
#
# v1:
#   POST + params 방식으로 검색 조건이 제대로
#   적용되지 않은 cache가 생성됨.
#
# v2:
#   NASA 공식 OpenAPI 방식대로 JSON body 사용.
#
# 따라서 기존 잘못된 cache를 자동으로 무시한다.
# ============================================================

CACHE_VERSION = "v2"


# ============================================================
# Pilot Settings
# ============================================================

RESULTS_PER_QUERY = 20


# ============================================================
# Search Queries
# ============================================================
#
# NASA NTRS 검색 문법:
#
# space = AND
# |     = OR
# "..." = exact phrase
#
# 목적:
# 각 axis마다 최종 5편을 사람이 선정할 수 있는
# 후보 pool 확보.
# ============================================================

NTRS_QUERIES = {

    # ========================================================
    # 1. Rover Autonomy
    # ========================================================

    "rover_autonomy": [

        {
            "name": "planetary_rover_navigation",
            "query": (
                '"planetary rover" '
                '"autonomous navigation"'
            ),
        },

        {
            "name": "mars_rover_navigation",
            "query": (
                '"Mars rover" '
                '"autonomous navigation"'
            ),
        },

        {
            "name": "lunar_rover_navigation",
            "query": (
                '"lunar rover" '
                '"autonomous navigation"'
            ),
        },

        {
            "name": "planetary_path_planning",
            "query": (
                '"planetary rover" '
                '"path planning"'
            ),
        },

        {
            "name": "rover_hazard_avoidance",
            "query": (
                'rover '
                '"hazard avoidance"'
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
                'spacecraft '
                '"onboard autonomy"'
            ),
        },

        {
            "name": "autonomous_spacecraft",
            "query": (
                '"autonomous spacecraft"'
            ),
        },

        {
            "name": "deep_space_autonomy",
            "query": (
                '"deep space" '
                'spacecraft '
                'autonomy'
            ),
        },

        {
            "name": "onboard_ai",
            "query": (
                'spacecraft '
                '("onboard AI"|'
                '"onboard artificial intelligence")'
            ),
        },

        {
            "name": "autonomous_decision",
            "query": (
                'spacecraft '
                '"autonomous decision"'
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
                'satellite '
                '"fault detection"'
            ),
        },

        {
            "name": "spacecraft_fault_detection",
            "query": (
                'spacecraft '
                '"fault detection"'
            ),
        },

        {
            "name": "fault_diagnosis",
            "query": (
                'spacecraft '
                '"fault diagnosis"'
            ),
        },

        {
            "name": "fault_recovery",
            "query": (
                'spacecraft '
                '"fault recovery"'
            ),
        },

        {
            "name": "fault_management",
            "query": (
                'spacecraft '
                '"fault management"'
            ),
        },

        {
            "name": "autonomous_operations",
            "query": (
                'spacecraft '
                '"autonomous operations"'
            ),
        },
    ],
}


# ============================================================
# Allowed STI Types
# ============================================================
#
# NASA 공식 STI Type domain values 중
# 논문/기술보고서 계열만 유지.
#
# Presentation / Poster / Video / Other는 제외.
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
# Paths
# ============================================================

CACHE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "ntrs"
    / CACHE_VERSION
)


# ============================================================
# HTTP Headers
# ============================================================

HEADERS = {
    "User-Agent": (
        "TEAM-B-University-Research-Project/1.0"
    ),
    "Accept": "application/json",
    "Content-Type": "application/json",
}


# ============================================================
# Text Helpers
# ============================================================

def _normalize_space(
    value,
) -> str:

    if value is None:
        return ""

    return " ".join(
        str(value).split()
    )


# ============================================================
# URL Helper
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
# Cache Helpers
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
    topic_axis: str,
    query_name: str,
    query: str,
    max_results: int,
) -> list[dict] | None:

    path = _cache_path(
        topic_axis,
        query_name,
        query,
    )

    if not path.exists():
        return None

    try:

        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except Exception:
        return None

    if (
        payload.get(
            "cache_version"
        )
        != CACHE_VERSION
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

    documents = payload.get(
        "documents"
    )

    if not isinstance(
        documents,
        list,
    ):
        return None

    print(
        f"[CACHE] Loaded: {path}"
    )

    return documents


def _save_query_cache(
    topic_axis: str,
    query_name: str,
    query: str,
    max_results: int,
    documents: list[dict],
    api_total: int | None,
    returned_count: int,
) -> None:

    path = _cache_path(
        topic_axis,
        query_name,
        query,
    )

    payload = {
        "cache_version": (
            CACHE_VERSION
        ),

        "source": "ntrs",

        "topic_axis": (
            topic_axis
        ),

        "query_name": (
            query_name
        ),

        "query": query,

        "max_results": (
            max_results
        ),

        "api_total": (
            api_total
        ),

        "returned_count": (
            returned_count
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
        f"[CACHE] Saved: {path}"
    )


# ============================================================
# NASA NTRS HTTP Search
# ============================================================

def _build_search_body(
    query: str,
    max_results: int,
) -> dict:
    """
    NASA OpenAPI POST /citations/search용
    JSON request body.

    핵심 수정:
    이전 params= 방식이 아니라
    JSON body 자체에 검색조건을 넣는다.
    """

    return {
        "q": query,

        "disseminated": (
            "DOCUMENT_AND_METADATA"
        ),

        "page": {
            "size": min(
                max_results,
                100,
            ),

            "from": 0,
        },
    }


def _do_post(
    body: dict,
) -> httpx.Response:

    return httpx.post(
        NTRS_SEARCH_URL,
        json=body,
        timeout=settings.request_timeout,
        follow_redirects=True,
        headers=HEADERS,
    )


def _request_ntrs_search(
    query: str,
    max_results: int,
) -> dict:
    """
    NASA NTRS 검색.

    POST + JSON body.

    429가 발생하고 Retry-After가 있을 경우
    한 번만 재시도한다.
    """

    body = _build_search_body(
        query=query,
        max_results=max_results,
    )

    response = _do_post(
        body
    )

    # ========================================================
    # Rate Limit
    # ========================================================

    if response.status_code == 429:

        print()
        print(
            "[NTRS] HTTP 429 rate limit detected."
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
                "NASA NTRS returned HTTP 429 "
                "without Retry-After."
            )

        wait_seconds = int(
            retry_after
        )

        print(
            f"[NTRS] Waiting "
            f"{wait_seconds} seconds..."
        )

        time.sleep(
            wait_seconds
        )

        response = _do_post(
            body
        )

        if response.status_code == 429:

            raise RuntimeError(
                "NASA NTRS rate limit "
                "is still active."
            )

    # ========================================================
    # Other HTTP Errors
    # ========================================================

    if not response.is_success:

        preview = (
            response.text[:1000]
            if response.text
            else ""
        )

        raise RuntimeError(
            f"NTRS HTTP "
            f"{response.status_code}\n"
            f"Response: {preview}"
        )

    # ========================================================
    # JSON
    # ========================================================

    try:

        payload = (
            response.json()
        )

    except Exception as exc:

        raise RuntimeError(
            "NASA NTRS response was not JSON."
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):

        raise RuntimeError(
            "Unexpected NASA NTRS "
            "response structure."
        )

    return payload


# ============================================================
# Metadata Helpers
# ============================================================

def _extract_authors(
    item: dict,
) -> list[str]:

    authors: list[str] = []

    affiliations = item.get(
        "authorAffiliations",
        [],
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

        meta = affiliation.get(
            "meta",
            {},
        )

        if not isinstance(
            meta,
            dict,
        ):
            continue

        author = meta.get(
            "author",
            {},
        )

        if not isinstance(
            author,
            dict,
        ):
            continue

        name = _normalize_space(
            author.get(
                "name"
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

    identifiers = item.get(
        "sourceIdentifiers",
        [],
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

            identifier_type = str(
                identifier.get(
                    "type",
                    "",
                )
            ).upper()

            if identifier_type != "DOI":
                continue

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

    publications = item.get(
        "publications",
        [],
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

            doi = publication.get(
                "doi"
            )

            if doi:
                return str(
                    doi
                )

    return None


def _extract_published_date(
    item: dict,
) -> str | None:

    publications = item.get(
        "publications",
        [],
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

            value = publication.get(
                "publicationDate"
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

        value = item.get(
            key
        )

        if value:

            return str(
                value
            )[:10]

    return None


# ============================================================
# Download Metadata
# ============================================================

def _extract_download_urls(
    item: dict,
) -> dict:
    """
    검색 응답 안에 downloads 정보가 있으면
    후보 URL을 미리 저장한다.

    실제 사용 가능 여부는 다음 resolver에서
    다시 확인한다.
    """

    fulltext_url = None
    pdf_url = None
    original_url = None

    downloads = item.get(
        "downloads",
        [],
    )

    if not isinstance(
        downloads,
        list,
    ):
        downloads = []

    summary = []

    for download in downloads:

        if not isinstance(
            download,
            dict,
        ):
            continue

        links = download.get(
            "links",
            {},
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

        summary.append(
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
            summary
        ),
    }


# ============================================================
# Candidate Filter
# ============================================================

def _candidate_is_usable(
    item: dict,
) -> tuple[bool, str]:
    """
    collector 단계에서는 너무 공격적으로
    full-text 여부를 필터링하지 않는다.

    필수 조건:
      - PUBLIC
      - DOCUMENT_AND_METADATA
      - 학술/기술 STI type
      - title 존재

    downloadsAvailable / onlyAbstract는
    후보 metadata로만 기록하고,
    실제 full text 여부는 resolver에서 확인한다.
    """

    distribution = str(
        item.get(
            "distribution",
            "",
        )
    ).upper()

    if distribution != "PUBLIC":

        return (
            False,
            "not_public",
        )

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

    title = _normalize_space(
        item.get(
            "title"
        )
    )

    if not title:

        return (
            False,
            "no_title",
        )

    return (
        True,
        "ok",
    )


# ============================================================
# Parse Candidate
# ============================================================

def _parse_candidate(
    item: dict,
    topic_axis: str,
    query_name: str,
) -> dict:

    source_id = str(
        item.get(
            "id"
        )
    )

    downloads = (
        _extract_download_urls(
            item
        )
    )

    categories = item.get(
        "subjectCategories",
        [],
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

    center = item.get(
        "center",
        {},
    )

    if not isinstance(
        center,
        dict,
    ):

        center = {}

    raw_score = (
        item.get(
            "_meta",
            {},
        )
        if isinstance(
            item.get(
                "_meta",
                {},
            ),
            dict,
        )
        else {}
    )

    api_score = (
        raw_score.get(
            "score"
        )
    )

    record_url = (
        f"https://ntrs.nasa.gov/"
        f"citations/{source_id}"
    )

    return {
        "source": "ntrs",

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

        "language": "en",

        "matched_queries": [
            query_name
        ],

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
                "downloadsAvailable"
            )
        ),

        "only_abstract": (
            item.get(
                "onlyAbstract"
            )
        ),

        "api_score": (
            api_score
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
                    []
                )
            ),

            "other_report_numbers": (
                item.get(
                    "otherReportNumbers",
                    []
                )
            ),

            "source_identifiers": (
                item.get(
                    "sourceIdentifiers",
                    []
                )
            ),
        },
    }


# ============================================================
# Search One Query
# ============================================================

def search_single_query(
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

    cached = _load_query_cache(
        topic_axis=topic_axis,
        query_name=query_name,
        query=query,
        max_results=max_results,
    )

    if cached is not None:

        print(
            f"[NTRS] Cached candidates: "
            f"{len(cached)}"
        )

        return cached

    # ========================================================
    # NASA API
    # ========================================================

    payload = (
        _request_ntrs_search(
            query=query,
            max_results=max_results,
        )
    )

    stats = payload.get(
        "stats",
        {},
    )

    if not isinstance(
        stats,
        dict,
    ):
        stats = {}

    api_total = stats.get(
        "total"
    )

    raw_results = payload.get(
        "results",
        [],
    )

    if not isinstance(
        raw_results,
        list,
    ):

        raise RuntimeError(
            "NASA NTRS response does not "
            "contain a results list."
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
    # Sanity Check
    # ========================================================
    #
    # 이전 잘못된 호출은 모든 query가
    # 전체 repository count 약 647k를 반환했다.
    #
    # 지금 query가 매우 구체적인데도 다시
    # 60만+가 나오면 요청이 적용되지 않은 것일
    # 가능성이 높으므로 경고.
    # ========================================================

    try:

        total_int = int(
            api_total
        )

    except (
        TypeError,
        ValueError,
    ):

        total_int = None

    if (
        total_int is not None
        and total_int > 600000
        and query.strip()
    ):

        print()
        print(
            "[WARNING] Search total is suspiciously large."
        )

        print(
            "[WARNING] Verify that NASA applied the query."
        )

    # ========================================================
    # Candidate Filter
    # ========================================================

    documents: list[dict] = []

    rejected_reasons: dict[
        str,
        int
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
    # Save Cache
    # ========================================================

    _save_query_cache(
        topic_axis=topic_axis,
        query_name=query_name,
        query=query,
        max_results=max_results,
        documents=documents,
        api_total=api_total,
        returned_count=len(
            raw_results
        ),
    )

    # ========================================================
    # Delay
    # ========================================================

    time.sleep(
        settings.request_delay
    )

    return documents


# ============================================================
# Merge
# ============================================================

def _merge_documents(
    documents: list[dict],
) -> list[dict]:
    """
    NTRS source_id 기준 dedup.

    같은 문서가 여러 query에서 검색되면
    matched_queries만 합친다.
    """

    merged: dict[
        str,
        dict
    ] = {}

    for document in documents:

        source_id = (
            document.get(
                "source_id"
            )
        )

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

        # 더 높은 API score가 있다면 보존
        old_score = (
            existing.get(
                "api_score"
            )
        )

        new_score = (
            document.get(
                "api_score"
            )
        )

        try:

            old_score_value = float(
                old_score
                or 0
            )

        except (
            TypeError,
            ValueError,
        ):

            old_score_value = 0.0

        try:

            new_score_value = float(
                new_score
                or 0
            )

        except (
            TypeError,
            ValueError,
        ):

            new_score_value = 0.0

        if (
            new_score_value
            > old_score_value
        ):

            existing[
                "api_score"
            ] = new_score

    # ========================================================
    # Human review priority
    # ========================================================
    #
    # 여러 query에 동시에 걸린 문서 우선.
    # 같은 경우 API relevance score 활용.
    # ========================================================

    merged_list = list(
        merged.values()
    )

    def _sort_key(
        document: dict,
    ):

        query_count = len(
            document.get(
                "matched_queries",
                [],
            )
        )

        try:

            score = float(
                document.get(
                    "api_score"
                )
                or 0
            )

        except (
            TypeError,
            ValueError,
        ):

            score = 0.0

        return (
            query_count,
            score,
        )

    merged_list.sort(
        key=_sort_key,
        reverse=True,
    )

    return merged_list


# ============================================================
# Search Axis
# ============================================================

def search_axis(
    topic_axis: str,
) -> list[dict]:

    queries = (
        NTRS_QUERIES[
            topic_axis
        ]
    )

    print()
    print("=" * 70)

    print(
        f"[AXIS] {topic_axis}"
    )

    print(
        f"[AXIS] Query count: "
        f"{len(queries)}"
    )

    print("=" * 70)

    raw_documents: list[dict] = []

    for query_spec in queries:

        documents = (
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
            documents
        )

    merged = _merge_documents(
        raw_documents
    )

    print()
    print(
        f"[AXIS] Raw usable candidates : "
        f"{len(raw_documents)}"
    )

    print(
        f"[AXIS] Unique candidates     : "
        f"{len(merged)}"
    )

    return merged


# ============================================================
# Save Axis Report
# ============================================================

def _save_axis_report(
    topic_axis: str,
    documents: list[dict],
) -> Path:

    axis_dir = (
        CACHE_ROOT
        / topic_axis
    )

    axis_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        axis_dir
        / "_merged_candidates.json"
    )

    payload = {
        "cache_version": (
            CACHE_VERSION
        ),

        "source": "ntrs",

        "topic_axis": (
            topic_axis
        ),

        "unique_candidates": (
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
        f"[REPORT] Saved: {path}"
    )

    return path


# ============================================================
# Console Candidate Preview
# ============================================================

def print_candidates(
    topic_axis: str,
    documents: list[dict],
) -> None:

    print()
    print("=" * 70)

    print(
        f"{topic_axis} - "
        f"{len(documents)} unique candidates"
    )

    print("=" * 70)

    for index, document in enumerate(
        documents,
        start=1,
    ):

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

        categories = (
            document.get(
                "categories",
                [],
            )
        )

        print(
            f"    Categories: "
            f"{', '.join(map(str, categories))}"
        )

        print(
            f"    Matched queries: "
            f"{', '.join(document['matched_queries'])}"
        )

        print(
            f"    API score: "
            f"{document.get('api_score')}"
        )

        print(
            f"    Downloads available: "
            f"{document.get('downloads_available')}"
        )

        print(
            f"    onlyAbstract: "
            f"{document.get('only_abstract')}"
        )

        abstract = (
            document.get(
                "abstract",
                "",
            )
        )

        if len(
            abstract
        ) > 500:

            abstract = (
                abstract[:500]
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


# ============================================================
# Main
# ============================================================

def main():
    """
    NASA NTRS metadata 후보 수집.

    현재 단계에서는:

    O metadata search
    O axis별 후보 생성
    O dedup
    O cache

    아직:

    X fulltext 다운로드
    X PDF parsing
    X cleaning
    X AWS DB insert

    최종 5편/axis는 사람이 검토 후 선정한다.
    """

    print()
    print("=" * 70)

    print(
        "TEAM B - NASA NTRS Metadata Collection v2"
    )

    print("=" * 70)

    print(
        f"API            : "
        f"{NTRS_SEARCH_URL}"
    )

    print(
        f"Cache version  : "
        f"{CACHE_VERSION}"
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

    all_results: dict[
        str,
        list[dict]
    ] = {}

    for topic_axis in NTRS_QUERIES:

        try:

            documents = (
                search_axis(
                    topic_axis
                )
            )

            _save_axis_report(
                topic_axis,
                documents,
            )

            print_candidates(
                topic_axis,
                documents,
            )

            all_results[
                topic_axis
            ] = documents

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
                "Completed v2 query caches "
                "have been preserved."
            )

            print("=" * 70)

            return

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 70)

    print(
        "NTRS metadata collection completed."
    )

    print("=" * 70)

    total = 0

    for (
        topic_axis,
        documents,
    ) in all_results.items():

        count = len(
            documents
        )

        total += count

        print(
            f"{topic_axis:22} : "
            f"{count}"
        )

    print("-" * 70)

    print(
        f"Axis-level unique total : "
        f"{total}"
    )

    print()
    print(
        "Next:"
    )

    print(
        "Review candidates and select "
        "5 papers per axis."
    )

    print(
        "Do NOT download full text yet."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()