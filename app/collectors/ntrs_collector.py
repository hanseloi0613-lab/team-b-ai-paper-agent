from __future__ import annotations

import json
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import quote, unquote

import httpx


# ============================================================
# TEAM B - NASA NTRS Collector
# ============================================================

NTRS_API_ROOT = "https://ntrs.nasa.gov/api"

HEADERS = {
    "User-Agent": "TEAM-B-University-Research-Project/1.0",
    "Accept": "application/json",
}

SPACE_ANCHORS = (
    "spacecraft",
    "satellite",
    "space mission",
    "space missions",
    "deep space",
    "interplanetary",
    "space system",
    "space systems",
    "orbital",
    "onboard",
    "on-board",
)

INTELLIGENCE_ANCHORS = (
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
    "mission management",
    "resource management",
    "guidance",
    "navigation",
    "control",
)

SKIP_STI_TYPES = (
    "PRESENTATION",
    "POSTER",
    "VIDEO",
    "IMAGE",
    "PATENT",
)


# ============================================================
# Helpers
# ============================================================

def safe_text(value: Any) -> str:
    if value is None:
        return ""

    return (
        str(value)
        .encode(
            "utf-8",
            errors="replace",
        )
        .decode(
            "utf-8",
            errors="replace",
        )
    )


def normalize_text(value: Any) -> str:
    return " ".join(
        safe_text(
            value
        ).split()
    )


def _request_json(
    client: httpx.Client,
    url: str,
    *,
    params: dict | None = None,
) -> Any:

    delays = (
        5,
        15,
        30,
        60,
    )

    last_error: Exception | None = None

    for attempt in range(
        len(delays) + 1
    ):
        try:
            response = client.get(
                url,
                params=params,
            )

            if response.status_code in (
                429,
                500,
                502,
                503,
                504,
            ):
                raise RuntimeError(
                    f"HTTP {response.status_code}"
                )

            response.raise_for_status()

            return response.json()

        except FileNotFoundError as exc:
            # 404 is permanent for this exact URL.
            # Do not waste 5+15+30+60 seconds retrying it.
            raise RuntimeError(
                f"NTRS PDF not found: {exc}"
            ) from exc

        except Exception as exc:
            last_error = exc

            if attempt >= len(delays):
                break

            wait_seconds = delays[
                attempt
            ]

            print(
                f"[NTRS RETRY] "
                f"attempt={attempt + 1} "
                f"wait={wait_seconds}s "
                f"reason={exc}",
                flush=True,
            )

            time.sleep(
                wait_seconds
            )

    raise RuntimeError(
        f"NTRS request failed: {last_error}"
    )


def _recursive_pdf_names(
    value: Any,
) -> list[str]:

    results: list[str] = []

    if isinstance(
        value,
        str,
    ):
        text = value.strip()

        if (
            text.lower()
            .split("?")[0]
            .endswith(".pdf")
        ):
            # URL인 경우 filename 부분만 이용.
            if "/" in text:
                text = (
                    text
                    .split("?")[0]
                    .rstrip("/")
                    .split("/")[-1]
                )

            results.append(
                text
            )

    elif isinstance(
        value,
        dict,
    ):
        for key, item in value.items():

            key_text = str(
                key
            ).lower()

            if (
                key_text
                in {
                    "name",
                    "filename",
                    "file",
                    "file_name",
                }
                and isinstance(
                    item,
                    str,
                )
                and item.lower()
                .split("?")[0]
                .endswith(".pdf")
            ):
                results.append(
                    item
                    .split("?")[0]
                    .rstrip("/")
                    .split("/")[-1]
                )

            results.extend(
                _recursive_pdf_names(
                    item
                )
            )

    elif isinstance(
        value,
        list,
    ):
        for item in value:
            results.extend(
                _recursive_pdf_names(
                    item
                )
            )

    # 순서 보존 dedup.
    seen: set[str] = set()
    output: list[str] = []

    for name in results:
        name = normalize_text(
            name
        )

        if (
            name
            and name not in seen
        ):
            seen.add(
                name
            )
            output.append(
                name
            )

    return output


# ============================================================
# Metadata Parsing
# ============================================================

def extract_authors(
    citation: dict,
) -> list[str]:

    output: list[str] = []

    affiliations = (
        citation.get(
            "authorAffiliations"
        )
        or []
    )

    if isinstance(
        affiliations,
        list,
    ):
        for item in affiliations:

            if not isinstance(
                item,
                dict,
            ):
                continue

            meta = (
                item.get(
                    "meta"
                )
                or {}
            )

            author = (
                meta.get(
                    "author"
                )
                or {}
            )

            if isinstance(
                author,
                dict,
            ):
                name = normalize_text(
                    author.get(
                        "name"
                    )
                )

                if (
                    name
                    and name not in output
                ):
                    output.append(
                        name
                    )

    # API 버전에 따라 authors 직접 반환 가능성도 처리.
    authors = (
        citation.get(
            "authors"
        )
        or []
    )

    if isinstance(
        authors,
        list,
    ):
        for author in authors:

            if isinstance(
                author,
                dict,
            ):
                name = normalize_text(
                    author.get(
                        "name"
                    )
                )
            else:
                name = normalize_text(
                    author
                )

            if (
                name
                and name not in output
            ):
                output.append(
                    name
                )

    return output


def extract_published_at(
    citation: dict,
) -> str | None:

    direct_candidates = (
        citation.get(
            "publicationDate"
        ),
        citation.get(
            "published"
        ),
        citation.get(
            "distributionDate"
        ),
    )

    for value in direct_candidates:

        text = normalize_text(
            value
        )

        if text:
            return text[:10]

    publications = (
        citation.get(
            "publications"
        )
        or []
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

            for key in (
                "publicationDate",
                "issuePublicationDate",
                "date",
            ):
                text = normalize_text(
                    publication.get(
                        key
                    )
                )

                if text:
                    return text[:10]

    return None


def metadata_text(
    citation: dict,
) -> str:

    keywords = (
        citation.get(
            "keywords"
        )
        or []
    )

    categories = (
        citation.get(
            "subjectCategories"
        )
        or []
    )

    pieces = [
        citation.get(
            "title"
        ),
        citation.get(
            "abstract"
        ),
    ]

    if isinstance(
        keywords,
        list,
    ):
        pieces.extend(
            keywords
        )

    if isinstance(
        categories,
        list,
    ):
        pieces.extend(
            categories
        )

    return " ".join(
        normalize_text(
            item
        )
        for item in pieces
        if item is not None
    ).lower()


def onboard_relevance(
    citation: dict,
) -> tuple[
    bool,
    int,
]:

    text = metadata_text(
        citation
    )

    space_hits = sum(
        1
        for term
        in SPACE_ANCHORS
        if term in text
    )

    intelligence_hits = sum(
        1
        for term
        in INTELLIGENCE_ANCHORS
        if term in text
    )

    score = (
        space_hits
        + intelligence_hits
    )

    passed = (
        space_hits >= 1
        and intelligence_hits >= 1
    )

    return (
        passed,
        score,
    )


def acceptable_document_type(
    citation: dict,
) -> bool:

    sti_type = normalize_text(
        citation.get(
            "stiType"
        )
    ).upper()

    if not sti_type:
        return True

    return not any(
        item in sti_type
        for item
        in SKIP_STI_TYPES
    )


# ============================================================
# Search
# ============================================================

def search_citations(
    client: httpx.Client,
    *,
    query: str,
    size: int = 50,
    offset: int = 0,
) -> tuple[
    list[dict],
    int,
]:

    params = {
        "q": query,
        "disseminated": (
            "DOCUMENT_AND_METADATA"
        ),
        "page.size": min(
            int(size),
            100,
        ),
        "page.from": max(
            int(offset),
            0,
        ),
        "sort.field": "id",
        "sort.order": "desc",
    }

    payload = _request_json(
        client,
        (
            f"{NTRS_API_ROOT}"
            "/citations/search"
        ),
        params=params,
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            "Unexpected NTRS search response"
        )

    results = (
        payload.get(
            "results"
        )
        or []
    )

    if not isinstance(
        results,
        list,
    ):
        results = []

    stats = (
        payload.get(
            "stats"
        )
        or {}
    )

    try:
        total = int(
            stats.get(
                "total",
                len(results),
            )
            or 0
        )
    except (
        TypeError,
        ValueError,
    ):
        total = len(
            results
        )

    print(
        f"[NTRS SEARCH] "
        f"q={query!r} "
        f"received={len(results)} "
        f"total={total}",
        flush=True,
    )

    return (
        results,
        total,
    )


# ============================================================
# Downloads
# ============================================================

def get_pdf_filenames(
    client: httpx.Client,
    citation_id: str,
) -> list[str]:

    payload = _request_json(
        client,
        (
            f"{NTRS_API_ROOT}"
            f"/citations/{citation_id}"
            "/downloads"
        ),
    )

    names = (
        _recursive_pdf_names(
            payload
        )
    )

    return [
        item
        for item in names
        if item.lower()
        .endswith(
            ".pdf"
        )
    ]


def download_pdf(
    client: httpx.Client,
    *,
    citation_id: str,
    filename: str,
    destination: Path,
) -> str:

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # NTRS download metadata may already contain URL-escaped names.
    # Decode once, then encode exactly once.
    decoded_name = unquote(filename)

    encoded_name = quote(
        decoded_name,
        safe="",
    )

    url = (
        f"{NTRS_API_ROOT}"
        f"/citations/{citation_id}"
        f"/downloads/{encoded_name}"
    )

    delays = (
        5,
        15,
        30,
        60,
    )

    last_error: Exception | None = None

    for attempt in range(
        len(delays) + 1
    ):
        try:
            with client.stream(
                "GET",
                url,
                headers={
                    "User-Agent": (
                        HEADERS[
                            "User-Agent"
                        ]
                    ),
                    "Accept": (
                        "application/pdf,"
                        "application/octet-stream;q=0.9,"
                        "*/*;q=0.8"
                    ),
                },
            ) as response:

                if response.status_code == 404:
                    raise FileNotFoundError(
                        f"HTTP 404: {url}"
                    )

                if response.status_code in (
                    429,
                    500,
                    502,
                    503,
                    504,
                ):
                    raise RuntimeError(
                        f"HTTP "
                        f"{response.status_code}"
                    )

                response.raise_for_status()

                tmp_path = (
                    destination
                    .with_suffix(
                        ".pdf.part"
                    )
                )

                with tmp_path.open(
                    "wb"
                ) as handle:

                    for chunk in (
                        response.iter_bytes(
                            chunk_size=(
                                1024
                                * 256
                            )
                        )
                    ):
                        if chunk:
                            handle.write(
                                chunk
                            )

            if (
                not tmp_path.exists()
                or tmp_path.stat().st_size
                < 1000
            ):
                raise RuntimeError(
                    "NTRS PDF too small"
                )

            with tmp_path.open(
                "rb"
            ) as handle:
                magic = handle.read(
                    5
                )

            if magic != b"%PDF-":
                raise RuntimeError(
                    "Downloaded NTRS file "
                    "is not a PDF"
                )

            tmp_path.replace(
                destination
            )

            return url

        except Exception as exc:
            last_error = exc

            if attempt >= len(delays):
                break

            wait_seconds = delays[
                attempt
            ]

            print(
                f"[NTRS PDF RETRY] "
                f"id={citation_id} "
                f"attempt={attempt + 1} "
                f"wait={wait_seconds}s "
                f"reason={exc}",
                flush=True,
            )

            time.sleep(
                wait_seconds
            )

    raise RuntimeError(
        f"NTRS PDF download failed: "
        f"{last_error}"
    )


def build_metadata(
    citation: dict,
    *,
    pdf_url: str,
    relevance_score: int,
) -> dict:

    citation_id = normalize_text(
        citation.get(
            "id"
        )
    )

    return {
        "source": "ntrs",
        "source_id": citation_id,
        "source_base_id": (
            f"ntrs:{citation_id}"
        ),
        "topic_axis": "onboard_ai",
        "title": normalize_text(
            citation.get(
                "title"
            )
        ),
        "abstract": normalize_text(
            citation.get(
                "abstract"
            )
        ),
        "authors": extract_authors(
            citation
        ),
        "categories": (
            citation.get(
                "subjectCategories"
            )
            or []
        ),
        "published_at": (
            extract_published_at(
                citation
            )
        ),
        "document_type": normalize_text(
            citation.get(
                "stiType"
            )
            or citation.get(
                "stiTypeDetails"
            )
            or "ntrs_document"
        ),
        "url": (
            "https://ntrs.nasa.gov/"
            f"citations/{citation_id}"
        ),
        "pdf_url": pdf_url,
        "language": "en",
        "relevance_score": (
            relevance_score
        ),
        "distribution": normalize_text(
            citation.get(
                "distribution"
            )
        ),
        "disseminated": normalize_text(
            citation.get(
                "disseminated"
            )
        ),
        "copyright": (
            citation.get(
                "copyright"
            )
            or {}
        ),
    }
