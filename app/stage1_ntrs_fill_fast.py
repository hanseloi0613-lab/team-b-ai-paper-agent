from __future__ import annotations

import os
import threading
import time

import httpx as _httpx

import app.stage1_ntrs_fill as base


# ============================================================
# TEAM B NTRS FAST / STRICT PRODUCTION RUNNER
#
# 핵심:
# - 기존 main corpus resume
# - 외부 pretrained model 없음
# - strong space-domain relevance
# - aviation-only 자료 차단
# - NTRS 전체 HTTP 요청 global pacing
# - blind post-request sleep 제거
# ============================================================


MIN_REQUEST_INTERVAL = float(
    os.environ.get(
        "NTRS_MIN_INTERVAL",
        "1.9",
    )
)


# ============================================================
# 1. High Precision Queries
# ============================================================

HIGH_PRECISION_QUERIES = (
    "spacecraft autonomy",
    "spacecraft onboard autonomy",
    "onboard autonomy spacecraft",
    "autonomous spacecraft",
    "autonomous spacecraft operations",
    "spacecraft mission autonomy",
    "deep space autonomy",
    "spacecraft artificial intelligence",
    "spacecraft machine learning",
    "onboard artificial intelligence spacecraft",
    "onboard machine learning spacecraft",
    "spacecraft reinforcement learning",
    "spacecraft decision making",
    "onboard decision making spacecraft",
    "autonomous mission planning spacecraft",
    "spacecraft mission planning",
    "spacecraft planning execution",
    "spacecraft scheduling autonomy",
    "spacecraft resource management autonomy",
    "autonomous navigation spacecraft",
    "spacecraft guidance navigation autonomy",
    "satellite autonomy",
    "satellite onboard autonomy",
    "satellite autonomous operations",
    "lunar spacecraft autonomy",
    "mars spacecraft autonomy",
    "planetary mission autonomy",
    "space robotics autonomy",
    "onboard science autonomy spacecraft",
    "spacecraft command execution autonomy",
    "cubesat autonomy",
    "distributed spacecraft autonomy",
    "spacecraft swarm autonomy",
)

base.QUERIES = HIGH_PRECISION_QUERIES


# ============================================================
# 2. Strict Domain Gate
# ============================================================

SPACE_STRONG = (
    "spacecraft",
    "satellite",
    "deep space",
    "space mission",
    "space missions",
    "space system",
    "space systems",
    "spaceflight",
    "space flight",
    "space station",
    "on-orbit",
    "on orbit",
    "orbital",
    "lunar",
    "moon mission",
    "mars mission",
    "mars rover",
    "planetary mission",
    "planetary exploration",
    "cubesat",
    "cube sat",
    "small satellite",
    "starling swarm",
    "space robotics",
    "space robot",
)

AUTONOMY_STRONG = (
    "autonomy",
    "autonomous",
    "onboard",
    "on-board",
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "reinforcement learning",
    "decision making",
    "decision-making",
    "mission planning",
    "planning",
    "scheduling",
    "guidance",
    "navigation",
    "command execution",
    "mission management",
    "resource management",
    "edge compute",
    "edge computing",
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

SPACE_CATEGORY_MARKERS = (
    "space sciences",
    "spacecraft",
    "lunar",
    "planetary",
    "satellite",
    "space transportation",
    "space communications",
)

AVIATION_CATEGORY_MARKERS = (
    "aeronautics",
    "air transportation",
    "aircraft",
    "aviation",
)


def _clean(value) -> str:
    return base.normalize(
        value
    ).lower()


def _list_text(value) -> str:
    if isinstance(value, list):
        return " ".join(
            _clean(item)
            for item in value
        )

    return _clean(value)


def strict_production_relevance(
    citation: dict,
    min_score: int,
):
    (
        base_pass,
        base_score,
    ) = base.onboard_relevance(
        citation
    )

    title = _clean(
        citation.get(
            "title"
        )
    )

    text = base.citation_text(
        citation
    )

    categories = _list_text(
        citation.get(
            "subjectCategories"
        )
        or []
    )

    space_hits = {
        term
        for term in SPACE_STRONG
        if term in text
    }

    autonomy_hits = {
        term
        for term in AUTONOMY_STRONG
        if term in text
    }

    title_space_hits = {
        term
        for term in SPACE_STRONG
        if term in title
    }

    title_autonomy_hits = {
        term
        for term in AUTONOMY_STRONG
        if term in title
    }

    aviation_title = any(
        term in title
        for term in AVIATION_MARKERS
    )

    aviation_category = any(
        term in categories
        for term in AVIATION_CATEGORY_MARKERS
    )

    space_category = any(
        term in categories
        for term in SPACE_CATEGORY_MARKERS
    )

    # --------------------------------------------
    # 반드시 실제 space-system anchor가 있어야 함.
    # 단순 NASA + autonomy는 불합격.
    # --------------------------------------------

    if not space_hits:
        return (
            False,
            base_score,
            "strict:no_space_system_anchor",
        )

    if not autonomy_hits:
        return (
            False,
            base_score,
            "strict:no_autonomy_ai_anchor",
        )

    # --------------------------------------------
    # 제목 자체가 aviation/UAS/air traffic이면
    # spacecraft/satellite 등이 제목에 없는 한 제외.
    # --------------------------------------------

    if (
        aviation_title
        and not title_space_hits
    ):
        return (
            False,
            base_score,
            "strict:aviation_title",
        )

    # --------------------------------------------
    # Aeronautics category만 있고
    # space category/title evidence가 없으면 제외.
    # --------------------------------------------

    if (
        aviation_category
        and not space_category
        and not title_space_hits
    ):
        return (
            False,
            base_score,
            "strict:aviation_category",
        )

    # --------------------------------------------
    # 제목이 generic이면 본문/metadata에서
    # 최소 두 개의 서로 다른 strong space anchor 요구.
    # --------------------------------------------

    if (
        not title_space_hits
        and len(space_hits) < 2
    ):
        return (
            False,
            base_score,
            "strict:weak_space_context",
        )

    # --------------------------------------------
    # 기존 relevance score도 유지.
    # 단 제목에 space + autonomy가 직접 있으면 허용.
    # --------------------------------------------

    if (
        base_score < min_score
        and not (
            title_space_hits
            and title_autonomy_hits
        )
    ):
        return (
            False,
            base_score,
            f"strict:score<{min_score}",
        )

    return (
        True,
        base_score,
        "passed",
    )


base.production_relevance = (
    strict_production_relevance
)


# ============================================================
# 3. Global HTTP Rate Pacer
#
# NTRS API:
# 500 requests / 15 min
#
# 1.9 sec minimum spacing:
# theoretical max ~474 / 15 min
#
# 기존 방식:
# HTTP 완료 -> 무조건 2~3초 sleep
#
# 개선:
# 다음 HTTP가 너무 빨리 올 때만 필요한 만큼 sleep
#
# PDF parsing 시간이 2초 걸렸다면
# 별도 2초를 또 기다리지 않음.
# ============================================================

OriginalClient = _httpx.Client


class PacedClient(
    OriginalClient
):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        self._pace_lock = (
            threading.Lock()
        )

        self._last_request_at = 0.0

        self._min_interval = (
            MIN_REQUEST_INTERVAL
        )

    def _pace(
        self,
    ) -> None:

        with self._pace_lock:

            now = (
                time.monotonic()
            )

            elapsed = (
                now
                - self._last_request_at
            )

            wait = (
                self._min_interval
                - elapsed
            )

            if wait > 0:
                time.sleep(
                    wait
                )

            self._last_request_at = (
                time.monotonic()
            )

    def request(
        self,
        *args,
        **kwargs,
    ):
        self._pace()

        return super().request(
            *args,
            **kwargs,
        )

    def stream(
        self,
        *args,
        **kwargs,
    ):
        self._pace()

        return super().stream(
            *args,
            **kwargs,
        )


base.httpx.Client = PacedClient


# ============================================================
# 4. 기존 main의 "요청 후 또 sleep" 제거
#
# HTTP 요청 자체가 PacedClient로 제한되므로
# api_called=True를 main에 넘겨 추가 sleep하지 않는다.
# ============================================================

_original_search_cached = (
    base.search_cached
)


def fast_search_cached(
    *args,
    **kwargs,
):
    (
        results,
        total,
        _api_called,
    ) = _original_search_cached(
        *args,
        **kwargs,
    )

    return (
        results,
        total,
        False,
    )


base.search_cached = (
    fast_search_cached
)


_original_get_downloads_cached = (
    base.get_downloads_cached
)


def fast_get_downloads_cached(
    *args,
    **kwargs,
):
    (
        names,
        _api_called,
    ) = _original_get_downloads_cached(
        *args,
        **kwargs,
    )

    return (
        names,
        False,
    )


base.get_downloads_cached = (
    fast_get_downloads_cached
)


# process_candidate 내부의
# download 후 blind sleep도 제거.
_original_process_candidate = (
    base.process_candidate
)


def fast_process_candidate(
    *args,
    **kwargs,
):
    kwargs[
        "request_delay"
    ] = 0.0

    return (
        _original_process_candidate(
            *args,
            **kwargs,
        )
    )


base.process_candidate = (
    fast_process_candidate
)


if __name__ == "__main__":
    base.main()
