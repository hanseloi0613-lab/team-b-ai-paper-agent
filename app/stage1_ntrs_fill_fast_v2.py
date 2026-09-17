from __future__ import annotations

from typing import Any

import app.stage1_ntrs_fill_fast as fast1


base = fast1.base


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
    " uas ",
    "uas in the nas",
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
    "satellite",
    "lunar",
    "planetary",
    "space transportation",
    "space communications",
    "space exploration",
    "space operations",
)

AVIATION_CATEGORY_MARKERS = (
    "aeronautics",
    "air transportation",
    "aircraft",
    "aviation",
)

# Title만 봐도 NASA space mission임이 분명한 것들.
MISSION_MARKERS = (
    "swift mission",
    "swift spacecraft",
    "eo-1",
    "earth observing-1",
    "gateway autonomy",
    "gateway spacecraft",
    "starling swarm",
    "international space station",
    "iss ",
)


def norm(value: Any) -> str:
    return " ".join(
        str(value or "")
        .lower()
        .split()
    )


def list_text(value: Any) -> str:

    if isinstance(
        value,
        list,
    ):
        output = []

        for item in value:

            if isinstance(
                item,
                dict,
            ):
                output.extend(
                    norm(v)
                    for v
                    in item.values()
                )

            else:
                output.append(
                    norm(item)
                )

        return " ".join(
            output
        )

    return norm(
        value
    )


def all_text(
    citation: dict,
) -> str:

    pieces = [
        norm(
            citation.get(
                "title"
            )
        ),
        norm(
            citation.get(
                "abstract"
            )
        ),
        list_text(
            citation.get(
                "keywords"
            )
        ),
        list_text(
            citation.get(
                "subjectCategories"
            )
        ),
        # canonical documents.jsonl에서는
        # subjectCategories가 categories로 저장된다.
        list_text(
            citation.get(
                "categories"
            )
        ),
        norm(
            citation.get(
                "document_type"
            )
        ),
    ]

    return " ".join(
        p
        for p in pieces
        if p
    )


def strict_production_relevance_v2(
    citation: dict,
    min_score: int,
):

    title = norm(
        citation.get(
            "title"
        )
    )

    text = all_text(
        citation
    )

    categories = " ".join(
        [
            list_text(
                citation.get(
                    "subjectCategories"
                )
            ),
            list_text(
                citation.get(
                    "categories"
                )
            ),
        ]
    )

    space_hits = {
        term
        for term
        in SPACE_STRONG
        if term in text
    }

    autonomy_hits = {
        term
        for term
        in AUTONOMY_STRONG
        if term in text
    }

    title_space_hits = {
        term
        for term
        in SPACE_STRONG
        if term in title
    }

    mission_hits = {
        term
        for term
        in MISSION_MARKERS
        if term in text
    }

    space_category = any(
        term in categories
        for term
        in SPACE_CATEGORY_MARKERS
    )

    aviation_title = any(
        term in f" {title} "
        for term
        in AVIATION_MARKERS
    )

    aviation_category = any(
        term in categories
        for term
        in AVIATION_CATEGORY_MARKERS
    )

    # --------------------------------------------
    # Space evidence:
    # 1) explicit spacecraft/satellite/etc
    # OR
    # 2) NASA space subject category
    # OR
    # 3) well-known explicit mission marker
    # --------------------------------------------

    has_space_evidence = bool(
        space_hits
        or space_category
        or mission_hits
    )

    if not has_space_evidence:
        return (
            False,
            0,
            "strict_v2:no_space_evidence",
        )

    if not autonomy_hits:
        return (
            False,
            len(space_hits),
            "strict_v2:no_autonomy_ai_anchor",
        )

    # --------------------------------------------
    # Aviation-only autonomy 차단
    #
    # title이 UAS/air traffic/aviation인데
    # spacecraft/satellite/space-category/mission
    # 증거가 없으면 제거.
    # --------------------------------------------

    if (
        aviation_title
        and not (
            title_space_hits
            or space_category
            or mission_hits
        )
    ):
        return (
            False,
            0,
            "strict_v2:aviation_only_title",
        )

    if (
        aviation_category
        and not (
            space_category
            or title_space_hits
            or mission_hits
        )
    ):
        return (
            False,
            0,
            "strict_v2:aviation_only_category",
        )

    score = (
        len(
            space_hits
        )
        + len(
            autonomy_hits
        )
        + (
            2
            if mission_hits
            else 0
        )
        + (
            1
            if space_category
            else 0
        )
    )

    # Generic title인데 metadata 전체에도
    # space evidence가 너무 약하면 제거.
    if (
        not title_space_hits
        and not mission_hits
        and not space_category
        and len(
            space_hits
        ) < 2
    ):
        return (
            False,
            score,
            "strict_v2:weak_space_context",
        )

    # strong evidence가 있으면 단순 숫자 score 때문에
    # Swift/EO-1 같은 자료를 탈락시키지 않는다.
    if (
        score < min_score
        and not (
            mission_hits
            or space_category
            or title_space_hits
        )
    ):
        return (
            False,
            score,
            f"strict_v2:score<{min_score}",
        )

    return (
        True,
        score,
        "passed",
    )


base.production_relevance = (
    strict_production_relevance_v2
)


if __name__ == "__main__":
    base.main()
