import json
import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from app.config import PROJECT_ROOT


# ============================================================
# Paths
# ============================================================

RESOLVED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "tmp"
    / "arxiv_resolved"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "reports"
)

REPORT_FILE = (
    REPORT_DIR
    / "arxiv_html_parsing_report.json"
)


# ============================================================
# Basic Text Helpers
# ============================================================

def _normalize_space(
    text: str,
) -> str:
    """
    한 줄 내부의 여러 공백/탭을 한 칸으로 정리한다.

    문단 자체를 합치는 함수가 아니다.
    """

    return re.sub(
        r"[ \t]+",
        " ",
        text,
    ).strip()


def _normalize_block_text(
    text: str,
) -> str:
    """
    하나의 HTML block 내부에서 생긴
    불필요한 줄바꿈과 공백을 정리한다.

    paragraph와 paragraph 사이의 구조는
    이 함수가 건드리지 않는다.
    """

    lines = []

    for line in text.splitlines():

        line = _normalize_space(
            line
        )

        if line:
            lines.append(
                line
            )

    return " ".join(
        lines
    ).strip()


def _clean_heading(
    text: str,
) -> str:
    """
    section heading 정리.
    """

    text = _normalize_block_text(
        text
    )

    text = text.strip(
        " ¶"
    )

    return text


def _get_classes(
    element: Tag,
) -> list[str]:
    """
    BeautifulSoup class 값을 항상
    list[str] 형태로 반환.
    """

    classes = element.get(
        "class",
        [],
    )

    if isinstance(
        classes,
        str,
    ):
        return [
            classes
        ]

    return list(
        classes
    )


# ============================================================
# Ancestor Helpers
# ============================================================

def _has_ancestor_with_class(
    element: Tag,
    class_names: set[str],
) -> bool:
    """
    부모/조상 중 특정 class를 가진 요소가 있는지 확인.
    """

    parent = element.parent

    while isinstance(
        parent,
        Tag,
    ):

        classes = set(
            _get_classes(
                parent
            )
        )

        if classes.intersection(
            class_names
        ):
            return True

        parent = parent.parent

    return False


def _is_inside_abstract(
    element: Tag,
) -> bool:
    """
    abstract 내부의 요소인지 확인.
    """

    return _has_ancestor_with_class(
        element,
        {
            "ltx_abstract",
        },
    )


def _is_inside_title_area(
    element: Tag,
) -> bool:
    """
    논문 document title 내부인지 확인.

    title은 metadata에서 별도로 추출하므로
    본문 block에 다시 포함하지 않는다.
    """

    return _has_ancestor_with_class(
        element,
        {
            "ltx_title_document",
        },
    )


def _is_inside_figure_caption(
    element: Tag,
) -> bool:
    """
    figure caption 내부의 p 등을
    중복 수집하지 않기 위한 검사.
    """

    parent = element.parent

    while isinstance(
        parent,
        Tag,
    ):

        if parent.name == "figcaption":
            return True

        classes = set(
            _get_classes(
                parent
            )
        )

        if "ltx_caption" in classes:
            return True

        parent = parent.parent

    return False


def _is_inside_bibliography(
    element: Tag,
) -> bool:
    """
    References / Bibliography 안에 있는 block인지 확인.

    raw 단계에서는 삭제하지 않고 표시만 한다.
    실제 제거 여부는 cleaner에서 결정한다.
    """

    parent = element

    while isinstance(
        parent,
        Tag,
    ):

        classes = " ".join(
            _get_classes(
                parent
            )
        ).lower()

        element_id = (
            parent.get(
                "id",
                "",
            )
            or ""
        ).lower()

        if (
            "bibliography" in classes
            or "biblist" in classes
            or "bibitem" in classes
            or "bibliography" in element_id
        ):
            return True

        parent = parent.parent

    return False


# ============================================================
# Noise Removal
# ============================================================

def _remove_non_content(
    soup: BeautifulSoup,
) -> None:
    """
    논문 본문과 관계없는 HTML UI 제거.

    raw 단계이므로 공격적인 내용 제거는 하지 않는다.
    """

    remove_selectors = [
        # Script / CSS
        "script",
        "style",
        "noscript",

        # Navigation
        "nav",

        # arXiv / LaTeXML UI
        ".ltx_page_logo",
        ".ltx_page_navbar",
        ".ltx_page_footer",
        ".ltx_TOC",
        ".ltx_error",

        # 저자/소속은 metadata로 관리할 것이므로
        # 본문 raw_content에는 넣지 않는다.
        ".ltx_authors",
        ".ltx_role_affiliation",
        ".ltx_dates",
    ]

    for selector in remove_selectors:

        for element in soup.select(
            selector
        ):

            element.decompose()


# ============================================================
# Title Extraction
# ============================================================

def _extract_title(
    soup: BeautifulSoup,
) -> str:
    """
    논문 제목 추출.

    우선순위:

    1. citation_title metadata
    2. arXiv/LaTeXML document title
    3. 일반 h1
    4. HTML <title>

    이전 버전의 문제:
    h1 안에 Thanks / copyright /
    project page 등이 같이 붙는 경우가 있었다.

    따라서 citation_title을 최우선으로 사용한다.
    """

    # ========================================================
    # 1. citation_title
    # ========================================================

    meta_title = soup.find(
        "meta",
        attrs={
            "name": "citation_title"
        },
    )

    if meta_title is not None:

        content = meta_title.get(
            "content"
        )

        if content:

            title = _normalize_space(
                content
            )

            if title:
                return title

    # ========================================================
    # 2. arXiv / LaTeXML title
    # ========================================================

    selectors = [
        "h1.ltx_title_document",
        ".ltx_title_document",
        "article h1.ltx_title",
        "main h1.ltx_title",
    ]

    for selector in selectors:

        element = soup.select_one(
            selector
        )

        if element is None:
            continue

        # 원본 soup를 훼손하지 않도록 복사
        clone = BeautifulSoup(
            str(element),
            "html.parser",
        )

        # 제목 내부에 섞이는 noise 제거
        noise_selectors = [
            ".ltx_role_thanks",
            ".ltx_note",
            ".ltx_dates",
            ".ltx_role_affiliation",
            ".ltx_role_author",
            "small",
        ]

        for noise_selector in noise_selectors:

            for noise in clone.select(
                noise_selector
            ):

                noise.decompose()

        title = _clean_heading(
            clone.get_text(
                " ",
                strip=True,
            )
        )

        # 혹시 plain text로 Thanks가 붙는 경우
        # 그 뒤를 보수적으로 제거
        title = re.split(
            r"\s+(?:Thanks|Project page)\s*:",
            title,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()

        if title:
            return title

    # ========================================================
    # 3. Generic h1
    # ========================================================

    element = soup.find(
        "h1"
    )

    if element is not None:

        title = _clean_heading(
            element.get_text(
                " ",
                strip=True,
            )
        )

        title = re.split(
            r"\s+(?:Thanks|Project page)\s*:",
            title,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()

        if title:
            return title

    # ========================================================
    # 4. <title>
    # ========================================================

    if soup.title:

        title = _normalize_space(
            soup.title.get_text(
                " ",
                strip=True,
            )
        )

        if title:
            return title

    return ""


# ============================================================
# Abstract Extraction
# ============================================================

def _extract_abstract(
    soup: BeautifulSoup,
) -> str:
    """
    arXiv HTML에서 abstract 추출.
    """

    selectors = [
        ".ltx_abstract",
        "section.ltx_abstract",
        "div.ltx_abstract",
    ]

    for selector in selectors:

        element = soup.select_one(
            selector
        )

        if element is None:
            continue

        clone = BeautifulSoup(
            str(element),
            "html.parser",
        )

        # Abstract heading 중복 제거
        for heading in clone.find_all(
            [
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
            ]
        ):

            heading.decompose()

        # abstract 안의 inline math를
        # alttext/TeX로 최대한 보존
        for math in clone.find_all(
            "math"
        ):

            replacement = (
                math.get(
                    "alttext"
                )
                or math.get_text(
                    " ",
                    strip=True,
                )
            )

            math.replace_with(
                replacement
            )

        abstract = _normalize_block_text(
            clone.get_text(
                "\n",
                strip=True,
            )
        )

        abstract = re.sub(
            r"^Abstract\s*[:.\-]?\s*",
            "",
            abstract,
            flags=re.IGNORECASE,
        )

        if abstract:
            return abstract

    return ""


# ============================================================
# Math / Equation Helpers
# ============================================================

def _extract_math_text(
    math: Tag,
) -> str:
    """
    math element에서 사람이 읽을 수 있는
    표현을 최대한 보존한다.

    우선순위:
    1. alttext
    2. TeX annotation
    3. visible text
    """

    alttext = math.get(
        "alttext"
    )

    if alttext:

        return _normalize_space(
            alttext
        )

    annotation = math.find(
        "annotation",
        attrs={
            "encoding": re.compile(
                r"tex",
                re.IGNORECASE,
            )
        },
    )

    if annotation:

        latex = _normalize_block_text(
            annotation.get_text(
                " ",
                strip=True,
            )
        )

        if latex:
            return latex

    return _normalize_block_text(
        math.get_text(
            " ",
            strip=True,
        )
    )


def _extract_text_preserving_math(
    element: Tag,
) -> str:
    """
    paragraph 안의 inline math를 없애지 않고
    alttext/TeX 표현으로 치환한 뒤 text 추출.

    예:
    speed is v = d/t

    같은 식을 paragraph 안에서 유지한다.
    """

    clone = BeautifulSoup(
        str(element),
        "html.parser",
    )

    for math in clone.find_all(
        "math"
    ):

        replacement = _extract_math_text(
            math
        )

        if replacement:

            math.replace_with(
                f" {replacement} "
            )

    return _normalize_block_text(
        clone.get_text(
            " ",
            strip=True,
        )
    )


def _is_equation_container(
    element: Tag,
) -> bool:
    """
    display equation container인지 판단.
    """

    classes = set(
        _get_classes(
            element
        )
    )

    equation_classes = {
        "ltx_equation",
        "ltx_equationgroup",
        "ltx_equation_table",
    }

    return bool(
        classes.intersection(
            equation_classes
        )
    )


def _has_equation_ancestor(
    element: Tag,
) -> bool:
    """
    이미 상위 equation container를 처리한 경우
    내부 equation을 다시 추가하지 않도록 함.
    """

    parent = element.parent

    while isinstance(
        parent,
        Tag,
    ):

        if _is_equation_container(
            parent
        ):
            return True

        parent = parent.parent

    return False


def _extract_equation(
    element: Tag,
) -> str:
    """
    display equation을 가능한 한
    TeX/alttext 형태로 추출.
    """

    math_elements = element.find_all(
        "math"
    )

    equations = []

    for math in math_elements:

        text = _extract_math_text(
            math
        )

        if (
            text
            and text not in equations
        ):
            equations.append(
                text
            )

    if equations:

        return " ".join(
            equations
        )

    annotation = element.find(
        "annotation",
        attrs={
            "encoding": re.compile(
                r"tex",
                re.IGNORECASE,
            )
        },
    )

    if annotation:

        text = _normalize_block_text(
            annotation.get_text(
                " ",
                strip=True,
            )
        )

        if text:
            return text

    return _normalize_block_text(
        element.get_text(
            " ",
            strip=True,
        )
    )


# ============================================================
# Document Root
# ============================================================

def _find_document_root(
    soup: BeautifulSoup,
) -> Tag:
    """
    실제 논문 본문 root 탐색.
    """

    selectors = [
        "article.ltx_document",
        "div.ltx_document",
        "main.ltx_document",
        "article",
        "main",
        "body",
    ]

    for selector in selectors:

        root = soup.select_one(
            selector
        )

        if root is not None:
            return root

    raise ValueError(
        "Could not find document root."
    )


# ============================================================
# Block Type Helpers
# ============================================================

def _is_document_title(
    element: Tag,
) -> bool:
    """
    document title heading인지 확인.
    """

    classes = set(
        _get_classes(
            element
        )
    )

    return (
        "ltx_title_document"
        in classes
    )


def _is_noise_heading(
    text: str,
) -> bool:
    """
    heading처럼 보이지만 실제 section이 아닌
    일부 UI/문서 부가 텍스트 필터.

    과도한 제거는 하지 않는다.
    """

    normalized = text.strip().lower()

    noise = {
        "contents",
        "table of contents",
    }

    return normalized in noise


# ============================================================
# DOM-Order Block Extraction
# ============================================================

def _extract_blocks(
    root: Tag,
) -> list[dict]:
    """
    DOM 순서를 그대로 따라가며 block 추출.

    매우 중요:

    이전 버전은
    heading/paragraph를 먼저 모은 뒤
    equation을 마지막에 append했다.

    그래서 실제 문서 순서:

        paragraph
        equation
        paragraph

    가

        paragraph
        paragraph
        ...
        equation

    으로 깨질 수 있었다.

    이 버전은 HTML DOM을 한 번 순회하면서:

    heading
    paragraph
    list
    figure caption
    equation

    을 실제 등장 순서대로 기록한다.
    """

    blocks: list[dict] = []

    seen_blocks: set[
        tuple[str, str]
    ] = set()

    heading_tags = {
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }

    # find_all(True)는 HTML 문서의 DOM 순서대로 반환
    for element in root.find_all(
        True
    ):

        if not isinstance(
            element,
            Tag,
        ):
            continue

        # --------------------------------------------
        # Abstract는 별도로 추출했으므로 제외
        # --------------------------------------------

        if _is_inside_abstract(
            element
        ):
            continue

        # --------------------------------------------
        # Document title 중복 제외
        # --------------------------------------------

        if (
            _is_document_title(
                element
            )
            or _is_inside_title_area(
                element
            )
        ):
            continue

        # --------------------------------------------
        # Display Equation
        # --------------------------------------------

        if _is_equation_container(
            element
        ):

            # equationgroup 안의 equation을
            # 두 번 수집하지 않는다.
            if _has_equation_ancestor(
                element
            ):
                continue

            text = _extract_equation(
                element
            )

            if not text:
                continue

            key = (
                "equation",
                text,
            )

            if key in seen_blocks:
                continue

            seen_blocks.add(
                key
            )

            blocks.append(
                {
                    "type": "equation",
                    "text": text,
                    "in_bibliography": (
                        _is_inside_bibliography(
                            element
                        )
                    ),
                }
            )

            continue

        # equation 안의 descendant들은
        # equation block에서 이미 처리했으므로 제외
        if _has_equation_ancestor(
            element
        ):
            continue

        tag_name = (
            element.name
            or ""
        ).lower()

        # --------------------------------------------
        # Heading
        # --------------------------------------------

        if tag_name in heading_tags:

            text = _clean_heading(
                _extract_text_preserving_math(
                    element
                )
            )

            if not text:
                continue

            if _is_noise_heading(
                text
            ):
                continue

            level = int(
                tag_name[1]
            )

            key = (
                "heading",
                text,
            )

            if key in seen_blocks:
                continue

            seen_blocks.add(
                key
            )

            blocks.append(
                {
                    "type": "heading",
                    "level": level,
                    "text": text,
                    "in_bibliography": (
                        _is_inside_bibliography(
                            element
                        )
                    ),
                }
            )

            continue

        # --------------------------------------------
        # Figure Caption
        # --------------------------------------------

        if (
            tag_name == "figcaption"
            or "ltx_caption"
            in _get_classes(
                element
            )
        ):

            # ltx_caption 안의 더 작은 nested caption
            # 중복 가능성 최소화
            if _is_inside_figure_caption(
                element
            ):
                continue

            text = _extract_text_preserving_math(
                element
            )

            if not text:
                continue

            key = (
                "figure_caption",
                text,
            )

            if key in seen_blocks:
                continue

            seen_blocks.add(
                key
            )

            blocks.append(
                {
                    "type": "figure_caption",
                    "text": text,
                    "in_bibliography": (
                        _is_inside_bibliography(
                            element
                        )
                    ),
                }
            )

            continue

        # --------------------------------------------
        # Paragraph
        # --------------------------------------------

        if tag_name == "p":

            # caption 안의 p는 caption 자체에서 처리
            if _is_inside_figure_caption(
                element
            ):
                continue

            text = _extract_text_preserving_math(
                element
            )

            if not text:
                continue

            if len(text) < 2:
                continue

            key = (
                "paragraph",
                text,
            )

            if key in seen_blocks:
                continue

            seen_blocks.add(
                key
            )

            blocks.append(
                {
                    "type": "paragraph",
                    "text": text,
                    "in_bibliography": (
                        _is_inside_bibliography(
                            element
                        )
                    ),
                }
            )

            continue

        # --------------------------------------------
        # List Item
        # --------------------------------------------

        if tag_name == "li":

            # li 안에 p가 있으면
            # p가 이미 DOM 순서에서 별도로 처리되므로
            # li 전체 text를 또 넣지 않는다.
            if element.find(
                "p"
            ) is not None:
                continue

            # nested list가 있으면 상위 li 전체를 넣으면
            # 자식 내용까지 중복될 수 있다.
            if element.find(
                [
                    "ul",
                    "ol",
                ]
            ) is not None:
                continue

            text = _extract_text_preserving_math(
                element
            )

            if not text:
                continue

            if len(text) < 2:
                continue

            key = (
                "list_item",
                text,
            )

            if key in seen_blocks:
                continue

            seen_blocks.add(
                key
            )

            blocks.append(
                {
                    "type": "list_item",
                    "text": text,
                    "in_bibliography": (
                        _is_inside_bibliography(
                            element
                        )
                    ),
                }
            )

    return blocks


# ============================================================
# Section Construction
# ============================================================

def _blocks_to_sections(
    blocks: list[dict],
) -> list[dict]:
    """
    heading을 기준으로 section을 구성한다.

    subsection / subsubsection 역시
    독립 section record로 유지한다.

    나중에 RAG chunking에서
    heading 정보를 그대로 활용할 수 있다.
    """

    sections: list[dict] = []

    current_section = {
        "heading": "Document Body",
        "level": 1,
        "blocks": [],
        "in_bibliography": False,
    }

    for block in blocks:

        if block[
            "type"
        ] == "heading":

            # 기존 section에 내용이 있는 경우 저장
            if current_section[
                "blocks"
            ]:

                sections.append(
                    current_section
                )

            current_section = {
                "heading": block[
                    "text"
                ],
                "level": block.get(
                    "level",
                    2,
                ),
                "blocks": [],
                "in_bibliography": block.get(
                    "in_bibliography",
                    False,
                ),
            }

            continue

        current_section[
            "blocks"
        ].append(
            block
        )

        if block.get(
            "in_bibliography",
            False,
        ):

            current_section[
                "in_bibliography"
            ] = True

    if current_section[
        "blocks"
    ]:

        sections.append(
            current_section
        )

    return sections


# ============================================================
# Raw Content Rendering
# ============================================================

def _render_raw_content(
    title: str,
    abstract: str,
    sections: list[dict],
) -> str:
    """
    DB에 저장하기 전 intermediate raw_content 생성.

    구조를 보존하기 위해 Markdown 스타일 heading 사용.

    아직 References 삭제나 caption 제거 등의
    cleaner 작업은 하지 않는다.
    """

    output: list[str] = []

    # ========================================================
    # Title
    # ========================================================

    if title:

        output.append(
            "# TITLE"
        )

        output.append(
            title
        )

        output.append(
            ""
        )

    # ========================================================
    # Abstract
    # ========================================================

    if abstract:

        output.append(
            "## ABSTRACT"
        )

        output.append(
            abstract
        )

        output.append(
            ""
        )

    # ========================================================
    # Sections
    # ========================================================

    for section in sections:

        heading = section[
            "heading"
        ]

        level = section.get(
            "level",
            2,
        )

        # document body용 level 1이라도
        # TITLE보다 아래 단계로 표현
        markdown_level = max(
            2,
            min(
                level,
                6,
            ),
        )

        output.append(
            (
                "#" * markdown_level
            )
            + " "
            + heading
        )

        output.append(
            ""
        )

        for block in section[
            "blocks"
        ]:

            block_type = block[
                "type"
            ]

            text = block[
                "text"
            ]

            if block_type == "list_item":

                output.append(
                    f"- {text}"
                )

            elif block_type == "figure_caption":

                output.append(
                    f"[FIGURE CAPTION] {text}"
                )

            elif block_type == "equation":

                output.append(
                    f"[EQUATION] {text}"
                )

            else:

                output.append(
                    text
                )

            output.append(
                ""
            )

    result = "\n".join(
        output
    )

    # 3줄 이상 연속 빈 줄 방지
    result = re.sub(
        r"\n{3,}",
        "\n\n",
        result,
    )

    return result.strip()


# ============================================================
# Statistics
# ============================================================

def _build_stats(
    raw_content: str,
    sections: list[dict],
) -> dict:
    """
    파일럿 QA용 통계.
    """

    word_count = len(
        raw_content.split()
    )

    paragraph_count = 0
    equation_count = 0
    figure_caption_count = 0
    list_item_count = 0
    bibliography_section_count = 0

    for section in sections:

        if section.get(
            "in_bibliography",
            False,
        ):

            bibliography_section_count += 1

        for block in section[
            "blocks"
        ]:

            block_type = block[
                "type"
            ]

            if block_type == "paragraph":

                paragraph_count += 1

            elif block_type == "equation":

                equation_count += 1

            elif block_type == "figure_caption":

                figure_caption_count += 1

            elif block_type == "list_item":

                list_item_count += 1

    return {
        "char_count": len(
            raw_content
        ),

        "word_count": (
            word_count
        ),

        "section_count": len(
            sections
        ),

        "paragraph_count": (
            paragraph_count
        ),

        "equation_count": (
            equation_count
        ),

        "figure_caption_count": (
            figure_caption_count
        ),

        "list_item_count": (
            list_item_count
        ),

        "bibliography_section_count": (
            bibliography_section_count
        ),
    }


# ============================================================
# Parse One HTML File
# ============================================================

def parse_html_file(
    html_path: Path,
) -> dict:
    """
    paper.html 1편을:

    title
    abstract
    sections
    blocks
    raw_content

    구조로 변환한다.
    """

    html = html_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    # 제목과 abstract는
    # noise element 제거 전에 먼저 추출
    title = _extract_title(
        soup
    )

    abstract = _extract_abstract(
        soup
    )

    # 본문 UI/noise 제거
    _remove_non_content(
        soup
    )

    root = _find_document_root(
        soup
    )

    # DOM 순서 보존
    blocks = _extract_blocks(
        root
    )

    sections = _blocks_to_sections(
        blocks
    )

    raw_content = _render_raw_content(
        title=title,
        abstract=abstract,
        sections=sections,
    )

    stats = _build_stats(
        raw_content,
        sections,
    )

    return {
        "title": title,
        "abstract": abstract,
        "sections": sections,
        "raw_content": raw_content,
        "stats": stats,
    }


# ============================================================
# Save One Parsed Document
# ============================================================

def _save_parsed_document(
    paper_dir: Path,
    parsed: dict,
) -> None:
    """
    각 논문 폴더 결과:

    paper.html
    resolution.json
    raw_content.txt
    parsed_document.json
    """

    raw_path = (
        paper_dir
        / "raw_content.txt"
    )

    json_path = (
        paper_dir
        / "parsed_document.json"
    )

    # 기존 파일이 있어도
    # 이번 수정 결과로 덮어쓴다.
    raw_path.write_text(
        parsed[
            "raw_content"
        ],
        encoding="utf-8",
    )

    json_payload = {
        "title": parsed[
            "title"
        ],

        "abstract": parsed[
            "abstract"
        ],

        "sections": parsed[
            "sections"
        ],

        "stats": parsed[
            "stats"
        ],
    }

    json_path.write_text(
        json.dumps(
            json_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# Find HTML Files
# ============================================================

def _find_html_files() -> list[Path]:
    """
    Content Resolver가 저장한
    모든 paper.html 탐색.

    구조:

    arxiv_resolved/
      axis/
        arxiv_id/
          paper.html
    """

    if not RESOLVED_ROOT.exists():

        raise FileNotFoundError(
            f"Resolved root not found: "
            f"{RESOLVED_ROOT}"
        )

    return sorted(
        RESOLVED_ROOT.glob(
            "*/*/paper.html"
        )
    )


# ============================================================
# Parsing Report
# ============================================================

def _save_report(
    results: list[dict],
) -> None:
    """
    전체 parser 실행 결과 report 저장.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    success_count = sum(
        1
        for item in results
        if item[
            "status"
        ] == "success"
    )

    failed_count = (
        len(results)
        - success_count
    )

    total_words = sum(
        item.get(
            "stats",
            {},
        ).get(
            "word_count",
            0,
        )
        for item in results
        if item[
            "status"
        ] == "success"
    )

    total_chars = sum(
        item.get(
            "stats",
            {},
        ).get(
            "char_count",
            0,
        )
        for item in results
        if item[
            "status"
        ] == "success"
    )

    payload = {
        "total": len(
            results
        ),

        "success": (
            success_count
        ),

        "failed": (
            failed_count
        ),

        "total_words": (
            total_words
        ),

        "total_chars": (
            total_chars
        ),

        "documents": results,
    }

    REPORT_FILE.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# Main
# ============================================================

def main():
    """
    arXiv HTML 파일럿 15편:

    paper.html
        ↓
    metadata/abstract
        ↓
    DOM-order parsing
        ↓
    title / sections / paragraphs /
    equations / captions
        ↓
    raw_content.txt
        ↓
    parsed_document.json

    아직:
    clean_content 생성 X
    DB INSERT X
    """

    print()
    print("=" * 70)

    print(
        "TEAM B - arXiv HTML Parser v2"
    )

    print("=" * 70)

    print(
        f"Input root : "
        f"{RESOLVED_ROOT}"
    )

    html_files = _find_html_files()

    print(
        f"HTML files : "
        f"{len(html_files)}"
    )

    if not html_files:

        print(
            "No paper.html files found."
        )

        return

    results: list[dict] = []

    for index, html_path in enumerate(
        html_files,
        start=1,
    ):

        paper_dir = (
            html_path.parent
        )

        topic_axis = (
            paper_dir
            .parent
            .name
        )

        source_id = (
            paper_dir.name
        )

        print()
        print("-" * 70)

        print(
            f"[{index}/{len(html_files)}]"
        )

        print(
            f"[AXIS] {topic_axis}"
        )

        print(
            f"[ID]   {source_id}"
        )

        print(
            f"[FILE] {html_path}"
        )

        print("-" * 70)

        try:

            parsed = parse_html_file(
                html_path
            )

            _save_parsed_document(
                paper_dir,
                parsed,
            )

            stats = parsed[
                "stats"
            ]

            print(
                f"[OK] Title      : "
                f"{parsed['title']}"
            )

            print(
                f"[OK] Sections   : "
                f"{stats['section_count']}"
            )

            print(
                f"[OK] Paragraphs : "
                f"{stats['paragraph_count']}"
            )

            print(
                f"[OK] Equations  : "
                f"{stats['equation_count']}"
            )

            print(
                f"[OK] Figures    : "
                f"{stats['figure_caption_count']}"
            )

            print(
                f"[OK] Words      : "
                f"{stats['word_count']}"
            )

            print(
                f"[OK] Chars      : "
                f"{stats['char_count']}"
            )

            print(
                f"[SAVE] "
                f"{paper_dir / 'raw_content.txt'}"
            )

            results.append(
                {
                    "status": "success",

                    "topic_axis": (
                        topic_axis
                    ),

                    "source_id": (
                        source_id
                    ),

                    "title": parsed[
                        "title"
                    ],

                    "stats": stats,

                    "raw_content_path": str(
                        paper_dir
                        / "raw_content.txt"
                    ),

                    "parsed_json_path": str(
                        paper_dir
                        / "parsed_document.json"
                    ),
                }
            )

        except Exception as exc:

            print(
                f"[ERROR] "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            results.append(
                {
                    "status": "failed",

                    "topic_axis": (
                        topic_axis
                    ),

                    "source_id": (
                        source_id
                    ),

                    "error": str(
                        exc
                    ),
                }
            )

        # 한 편 끝날 때마다 report 저장
        _save_report(
            results
        )

    # ========================================================
    # Summary
    # ========================================================

    success_count = sum(
        1
        for item in results
        if item[
            "status"
        ] == "success"
    )

    failed_count = (
        len(results)
        - success_count
    )

    total_words = sum(
        item.get(
            "stats",
            {},
        ).get(
            "word_count",
            0,
        )
        for item in results
        if item[
            "status"
        ] == "success"
    )

    print()
    print("=" * 70)

    print(
        "HTML parsing completed."
    )

    print("=" * 70)

    print(
        f"Total       : {len(results)}"
    )

    print(
        f"Success     : {success_count}"
    )

    print(
        f"Failed      : {failed_count}"
    )

    print(
        f"Total words : {total_words}"
    )

    print(
        f"Report      : {REPORT_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()