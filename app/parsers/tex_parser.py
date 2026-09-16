import json
import re
from pathlib import Path

from app.config import PROJECT_ROOT


# ============================================================
# TEAM B - arXiv TeX Parser
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
    / "arxiv_tex_parsing_report.json"
)


# ============================================================
# JSON
# ============================================================

def _load_json(
    path: Path,
) -> dict:

    if not path.exists():
        return {}

    try:

        return json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )

    except Exception:

        return {}


# ============================================================
# Text Helpers
# ============================================================

def _normalize_space(
    text: str,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def _strip_comments(
    text: str,
) -> str:
    """
    LaTeX comment 제거.

    escaped \% 는 유지한다.
    """

    cleaned_lines = []

    for line in text.splitlines():

        chars = []

        i = 0

        while i < len(line):

            char = line[i]

            if char == "%":

                # 직전 backslash 개수
                slash_count = 0
                j = i - 1

                while (
                    j >= 0
                    and line[j] == "\\"
                ):

                    slash_count += 1
                    j -= 1

                # escaped percent
                if slash_count % 2 == 1:

                    chars.append(
                        char
                    )

                    i += 1
                    continue

                # comment 시작
                break

            chars.append(
                char
            )

            i += 1

        cleaned_lines.append(
            "".join(
                chars
            )
        )

    return "\n".join(
        cleaned_lines
    )


# ============================================================
# Balanced Braces
# ============================================================

def _extract_balanced_braces(
    text: str,
    open_index: int,
) -> tuple[str, int] | None:
    """
    text[open_index] == "{"

    nested brace를 고려해 내부 text와
    닫는 brace 다음 index를 반환한다.
    """

    if (
        open_index >= len(text)
        or text[open_index] != "{"
    ):

        return None

    depth = 0

    content_start = (
        open_index + 1
    )

    i = open_index

    while i < len(text):

        char = text[i]

        if char == "{":

            depth += 1

        elif char == "}":

            depth -= 1

            if depth == 0:

                return (
                    text[
                        content_start:i
                    ],
                    i + 1,
                )

        i += 1

    return None


def _extract_command_argument(
    text: str,
    command_name: str,
) -> str:
    """
    예:

    \\title{My Paper}
    """

    pattern = re.compile(
        rf"\\{re.escape(command_name)}"
        rf"\s*(?:\[[^\]]*\]\s*)?\{{",
        flags=re.DOTALL,
    )

    match = pattern.search(
        text
    )

    if not match:

        return ""

    open_index = (
        match.end() - 1
    )

    result = (
        _extract_balanced_braces(
            text,
            open_index,
        )
    )

    if result is None:

        return ""

    value, _ = result

    return value


# ============================================================
# TeX -> Plain Text
# ============================================================

def _remove_environment(
    text: str,
    environment: str,
) -> str:

    pattern = re.compile(
        rf"\\begin\{{{re.escape(environment)}\}}"
        rf".*?"
        rf"\\end\{{{re.escape(environment)}\}}",
        flags=(
            re.DOTALL
            | re.IGNORECASE
        ),
    )

    return pattern.sub(
        "\n\n",
        text,
    )


def _replace_formatting_commands(
    text: str,
) -> str:
    """
    \\textbf{abc}
    \\emph{abc}
    등의 wrapper만 제거하고 내용은 유지.
    """

    commands = (
        "textbf",
        "textit",
        "emph",
        "texttt",
        "textrm",
        "textsf",
        "textsc",
        "underline",
        "mbox",
    )

    for _ in range(5):

        previous = text

        for command in commands:

            pattern = re.compile(
                rf"\\{command}\s*\{{([^{{}}]*)\}}"
            )

            text = pattern.sub(
                r"\1",
                text,
            )

        if text == previous:
            break

    return text


def _tex_to_plain(
    text: str,
) -> str:

    text = (
        _strip_comments(
            text
        )
    )

    # --------------------------------------------------------
    # remove non-body environments
    # --------------------------------------------------------

    for environment in (
        "figure",
        "figure*",
        "table",
        "table*",
        "tikzpicture",
        "algorithm",
        "algorithmic",
        "lstlisting",
        "verbatim",
        "thebibliography",
    ):

        text = _remove_environment(
            text,
            environment,
        )

    # --------------------------------------------------------
    # equations
    # --------------------------------------------------------

    equation_patterns = (
        r"\\begin\{equation\*?\}.*?\\end\{equation\*?\}",
        r"\\begin\{align\*?\}.*?\\end\{align\*?\}",
        r"\\begin\{gather\*?\}.*?\\end\{gather\*?\}",
        r"\\begin\{multline\*?\}.*?\\end\{multline\*?\}",
        r"\\\[.*?\\\]",
        r"\$\$.*?\$\$",
    )

    for pattern in equation_patterns:

        text = re.sub(
            pattern,
            "\n\n[EQUATION]\n\n",
            text,
            flags=re.DOTALL,
        )

    # inline math
    text = re.sub(
        r"(?<!\\)\$(.+?)(?<!\\)\$",
        r"\1",
        text,
        flags=re.DOTALL,
    )

    # --------------------------------------------------------
    # href
    # --------------------------------------------------------

    text = re.sub(
        r"\\href\s*\{[^{}]*\}\s*\{([^{}]*)\}",
        r"\1",
        text,
    )

    text = re.sub(
        r"\\url\s*\{([^{}]*)\}",
        r"\1",
        text,
    )

    # --------------------------------------------------------
    # commands we do not want in training text
    # --------------------------------------------------------

    text = re.sub(
        r"\\(?:cite|citep|citet|citealp|ref|eqref|autoref|label)"
        r"\s*(?:\[[^\]]*\]\s*)*\{[^{}]*\}",
        " ",
        text,
    )

    # --------------------------------------------------------
    # begin/end
    # --------------------------------------------------------

    text = re.sub(
        r"\\(?:begin|end)\{[^{}]*\}",
        "\n",
        text,
    )

    text = (
        _replace_formatting_commands(
            text
        )
    )

    # --------------------------------------------------------
    # common TeX escapes
    # --------------------------------------------------------

    replacements = {
        r"\%": "%",
        r"\&": "&",
        r"\_": "_",
        r"\#": "#",
        r"\$": "$",
        r"\{": "{",
        r"\}": "}",
        "~": " ",
        "``": '"',
        "''": '"',
        "---": "—",
        "--": "–",
    }

    for old, new in (
        replacements.items()
    ):

        text = text.replace(
            old,
            new,
        )

    # --------------------------------------------------------
    # remove remaining commands
    # --------------------------------------------------------

    text = re.sub(
        r"\\[A-Za-z@]+\*?"
        r"(?:\[[^\]]*\])?",
        " ",
        text,
    )

    # braces
    text = text.replace(
        "{",
        ""
    ).replace(
        "}",
        ""
    )

    # spaces but keep paragraph boundaries
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n[ \t]+",
        "\n",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ============================================================
# Metadata
# ============================================================

def _load_metadata(
    paper_dir: Path,
) -> dict:

    return _load_json(
        paper_dir
        / "metadata.json"
    )


# ============================================================
# Title / Abstract
# ============================================================

def _extract_title(
    tex: str,
    metadata: dict,
) -> str:

    metadata_title = str(
        metadata.get(
            "title",
            "",
        )
        or ""
    ).strip()

    if metadata_title:

        return metadata_title

    raw = (
        _extract_command_argument(
            tex,
            "title",
        )
    )

    return _normalize_space(
        _tex_to_plain(
            raw
        )
    )


def _extract_abstract(
    tex: str,
    metadata: dict,
) -> str:

    metadata_abstract = str(
        metadata.get(
            "abstract",
            "",
        )
        or ""
    ).strip()

    if metadata_abstract:

        return _normalize_space(
            metadata_abstract
        )

    match = re.search(
        r"\\begin\{abstract\}"
        r"(.*?)"
        r"\\end\{abstract\}",
        tex,
        flags=(
            re.DOTALL
            | re.IGNORECASE
        ),
    )

    if not match:

        return ""

    return _normalize_space(
        _tex_to_plain(
            match.group(
                1
            )
        )
    )


# ============================================================
# Sections
# ============================================================

SECTION_LEVELS = {
    "section": 1,
    "subsection": 2,
    "subsubsection": 3,
}


def _find_section_commands(
    tex: str,
) -> list[dict]:

    pattern = re.compile(
        r"\\(section|subsection|subsubsection)"
        r"\*?"
        r"\s*"
        r"(?:\[[^\]]*\]\s*)?"
        r"\{"
    )

    results = []

    for match in pattern.finditer(
        tex
    ):

        command = (
            match.group(
                1
            )
        )

        open_index = (
            match.end() - 1
        )

        extracted = (
            _extract_balanced_braces(
                tex,
                open_index,
            )
        )

        if extracted is None:

            continue

        heading_raw, end_index = (
            extracted
        )

        heading = _normalize_space(
            _tex_to_plain(
                heading_raw
            )
        )

        if not heading:

            continue

        results.append(
            {
                "command": (
                    command
                ),

                "level": (
                    SECTION_LEVELS[
                        command
                    ]
                ),

                "heading": (
                    heading
                ),

                "start": (
                    match.start()
                ),

                "body_start": (
                    end_index
                ),
            }
        )

    return results


def _paragraph_blocks(
    body: str,
) -> list[dict]:

    plain = (
        _tex_to_plain(
            body
        )
    )

    if not plain:

        return []

    blocks = []

    paragraphs = re.split(
        r"\n\s*\n",
        plain,
    )

    for paragraph in paragraphs:

        paragraph = (
            _normalize_space(
                paragraph
            )
        )

        if not paragraph:

            continue

        if (
            paragraph
            == "[EQUATION]"
        ):

            block_type = (
                "equation"
            )

        else:

            block_type = (
                "paragraph"
            )

        blocks.append(
            {
                "type": (
                    block_type
                ),

                "text": (
                    paragraph
                ),
            }
        )

    return blocks


def _extract_sections(
    tex: str,
) -> list[dict]:

    commands = (
        _find_section_commands(
            tex
        )
    )

    sections = []

    # --------------------------------------------------------
    # no section commands
    # --------------------------------------------------------

    if not commands:

        body = tex

        document_match = re.search(
            r"\\begin\{document\}",
            tex,
        )

        if document_match:

            body = tex[
                document_match.end():
            ]

        blocks = (
            _paragraph_blocks(
                body
            )
        )

        if blocks:

            sections.append(
                {
                    "heading": (
                        "Document Body"
                    ),

                    "level": 1,

                    "blocks": (
                        blocks
                    ),
                }
            )

        return sections

    # --------------------------------------------------------
    # content before first section
    # --------------------------------------------------------

    prefix = tex[
        :commands[0][
            "start"
        ]
    ]

    document_match = re.search(
        r"\\begin\{document\}",
        prefix,
    )

    if document_match:

        prefix = prefix[
            document_match.end():
        ]

    # abstract and title macros should not duplicate
    prefix = re.sub(
        r"\\begin\{abstract\}.*?\\end\{abstract\}",
        " ",
        prefix,
        flags=(
            re.DOTALL
            | re.IGNORECASE
        ),
    )

    prefix = re.sub(
        r"\\title\s*\{.*?\}",
        " ",
        prefix,
        flags=re.DOTALL,
    )

    prefix_blocks = (
        _paragraph_blocks(
            prefix
        )
    )

    if prefix_blocks:

        sections.append(
            {
                "heading": (
                    "Document Body"
                ),

                "level": 1,

                "blocks": (
                    prefix_blocks
                ),
            }
        )

    # --------------------------------------------------------
    # actual sections
    # --------------------------------------------------------

    for index, item in enumerate(
        commands
    ):

        if (
            index + 1
            < len(commands)
        ):

            body_end = (
                commands[
                    index + 1
                ][
                    "start"
                ]
            )

        else:

            body_end = len(
                tex
            )

        body = tex[
            item[
                "body_start"
            ]:body_end
        ]

        blocks = (
            _paragraph_blocks(
                body
            )
        )

        sections.append(
            {
                "heading": (
                    item[
                        "heading"
                    ]
                ),

                "level": (
                    item[
                        "level"
                    ]
                ),

                "blocks": (
                    blocks
                ),
            }
        )

    return sections


# ============================================================
# Render Raw Content
# ============================================================

def _render_raw_content(
    *,
    title: str,
    abstract: str,
    sections: list[dict],
) -> str:

    parts = []

    if title:

        parts.append(
            title
        )

    if abstract:

        parts.append(
            "Abstract"
        )

        parts.append(
            abstract
        )

    for section in sections:

        heading = str(
            section.get(
                "heading",
                "",
            )
            or ""
        ).strip()

        if heading:

            parts.append(
                heading
            )

        for block in section.get(
            "blocks",
            [],
        ):

            text = str(
                block.get(
                    "text",
                    "",
                )
                or ""
            ).strip()

            if text:

                parts.append(
                    text
                )

    return "\n\n".join(
        parts
    ).strip()


# ============================================================
# Stats
# ============================================================

def _build_stats(
    raw_content: str,
    sections: list[dict],
) -> dict:

    paragraph_count = 0
    equation_count = 0
    figure_caption_count = 0

    for section in sections:

        for block in section.get(
            "blocks",
            [],
        ):

            block_type = (
                block.get(
                    "type"
                )
            )

            if (
                block_type
                == "paragraph"
            ):

                paragraph_count += 1

            elif (
                block_type
                == "equation"
            ):

                equation_count += 1

            elif (
                block_type
                == "figure_caption"
            ):

                figure_caption_count += 1

    return {
        "char_count": len(
            raw_content
        ),

        "word_count": len(
            raw_content.split()
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
    }


# ============================================================
# Parse One TeX
# ============================================================

def parse_tex_file(
    tex_path: Path,
) -> dict:

    tex = tex_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    metadata = (
        _load_metadata(
            tex_path.parent
        )
    )

    title = (
        _extract_title(
            tex,
            metadata,
        )
    )

    abstract = (
        _extract_abstract(
            tex,
            metadata,
        )
    )

    sections = (
        _extract_sections(
            tex
        )
    )

    raw_content = (
        _render_raw_content(
            title=title,
            abstract=abstract,
            sections=sections,
        )
    )

    stats = (
        _build_stats(
            raw_content,
            sections,
        )
    )

    return {
        "title": (
            title
        ),

        "abstract": (
            abstract
        ),

        "sections": (
            sections
        ),

        "raw_content": (
            raw_content
        ),

        "stats": (
            stats
        ),
    }


# ============================================================
# Save
# ============================================================

def _save_parsed_document(
    paper_dir: Path,
    parsed: dict,
) -> None:

    raw_path = (
        paper_dir
        / "raw_content.txt"
    )

    parsed_path = (
        paper_dir
        / "parsed_document.json"
    )

    raw_path.write_text(
        parsed[
            "raw_content"
        ],
        encoding="utf-8",
    )

    payload = {
        "title": (
            parsed[
                "title"
            ]
        ),

        "abstract": (
            parsed[
                "abstract"
            ]
        ),

        "sections": (
            parsed[
                "sections"
            ]
        ),

        "stats": (
            parsed[
                "stats"
            ]
        ),
    }

    parsed_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# Find Files
# ============================================================

def _find_tex_files() -> list[
    Path
]:

    if not RESOLVED_ROOT.exists():

        raise FileNotFoundError(
            f"Resolved root not found: "
            f"{RESOLVED_ROOT}"
        )

    return sorted(
        RESOLVED_ROOT.glob(
            "*/*/paper.tex"
        )
    )


# ============================================================
# Report
# ============================================================

def _save_report(
    results: list[dict],
) -> None:

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

    REPORT_FILE.write_text(
        json.dumps(
            {
                "total": (
                    len(results)
                ),

                "success": (
                    success_count
                ),

                "failed": (
                    len(results)
                    - success_count
                ),

                "documents": (
                    results
                ),
            },
            ensure_ascii=False,
            indent=2,
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
        "TEAM B - arXiv TeX Parser"
    )

    print("=" * 70)

    print(
        f"Input root : "
        f"{RESOLVED_ROOT}"
    )

    tex_files = (
        _find_tex_files()
    )

    print(
        f"TeX files  : "
        f"{len(tex_files)}"
    )

    results = []

    for index, tex_path in enumerate(
        tex_files,
        start=1,
    ):

        paper_dir = (
            tex_path.parent
        )

        topic_axis = (
            paper_dir.parent.name
        )

        source_id = (
            paper_dir.name
        )

        print()
        print("-" * 70)

        print(
            f"[{index}/"
            f"{len(tex_files)}]"
        )

        print(
            f"[AXIS] "
            f"{topic_axis}"
        )

        print(
            f"[ID]   "
            f"{source_id}"
        )

        print(
            f"[FILE] "
            f"{tex_path}"
        )

        try:

            parsed = (
                parse_tex_file(
                    tex_path
                )
            )

            if not parsed[
                "raw_content"
            ].strip():

                raise RuntimeError(
                    "Parsed content is empty."
                )

            _save_parsed_document(
                paper_dir,
                parsed,
            )

            stats = (
                parsed[
                    "stats"
                ]
            )

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
                    "topic_axis": (
                        topic_axis
                    ),

                    "source_id": (
                        source_id
                    ),

                    "status": (
                        "success"
                    ),

                    "stats": (
                        stats
                    ),
                }
            )

        except Exception as exc:

            print(
                f"[FAIL] "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            results.append(
                {
                    "topic_axis": (
                        topic_axis
                    ),

                    "source_id": (
                        source_id
                    ),

                    "status": (
                        "failed"
                    ),

                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                }
            )

    _save_report(
        results
    )

    success_count = sum(
        1
        for item in results
        if item[
            "status"
        ] == "success"
    )

    print()
    print("=" * 70)

    print(
        "TeX parsing completed."
    )

    print("=" * 70)

    print(
        f"Total   : "
        f"{len(results)}"
    )

    print(
        f"Success : "
        f"{success_count}"
    )

    print(
        f"Failed  : "
        f"{len(results) - success_count}"
    )

    print(
        f"Report  : "
        f"{REPORT_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()