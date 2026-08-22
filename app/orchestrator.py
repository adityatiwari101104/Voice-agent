import re
from dataclasses import asdict

from .agent import create_paper_request
from .schemas import Section
from .tools.validator import validate_marks
from .tools.search import search_web
from .tools.reference_loader import load_reference_pages
from .tools.format_analyzer import analyze_format, PaperFormat
from .tools.board_pattern import build_board_pattern
from .tools.question_generator import generate_questions
from .tools.pdf_builder import build_pdf


ROMAN_NUMERALS = {
    1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI",
    7: "VII", 8: "VIII", 9: "IX", 10: "X", 11: "XI", 12: "XII"
}


def _class_roman(class_level: str) -> str | None:
    """'Class 10' -> 'X'. CBSE SQP/marking-scheme URLs use roman
    numerals for the class (ClassX_, ClassXII_, etc)."""

    match = re.search(r"(\d{1,2})", class_level)

    if not match:
        return None

    number = int(match.group(1))

    return ROMAN_NUMERALS.get(number)


def _class_match_score(url: str, target_roman: str | None) -> int:
    """Score (or heavily penalize) a candidate URL based on whether its
    class-level roman numeral actually matches the requested class.

    Without this, 'Class 10' and 'Class 12' Maths PDFs both look like
    valid matches on subject alone, and the wrong one can silently win.
    """

    if not target_roman:
        return 0

    match = re.search(r"class[_\-]?([ivxIVX]+)[_\-]", url, re.IGNORECASE)

    if not match:
        return 0

    found_roman = match.group(1).upper()

    if found_roman == target_roman:
        return 100

    return -1000


# CBSE doesn't split every subject out separately at every class level --
# most notably, Class 10 has no standalone Physics/Chemistry/Biology
# board paper, only a combined "Science" paper. Without accounting for
# this, a search for "Class 10 Physics" finds no real match and can
# silently fall back to picking the best-scoring WRONG subject/class
# entirely (e.g. a Class 12 Maths paper), rather than correctly
# recognizing "Science" as the right official document.
SUBJECT_SYNONYMS = {
    "physics": ["physics", "science"],
    "chemistry": ["chemistry", "science"],
    "biology": ["biology", "science"],
    "science": ["science"],
    "maths": ["maths", "math", "mathematics"],
    "mathematics": ["maths", "math", "mathematics"],
    "social science": ["social science", "sst"],
    "computer science": ["computer science", "computer applications"],
}


def _subject_keywords(subject: str) -> list[str]:
    normalized = subject.lower().strip()

    for key, variants in SUBJECT_SYNONYMS.items():
        if key in normalized:
            return variants

    return [normalized.replace(" ", "")]


def _subject_matches(subject: str, text: str) -> bool:
    """Hard requirement, not a soft bonus: does this candidate document
    plausibly correspond to the requested subject at all? If nothing
    found passes this, the caller should treat it as no-match-found and
    fall back gracefully, rather than accepting the least-bad wrong
    subject."""

    normalized_text = (
        text.lower().replace("-", "").replace("_", "").replace(" ", "")
    )

    for keyword in _subject_keywords(subject):
        if keyword.replace(" ", "") in normalized_text:
            return True

    return False


def _needs_board_format(user_input: str) -> bool:
    text = user_input.lower()

    keywords = [
        "board paper",
        "board exam",
        "board format",
        "board pattern",
        "board-style",
        "board style",
        "cbse",
        "ncert",
        "icse",
        "state board",
        "sample question paper",
        "previous year paper",
        "previous-year paper",
        "pyq"
    ]

    return any(keyword in text for keyword in keywords)


def _find_reference_pdf(request):
    """Find the official Sample Question Paper (SQP) PDF, generically
    for whatever subject/class was actually requested."""

    query = (
        f"official CBSE {request.class_level} "
        f"{request.subject} sample question paper SQP PDF"
    )

    results = search_web(
        query=query,
        max_results=10,
        domains=["cbseacademic.nic.in"]
    )

    target_roman = _class_roman(request.class_level)

    candidates = []

    for result in results:
        title = result.get("title", "")
        url = result.get("url", "")
        content = result.get("content", "")

        if ".pdf" not in url.lower():
            continue

        text = f"{title} {url} {content}".lower()

        if "sample question paper" not in text and "sqp" not in text:
            continue

        if "marking scheme" in text or "-ms.pdf" in url.lower():
            continue

        # Hard requirement: reject candidates that don't plausibly
        # match the subject at all, instead of just soft-scoring them.
        if not _subject_matches(request.subject, text):
            continue

        score = 0

        score += _class_match_score(url, target_roman)

        if "sqp.pdf" in url.lower():
            score += 50

        if request.subject.lower().replace(" ", "") in url.lower().replace("-", "").replace("_", ""):
            score += 40

        if "2025" in url.lower() or "2026" in url.lower():
            score += 20

        if request.subject.lower() in text:
            score += 20

        candidates.append((score, url))

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    best_score, best_url = candidates[0]

    # If even the best candidate is a confirmed wrong-class match,
    # refuse rather than silently returning the wrong grade's paper.
    if best_score <= -1000:
        return None

    return best_url


def _find_marking_scheme_pdf(request, sqp_url: str | None = None):
    """Find the official Marking Scheme PDF that pairs with the SQP,
    generically for whatever subject/class was actually requested.

    If sqp_url is given, prefer a marking scheme matching the same
    exam variant (e.g. CBSE Maths Standard vs Maths Basic use separate
    SQP + MS pairs -- mixing them would be internally inconsistent).
    """

    query = (
        f"official CBSE {request.class_level} "
        f"{request.subject} marking scheme PDF"
    )

    preferred_variant = None
    conflicting_variant = None

    if sqp_url:
        sqp_url_lower = sqp_url.lower()
        if "standard" in sqp_url_lower:
            preferred_variant = "standard"
            conflicting_variant = "basic"
        elif "basic" in sqp_url_lower:
            preferred_variant = "basic"
            conflicting_variant = "standard"

    results = search_web(
        query=query,
        max_results=10,
        domains=["cbseacademic.nic.in"]
    )

    target_roman = _class_roman(request.class_level)

    candidates = []

    for result in results:
        title = result.get("title", "")
        url = result.get("url", "")
        content = result.get("content", "")

        if ".pdf" not in url.lower():
            continue

        text = f"{title} {url} {content}".lower()

        if "marking scheme" not in text and "-ms.pdf" not in url.lower():
            continue

        # Hard requirement, same as the SQP search: don't accept a
        # marking scheme for a subject that doesn't plausibly match.
        if not _subject_matches(request.subject, text):
            continue

        score = 0

        score += _class_match_score(url, target_roman)

        if preferred_variant:
            if preferred_variant in url.lower():
                score += 60
            elif conflicting_variant and conflicting_variant in url.lower():
                score -= 200

        if "ms.pdf" in url.lower():
            score += 50

        if request.subject.lower().replace(" ", "") in url.lower().replace("-", "").replace("_", ""):
            score += 40

        if "2025" in url.lower() or "2026" in url.lower():
            score += 20

        if request.subject.lower() in text:
            score += 20

        candidates.append((score, url))

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    best_score, best_url = candidates[0]

    if best_score <= -1000:
        return None

    return best_url


def _board_pattern_to_paper_format(board_pattern) -> PaperFormat:
    """Build a PaperFormat from board_pattern's own extracted data.

    Used as a fallback when analyze_format() fails -- Groq's structured
    output for the full nested PaperFormat schema is occasionally
    unreliable on complex board papers (it can return stringified JSON
    instead of real nested objects for later array items). board_pattern
    extracts header_structure/general_instructions via plain text
    parsing, which doesn't have that failure mode, so it's a safe
    fallback for what pdf_builder actually needs.
    """

    return PaperFormat(
        exam_name=board_pattern.exam_name,
        board_name=board_pattern.board_name,
        class_level=board_pattern.class_level,
        subject=board_pattern.subject,
        duration_minutes=board_pattern.duration_minutes,
        total_marks=board_pattern.total_marks,
        header_structure=board_pattern.header_structure,
        general_instructions=board_pattern.general_instructions,
        total_questions=board_pattern.total_questions,
        sections=[],
        question_numbering_style="numeric continuous",
        marks_display_style="marks next to question",
        options_style="lettered options (A) (B) (C) (D)",
        internal_choice_style=None,
        layout_notes=[]
    )


def _create_board_sections(board_pattern):
    sections = []

    for pattern in board_pattern.patterns:
        sections.append(
            Section(
                type=pattern.question_type,
                count=pattern.count,
                marks_each=pattern.marks_each
            )
        )

    return sections


def run_orchestrator(user_input: str):
    request = create_paper_request(user_input)

    is_board_request = _needs_board_format(user_input)

    paper_format = None
    reference_url = None
    marking_scheme_url = None
    board_pattern = None
    board_error = None

    if is_board_request:
        reference_url = _find_reference_pdf(request)
        marking_scheme_url = _find_marking_scheme_pdf(request, sqp_url=reference_url)

        if reference_url and marking_scheme_url:
            board_pattern = build_board_pattern(
                sqp_url=reference_url,
                marking_scheme_url=marking_scheme_url,
                board_name="Central Board of Secondary Education",
                class_level=request.class_level,
                subject=request.subject
            )

            reference_pages = load_reference_pages(reference_url)

            try:
                paper_format = analyze_format(reference_pages)
            except Exception as exc:
                # format_analyzer's nested schema is occasionally
                # unreliable on complex board papers. Don't let a
                # flaky formatting call take down the whole pipeline --
                # fall back to board_pattern's own (more reliable)
                # header/instructions extraction instead.
                board_error = (
                    f"Format analysis failed ({exc}); using board "
                    f"pattern's own layout data instead."
                )
                paper_format = _board_pattern_to_paper_format(board_pattern)

            teacher_specified_sections = bool(request.sections)

            if not teacher_specified_sections:
                # Teacher didn't dictate their own structure (e.g. "make
                # me a board paper on Real Numbers, 80 marks") -- use
                # the official document's own section/marks pattern as
                # the sensible default.
                request.sections = _create_board_sections(board_pattern)
                request.total_marks = board_pattern.total_marks
                request.duration_minutes = board_pattern.duration_minutes
            else:
                # Teacher explicitly dictated their own section/marks
                # structure -- respect it. The official document's
                # header/instructions describe a DIFFERENT exam (its
                # own marks total, its own class label), so using them
                # here would produce a header that contradicts the
                # actual paper's content (e.g. "Max Marks: 24" on an
                # 80-mark paper). Discard paper_format and let
                # pdf_builder fall back to its generic header, which
                # accurately reflects the teacher's real request.
                paper_format = None
                board_error = (
                    board_error
                    or "Using the teacher's own section structure; "
                    "official board reference was not used for "
                    "header/instructions, since it describes a "
                    "different paper's marks and class."
                )
        else:
            # Could not find an official SQP + marking scheme pair for
            # this board/class/subject. Fall back to the teacher's own
            # requested sections instead of hard-failing, and flag it.
            missing = []
            if not reference_url:
                missing.append("sample question paper")
            if not marking_scheme_url:
                missing.append("marking scheme")

            board_error = (
                f"Could not find an official {' and '.join(missing)} "
                f"for {request.class_level} {request.subject}. "
                f"Falling back to the requested paper structure instead "
                f"of a board-accurate pattern."
            )

    validation = validate_marks(
        request.total_marks,
        [
            {
                "type": section.type,
                "count": section.count,
                "marks_each": section.marks_each
            }
            for section in request.sections
        ]
    )

    if not validation["valid"]:
        return {
            "success": False,
            "stage": "needs_clarification",
            "request": request.model_dump(),
            "validation": validation,
            "board_reference_used": is_board_request,
            "reference_url": reference_url,
            "marking_scheme_url": marking_scheme_url,
            "board_pattern": (
                asdict(board_pattern)
                if board_pattern
                else None
            ),
            "paper_format": (
                paper_format.model_dump()
                if paper_format
                else None
            ),
            "board_error": board_error,
            "message": (
                f"The generated paper pattern adds up to "
                f"{validation['calculated_marks']} marks, but the "
                f"requested paper is {validation['requested_marks']} marks."
            )
        }

    generated_paper = generate_questions(request)

    pdf_path = build_pdf(
        paper_request=request,
        generated_paper=generated_paper,
        paper_format=paper_format
    )

    return {
        "success": True,
        "stage": "completed",
        "request": request.model_dump(),
        "validation": validation,
        "board_reference_used": is_board_request,
        "reference_url": reference_url,
        "marking_scheme_url": marking_scheme_url,
        "board_pattern": (
            asdict(board_pattern)
            if board_pattern
            else None
        ),
        "paper_format": (
            paper_format.model_dump()
            if paper_format
            else None
        ),
        "board_error": board_error,
        "generated_paper": generated_paper.model_dump(),
        "pdf_path": pdf_path,
        "message": "Question paper generated successfully."
    }