from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    KeepTogether,
)


OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def _safe(text) -> str:
    """Escape text for ReportLab's Paragraph, which parses its input
    as mini-XML (tags like <b>, <br/>). Any raw '<', '>', or '&' in
    generated question/instruction text -- e.g. math like "0 < x < 2"
    -- would otherwise be misread as a broken tag and crash the whole
    PDF build. Always route dynamic text through this before it
    reaches Paragraph()."""

    return escape(str(text))


def build_pdf(paper_request, generated_paper, paper_format=None):
    output_path = OUTPUT_DIR / "question_paper.pdf"

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "PaperTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=14,
        leading=18,
        spaceAfter=6,
    )

    header_style = ParagraphStyle(
        "Header",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        leading=14,
        spaceAfter=3,
    )

    instruction_style = ParagraphStyle(
        "Instruction",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        spaceAfter=3,
    )

    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=11,
        leading=14,
        spaceBefore=8,
        spaceAfter=6,
    )

    question_style = ParagraphStyle(
        "Question",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        spaceAfter=5,
    )

    option_style = ParagraphStyle(
        "Option",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13,
        leftIndent=12,
        spaceAfter=2,
    )

    story = []

    if paper_format and paper_format.header_structure:
        for line in paper_format.header_structure:
            if line and str(line).lower() != "null":
                story.append(
                    Paragraph(
                        _safe(line),
                        header_style,
                    )
                )
    else:
        story.append(
            Paragraph(
                f"Class {_safe(paper_request.class_level)} - "
                f"{_safe(paper_request.subject)}",
                title_style,
            )
        )

    story.append(Spacer(1, 4))

    # If paper_format.header_structure was used above, it already
    # includes its own "Max. Marks" / "Time Allowed" lines -- adding
    # these again duplicates them. Only add the generic versions when
    # we fell back to the plain "Class X - Subject" title.
    used_format_header = bool(paper_format and paper_format.header_structure)

    if not used_format_header:
        if paper_request.total_marks is not None:
            story.append(
                Paragraph(
                    f"Maximum Marks: {paper_request.total_marks}",
                    header_style,
                )
            )

        if paper_request.duration_minutes is not None:
            story.append(
                Paragraph(
                    f"Time Allowed: {paper_request.duration_minutes} minutes",
                    header_style,
                )
            )

    story.append(Spacer(1, 8))

    if paper_format and paper_format.general_instructions:
        story.append(
            Paragraph(
                "General Instructions",
                section_style,
            )
        )

        for instruction in paper_format.general_instructions:
            if instruction and str(instruction).lower() != "null":
                story.append(
                    Paragraph(
                        f"- {_safe(instruction)}",
                        instruction_style,
                    )
                )

    question_number = 1

    for section in generated_paper.sections:
        is_case_study = "case" in section.type.lower()

        story.append(
            Paragraph(
                f"Section - {_safe(section.type)}",
                section_style,
            )
        )

        for question in section.questions:
            block = []

            # Case-study questions embed their own line-broken passage
            # + labeled sub-parts in the text; preserve those line
            # breaks (ReportLab collapses raw '\n' into a single space
            # otherwise -- it needs an explicit <br/> tag).
            question_text = _safe(question.question)

            if is_case_study:
                question_text = question_text.replace("\n", "<br/>")

            block.append(
                Paragraph(
                    f"{question_number}. {question_text}",
                    question_style,
                )
            )

            if question.options:
                for index, option in enumerate(question.options):
                    label = chr(65 + index)

                    block.append(
                        Paragraph(
                            f"({label}) {_safe(option)}",
                            option_style,
                        )
                    )

            marks_label = (
                f"[Total: {section.marks_each} marks]"
                if is_case_study
                else f"[{section.marks_each} marks]"
            )

            block.append(
                Paragraph(
                    marks_label,
                    instruction_style,
                )
            )

            story.append(
                KeepTogether(block)
            )

            question_number += 1

    doc.build(story)

    return str(output_path)