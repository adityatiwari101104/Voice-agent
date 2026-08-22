"""
Run this from your project root (venv activated) to see exactly what's
happening in the marking-scheme pipeline before it hits build_board_pattern.

    python debug_board_pattern.py
"""

from app.agent import create_paper_request
from app.orchestrator import _find_reference_pdf, _find_marking_scheme_pdf
from app.tools.reference_loader import load_reference

user_input = "Create a Class 10 CBSE Maths board paper on Real Numbers, 80 marks"

request = create_paper_request(user_input)
print("=== PaperRequest ===")
print(request.model_dump())

sqp_url = _find_reference_pdf(request)
ms_url = _find_marking_scheme_pdf(request, sqp_url=sqp_url)

print("\n=== URLs found ===")
print("SQP URL:", sqp_url)
print("Marking Scheme URL:", ms_url)

if not ms_url:
    print("\nNo marking scheme URL found at all -- this is a search problem, not extraction.")
    raise SystemExit

print("\n=== Loading SQP text (for general instructions) ===")
from app.tools.reference_loader import load_reference as _load_ref
sqp_text = _load_ref(sqp_url)
from app.tools.board_pattern import _extract_instructions, _analyze_general_instructions
instructions = _extract_instructions(sqp_text)
print("Instructions extracted:", len(instructions), "lines")
for line in instructions:
    print(" -", line)

if instructions:
    print("\n=== Running Groq extraction on general instructions ===")
    patterns = _analyze_general_instructions("\n".join(instructions))
    print("Patterns returned:", len(patterns))
    for p in patterns:
        print(p)
else:
    print("\nNo general instructions extracted -- regex boundary probably didn't match this SQP's wording. Need to see raw SQP text around 'General Instructions'.")
    print("\n--- SQP text search for 'General Instructions' ---")
    idx = sqp_text.lower().find("general instructions")
    if idx >= 0:
        print(sqp_text[idx:idx+2000])
    else:
        print("'General Instructions' not found in SQP text at all.")