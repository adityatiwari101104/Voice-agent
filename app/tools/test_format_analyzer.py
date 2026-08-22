from .search import search_web
from .reference_loader import load_reference_pages
from .format_analyzer import analyze_format


results = search_web(
    "CBSE Class 10 Science sample question paper SQP PDF",
    max_results=10,
    domains=["cbseacademic.nic.in"]
)

pdf_url = None

for result in results:
    title = result.get("title", "").lower()
    url = result.get("url", "").lower()

    if ".pdf" not in url:
        continue

    if "science-sqp.pdf" in url or "sample question paper" in title:
        pdf_url = result["url"]
        break

if not pdf_url:
    raise RuntimeError("Official CBSE Science SQP PDF not found")

print("Using reference:", pdf_url)

reference_pages = load_reference_pages(pdf_url)

print("Pages extracted:", len(reference_pages))

paper_format = analyze_format(reference_pages)

print("\nFORMAT ANALYSIS:\n")
print(paper_format.model_dump_json(indent=2))