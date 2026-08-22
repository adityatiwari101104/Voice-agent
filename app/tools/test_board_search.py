from .search import search_web


results = search_web(
    "CBSE Class 10 Science sample question paper PDF",
    max_results=5,
    domains=["cbseacademic.nic.in"]
)

for i, result in enumerate(results, 1):
    print(f"\nRESULT {i}")
    print("TITLE:", result["title"])
    print("URL:", result["url"])
    print("CONTENT:", result["content"][:1000])