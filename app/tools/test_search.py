from .search import search_web


query = "CBSE Class 10 Science Light Human Eye question paper PDF"

results = search_web(query)

for result in results:
    print("\nTITLE:", result["title"])
    print("URL:", result["url"])
    print("CONTENT:", result["content"][:500])