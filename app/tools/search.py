import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

api_key = os.getenv("TAVILY_API_KEY")

if not api_key:
    raise RuntimeError("TAVILY_API_KEY not found in .env")

client = TavilyClient(api_key=api_key)


def search_web(query: str, max_results: int = 5, domains: list[str] | None = None):
    search_kwargs = {
        "query": query,
        "search_depth": "advanced",
        "max_results": max_results
    }

    if domains:
        search_kwargs["include_domains"] = domains

    response = client.search(**search_kwargs)

    return [
        {
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "content": result.get("content", "")
        }
        for result in response.get("results", [])
    ]