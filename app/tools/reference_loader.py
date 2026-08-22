import io
import requests
import urllib3
from pypdf import PdfReader

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def load_reference_pages(url: str) -> list[str]:
    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"},
        verify=False
    )

    response.raise_for_status()

    reader = PdfReader(io.BytesIO(response.content))

    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text.strip())

    return [page for page in pages if page]


def load_reference(url: str) -> str:
    pages = load_reference_pages(url)
    return "\n\n".join(pages)