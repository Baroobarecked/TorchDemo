from html.parser import HTMLParser
from io import BytesIO
import re
from urllib.parse import urljoin, urlparse
import httpx
from pypdf import PdfReader
from langchain_core.documents import Document
from config import DATA_PATH
from pathlib import Path
from prefect import flow, task, get_run_logger
from database import open_database
from populate_database import split_documents, calculate_chunk_ids, add_to_postgres

INDEX_URL = "https://www.boardgamecapital.com/board-game-rules.htm"


def game_key(name): # Machine Generated
    key = "-".join(re.findall(r"[a-z0-9]+", name.lower()))
    if not key:
        raise ValueError("Enter a game name")
    return key


class Links(HTMLParser): # Machine Generated
    def __init__(self, html):
        super().__init__()
        self.links, self.href, self.label = [], None, []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.href, self.label = dict(attrs).get("href"), []

    def handle_data(self, data):
        if self.href:
            self.label.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self.href:
            self.links.append((self.href, " ".join(" ".join(self.label).split())))
            self.href = None


def fetch(url, skip_url_check=False): # Machine Generated
    for _ in range(5):
        parsed = urlparse(url)
        print(parsed)
        if (parsed.scheme != "https" or parsed.hostname not in {"www.boardgamecapital.com", "boardgamecapital.com"}) and not skip_url_check:
            raise ValueError("Rulebook links must stay on BoardGameCapital over HTTPS")
        response = httpx.get(url, timeout=60, follow_redirects=False)
        if response.is_redirect:
            url = urljoin(url, response.headers["location"])
            continue
        response.raise_for_status()
        return response
    raise ValueError("Too many source redirects")


@task(persist_result=False)
def download_rulebook(game): # Partial Machine Generated, Partial Human Generated
    folder = DATA_PATH
    path = folder / (game_key(game) + "_0.pdf")
    # Machine Generated
    if path.exists():
        get_run_logger().info("Reusing downloaded rulebook: %s", path.name)
        return {"path": str(path), "downloaded": False}
    index = Links(fetch(INDEX_URL).text)
    match = next((href for href, label in index.links if re.search(r"[a-z0-9]", label.lower()) and game_key(label) == game_key(game)), None)
    if not match:
        raise ValueError(f"No exact game match on BoardGameCapital: {game}")
    print(match)
    page_url : str = urljoin(INDEX_URL, match)
    links = Links(fetch(page_url).text)

    # Human Generated
    pdf_urls : list[str] = [
        urljoin(page_url, href) 
        for href, _ in links.links
        if urlparse(href).path.lower().endswith(".pdf")
    ]

    if not len(pdf_urls) > 0:
        raise ValueError(f"No rulebook PDF found for {game}")

    pdf_count_id : int = 0
    for pdf_url in pdf_urls:
        content = path.read_bytes() if path.exists() else fetch(pdf_url, True).content
        if not content.startswith(b"%PDF-"):
            continue
        reader = PdfReader(BytesIO(content))
        if not any((page.extract_text() or "").strip() for page in reader.pages):
            continue
        if not path.exists():
            temporary = path.with_suffix(".part")
            temporary.write_bytes(content)
            temporary.replace(path)
        get_run_logger().info("Rulebook ready: %s", pdf_url)
        pdf_count_id += 1

    if not pdf_count_id > 0:
        raise ValueError("No valid PDFs found")
    
    return {"path": str(path), "downloaded": True}


@task(persist_result=False, log_prints=True)
def extract_and_store(info): # Machine Generated
    reader = PdfReader(info["path"])
    pages = [Document(page_content=page.extract_text() or "", metadata={
        "source": info["path"], "page": number, "game": info["game"],
        "title": info["title"], "source_url": info["source_url"],
    }) for number, page in enumerate(reader.pages)]
    chunks = calculate_chunk_ids(split_documents(pages))
    if not chunks:
        raise ValueError("Rulebook produced no text chunks")
    with open_database() as db:
        add_to_postgres(db, chunks)
    return len(chunks)


@flow(name="ingest-game-rulebook", persist_result=False)
def ingest_rulebook(game): # Machine Generated
    directory = Path(DATA_PATH)
    filename = game_key(game)
    file_path = directory / filename

    if not file_path.is_file():
        info = download_rulebook(game)
        count = extract_and_store(info)
    return {"downloaded": info["downloaded"], "chunks": count, "source_url": info["source_url"]}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--game")
    ingest_rulebook(parser.parse_args().game)
