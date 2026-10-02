"""Load a student's PDF notes into Chroma: PDF -> pages -> topic per page -> chunks -> Chroma.

Run:  python -m notes_loader.load_notes data/sample_notes/biology.pdf sara
"""
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from pypdf import PdfReader
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"
COLLECTION = "notes"
PAGES_PER_CALL = 10       # pages per LLM call when assigning topics (fewer calls = cheaper)
PAGES_PER_TOPIC = 6       # about 1 topic per 6 pages, so topics stay broad like sections
PREVIEW_CHARS = 400       # how much of each page the LLM sees when making the topic list


# ---------- topic tagging in 2 passes (structured output) ----------
# Pass 1: look at the start of EVERY page at once -> the topic list (like a table of contents)
# Pass 2: give every page one topic FROM that list

class TopicList(BaseModel):
    topics: list[str]


class PageTopic(BaseModel):
    page: int
    topic: str


class PageTopics(BaseModel):
    pages: list[PageTopic]


llm = ChatOpenAI(model="gpt-5.4-nano")
list_llm = llm.with_structured_output(TopicList)
page_llm = llm.with_structured_output(PageTopics)

LIST_PROMPT = """Here is the start of every page of a student's study notes.
Make the list of the main topics a student would be quizzed on, like sections of a textbook.
- Use at most {max_topics} topics, each a short name (1-4 words), in the same language as the notes.
- Ignore page headers and footers (book or chapter titles repeated on every page): the name of
  the whole document or chapter is NOT a topic. Use the section headings inside the text.

{pages}"""

PAGE_PROMPT = """Give every page of a student's study notes exactly one topic from this list
(copy the name exactly): {topics}

{pages}"""


def find_topics(pages: dict[int, str]) -> list[str]:
    """Pass 1: the topic list for the whole document."""
    max_topics = max(2, len(pages) // PAGES_PER_TOPIC)
    preview = "\n".join(f"page {n}: {' '.join(text.split())[:PREVIEW_CHARS]}" for n, text in pages.items())
    result = list_llm.invoke(LIST_PROMPT.format(max_topics=max_topics, pages=preview))
    return [t.strip().lower() for t in result.topics]


def tag_topics(pages: dict[int, str], topic_list: list[str] | None = None) -> dict[int, str]:
    """{page_number: text} -> {page_number: topic}. Pass topic_list to reuse known topic names."""
    topic_list = topic_list or find_topics(pages)
    topics: dict[int, str] = {}
    numbers = list(pages)
    for i in range(0, len(numbers), PAGES_PER_CALL):
        batch = numbers[i:i + PAGES_PER_CALL]
        text = "\n\n".join(f"--- page {n} ---\n{pages[n][:1500]}" for n in batch)
        result = page_llm.invoke(PAGE_PROMPT.format(topics=", ".join(topic_list), pages=text))
        for item in result.pages:
            topics[item.page] = item.topic.strip().lower()

    # if the LLM made up a name that is not in the list, use the topic of the page before
    previous = topic_list[0]
    for n in numbers:
        if topics.get(n) not in topic_list:
            topics[n] = previous
        previous = topics[n]
    return topics


# ---------- Chroma ----------

def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=str(CHROMA_DIR),
    )


def load_notes(pdf_path: str, student_id: str) -> dict[str, int]:
    """Load one PDF for one student. Returns {topic: number_of_pages}."""
    file_name = Path(pdf_path).name

    # 1. Read the text of every page (page numbers start at 1, like in a PDF viewer)
    reader = PdfReader(pdf_path)
    pages = {n: page.extract_text() or "" for n, page in enumerate(reader.pages, start=1)}
    pages = {n: text for n, text in pages.items() if text.strip()}     # skip empty pages
    print(f"Read {len(pages)} pages with text from {file_name}")

    # 2. Give every page a topic. Uploaded this file before? Reuse its topic names,
    #    so the student's mastery (which is counted per topic name) stays connected.
    vectorstore = get_vectorstore()
    old = vectorstore.get(where={"$and": [{"student_id": student_id}, {"file": file_name}]}, include=["metadatas"])
    old_topics = sorted({m["topic"] for m in old["metadatas"]})
    topics = tag_topics(pages, old_topics or None)

    # 3. Split pages into chunks; each chunk remembers student, file, page and topic
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks, ids = [], []
    for n, text in pages.items():
        for i, piece in enumerate(splitter.split_text(text)):
            chunks.append(Document(
                page_content=piece,
                metadata={"student_id": student_id, "file": file_name, "page": n,
                          "topic": topics.get(n, "general"),
                          "path": str(Path(pdf_path).resolve())},      # so the app can show the page
            ))
            ids.append(f"{student_id}:{file_name}:p{n}:c{i}")
    print(f"Split into {len(chunks)} chunks")

    # 4. Save to Chroma (remove the older upload of the same file first, so nothing is doubled)
    if old["ids"]:
        vectorstore.delete(ids=old["ids"])
        print(f"Removed {len(old['ids'])} chunks from an older upload")
    vectorstore.add_documents(chunks, ids=ids)

    pages_per_topic = dict(Counter(topics.get(n, "general") for n in pages))
    print(f"Saved to Chroma. Topics: {pages_per_topic}")
    return pages_per_topic


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m notes_loader.load_notes <pdf_path> <student_id>")
        sys.exit(1)
    load_notes(sys.argv[1], sys.argv[2])
