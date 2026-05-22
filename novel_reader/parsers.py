from __future__ import annotations

import base64
import posixpath
import re
from html import unescape
from pathlib import Path
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup

from .models import ParsedBook, ParsedSection, BookCoverInfo


CHAPTER_RE = re.compile(
    r"^\s*((chapter|book|part)\s+([ivxlcdm]+|\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b.*|prologue\b.*|epilogue\b.*)\s*$",
    re.IGNORECASE,
)


def extract_cover_info(path: Path) -> BookCoverInfo:
    """Extract cover image (base64 data-URI) and extra metadata for a book file."""
    suffix = path.suffix.lower()
    if suffix == ".epub":
        return _epub_cover_info(path)
    if suffix == ".pdf":
        return _pdf_cover_info(path)
    return _txt_cover_info(path)


def _epub_cover_info(path: Path) -> BookCoverInfo:
    """Extract cover image + series/genres from an EPUB."""
    try:
        from ebooklib import ITEM_IMAGE, epub
        book = epub.read_epub(str(path))

        # --- cover image ---
        cover_b64 = ""
        # Try common cover identifiers
        for item in book.get_items():
            if item.get_type() == ITEM_IMAGE:
                name_lower = (item.get_name() or "").lower()
                item_id_lower = (item.id or "").lower()
                if any(k in name_lower or k in item_id_lower for k in ("cover", "title")):
                    mt = item.media_type or "image/jpeg"
                    cover_b64 = f"data:{mt};base64,{base64.b64encode(item.get_content()).decode()}"
                    break
        # Fallback: first image item
        if not cover_b64:
            for item in book.get_items_of_type(ITEM_IMAGE):
                mt = item.media_type or "image/jpeg"
                cover_b64 = f"data:{mt};base64,{base64.b64encode(item.get_content()).decode()}"
                break

        # --- series metadata ---
        series = ""
        try:
            belongs = book.get_metadata("OPF", "belongs-to-collection")
            if belongs:
                series = str(belongs[0][0]).strip()
        except Exception:
            pass
        if not series:
            try:
                series_meta = book.get_metadata("DC", "relation")
                if series_meta:
                    series = str(series_meta[0][0]).strip()
            except Exception:
                pass

        # --- genres/subjects ---
        genres = ""
        try:
            subjects = book.get_metadata("DC", "subject")
            if subjects:
                genres = ", ".join(str(s[0]).strip() for s in subjects[:5])
        except Exception:
            pass

        return BookCoverInfo(cover_b64=cover_b64, series=series, genres=genres)
    except Exception:
        return BookCoverInfo()


def _pdf_cover_info(path: Path) -> BookCoverInfo:
    """Render first PDF page as a JPEG cover thumbnail."""
    try:
        import fitz
        doc = fitz.open(str(path))
        page = doc.load_page(0)
        # Render at 2x scale for a decent thumbnail
        mat = fitz.Matrix(1.5, 1.5)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("jpeg")
        doc.close()
        cover_b64 = f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode()}"
        return BookCoverInfo(cover_b64=cover_b64)
    except Exception:
        return BookCoverInfo()


def _txt_cover_info(path: Path) -> BookCoverInfo:
    """Generate a styled SVG placeholder cover for text files."""
    # We return an empty string so the UI generates a CSS-based cover
    return BookCoverInfo()


def parse_book(path: Path) -> ParsedBook:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return parse_txt(path)
    if suffix == ".epub":
        return parse_epub(path)
    if suffix == ".pdf":
        return parse_pdf(path)
    raise ValueError(f"Unsupported book format: {suffix}")


def parse_txt(path: Path) -> ParsedBook:
    raw = path.read_text(encoding="utf-8", errors="replace")
    text = _normalize_text(raw)
    lines = text.splitlines()
    chapter_starts = [i for i, line in enumerate(lines) if CHAPTER_RE.match(line)]

    sections: list[ParsedSection] = []
    if chapter_starts:
        starts = chapter_starts + [len(lines)]
        preface = "\n".join(lines[: chapter_starts[0]]).strip()
        if preface:
            sections.append(ParsedSection(0, "Opening", preface, "txt:opening"))
        for pos, start in enumerate(chapter_starts):
            end = starts[pos + 1]
            title = lines[start].strip() or f"Chapter {pos + 1}"
            body = "\n".join(lines[start + 1 : end]).strip()
            if body:
                sections.append(ParsedSection(len(sections), title, body, f"txt:{start + 1}"))
    else:
        chunks = _chunk_text(text, max_chars=6500)
        sections = [
            ParsedSection(index=i, title=f"Section {i + 1}", text=chunk, source_locator=f"txt:section:{i + 1}")
            for i, chunk in enumerate(chunks)
        ]

    return ParsedBook(_clean_title(path.stem), "Unknown author", "txt", sections or [_empty_section()])


def parse_epub(path: Path) -> ParsedBook:
    from ebooklib import ITEM_DOCUMENT, ITEM_IMAGE
    from ebooklib import epub

    book = epub.read_epub(str(path))
    title = _metadata_value(book, "title") or _clean_title(path.stem)
    author = _metadata_value(book, "creator") or "Unknown author"
    images = _epub_images(book, ITEM_IMAGE)
    sections: list[ParsedSection] = []

    for item in _epub_documents(book, ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        for tag in soup(["script", "style", "nav"]):
            tag.decompose()
        body = soup.body or soup
        html = _clean_epub_html(body, item.get_name(), images)
        title_tag = soup.find(["h1", "h2", "h3"])
        section_title = title_tag.get_text(" ", strip=True) if title_tag else item.get_name()
        text = _normalize_text(body.get_text("\n"))
        if text or "<img" in html or "<hr" in html:
            sections.append(
                ParsedSection(
                    index=len(sections),
                    title=section_title or f"Chapter {len(sections) + 1}",
                    text=text or section_title or "Illustration",
                    source_locator=f"epub:{item.get_name()}",
                    html=html,
                )
            )

    if not sections:
        sections = [_empty_section()]
    return ParsedBook(unescape(title), unescape(author), "epub", sections)


def _epub_documents(book, item_document) -> list:
    documents = []
    seen = set()
    for spine_item in book.spine:
        item_id = spine_item[0] if isinstance(spine_item, tuple) else spine_item
        item = book.get_item_with_id(item_id)
        if item and item.get_type() == item_document:
            documents.append(item)
            seen.add(item.get_name())
    for item in book.get_items_of_type(item_document):
        if item.get_name() not in seen:
            documents.append(item)
    return documents


def _epub_images(book, item_image) -> dict[str, str]:
    images = {}
    for item in book.get_items_of_type(item_image):
        media_type = item.media_type or "image/png"
        data = base64.b64encode(item.get_content()).decode("ascii")
        images[item.get_name()] = f"data:{media_type};base64,{data}"
        images[posixpath.basename(item.get_name())] = images[item.get_name()]
    return images


def _clean_epub_html(body, document_name: str, images: dict[str, str]) -> str:
    allowed = {
        "a", "b", "blockquote", "br", "code", "div", "em", "h1", "h2", "h3", "h4",
        "hr", "i", "img", "li", "ol", "p", "pre", "section", "span", "strong", "u", "ul",
    }
    for tag in body.find_all(True):
        if tag.name not in allowed:
            tag.unwrap()
            continue
        if tag.name == "img":
            src = _resolve_epub_src(document_name, tag.get("src") or "", images)
            if not src:
                tag.decompose()
                continue
            tag.attrs = {"src": src, "alt": tag.get("alt", "")}
        elif tag.name == "a":
            tag.attrs = {}
        else:
            tag.attrs = {}
    return "".join(str(child) for child in body.contents).strip()


def _resolve_epub_src(document_name: str, src: str, images: dict[str, str]) -> str:
    clean = unquote(urlsplit(src).path)
    if not clean:
        return ""
    candidates = [
        clean,
        clean.lstrip("/"),
        posixpath.normpath(posixpath.join(posixpath.dirname(document_name), clean)),
        posixpath.basename(clean),
    ]
    for candidate in candidates:
        if candidate in images:
            return images[candidate]
    return ""


def parse_pdf(path: Path) -> ParsedBook:
    import fitz

    document = fitz.open(path)
    sections: list[ParsedSection] = []
    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        text = _normalize_text(page.get_text("text"))
        if text:
            sections.append(
                ParsedSection(
                    index=len(sections),
                    title=f"Page {page_index + 1}",
                    text=text,
                    source_locator=f"pdf:page:{page_index + 1}",
                )
            )
    document.close()
    return ParsedBook(_clean_title(path.stem), "Unknown author", "pdf", sections or [_empty_section()])


def _metadata_value(book, key: str) -> str:
    values = book.get_metadata("DC", key)
    if not values:
        return ""
    return str(values[0][0]).strip()


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_text(text: str, max_chars: int) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for paragraph in paragraphs:
        if current and current_size + len(paragraph) > max_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_size = 0
        current.append(paragraph)
        current_size += len(paragraph)
    if current:
        chunks.append("\n\n".join(current))
    return chunks or [text[:max_chars]]


def _clean_title(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").strip().title() or "Untitled"


def _empty_section() -> ParsedSection:
    return ParsedSection(0, "Empty Book", "No readable text was found in this file.", "empty")
