from __future__ import annotations

import posixpath
import re
from urllib.parse import quote_plus

import gradio as gr

from .ingestion import IngestionPipeline
from .storage import LibraryStore


store = LibraryStore()
pipeline = IngestionPipeline(store)


def build_dashboard_app() -> gr.Blocks:
    with gr.Blocks(title="Dashboard") as demo:
        selected_id = gr.State(None)
        library = gr.HTML()
        detail = gr.HTML()
        with gr.Row(elem_classes=["hidden"]):
            select = gr.Button("Select", elem_id="action-select-novel")
            clear = gr.Button("Clear", elem_id="action-clear-select")
            star = gr.Button("Star", elem_id="action-star")
            archive = gr.Button("Archive", elem_id="action-archive")
            delete = gr.Button("Delete", elem_id="action-delete")
        upload = gr.File(label="Add TXT, EPUB, or PDF", file_types=[".txt", ".epub", ".pdf"])
        refresh = gr.Button("Refresh")
        msg = gr.HTML()

        demo.load(_dashboard_load, outputs=[library, detail, selected_id, msg])
        refresh.click(_dashboard_refresh, outputs=[library, detail, selected_id, msg])
        upload.upload(_upload, upload, [library, detail, selected_id, msg])
        select.click(_dashboard_select, outputs=[library, detail, selected_id, msg])
        clear.click(_dashboard_clear, outputs=[library, detail, selected_id, msg])
        star.click(_toggle_star, selected_id, [library, detail, selected_id, msg])
        archive.click(_toggle_archive, selected_id, [library, detail, selected_id, msg])
        delete.click(_delete, selected_id, [library, detail, selected_id, msg])
    return demo


def build_reader_app() -> gr.Blocks:
    with gr.Blocks(title="Reader") as demo:
        state = gr.State(_state())
        with gr.Row(elem_classes=["reader-grid"]):
            with gr.Column(scale=1, min_width=230, elem_classes=["chapter-rail"]):
                gr.HTML("<a class='rail-link' href='/dashboard/'>Dashboard</a>")
                chapters = gr.Radio(label="Chapters", choices=[], interactive=True, elem_classes=["chapters"])
            with gr.Column(scale=5, min_width=560, elem_classes=["reader-body"]):
                top = gr.HTML()
                page = gr.HTML(_empty_page())
                settings_panel = gr.HTML(elem_id="settings-popup-panel")
                msg = gr.HTML()
                with gr.Row(elem_classes=["hidden"]):
                    prev = gr.Button("prev", elem_id="nr-prev")
                    next_ = gr.Button("next", elem_id="nr-next")
                    back = gr.Button("back", elem_id="nr-back")
                    forward = gr.Button("forward", elem_id="nr-forward")
                    font_up = gr.Button("font up", elem_id="nr-font-up")
                    font_down = gr.Button("font down", elem_id="nr-font-down")
                    sepia = gr.Button("sepia", elem_id="nr-sepia")
                    light = gr.Button("light", elem_id="nr-light")
                    dark = gr.Button("dark", elem_id="nr-dark")
                    speak = gr.Button("speak", elem_id="nr-speak")
                    selected = gr.Textbox(elem_id="nr-selected-text", container=False, show_label=False)
                    bookmark = gr.Button("bookmark", elem_id="nr-bookmark")
                    define = gr.Button("define", elem_id="nr-define")
                    speak_selected = gr.Button("speak selected", elem_id="nr-speak-selected")
                    target_jump = gr.Textbox(elem_id="nr-target-jump", container=False, show_label=False)
                    do_jump = gr.Button("jump", elem_id="nr-do-jump")

        demo.load(_reader_load, outputs=[page, top, chapters, settings_panel, msg, state])
        chapters.change(_jump, [chapters, state], [page, top, chapters, settings_panel, msg, state])
        prev.click(_move, [state, gr.State(-1)], [page, top, chapters, settings_panel, msg, state])
        next_.click(_move, [state, gr.State(1)], [page, top, chapters, settings_panel, msg, state])
        back.click(_history, [state, gr.State("back")], [page, top, chapters, settings_panel, msg, state])
        forward.click(_history, [state, gr.State("forward")], [page, top, chapters, settings_panel, msg, state])
        font_up.click(_font, [state, gr.State(1)], [page, top, chapters, settings_panel, msg, state])
        font_down.click(_font, [state, gr.State(-1)], [page, top, chapters, settings_panel, msg, state])
        sepia.click(_theme, [state, gr.State("sepia")], [page, top, chapters, settings_panel, msg, state])
        light.click(_theme, [state, gr.State("light")], [page, top, chapters, settings_panel, msg, state])
        dark.click(_theme, [state, gr.State("dark")], [page, top, chapters, settings_panel, msg, state])
        speak.click(_speak, [state, gr.State("")], msg)
        speak_selected.click(_speak, [state, selected], msg)
        bookmark.click(_bookmark, [state, selected], [page, top, chapters, settings_panel, msg, state])
        define.click(_define, [state, selected], [page, top, chapters, settings_panel, msg, state])
        do_jump.click(_jump, [target_jump, state], [page, top, chapters, settings_panel, msg, state])
    return demo


def _dashboard_load(request: gr.Request):
    novel_id = _request_id(request)
    return _dashboard_outputs(novel_id)


def _dashboard_refresh():
    return _dashboard_outputs(None, "Dashboard refreshed.")


def _upload(file):
    if not file:
        return _dashboard_outputs(None, "Choose a file first.", "warn")
    try:
        pipeline.ingest(file.name)
        return _dashboard_outputs(None, "Added. Parsing continues in the background.")
    except Exception as exc:
        return _dashboard_outputs(None, f"Upload failed: {exc}", "warn")


def _toggle_star(novel_id):
    novel_id = _int(novel_id)
    if novel_id:
        store.toggle_starred(novel_id)
        return _dashboard_outputs(novel_id, "Star status updated.")
    return _dashboard_outputs(None, "Choose a book first.", "warn")


def _toggle_archive(novel_id):
    novel_id = _int(novel_id)
    if novel_id:
        store.toggle_archived(novel_id)
        return _dashboard_outputs(novel_id, "Read status updated.")
    return _dashboard_outputs(None, "Choose a book first.", "warn")


def _delete(novel_id):
    novel_id = _int(novel_id)
    if novel_id:
        store.delete_novel(novel_id)
        return _dashboard_outputs(None, "Book deleted.")
    return _dashboard_outputs(None, "Choose a book first.", "warn")


def _dashboard_select(request: gr.Request):
    novel_id = _request_id(request)
    return _dashboard_outputs(novel_id)


def _dashboard_clear():
    return _dashboard_outputs(None)



def _dashboard_outputs(novel_id: int | None, message: str = "", tone: str = "ok"):
    novel = store.get_novel(novel_id) if novel_id else None
    return (
        _dashboard(novel_id),
        _detail(novel),
        novel["id"] if novel else None,
        _notice(message, tone),
    )


def _dashboard(selected_id: int | None = None) -> str:
    active = [n for n in store.list_novels(include_archived=True) if not n.get("archived")]
    archived = [n for n in store.list_novels(include_archived=True) if n.get("archived")]
    return f"""
    <main class="dashboard">
      <header>
        <div>
          <p>Novel Reader</p>
          <h1>Dashboard</h1>
          <small>{len(active)} active · {len(archived)} archived</small>
        </div>
        <div class="dashboard-theme-selector">
          <button data-theme-set="sepia" class="theme-btn sepia-btn">Sepia</button>
          <button data-theme-set="light" class="theme-btn light-btn">Light</button>
          <button data-theme-set="dark" class="theme-btn dark-btn">Dark</button>
        </div>
      </header>
      <h2>Library</h2>
      <section class="cards">{''.join(_card(n, selected_id) for n in active) or "<div class='empty'>No books yet. Add a novel to begin.</div>"}</section>
      <h2>Archive</h2>
      <section class="cards archive">{''.join(_card(n, selected_id) for n in archived) or "<div class='empty'>Completed books will appear here.</div>"}</section>
    </main>
    """


def _get_cover_html(novel: dict) -> str:
    cover_image = novel.get("cover_image")
    if cover_image:
        return f'<img class="book-cover-img" src="{cover_image}" alt="Cover" />'
        
    # Beautiful text fallback cover
    first_text = novel.get("first_section_text") or ""
    # Strip HTML tags just in case
    first_text = re.sub(r'<[^>]*>', '', first_text)
    preview = first_text.strip()[:140]
    if len(first_text) > 140:
        preview += "..."
    if not preview:
        preview = "No text content available."
        
    return f"""
    <div class="book-cover-fallback">
      <div class="fallback-header">{_escape(novel['file_format'].upper())}</div>
      <div class="fallback-title">{_escape(novel['title'])}</div>
      <div class="fallback-divider"></div>
      <div class="fallback-body">{_escape(preview)}</div>
      <div class="fallback-footer">{_escape(novel['author'])}</div>
    </div>
    """


def _card(novel: dict, selected_id: int | None) -> str:
    count = int(novel.get("section_count") or 0)
    done = int(novel.get("progress_section") or 0)
    progress = int(((done + 1) / count) * 100) if count else 0
    selected = " selected" if novel["id"] == selected_id else ""
    
    cover_html = _get_cover_html(novel)
    
    star_badge = ""
    if novel.get("starred"):
        star_badge = '<div class="star-badge" title="Starred">★</div>'
        
    return f"""
    <a class="book-card{selected}" href="/dashboard?novel_id={novel['id']}">
      <div class="card-cover-container">
        {cover_html}
        {star_badge}
      </div>
      <div class="card-info">
        <span class="card-format">{_escape(novel['file_format'].upper())}</span>
        <strong class="card-title">{_escape(novel['title'])}</strong>
        <small class="card-author">{_escape(novel['author'])}</small>
        <div class="card-progress"><b style="width:{progress}%"></b></div>
        <span class="card-progress-text">{progress}% read</span>
      </div>
    </a>
    """


def _detail(novel: dict | None) -> str:
    if not novel:
        return ""
        
    novel_id = novel["id"]
    starred = bool(novel.get("starred"))
    archived = bool(novel.get("archived"))
    
    # SVG icons for logo buttons
    star_icon = """<svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z"/></svg>"""
    if starred:
        star_icon = """<svg class="icon-svg filled" viewBox="0 0 24 24" fill="currentColor"><path d="M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z"/></svg>"""
        
    archive_icon = """<svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>"""
    if archived:
        archive_icon = """<svg class="icon-svg filled" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>"""
        
    delete_icon = """<svg class="icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg>"""
    
    continue_icon = """<svg class="icon-svg" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>"""
    
    meta_items = []
    
    author = novel.get("author")
    if author and author != "Unknown author":
        meta_items.append(f"""
        <div class="meta-row">
          <span class="meta-label">Author</span>
          <span class="meta-value">{_escape(author)}</span>
        </div>
        """)
        
    series = novel.get("series")
    if series:
        meta_items.append(f"""
        <div class="meta-row">
          <span class="meta-label">Series</span>
          <span class="meta-value">{_escape(series)}</span>
        </div>
        """)
        
    genres = novel.get("genres")
    if genres:
        meta_items.append(f"""
        <div class="meta-row">
          <span class="meta-label">Genres</span>
          <span class="meta-value">{_escape(genres)}</span>
        </div>
        """)
        
    count = int(novel.get("section_count") or 0)
    done = int(novel.get("progress_section") or 0)
    percent = int(((done + 1) / count) * 100) if count else 0
    meta_items.append(f"""
    <div class="meta-row">
      <span class="meta-label">Progress</span>
      <span class="meta-value">{percent}% ({done + 1} / {count} sections)</span>
    </div>
    """)
    
    meta_items.append(f"""
    <div class="meta-row">
      <span class="meta-label">Last Read</span>
      <span class="meta-value">{_format_date(novel.get("updated_at"))}</span>
    </div>
    """)
    
    file_size = novel.get("file_size") or 0
    if file_size:
        meta_items.append(f"""
        <div class="meta-row">
          <span class="meta-label">Size</span>
          <span class="meta-value">{_format_size(file_size)}</span>
        </div>
        """)
        
    metadata_html = "".join(meta_items)
    cover_html = _get_cover_html(novel)
    
    starred_class = "active" if starred else ""
    archived_class = "active" if archived else ""
    read_label = "Read Again" if archived else "Continue Reading"
    
    return f"""
    <div id="detail-modal" class="modal-overlay show" onclick="if(event.target === this) closeModal();">
      <div class="modal-content">
        <button class="modal-close" onclick="closeModal();" title="Close modal">&times;</button>
        
        <div class="modal-cover-wrapper">
          {cover_html}
        </div>
        
        <h1 class="modal-title">{_escape(novel['title'])}</h1>
        
        <div class="modal-actions">
          <!-- Star button -->
          <button class="action-btn star-btn {starred_class}" onclick="document.getElementById('action-star').click();" title="Toggle Star">
            {star_icon}
          </button>
          
          <!-- Have read button -->
          <button class="action-btn archive-btn {archived_class}" onclick="document.getElementById('action-archive').click();" title="Toggle Read Status">
            {archive_icon}
          </button>
          
          <!-- Delete button -->
          <button class="action-btn delete-btn" onclick="if(confirm('Are you sure you want to delete this book?')) document.getElementById('action-delete').click();" title="Delete Book">
            {delete_icon}
          </button>
          
          <!-- Continue reading button -->
          <a class="action-btn continue-btn" href="/reader/?novel_id={novel_id}" title="{read_label}">
            {continue_icon}
          </a>
        </div>
        
        <div class="modal-divider"></div>
        
        <div class="modal-metadata">
          {metadata_html}
        </div>
      </div>
    </div>
    """


def _format_size(bytes_val: int) -> str:
    if not bytes_val:
        return "Unknown size"
    val = float(bytes_val)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if val < 1024:
            return f"{val:.1f} {unit}"
        val /= 1024
    return f"{val:.1f} TB"


def _format_date(iso_str: str | None) -> str:
    if not iso_str:
        return "Never"
    try:
        parts = iso_str.split("T")
        date_part = parts[0]
        time_part = parts[1][:5] if len(parts) > 1 else ""
        return f"{date_part} {time_part}".strip()
    except Exception:
        return str(iso_str)



def _reader_load(request: gr.Request):
    novel_id = _request_id(request)
    if not novel_id:
        return _empty_page(), "", gr.update(choices=[]), "", _notice("Open a book from the dashboard.", "warn"), _state()
    novel = store.get_novel(novel_id)
    if not novel:
        return _empty_page(), "", gr.update(choices=[]), "", _notice("Book not found.", "warn"), _state()
    state = {**_state(), "novel_id": novel_id, "section": int(novel.get("progress_section") or 0)}
    return _render(state)


def _state() -> dict:
    return {"novel_id": None, "section": 0, "history": [], "future": [], "theme": "sepia", "font": 22}


def _move(state, delta):
    novel_id = state.get("novel_id")
    if not novel_id:
        return _render(state, "Open a book first.")
    count = store.section_count(novel_id)
    current = int(state["section"])
    target = max(0, min(count - 1, current + int(delta)))
    if target != current:
        state = {**state, "section": target, "history": state["history"] + [current], "future": []}
    return _render(state, save_progress=True)


def _jump(section, state):
    target = _int(section)
    if target is None:
        return _render(state)
    current = int(state["section"])
    if target != current:
        state = {**state, "section": target, "history": state["history"] + [current], "future": []}
    return _render(state, save_progress=True)


def _history(state, direction):
    history, future = list(state["history"]), list(state["future"])
    current = int(state["section"])
    if direction == "back" and history:
        state = {**state, "section": history.pop(), "history": history, "future": [current] + future}
    if direction == "forward" and future:
        state = {**state, "section": future.pop(0), "history": history + [current], "future": future}
    return _render(state, save_progress=True)


def _font(state, delta):
    novel_id = state.get("novel_id")
    novel = store.get_novel(novel_id) if novel_id else None
    is_pdf = novel and novel.get("file_format") == "pdf"
    
    if is_pdf:
        current_zoom = int(state.get("zoom", 100))
        zoom_change = int(delta) * 10
        new_zoom = max(50, min(200, current_zoom + zoom_change))
        return _render({**state, "zoom": new_zoom})
    else:
        return _render({**state, "font": max(16, min(32, int(state["font"]) + int(delta)))})


def _theme(state, theme):
    return _render({**state, "theme": theme})


def _bookmark(state, text):
    text = (text or "").strip()
    novel_id = state.get("novel_id")
    if novel_id and text:
        store.add_bookmark(novel_id, int(state["section"]), text)
        return _render(state, "Bookmarked.")
    return _render(state, "Select text first.")


def _define(state, text):
    text = (text or "").strip()
    novel_id = state.get("novel_id")
    if novel_id and text:
        store.add_dictionary_lookup(novel_id, int(state["section"]), text, _meaning_url(text))
        return _render(state, f"Defined: {text}")
    return _render(state, "Select word first.")


def _speak(state, text):
    detail = f": {_escape((text or '').strip()[:48])}" if (text or "").strip() else ""
    return _notice(f"Speak queued for future TTS{detail}.")


def _render(state, message: str = "", save_progress: bool = False):
    novel_id = state.get("novel_id")
    if not novel_id:
        return _empty_page(), "", gr.update(choices=[]), "", _notice(message, "warn") if message else "", state
    novel = store.get_novel(novel_id)
    if not novel:
        return _empty_page(), "", gr.update(choices=[]), "", _notice("Book not found.", "warn"), state
    if novel["status"] != "complete":
        return _waiting_page(novel, state), _top(novel, None, state), _chapter_update(novel_id, state), _settings_panel(novel_id, state), _notice(novel["status_message"]), state

    count = store.section_count(novel_id)
    state = {**state, "section": max(0, min(count - 1, int(state["section"])))}
    if save_progress:
        store.update_progress(novel_id, int(state["section"]))
    section = store.get_section(novel_id, int(state["section"]))
    return _page(section, count, state), _top(novel, section, state), _chapter_update(novel_id, state), _settings_panel(novel_id, state), _notice(message), state


def _clean_chapter_title(title: str, index: int) -> str:
    if not title:
        return f"Chapter {index + 1}"
    title = title.strip()
    # Check if it looks like a filepath or ends with file extensions
    if "/" in title or "\\" in title or title.lower().endswith((".html", ".xhtml", ".xml", ".htm")):
        basename = posixpath.basename(title.replace("\\", "/"))
        # split_003 -> Chapter 4
        num_match = re.search(r"split_(\d+)", basename)
        if num_match:
            return f"Chapter {int(num_match.group(1)) + 1}"
        # generic number search, e.g. chapter_04 -> Chapter 4
        num_match_generic = re.search(r"(\d+)", basename)
        if num_match_generic:
            return f"Chapter {int(num_match_generic.group(1))}"
        # stem title fallback
        stem = posixpath.splitext(basename)[0]
        stem = stem.replace("_", " ").replace("-", " ").title()
        return stem or f"Chapter {index + 1}"
    return title


def _top(novel: dict, section: dict | None, state: dict) -> str:
    count = store.section_count(novel["id"]) if novel and novel["status"] == "complete" else 0
    progress = f"{section['section_index'] + 1}/{count}" if section else ""
    title = _clean_chapter_title(section["title"], section["section_index"]) if section else novel["title"]
    return f"""
    <header class="top theme-{state['theme']}">
      <div><small>{_escape(novel['title'])}</small><strong>{_escape(title)}</strong></div>
      <nav>
        <a href="/dashboard/">Dashboard</a>
        <button data-click="nr-back">Back</button><button data-click="nr-prev">Prev</button>
        <span>{_escape(progress)}</span>
        <button data-click="nr-next">Next</button><button data-click="nr-forward">Forward</button>
        <button class="settings-trigger" title="Settings">
          <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: block;">
            <circle cx="12" cy="12" r="3"></circle>
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
          </svg>
        </button>
      </nav>
    </header>
    """


def _chapter_update(novel_id: int, state: dict):
    choices = [(_clean_chapter_title(s["title"], s["section_index"]), s["section_index"]) for s in store.list_sections(novel_id)]
    return gr.update(choices=choices, value=int(state["section"]))


def _highlight_phrases(html: str, phrases: list[str], css_class: str) -> str:
    if not phrases:
        return html
    phrases = sorted(list(set(phrases)), key=len, reverse=True)
    parts = re.split(r'(<[^>]+>)', html)
    for i in range(len(parts)):
        if parts[i].startswith('<') and parts[i].endswith('>'):
            continue
        for phrase in phrases:
            if not phrase.strip():
                continue
            escaped = re.escape(phrase)
            replacement = f'<span class="{css_class}" data-phrase="{_escape(phrase)}">{phrase}</span>'
            parts[i] = re.sub(escaped, replacement, parts[i])
    return "".join(parts)


def _page(section: dict | None, count: int, state: dict) -> str:
    if not section:
        return _empty_page()
    progress = int(((section["section_index"] + 1) / max(count, 1)) * 100)
    content = section["html"] or _paragraphs(section["text"])
    
    novel_id = state.get("novel_id")
    if novel_id:
        bookmarks = store.get_section_bookmarks(novel_id, section["section_index"])
        lookups = store.get_section_lookups(novel_id, section["section_index"])
        content = _highlight_phrases(content, bookmarks, "bookmark-dotted")
        content = _highlight_phrases(content, lookups, "define-dotted")

    zoom_val = state.get("zoom", 100) / 100.0
    return f"""
    <article class="page theme-{state['theme']}" style="--font:{state['font']}px; --pdf-zoom:{zoom_val}">
      <div class="progress"><span style="width:{progress}%"></span></div>
      <div class="text">{content}</div>
    </article>
    """


def _waiting_page(novel: dict, state: dict) -> str:
    return f"<article class='page theme-{state['theme']}'><div class='empty'><h2>{_escape(novel['title'])}</h2><p>{_escape(novel['status_message'])}</p></div></article>"


def _empty_page() -> str:
    return "<article class='page theme-sepia'><div class='empty'><h2>Open a book from your dashboard.</h2></div></article>"


def _settings_panel(novel_id: int | None, state: dict) -> str:
    if not novel_id:
        return ""
    bookmarks = store.list_all_bookmarks(novel_id)
    bookmarks_html = ""
    for b in bookmarks:
        section_title = f"Ch {b['section_index'] + 1}"
        preview = _escape(b['label'][:45])
        if len(b['label']) > 45:
            preview += "..."
        bookmarks_html += f"""
        <div class="bookmark-item" data-goto-section="{b['section_index']}" data-goto-text="{_escape(b['label'])}">
          <span class="b-sec">{section_title}</span>
          <span class="b-text">"{preview}"</span>
        </div>
        """
    if not bookmarks:
        bookmarks_html = "<p class='no-bookmarks'>No bookmarks in this book.</p>"
        
    novel = store.get_novel(novel_id)
    is_pdf = novel and novel.get("file_format") == "pdf"
        
    theme = state.get("theme", "sepia")
    font = state.get("font", 22)
    active_sepia = "active" if theme == "sepia" else ""
    active_light = "active" if theme == "light" else ""
    active_dark = "active" if theme == "dark" else ""
    
    font_group_html = ""
    if is_pdf:
        zoom = state.get("zoom", 100)
        font_group_html = f"""
        <div class="settings-group">
          <h4>Zoom</h4>
          <div class="settings-row font-adjust">
            <button class="settings-action" data-click="nr-font-down">Zoom-</button>
            <span class="font-display">{zoom}%</span>
            <button class="settings-action" data-click="nr-font-up">Zoom+</button>
          </div>
        </div>
        """
    else:
        font = state.get("font", 22)
        font_group_html = f"""
        <div class="settings-group">
          <h4>Font Size</h4>
          <div class="settings-row font-adjust">
            <button class="settings-action" data-click="nr-font-down">A-</button>
            <span class="font-display">{font}px</span>
            <button class="settings-action" data-click="nr-font-up">A+</button>
          </div>
        </div>
        """
    
    return f"""
    <div class="settings-popover theme-{theme}">
      <!-- Screen 1: Main Settings -->
      <div class="settings-main-screen">
        {font_group_html}
        <div class="settings-group">
          <h4>Theme</h4>
          <div class="settings-row themes">
            <button class="theme-btn sepia {active_sepia}" data-theme-set="sepia">Sepia</button>
            <button class="theme-btn light {active_light}" data-theme-set="light">Light</button>
            <button class="theme-btn dark {active_dark}" data-theme-set="dark">Dark</button>
          </div>
        </div>
        <div class="settings-group" style="border-bottom: none; padding-bottom: 0; margin-bottom: 0;">
          <button class="theme-btn view-bookmarks-btn" style="display: flex; justify-content: space-between; align-items: center; width: 100% !important; background: var(--blue) !important; color: #ffffff !important; border-color: var(--blue) !important;">
            <span>View Bookmarks ({len(bookmarks)})</span>
            <span style="font-weight: bold; margin-left: 8px;">&rarr;</span>
          </button>
        </div>
      </div>
      
      <!-- Screen 2: Bookmarks Screen -->
      <div class="settings-bookmarks-screen" style="display: none;">
        <div class="settings-group" style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px; border-bottom: 1px dashed var(--line); padding-bottom: 8px;">
          <button class="back-to-settings-btn" style="background: transparent; border: none; color: var(--blue); cursor: pointer; font-size: 16px; padding: 0 4px 0 0; font-weight: bold; display: inline-flex; align-items: center;">&larr;</button>
          <h4 style="margin: 0 !important; color: var(--muted) !important; font-size: 11px !important; text-transform: uppercase !important; letter-spacing: 0.5px !important;">Bookmarks ({len(bookmarks)})</h4>
        </div>
        <div class="bookmarks-list">
          {bookmarks_html}
        </div>
      </div>
    </div>
    """


def _note(title: str, rows: list[dict], key: str) -> str:
    body = "".join(f"<p><b>{r['section_index'] + 1}</b>{_escape(str(r.get(key, ''))[:120])}</p>" for r in rows[:8]) or "<p>Nothing yet.</p>"
    return f"<div><h3>{title}</h3>{body}</div>"


def _paragraphs(text: str) -> str:
    return "".join(f"<p>{_escape(p).replace(chr(10), '<br>')}</p>" for p in text.split("\n\n") if p.strip())


def _request_id(request: gr.Request) -> int | None:
    params = dict(request.query_params) if request else {}
    return _int(params.get("novel_id"))


def _notice(text: str, tone: str = "ok") -> str:
    return f"<p class='notice {tone}'>{_escape(text)}</p>" if text else ""


def _meaning_url(text: str) -> str:
    return f"https://www.google.com/search?q={quote_plus(text + ' meaning')}"


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _escape(value) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


READER_JS = r"""
// Apply saved theme immediately so there's no flash
(function() {
  var t = localStorage.getItem('nr-theme') || 'sepia';
  document.documentElement.setAttribute('data-theme', t);
})();

const q = (sel, root = document) => root.querySelector(sel);
const setBox = (id, value) => {
  const field = q(`#${id} textarea, #${id} input`);
  if (field) { field.value = value; field.dispatchEvent(new Event("input", { bubbles: true })); }
};

// Fixed Gradio 4 element-level click interceptor mapping
const click = (id) => {
  const el = q(`#${id}`);
  if (!el) return;
  if (el.tagName === "BUTTON") {
    el.click();
  } else {
    el.querySelector("button")?.click();
  }
};

const selectionText = () => {
  const sel = getSelection();
  const text = sel && !sel.isCollapsed ? sel.toString().trim().replace(/\s+/g, " ") : "";
  const reader = q(".text");
  return text && reader?.contains(sel.anchorNode) && reader.contains(sel.focusNode) ? text : "";
};

// Theme ids that map to data-click -> localStorage key
const THEMES = { 'nr-sepia': 'sepia', 'nr-light': 'light', 'nr-dark': 'dark' };

function applyTheme(name) {
  localStorage.setItem('nr-theme', name);
  document.documentElement.setAttribute('data-theme', name);
}

function checkAndScrollBookmark() {
  if (!window.__gotoBookmarkText) return;
  const phrase = window.__gotoBookmarkText;
  const el = Array.from(document.querySelectorAll(".bookmark-dotted")).find(
    span => span.dataset.phrase === phrase || span.textContent.trim() === phrase
  );
  if (el) {
    el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    el.style.backgroundColor = 'rgba(255, 215, 0, 0.4)';
    setTimeout(() => { el.style.backgroundColor = 'transparent'; }, 2000);
    window.__gotoBookmarkText = null;
  }
}

// Seamlessly preserve settings popup state across Svelte server re-renders
function syncSettingsPopupState() {
  const panel = q(".settings-popover");
  if (!panel) return;
  if (window.__settingsPanelOpen) {
    if (!panel.classList.contains("show")) {
      panel.classList.add("show");
      const trigger = q(".settings-trigger");
      if (trigger) {
        const rect = trigger.getBoundingClientRect();
        panel.style.top = `${rect.bottom + 8}px`; // Fixed position
        panel.style.left = `${Math.max(10, Math.min(innerWidth - 320, rect.left + rect.width - 290))}px`;
      }
    }

    // Toggle main settings screen vs bookmarks list sub-screen
    const mainScreen = q(".settings-main-screen", panel);
    const bScreen = q(".settings-bookmarks-screen", panel);
    if (mainScreen && bScreen) {
      if (window.__settingsPanelBookmarksScreen) {
        mainScreen.style.display = "none";
        bScreen.style.display = "block";
      } else {
        mainScreen.style.display = "block";
        bScreen.style.display = "none";
      }
    }
  } else {
    panel.classList.remove("show");
  }
}

function bootReader() {
  if (window.__readerBooted) return;
  window.__readerBooted = true;
  
  const bar = document.createElement("div");
  bar.id = "select-bar";
  document.body.appendChild(bar);

  // Prevent selection clearing when clicking buttons in selection bar
  bar.addEventListener("mousedown", (e) => {
    e.preventDefault();
  });

  // Set up instant MutationObserver on settings popup container to prevent any flicker during Svelte DOM updates
  const observer = new MutationObserver(() => {
    syncSettingsPopupState();
  });
  const container = q("#settings-popup-panel") || document.body;
  observer.observe(container, { childList: true, subtree: true });

  // Poll for bookmark targets and keep settings open state synchronized
  setInterval(checkAndScrollBookmark, 300);
  setInterval(syncSettingsPopupState, 50);

  document.addEventListener("click", (event) => {
    // Intercept bookmarks clicks inside the settings popup
    const bookmarkItem = event.target.closest(".bookmark-item");
    if (bookmarkItem) {
      const section = bookmarkItem.dataset.gotoSection;
      const text = bookmarkItem.dataset.gotoText;
      window.__gotoBookmarkText = text;
      setBox("nr-target-jump", section);
      click("nr-do-jump");
      return;
    }

    // Toggle settings popup visibility & positioning
    const trigger = event.target.closest(".settings-trigger");
    const panel = q(".settings-popover");
    if (trigger) {
      event.preventDefault();
      if (panel) {
        panel.classList.toggle("show");
        window.__settingsPanelOpen = panel.classList.contains("show");
        if (window.__settingsPanelOpen) {
          const rect = trigger.getBoundingClientRect();
          panel.style.top = `${rect.bottom + 8}px`; // Fixed position
          panel.style.left = `${Math.max(10, Math.min(innerWidth - 320, rect.left + rect.width - 290))}px`;
        }
      }
      return;
    }

    // Sub-screen transition triggers inside settings panel
    if (event.target.closest(".view-bookmarks-btn")) {
      event.preventDefault();
      window.__settingsPanelBookmarksScreen = true;
      syncSettingsPopupState();
      return;
    }
    if (event.target.closest(".back-to-settings-btn")) {
      event.preventDefault();
      window.__settingsPanelBookmarksScreen = false;
      syncSettingsPopupState();
      return;
    }

    // Double-guard settings open state when clicking elements inside settings popover
    if (event.target.closest(".settings-popover")) {
      window.__settingsPanelOpen = true;
    }
    
    // Clicking outside closes the settings popup
    if (panel && !event.target.closest(".settings-popover") && !event.target.closest(".settings-trigger")) {
      // ONLY close if the click target is still in the document body (prevents closing on detached Svelte re-renders)
      if (document.body.contains(event.target)) {
        panel.classList.remove("show");
        window.__settingsPanelOpen = false;
      }
    }

    const proxy = event.target.closest("[data-click]");
    if (proxy) {
      const themeKey = THEMES[proxy.dataset.click];
      if (themeKey) applyTheme(themeKey);
      click(proxy.dataset.click);
      return;
    }

    const themeSet = event.target.closest("[data-theme-set]");
    if (themeSet) {
      const theme = themeSet.dataset.themeSet;
      applyTheme(theme);
      click(`nr-${theme}`);
      return;
    }

    const action = event.target.closest("#select-bar button");
    if (action) {
      const text = selectionText();
      setBox("nr-selected-text", text);
      if (action.dataset.a === "define" && text) {
        window.open(`https://www.google.com/search?q=${encodeURIComponent(text + " meaning")}`, "_blank", "noopener");
        click("nr-define");
      } else if (action.dataset.a === "bookmark" && text) {
        click("nr-bookmark");
      } else if (action.dataset.a === "speak-selected" && text) {
        click("nr-speak-selected");
      }
      bar.classList.remove("show");
      return;
    }
  });

  document.addEventListener("mouseup", (e) => {
    if (e.target.closest("#select-bar")) return; // ignore mouseup when interacting with the select bar itself!
    const text = selectionText();
    if (!text) return bar.classList.remove("show");

    // Dynamic selection popover button filters
    const hasSpace = /\s/.test(text);
    let buttonsHtml = `<button data-a='bookmark'>Bookmark</button>`;
    if (!hasSpace && text.length > 0) {
      buttonsHtml += `<button data-a='define'>Define</button>`;
    }
    buttonsHtml += `<button data-a='speak-selected'>Speak</button>`;
    bar.innerHTML = buttonsHtml;

    const rect = getSelection().getRangeAt(0).getBoundingClientRect();
    bar.style.left = `${Math.max(10, Math.min(innerWidth - 260, rect.left + rect.width / 2 - 120))}px`;
    bar.style.top = `${Math.max(10, scrollY + rect.top - 46)}px`;
    bar.classList.add("show");
  });

  // Force Svelte layout reflows so observers position dynamic chapters perfectly at 100% zoom
  [50, 150, 300, 600, 1200, 2500, 4000].forEach(delay => {
    setTimeout(() => {
      window.dispatchEvent(new Event('resize'));
      const rail = q('.chapter-rail');
      if (rail) {
        const d = rail.style.display;
        rail.style.display = 'none';
        rail.offsetHeight;
        rail.style.display = d;
      }
    }, delay);
  });
}
bootReader();
"""


# JS injected on both pages: reads localStorage theme and applies data-theme to <html>
THEME_JS = """
(function() {
  var t = localStorage.getItem('nr-theme') || 'sepia';
  document.documentElement.setAttribute('data-theme', t);

  document.addEventListener("click", function(e) {
    var card = e.target.closest(".book-card");
    if (card) {
      e.preventDefault();
      var href = card.getAttribute("href");
      window.history.pushState(null, "", href);
      var btn = document.getElementById("action-select-novel");
      if (btn) {
        btn.click();
      }
      return;
    }

    var btn = e.target.closest("[data-theme-set]");
    if (btn) {
      var theme = btn.getAttribute("data-theme-set");
      localStorage.setItem('nr-theme', theme);
      document.documentElement.setAttribute('data-theme', theme);
    }
  });
})();

function closeModal() {
  var modal = document.getElementById('detail-modal');
  if (modal) {
    modal.classList.remove('show');
  }
  window.history.pushState(null, "", "/dashboard/");
  var btn = document.getElementById("action-clear-select");
  if (btn) {
    btn.click();
  }
}
"""

CSS = """
/* ── Unified theme tokens ────────────────────────────────────────── */
:root,
html[data-theme=sepia] {
  --bg:#f4ead0; --paper:#f7edcf; --ink:#2b2118; --muted:#756854;
  --line:rgba(43,33,24,.18); --card-bg:rgba(244,235,208,.72);
  --card-hover:rgba(244,235,208,.96); --rail-bg:#2e2416;
  --rail-fg:#e8ddc8; --rail-border:rgba(255,255,255,.10);
  --blue:#2f80ed; --warn:#b84a18;
  --reader-bg:#f4ead0; --reader-ink:#2b2118;
  --reader-muted:#756854; --reader-line:rgba(43,33,24,.18);
  /* Gradio var overrides */
  --body-background-fill:#f4ead0; --body-text-color:#2b2118;
  --background-fill-primary:#f7edcf; --background-fill-secondary:#f0e3be;
  --block-background-fill:transparent; --block-border-color:rgba(43,33,24,.18);
  --border-color-primary:rgba(43,33,24,.18);
  --neutral-50:#f7edcf; --neutral-100:#f0e3be; --neutral-200:rgba(43,33,24,.18);
  --neutral-300:#a89880; --neutral-400:#756854; --neutral-500:#4a3828;
  --neutral-600:#2b2118; --neutral-700:#2b2118; --neutral-800:#2b2118; --neutral-900:#1a140f;
  --button-secondary-background-fill:#e8ddc8; --button-secondary-text-color:#2b2118;
  --button-secondary-text-color-hover:#2b2118;
  --input-background-fill:#f0e3be; --checkbox-label-text-color:#2b2118;
  --upload-text-color:#4a3828; --prose-text-color:#2b2118;
}
html[data-theme=light] {
  --bg:#f5f5f0; --paper:#ffffff; --ink:#1a1a1a; --muted:#5a5a5a;
  --line:rgba(0,0,0,.13); --card-bg:rgba(255,255,255,.80);
  --card-hover:rgba(255,255,255,.98); --rail-bg:#1e1e1e;
  --rail-fg:#e8e8e8; --rail-border:rgba(255,255,255,.10);
  --blue:#1a6ed8; --warn:#c0340d;
  --reader-bg:#f5f5f0; --reader-ink:#1a1a1a;
  --reader-muted:#5a5a5a; --reader-line:rgba(0,0,0,.13);
  /* Gradio var overrides */
  --body-background-fill:#f5f5f0; --body-text-color:#1a1a1a;
  --background-fill-primary:#ffffff; --background-fill-secondary:#f0f0eb;
  --block-background-fill:transparent; --block-border-color:rgba(0,0,0,.13);
  --border-color-primary:rgba(0,0,0,.13);
  --neutral-50:#ffffff; --neutral-100:#f0f0eb; --neutral-200:rgba(0,0,0,.13);
  --neutral-300:#999999; --neutral-400:#5a5a5a; --neutral-500:#3a3a3a;
  --neutral-600:#1a1a1a; --neutral-700:#1a1a1a; --neutral-800:#1a1a1a; --neutral-900:#000000;
  --button-secondary-background-fill:#e4e4e0; --button-secondary-text-color:#1a1a1a;
  --button-secondary-text-color-hover:#1a1a1a;
  --input-background-fill:#ececec; --checkbox-label-text-color:#1a1a1a;
  --upload-text-color:#3a3a3a; --prose-text-color:#1a1a1a;
}
html[data-theme=dark] {
  --bg:#141414; --paper:#1e1e1e; --ink:#e8e2d8; --muted:#9a9188;
  --line:rgba(232,226,216,.14); --card-bg:rgba(38,34,28,.80);
  --card-hover:rgba(50,45,38,.95); --rail-bg:#0d0d0d;
  --rail-fg:#d4cec4; --rail-border:rgba(255,255,255,.08);
  --blue:#5ba3f5; --warn:#f0956a;
  --reader-bg:#141414; --reader-ink:#e8e2d8;
  --reader-muted:#9a9188; --reader-line:rgba(232,226,216,.14);
  /* Gradio var overrides */
  --body-background-fill:#141414; --body-text-color:#e8e2d8;
  --background-fill-primary:#1e1e1e; --background-fill-secondary:#1a1a1a;
  --block-background-fill:transparent; --block-border-color:rgba(232,226,216,.14);
  --border-color-primary:rgba(232,226,216,.14); --neutral-100:#e8e2d8;
  --neutral-200:rgba(232,226,216,.14); --neutral-800:#e8e2d8;
  --button-secondary-background-fill:#2a2520; --button-secondary-text-color:#e8e2d8;
  --input-background-fill:#252220; --checkbox-label-text-color:#e8e2d8;
}
/* ── Base ─────────────────────────────────────────────────────────── */
body, .gradio-container { margin:0!important; padding:0!important; max-width:none!important; background:var(--bg)!important; color:var(--ink)!important; font-family:Inter,system-ui,sans-serif; transition:background .25s,color .25s; }
/* Override Gradio base-theme text-color resets on common elements */
.gradio-container p, .gradio-container span, .gradio-container h1, .gradio-container h2, .gradio-container h3, .gradio-container h4, .gradio-container a, .gradio-container label, .gradio-container dt, .gradio-container dd, .gradio-container li, .gradio-container summary, .gradio-container small, .gradio-container strong, .gradio-container em { color:inherit; }
/* Override Gradio widget backgrounds so they adapt to theme */
.gradio-container .block { background:transparent!important; border-color:var(--line)!important; }
/* File upload: force all text inside to be readable — Gradio uses neutral-400/500 which are light-gray by default */
.gradio-container .upload-container, .gradio-container .file-upload, .gradio-container .upload-btn { background:var(--card-bg)!important; border-color:var(--line)!important; color:var(--ink)!important; }
.gradio-container .upload-container *, .gradio-container .file-upload * { color:var(--ink)!important; }
/* upload-area .wrap blocks (NOT chapter-rail .wrap) */
.gradio-container .upload-container .wrap, .gradio-container .file-upload .wrap { background:var(--card-bg)!important; border-color:var(--line)!important; color:var(--ink)!important; }
.gradio-container .upload-container .wrap .icon-wrap, .gradio-container .upload-container .wrap .title, .gradio-container .upload-container .wrap p, .gradio-container .upload-container .wrap span,
.gradio-container .file-upload .wrap .icon-wrap, .gradio-container .file-upload .wrap .title, .gradio-container .file-upload .wrap p, .gradio-container .file-upload .wrap span { color:var(--ink)!important; }
.gradio-container button.primary { background:var(--blue)!important; color:#fff!important; border:none!important; }
.gradio-container button.secondary { background:var(--card-bg)!important; color:var(--ink)!important; border:1px solid var(--line)!important; }
/* Gradio svelte components use these classes for upload hint text */
.gradio-container .icon-wrap svg, .gradio-container .upload-icon { color:var(--muted)!important; opacity:0.7; }
.gradio-container .file-name, .gradio-container .file-size, .gradio-container .or, .gradio-container .subtitle { color:var(--muted)!important; }
/* Hide Gradio default footer entirely to prevent standard Gradio settings from interfering with our persistent theme system */
footer, .footer, .gradio-button.theme-toggle, button.theme-toggle, .settings-btn, .show-api { display:none!important; }
/* ── Dashboard ────────────────────────────────────────────────────── */
.dashboard { width:min(1120px,calc(100vw - 48px)); margin:auto; padding:48px 0; }
.dashboard header { display:flex; justify-content:space-between; align-items:flex-end; border-bottom:1px solid var(--line); padding-bottom:20px; margin-bottom:28px; }
.dashboard header p, .dashboard h1, .dashboard small { margin:0; }
.dashboard header p { color:var(--muted); text-transform:uppercase; font-size:12px; }
.dashboard h1 { font:400 clamp(44px,7vw,82px)/1 Georgia,serif; color:var(--ink); }
.dashboard-theme-selector { display:flex; gap:8px; align-items:center; }
.dashboard-theme-selector button { border:1px solid var(--line)!important; background:transparent!important; border-radius:6px!important; color:var(--ink)!important; box-shadow:none!important; text-decoration:none; padding:7px 14px; cursor:pointer; font-size:13px; font-weight:500; transition:all .2s; }
.dashboard-theme-selector button:hover { background:var(--line)!important; }
.dashboard h2 { margin:36px 0 12px; font-size:14px; color:var(--muted); text-transform:uppercase; }
.cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:24px 18px; }

/* Book Card */
.book-card { display:flex; flex-direction:column; text-decoration:none; border:1px solid var(--line); border-radius:12px; background:var(--card-bg); color:var(--ink)!important; overflow:hidden; transition:all 0.25s cubic-bezier(0.4, 0, 0.2, 1); box-shadow:0 2px 8px rgba(0,0,0,0.04); height: 100%; position: relative; }
.book-card:hover, .book-card.selected { background:var(--card-hover); transform:translateY(-4px); box-shadow:0 8px 24px rgba(0,0,0,0.12); border-color: var(--blue); }

.card-cover-container { width:100%; aspect-ratio:3/4; overflow:hidden; background:var(--paper); border-bottom:1px solid var(--line); position:relative; display:flex; align-items:center; justify-content:center; }
.book-cover-img { width:100%; height:100%; object-fit:cover; transition: transform 0.3s ease; }
.book-card:hover .book-cover-img { transform: scale(1.03); }

/* Star badge inside card cover */
.star-badge { position:absolute; top:8px; right:8px; background:rgba(242, 201, 76, 0.95); color:#2b2118; width:26px; height:26px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:14px; font-weight:bold; box-shadow:0 2px 6px rgba(0,0,0,0.15); z-index: 5; }

/* Fallback Text Cover */
.book-cover-fallback { width:100%; height:100%; padding:16px; display:flex; flex-direction:column; justify-content:space-between; box-sizing:border-box; background:var(--paper); font-family:Georgia, serif; color:var(--ink); overflow:hidden; position:relative; }
.book-cover-fallback::before { content:''; position:absolute; top:0; left:0; right:0; bottom:0; background:linear-gradient(90deg, rgba(0,0,0,0.04) 0%, rgba(255,255,255,0.06) 1.5%, rgba(0,0,0,0.02) 3%, transparent 4%, rgba(0,0,0,0.03) 98%, rgba(0,0,0,0.08) 100%); pointer-events:none; }
.fallback-header { font-size:10px; text-transform:uppercase; letter-spacing:1px; color:var(--muted); text-align:center; }
.fallback-title { font-size:16px; font-weight:bold; text-align:center; margin:8px 0 4px; line-height:1.2; display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden; max-height: 3.6em; }
.fallback-divider { width:30px; height:1px; background:var(--muted); margin:4px auto; opacity:0.5; }
.fallback-body { font-size:10px; line-height:1.4; color:var(--muted); text-align:justify; opacity:0.85; overflow:hidden; display:-webkit-box; -webkit-line-clamp:6; -webkit-box-orient:vertical; height: 8.4em; }
.fallback-footer { font-size:10px; font-style:italic; text-align:center; color:var(--muted); margin-top: auto; }

.card-info { padding:14px; display:flex; flex-direction:column; gap:4px; flex-grow:1; }
.card-format { color:var(--blue); font-size:10px; font-weight:bold; text-transform:uppercase; letter-spacing:0.5px; }
.card-title { font:500 16px/1.25 Georgia,serif; color:var(--ink); display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; min-height: 2.5em; }
.card-author { color:var(--muted); font-size:12px; display:-webkit-box; -webkit-line-clamp:1; -webkit-box-orient:vertical; overflow:hidden; }
.card-progress { height:3px; background:var(--line); overflow:hidden; border-radius:1.5px; margin-top:8px; }
.card-progress b { display:block; height:100%; background:var(--blue); }
.card-progress-text { font-size:10px; color:var(--muted); align-self:flex-end; margin-top:2px; }

/* ── Modal Dialog ──────────────────────────────────────────────────── */
.modal-overlay { position:fixed; top:0; left:0; right:0; bottom:0; background:rgba(0, 0, 0, 0.4); display:none; align-items:center; justify-content:center; z-index:99999; padding:20px; box-sizing:border-box; backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px); animation:fadeIn 0.25s ease-out; }
.modal-overlay.show { display:flex; }

.modal-content { background:var(--paper); color:var(--ink); width:100%; max-width:520px; border-radius:20px; box-shadow:0 20px 50px rgba(0,0,0,0.3); border:1px solid var(--line); position:relative; box-sizing:border-box; padding:36px 32px 28px; text-align:center; max-height:90vh; overflow-y:auto; animation:slideUp 0.3s cubic-bezier(0.16, 1, 0.3, 1); }

/* Scrollbar styling for modal */
.modal-content::-webkit-scrollbar { width:6px; }
.modal-content::-webkit-scrollbar-track { background:transparent; }
.modal-content::-webkit-scrollbar-thumb { background:var(--line); border-radius:3px; }

.modal-close { position:absolute; top:16px; right:16px; width:32px; height:32px; border-radius:50%; border:none; background:rgba(0,0,0,0.03); color:var(--ink); font-size:24px; font-weight:300; display:flex; align-items:center; justify-content:center; cursor:pointer; transition:all 0.2s; z-index:10; }
.modal-close:hover { background:var(--line); transform:scale(1.05); }

.modal-cover-wrapper { width:190px; aspect-ratio:3/4; margin:0 auto 24px; border-radius:12px; overflow:hidden; box-shadow:0 8px 24px rgba(0,0,0,0.15); border:1px solid var(--line); background:var(--bg); }
.modal-cover-wrapper .book-cover-fallback { padding: 16px; }
.modal-cover-wrapper .fallback-title { font-size: 15px; }
.modal-cover-wrapper .fallback-body { font-size: 9.5px; -webkit-line-clamp: 5; height: 7em; }

.modal-title { font:400 28px/1.25 Georgia,serif; color:var(--ink); margin:0 0 24px; padding:0 10px; }

.modal-actions { display:flex; justify-content:center; align-items:center; gap:20px; margin:0 auto 28px; }
.action-btn { width:54px!important; height:54px!important; border-radius:50%!important; border:1.5px solid var(--muted)!important; background:var(--bg)!important; color:var(--ink)!important; display:flex!important; align-items:center!important; justify-content:center!important; cursor:pointer!important; transition:all 0.2s cubic-bezier(0.4, 0, 0.2, 1)!important; text-decoration:none!important; box-sizing:border-box!important; box-shadow:0 2px 10px rgba(0,0,0,0.05)!important; padding:0!important; }
.action-btn:hover { transform:translateY(-2px)!important; box-shadow:0 6px 16px rgba(0,0,0,0.12)!important; border-color:var(--ink)!important; background:var(--paper)!important; }

/* Force standard action icons to follow the ink color on all internal strokes */
.action-btn .icon-svg { width:24px!important; height:24px!important; stroke:var(--ink)!important; color:var(--ink)!important; }
.action-btn .icon-svg path,
.action-btn .icon-svg circle { stroke:var(--ink)!important; stroke-width:2.2px!important; fill:none!important; }

/* Star Button States */
.star-btn.active .icon-svg,
.star-btn.active .icon-svg path { fill:#f2c94c!important; stroke:#f2c94c!important; }
.star-btn:hover, .star-btn.active { border-color:#f2c94c!important; }
.star-btn:hover .icon-svg,
.star-btn:hover .icon-svg path { stroke:#f2c94c!important; }
.star-btn.active { background:rgba(242, 201, 76, 0.15)!important; }

/* Archive Button States */
.archive-btn.active .icon-svg,
.archive-btn.active .icon-svg path,
.archive-btn.active .icon-svg circle { fill:var(--blue)!important; stroke:var(--blue)!important; }
.archive-btn:hover, .archive-btn.active { border-color:var(--blue)!important; }
.archive-btn:hover .icon-svg,
.archive-btn:hover .icon-svg path,
.archive-btn:hover .icon-svg circle { stroke:var(--blue)!important; }
.archive-btn.active { background:rgba(47, 128, 237, 0.15)!important; }

/* Delete Button States */
.delete-btn:hover { border-color:#eb5757!important; background:rgba(235, 87, 87, 0.15)!important; }
.delete-btn:hover .icon-svg,
.delete-btn:hover .icon-svg path { stroke:#eb5757!important; }

/* Continue Button (Play Icon) */
.continue-btn { background:var(--blue)!important; border-color:var(--blue)!important; }
.continue-btn .icon-svg { stroke:none!important; color:#ffffff!important; }
.continue-btn .icon-svg path { fill:#ffffff!important; stroke:none!important; }
.continue-btn:hover { background:var(--blue)!important; opacity:0.95!important; }

.modal-divider { height:1px; background:var(--line); width:100%; margin-bottom:24px; }

.modal-metadata { display:flex; flex-direction:column; gap:12px; text-align:left; padding:0 6px; }
.meta-row { display:flex; justify-content:space-between; align-items:baseline; border-bottom:1px dashed rgba(0,0,0,0.05); padding-bottom:6px; font-size:14px; }
html[data-theme=dark] .meta-row { border-bottom-color:rgba(255,255,255,0.05); }
.meta-label { color:var(--muted); font-weight:normal; }
.meta-value { color:var(--ink); font-weight:500; text-align:right; max-width:70%; word-break:break-word; }

@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes slideUp { from { transform: translateY(20px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }

/* ── Reader layout ────────────────────────────────────────────────── */
@media (min-width: 861px) {
  .reader-grid { display:flex!important; flex-direction:row!important; align-items:stretch!important; min-height:100vh!important; gap:0!important; }
  .chapter-rail { display:flex!important; flex-direction:column!important; flex-wrap:nowrap!important; position:sticky!important; top:0!important; height:100vh!important; overflow-y:auto!important; overflow-x:hidden!important; flex:0 0 280px!important; min-width:280px!important; max-width:280px!important; background:var(--rail-bg)!important; color:var(--rail-fg)!important; padding:18px 0!important; border-right:1px solid var(--rail-border)!important; box-sizing:border-box!important; z-index:100!important; }
  .reader-body { flex:1!important; min-width:0!important; box-sizing:border-box!important; display:flex!important; flex-direction:column!important; }
}

@media (max-width: 860px) {
  .reader-grid { min-height:100vh; gap:0!important; }
  .chapter-rail { display:none!important; }
}

/* Force each chapter choice inside the radio group to render on a fresh single line spanning 100% width */
.chapters .gr-radio-group {
  display:flex!important;
  flex-direction:column!important;
  align-items:stretch!important;
  width:100%!important;
}

.chapters label {
  width:100%!important;
  flex:1 1 100%!important;
  display:flex!important;
  align-items:center!important;
  box-sizing:border-box!important;
  margin:0!important;
  padding:14px 18px!important;
  border-top:1px solid var(--rail-border)!important;
  border-radius:0!important;
  color:var(--rail-fg)!important;
  background:transparent!important;
}
.chapters label:has(input:checked) { color:var(--blue)!important; }

/* Force ALL children of chapter-rail to use rail colors regardless of any other color overrides */
.chapter-rail * { color:var(--rail-fg)!important; background:transparent!important; }
.chapter-rail .block, .chapter-rail .wrap, .chapter-rail .panel, .chapter-rail fieldset, .chapter-rail .form { border:none!important; box-shadow:none!important; }
.chapter-rail span.group-text { color:var(--rail-fg)!important; }
.rail-link { display:block; margin:0 18px 18px; color:var(--rail-fg)!important; text-decoration:none; }
/* theme-* classes kept on elements for state tracking; all reader vars now flow from html[data-theme] */
.top { height:64px; display:flex; align-items:center; justify-content:space-between; gap:18px; padding:0 28px; color:var(--reader-ink)!important; background:var(--reader-bg)!important; border-bottom:1px solid var(--reader-line); }
.top small { display:block!important; color:var(--reader-muted)!important; font-size:12px!important; }
.top strong { display:block!important; font:400 20px/1.15 Georgia,serif!important; color:var(--reader-ink)!important; }
.top nav { display:flex; gap:7px; align-items:center; flex-wrap:wrap; }
.top a, .top button { border:1px solid var(--reader-line)!important; background:transparent!important; border-radius:6px!important; color:var(--reader-ink)!important; box-shadow:none!important; text-decoration:none; padding:7px 10px; }
.top span { min-width:54px; text-align:center; color:var(--reader-muted); font-size:13px; }
.page { min-height:calc(100vh - 64px); padding:64px clamp(26px,8vw,130px); color:var(--reader-ink)!important; background:var(--reader-bg)!important; }
.text { max-width:930px; margin:auto; text-align:center; font:var(--font,22px)/1.32 Georgia,serif; color:var(--reader-ink)!important; }
/* Force ALL descendants inside .text to use reader-ink — prevents EPUB inline color styles from creating unreadable light text on light backgrounds */
.text * { color:var(--reader-ink)!important; }
.text p { margin:0 0 .85em; }
.text img { display:block; max-width:100%; height:auto; margin:1.2em auto; color:unset!important; }
.text hr { width:min(320px,60%); margin:1.6em auto; border:0; border-top:1px solid currentColor; opacity:.38; }
.text h1, .text h2, .text h3, .text h4 { font-weight:400; line-height:1.15; margin:1.2em 0 .7em; }
.text blockquote { margin:1.2em auto; max-width:760px; opacity:.86; }
.text ::selection { background:rgba(47,128,237,.22); }
.notes { padding:0 28px 28px; color:var(--reader-ink)!important; background:var(--reader-bg)!important; border-top:1px solid var(--reader-line); }
.notes summary { padding:16px 0; color:var(--reader-muted)!important; cursor:pointer; }
.notes section { display:grid; grid-template-columns:repeat(3,1fr); gap:18px; }
.notes h3 { margin:0 0 8px; color:var(--reader-muted)!important; font-size:12px; text-transform:uppercase; }
.notes p { margin:0 0 10px; color:var(--reader-ink)!important; }
.notes b { margin-right:8px; color:var(--blue); }
.notice { width:fit-content; margin:10px auto; color:var(--blue); font-size:13px; }
.notice.warn { color:var(--warn); }
.hidden { position:fixed!important; left:-10000px!important; width:1px!important; height:1px!important; overflow:hidden!important; }
@media (max-width:860px) { .reader-grid{flex-direction:column}.chapter-rail{display:none!important}.top{height:auto;align-items:flex-start;flex-direction:column;padding:14px}.page{padding:42px 20px}.text{text-align:left}.notes section,.detail dl{grid-template-columns:1fr} }
"""


READER_CSS = CSS + """
/* select-bar uses reader theme vars so it adapts automatically */
#select-bar { position:absolute; z-index:9999; display:none; gap:4px; padding:5px; border-radius:7px; background:var(--reader-bg); border:1px solid var(--reader-line); box-shadow:0 10px 26px rgba(0,0,0,.30); }
#select-bar.show { display:flex; }
#select-bar button { border:0; border-radius:5px; background:transparent; color:var(--reader-ink); padding:7px 10px; cursor:pointer; font-size:13px; }
#select-bar button:hover { background:var(--reader-line); }

/* Settings Popover Dropdown */
.settings-popover {
  position: fixed !important;
  z-index: 1000;
  display: none;
  width: 300px;
  background: var(--paper) !important;
  color: var(--ink) !important;
  border: 1px solid var(--line) !important;
  border-radius: 14px !important;
  box-shadow: 0 12px 36px rgba(0,0,0,0.18) !important;
  padding: 20px !important;
  box-sizing: border-box !important;
  animation: slideUp 0.2s ease-out;
}
.settings-popover.show {
  display: block !important;
}
.settings-group {
  margin-bottom: 16px;
  border-bottom: 1px dashed var(--line);
  padding-bottom: 12px;
}
.settings-group:last-child {
  margin-bottom: 0;
  border-bottom: none;
  padding-bottom: 0;
}
.settings-group h4 {
  margin: 0 0 10px 0 !important;
  color: var(--muted) !important;
  font-size: 11px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.5px !important;
}
.settings-row {
  display: flex;
  gap: 8px;
  align-items: center;
}
.settings-row.themes {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 6px;
}
.font-adjust {
  justify-content: space-between;
}
.font-display {
  font-size: 16px;
  font-weight: 600;
  color: var(--ink);
}
.settings-action, .theme-btn {
  background: var(--bg) !important;
  color: var(--ink) !important;
  border: 1.5px solid var(--line) !important;
  border-radius: 8px !important;
  padding: 8px 12px !important;
  cursor: pointer !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  transition: all 0.15s ease !important;
  text-align: center !important;
  width: 100% !important;
  box-sizing: border-box !important;
}
.settings-action:hover, .theme-btn:hover {
  border-color: var(--ink) !important;
  transform: translateY(-1px);
}
.theme-btn.active {
  background: var(--blue) !important;
  color: #ffffff !important;
  border-color: var(--blue) !important;
}
/* Bookmarks list section inside popup */
.bookmarks-list {
  max-height: 180px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding-right: 4px;
}
.bookmarks-list::-webkit-scrollbar {
  width: 4px;
}
.bookmarks-list::-webkit-scrollbar-thumb {
  background: var(--line);
  border-radius: 2px;
}
.bookmark-item {
  display: flex;
  flex-direction: column;
  padding: 8px 10px;
  background: var(--bg);
  border: 1.5px solid var(--line);
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.15s ease;
}
.bookmark-item:hover {
  border-color: var(--blue);
  background: var(--paper);
  transform: translateX(2px);
}
.bookmark-item .b-sec {
  font-size: 10px;
  font-weight: bold;
  color: var(--blue);
  text-transform: uppercase;
  margin-bottom: 2px;
}
.bookmark-item .b-text {
  font-size: 12px;
  font-style: italic;
  color: var(--ink);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.no-bookmarks {
  font-size: 12px;
  color: var(--muted);
  text-align: center;
  margin: 12px 0 0 0;
}

/* Persistent dotted annotation underlines */
.bookmark-dotted {
  border-bottom: 2.2px dotted var(--blue) !important;
  background: transparent !important;
  cursor: pointer !important;
  transition: background-color 0.3s ease;
}
.bookmark-dotted:hover {
  background: rgba(47, 128, 237, 0.12) !important;
}
.define-dotted {
  border-bottom: 2.2px dotted #9b51e0 !important; /* purple dotted */
  background: transparent !important;
  cursor: pointer !important;
  transition: background-color 0.3s ease;
}
.define-dotted:hover {
  background: rgba(155, 81, 224, 0.12) !important;
}
.settings-trigger {
  width: 36px !important;
  height: 36px !important;
  border-radius: 50% !important;
  padding: 0 !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  cursor: pointer !important;
  border: 1.5px solid var(--reader-line) !important;
  background: transparent !important;
}
.settings-trigger svg {
  display: block;
  stroke: var(--reader-ink) !important;
  color: var(--reader-ink) !important;
}
.settings-trigger svg path,
.settings-trigger svg circle {
  stroke: var(--reader-ink) !important;
  stroke-width: 2.2px !important;
  fill: none !important;
}
.pdf-page-container {
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12) !important;
  border: 1px solid var(--reader-line) !important;
  border-radius: 8px !important;
  overflow: hidden !important;
  background: #ffffff !important;
  zoom: var(--pdf-zoom, 1) !important;
  transition: zoom 0.2s ease !important;
}
.pdf-page-img {
  transition: filter 0.3s ease !important;
}
[data-theme="dark"] .pdf-page-img {
  filter: invert(0.9) hue-rotate(180deg) contrast(0.95) !important;
}
[data-theme="sepia"] .pdf-page-img {
  filter: sepia(0.55) contrast(0.95) brightness(0.98) !important;
}
.pdf-text-layer {
  color: transparent !important;
  font-family: system-ui, -apple-system, sans-serif !important;
  white-space: pre-wrap !important;
  line-height: 1.15 !important;
  cursor: text !important;
  user-select: text !important;
  -webkit-user-select: text !important;
}
.pdf-text-layer::selection {
  background: rgba(0, 120, 215, 0.33) !important;
  color: transparent !important;
}
.pdf-text-layer::-moz-selection {
  background: rgba(0, 120, 215, 0.33) !important;
  color: transparent !important;
}
.pdf-page-error {
  display: flex !important;
  justify-content: center !important;
  align-items: center !important;
  width: 100% !important;
  padding: 40px !important;
  background: rgba(235, 94, 85, 0.1) !important;
  border: 1px dashed #eb5e55 !important;
  border-radius: 8px !important;
  color: #eb5e55 !important;
}
"""
