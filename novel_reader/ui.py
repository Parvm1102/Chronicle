from __future__ import annotations

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
        with gr.Row(visible=False) as actions:
            archive = gr.Button("Completed reading")
            delete = gr.Button("Delete")
        upload = gr.File(label="Add TXT, EPUB, or PDF", file_types=[".txt", ".epub", ".pdf"])
        refresh = gr.Button("Refresh")
        msg = gr.HTML()

        demo.load(_dashboard_load, outputs=[library, detail, selected_id, actions, msg])
        refresh.click(_dashboard_refresh, outputs=[library, detail, selected_id, actions, msg])
        upload.upload(_upload, upload, [library, detail, selected_id, actions, msg])
        archive.click(_archive, selected_id, [library, detail, selected_id, actions, msg])
        delete.click(_delete, selected_id, [library, detail, selected_id, actions, msg])
    return demo


def build_reader_app() -> gr.Blocks:
    with gr.Blocks(title="Reader") as demo:
        state = gr.State(_state())
        with gr.Row(elem_classes=["reader-grid"]):
            with gr.Column(scale=1, min_width=230, elem_classes=["chapter-rail"]):
                gr.HTML("<a class='rail-link' href='/dashboard/'>Dashboard</a>")
                chapters = gr.Radio(label="Chapters", choices=[], interactive=True, elem_classes=["chapters"])
            with gr.Column(scale=5, min_width=560):
                top = gr.HTML()
                page = gr.HTML(_empty_page())
                notes = gr.HTML()
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
                    mark = gr.Button("mark", elem_id="nr-mark")
                    define = gr.Button("define", elem_id="nr-define")
                    save = gr.Button("save", elem_id="nr-save")
                    speak_selected = gr.Button("speak selected", elem_id="nr-speak-selected")

        demo.load(_reader_load, outputs=[page, top, chapters, notes, msg, state])
        chapters.change(_jump, [chapters, state], [page, top, chapters, notes, msg, state])
        prev.click(_move, [state, gr.State(-1)], [page, top, chapters, notes, msg, state])
        next_.click(_move, [state, gr.State(1)], [page, top, chapters, notes, msg, state])
        back.click(_history, [state, gr.State("back")], [page, top, chapters, notes, msg, state])
        forward.click(_history, [state, gr.State("forward")], [page, top, chapters, notes, msg, state])
        font_up.click(_font, [state, gr.State(1)], [page, top, chapters, notes, msg, state])
        font_down.click(_font, [state, gr.State(-1)], [page, top, chapters, notes, msg, state])
        sepia.click(_theme, [state, gr.State("sepia")], [page, top, chapters, notes, msg, state])
        light.click(_theme, [state, gr.State("light")], [page, top, chapters, notes, msg, state])
        dark.click(_theme, [state, gr.State("dark")], [page, top, chapters, notes, msg, state])
        speak.click(_speak, [state, gr.State("")], msg)
        speak_selected.click(_speak, [state, selected], msg)
        mark.click(_mark, [state, selected], [notes, msg])
        define.click(_define, [state, selected], [notes, msg])
        save.click(_save, [state, selected], [notes, msg])
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


def _archive(novel_id):
    novel_id = _int(novel_id)
    if novel_id:
        store.archive_novel(novel_id)
        return _dashboard_outputs(None, "Moved to archive.")
    return _dashboard_outputs(None, "Choose a book first.", "warn")


def _delete(novel_id):
    novel_id = _int(novel_id)
    if novel_id:
        store.delete_novel(novel_id)
        return _dashboard_outputs(None, "Book deleted.")
    return _dashboard_outputs(None, "Choose a book first.", "warn")


def _dashboard_outputs(novel_id: int | None, message: str = "", tone: str = "ok"):
    novel = store.get_novel(novel_id) if novel_id else None
    return (
        _dashboard(novel_id),
        _detail(novel),
        novel["id"] if novel else None,
        gr.update(visible=bool(novel)),
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


def _card(novel: dict, selected_id: int | None) -> str:
    count = int(novel.get("section_count") or 0)
    done = int(novel.get("progress_section") or 0)
    progress = int(((done + 1) / count) * 100) if count else 0
    selected = " selected" if novel["id"] == selected_id else ""
    return f"""
    <a class="book-card{selected}" href="/dashboard?novel_id={novel['id']}">
      <span>{_escape(novel['status'])}</span>
      <strong>{_escape(novel['title'])}</strong>
      <small>{_escape(novel['author'])}</small>
      <em>{_escape(novel['original_filename'])}</em>
      <i><b style="width:{progress}%"></b></i>
    </a>
    """


def _detail(novel: dict | None) -> str:
    if not novel:
        return "<aside class='detail empty'>Select a novel to see details.</aside>"
    count = int(novel.get("section_count") or 0)
    done = int(novel.get("progress_section") or 0)
    percent = int(((done + 1) / count) * 100) if count else 0
    summary = _summary(novel["id"]) if novel["status"] == "complete" else novel["status_message"]
    read_label = "Read again" if novel.get("archived") else "Continue reading"
    return f"""
    <aside class="detail">
      <p>{_escape(novel['file_format'].upper())}</p>
      <h1>{_escape(novel['title'])}</h1>
      <h2>{_escape(novel['author'])}</h2>
      <dl>
        <dt>Original file</dt><dd>{_escape(novel['original_filename'])}</dd>
        <dt>Progress</dt><dd>{done + 1 if count else 0} / {count} sections · {percent}%</dd>
      </dl>
      <div class="card-progress"><span style="width:{percent}%"></span></div>
      <h3>Summary</h3>
      <p class="summary">{_escape(summary or 'Summary will be available after parsing.')}</p>
      <a class="primary-link" href="/reader/?novel_id={novel['id']}">{read_label}</a>
    </aside>
    """


def _summary(novel_id: int) -> str:
    text = re.sub(r"\s+", " ", store.first_section_text(novel_id)).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    summary = " ".join(sentences[:3]).strip()
    return summary[:700] + ("..." if len(summary) > 700 else "")


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
    return _render({**state, "font": max(16, min(32, int(state["font"]) + int(delta)))})


def _theme(state, theme):
    return _render({**state, "theme": theme})


def _mark(state, text):
    text = (text or "").strip()
    novel_id = state.get("novel_id")
    if novel_id and text:
        store.add_highlight(novel_id, int(state["section"]), text)
        return _notes(novel_id, state["theme"]), _notice("Highlighted.")
    return _notes(novel_id, state["theme"]), _notice("Select text first.", "warn")


def _define(state, text):
    text = (text or "").strip()
    novel_id = state.get("novel_id")
    if novel_id and text:
        store.add_dictionary_lookup(novel_id, int(state["section"]), text, _meaning_url(text))
        return _notes(novel_id, state["theme"]), _notice("Definition saved.")
    return _notes(novel_id, state["theme"]), _notice("Select text first.", "warn")


def _save(state, text):
    novel_id = state.get("novel_id")
    if not novel_id:
        return "", _notice("Open a book first.", "warn")
    section = store.get_section(novel_id, int(state["section"]))
    label = (text or "").strip()[:80] or (section["title"] if section else "Saved position")
    store.add_bookmark(novel_id, int(state["section"]), label)
    return _notes(novel_id, state["theme"]), _notice("Bookmark saved.")


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
        return _waiting_page(novel, state), _top(novel, None, state), _chapter_update(novel_id, state), _notes(novel_id, state["theme"]), _notice(novel["status_message"]), state

    count = store.section_count(novel_id)
    state = {**state, "section": max(0, min(count - 1, int(state["section"])))}
    if save_progress:
        store.update_progress(novel_id, int(state["section"]))
    section = store.get_section(novel_id, int(state["section"]))
    return _page(section, count, state), _top(novel, section, state), _chapter_update(novel_id, state), _notes(novel_id, state["theme"]), _notice(message), state


def _top(novel: dict, section: dict | None, state: dict) -> str:
    count = store.section_count(novel["id"]) if novel and novel["status"] == "complete" else 0
    progress = f"{section['section_index'] + 1}/{count}" if section else ""
    title = section["title"] if section else novel["title"]
    return f"""
    <header class="top theme-{state['theme']}">
      <div><small>{_escape(novel['title'])}</small><strong>{_escape(title)}</strong></div>
      <nav>
        <a href="/dashboard/">Dashboard</a>
        <button data-click="nr-back">Back</button><button data-click="nr-prev">Prev</button>
        <span>{_escape(progress)}</span>
        <button data-click="nr-next">Next</button><button data-click="nr-forward">Forward</button>
        <button data-click="nr-font-down">A-</button><button data-click="nr-font-up">A+</button>
        <button data-click="nr-sepia">Sepia</button><button data-click="nr-light">Light</button><button data-click="nr-dark">Dark</button>
        <button data-click="nr-speak">Speak</button>
      </nav>
    </header>
    """


def _chapter_update(novel_id: int, state: dict):
    choices = [(s["title"], s["section_index"]) for s in store.list_sections(novel_id)]
    return gr.update(choices=choices, value=int(state["section"]))


def _page(section: dict | None, count: int, state: dict) -> str:
    if not section:
        return _empty_page()
    progress = int(((section["section_index"] + 1) / max(count, 1)) * 100)
    content = section["html"] or _paragraphs(section["text"])
    return f"""
    <article class="page theme-{state['theme']}" style="--font:{state['font']}px">
      <div class="progress"><span style="width:{progress}%"></span></div>
      <div class="text">{content}</div>
    </article>
    """


def _waiting_page(novel: dict, state: dict) -> str:
    return f"<article class='page theme-{state['theme']}'><div class='empty'><h2>{_escape(novel['title'])}</h2><p>{_escape(novel['status_message'])}</p></div></article>"


def _empty_page() -> str:
    return "<article class='page theme-sepia'><div class='empty'><h2>Open a book from your dashboard.</h2></div></article>"


def _notes(novel_id: int | None, theme: str) -> str:
    bookmarks, highlights, lookups = store.sidebar_items(novel_id)
    return f"""
    <details class="notes theme-{theme}"><summary>Saved</summary>
      <section>{_note('Bookmarks', bookmarks, 'label')}{_note('Highlights', highlights, 'quote')}{_note('Dictionary', lookups, 'query')}</section>
    </details>
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
const click = (id) => q(`#${id} button`)?.click();
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

function bootReader() {
  if (window.__readerBooted) return;
  window.__readerBooted = true;
  const bar = document.createElement("div");
  bar.id = "select-bar";
  bar.innerHTML = "<button data-a='mark'>Mark</button><button data-a='define'>Define</button><button data-a='save'>Save</button><button data-a='speak-selected'>Speak</button>";
  document.body.appendChild(bar);

  document.addEventListener("click", (event) => {
    const proxy = event.target.closest("[data-click]");
    if (proxy) {
      const themeKey = THEMES[proxy.dataset.click];
      if (themeKey) applyTheme(themeKey);
      click(proxy.dataset.click);
      return;
    }
    const action = event.target.closest("#select-bar button");
    if (!action) return;
    const text = selectionText();
    setBox("nr-selected-text", text);
    if (action.dataset.a === "define" && text) window.open(`https://www.google.com/search?q=${encodeURIComponent(text + " meaning")}`, "_blank", "noopener");
    click(`nr-${action.dataset.a}`);
    bar.classList.remove("show");
  });

  document.addEventListener("mouseup", () => {
    const text = selectionText();
    if (!text) return bar.classList.remove("show");
    const rect = getSelection().getRangeAt(0).getBoundingClientRect();
    bar.style.left = `${Math.max(10, Math.min(innerWidth - 260, rect.left + rect.width / 2 - 120))}px`;
    bar.style.top = `${Math.max(10, scrollY + rect.top - 46)}px`;
    bar.classList.add("show");
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
    var btn = e.target.closest("[data-theme-set]");
    if (btn) {
      var theme = btn.getAttribute("data-theme-set");
      localStorage.setItem('nr-theme', theme);
      document.documentElement.setAttribute('data-theme', theme);
    }
  });
})();
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
.cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:14px; }
.book-card { min-height:190px; display:grid; gap:8px; text-align:left; text-decoration:none; border:1px solid var(--line); border-radius:8px; background:var(--card-bg); color:var(--ink)!important; padding:18px; transition:background .2s,transform .15s,box-shadow .15s; }
.book-card:hover, .book-card.selected { background:var(--card-hover); transform:translateY(-2px); box-shadow:0 4px 18px rgba(0,0,0,.12); }
.book-card strong { font:400 24px/1.1 Georgia,serif; color:var(--ink); }
.book-card span { color:var(--blue); text-transform:uppercase; font-size:12px; }
.book-card small, .book-card em, .empty, .summary { color:var(--muted); font-style:normal; }
.book-card i, .card-progress, .progress { height:2px; background:var(--line); overflow:hidden; }
.book-card b, .card-progress span, .progress span { display:block; height:100%; background:var(--blue); }
.detail { width:min(1120px,calc(100vw - 48px)); margin:8px auto 24px; padding:24px; border:1px solid var(--line); border-radius:8px; background:var(--card-bg); color:var(--ink); }
.detail p, .detail h1, .detail h2 { margin:0; }
.detail > p { color:var(--blue); text-transform:uppercase; font-size:12px; }
.detail h1 { font:400 42px/1.05 Georgia,serif; margin-top:6px; color:var(--ink); }
.detail h2 { color:var(--muted); font-size:16px; font-weight:400; }
.detail dl { display:grid; grid-template-columns:120px 1fr; gap:8px 16px; margin:20px 0; color:var(--muted); }
.detail dt { color:var(--ink); }
.detail h3 { margin:22px 0 8px; font-size:13px; color:var(--muted); text-transform:uppercase; }
.primary-link { display:inline-block; margin-top:18px; color:var(--blue)!important; text-decoration:none; }
/* ── Reader layout ────────────────────────────────────────────────── */
.reader-grid { min-height:100vh; gap:0!important; }
.chapter-rail { background:var(--rail-bg)!important; color:var(--rail-fg)!important; padding:18px 0!important; border-right:1px solid var(--rail-border); }
/* Force ALL children of chapter-rail to use rail colors regardless of any other color overrides */
.chapter-rail * { color:var(--rail-fg)!important; background:transparent!important; }
.chapter-rail .block, .chapter-rail .wrap, .chapter-rail .panel, .chapter-rail fieldset, .chapter-rail .form { border:none!important; box-shadow:none!important; }
.chapter-rail span.group-text { color:var(--rail-fg)!important; }
.rail-link { display:block; margin:0 18px 18px; color:var(--rail-fg)!important; text-decoration:none; }
.chapters label { margin:0!important; padding:14px 18px!important; border-top:1px solid var(--rail-border)!important; border-radius:0!important; color:var(--rail-fg)!important; background:transparent!important; }
.chapters label:has(input:checked) { color:var(--blue)!important; }
/* theme-* classes kept on elements for state tracking; all reader vars now flow from html[data-theme] */
.top { height:64px; display:flex; align-items:center; justify-content:space-between; gap:18px; padding:0 28px; color:var(--reader-ink)!important; background:var(--reader-bg)!important; border-bottom:1px solid var(--reader-line); }
.top small { display:block; color:var(--reader-muted); font-size:12px; }
.top strong { display:block; font:400 20px/1.15 Georgia,serif; }
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
"""
