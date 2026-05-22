import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from novel_reader.ui import CSS, READER_CSS, READER_JS, THEME_JS, build_dashboard_app, build_reader_app


app = FastAPI()


@app.get("/")
def root():
    return RedirectResponse("/dashboard/")


@app.head("/")
def root_head():
    return RedirectResponse("/dashboard/")


gr.mount_gradio_app(app, build_dashboard_app().queue(default_concurrency_limit=4), path="/dashboard", css=CSS, js=THEME_JS, theme=gr.themes.Base())
gr.mount_gradio_app(app, build_reader_app().queue(default_concurrency_limit=4), path="/reader", css=READER_CSS, js=READER_JS, theme=gr.themes.Base())


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8060)
