"""Gradio UI: deals table, live colour-coded agent logs, 3-D t-SNE plot of the vector store, periodic timer."""

from __future__ import annotations

import queue
import threading

import gradio as gr  # module-level: gradio resolves type hints (gr.SelectData) via module globals
import plotly.graph_objects as go

from .config import Settings
from .framework import DealAgentFramework
from .log_utils import attach_queue_handler, detach_handler
from .ui_helpers import stream_run, table_for


class App:
    def __init__(self, settings: Settings | None = None, demo: bool = False):
        self.settings = settings or Settings()
        self.demo = demo
        self.agent_framework: DealAgentFramework | None = None

    def get_agent_framework(self) -> DealAgentFramework:
        if not self.agent_framework:
            self.agent_framework = DealAgentFramework(self.settings, demo=self.demo)
        return self.agent_framework

    def build(self):
        def get_plot():
            fig = go.Figure()
            try:
                if self.demo:
                    raise RuntimeError("demo mode has no vector store")
                documents, vectors, colors = DealAgentFramework.get_plot_data(self.settings, self.settings.ui_plot_points)
            except Exception as exc:  # noqa: BLE001 - missing store must not kill the UI
                fig.update_layout(title=f"Vector store unavailable ({exc})", height=400)
                return fig
            if len(vectors) == 0:
                fig.update_layout(title="Vector store is empty - run `deal-hunter build-vectorstore`", height=400)
                return fig
            fig.add_trace(
                go.Scatter3d(
                    x=vectors[:, 0], y=vectors[:, 1], z=vectors[:, 2], mode="markers",
                    marker=dict(size=2, color=colors, opacity=0.7),
                    text=[d[:60] for d in documents], hoverinfo="text",
                )
            )
            fig.update_layout(
                scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="manual",
                           aspectratio=dict(x=2.2, y=2.2, z=1), camera=dict(eye=dict(x=1.6, y=1.6, z=0.8))),
                height=400, margin=dict(r=5, b=1, l=5, t=2),
            )
            return fig

        def run_with_logging(initial_log_data):
            log_queue, result_queue = queue.Queue(), queue.Queue()
            handler = attach_queue_handler(log_queue)

            def worker():
                try:
                    result_queue.put(table_for(self.get_agent_framework().run()))
                except Exception as exc:  # noqa: BLE001
                    log_queue.put(f"<span style='color:#dd0000'>Run failed: {exc}</span>")
                    result_queue.put(table_for(self.get_agent_framework().memory))

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            try:
                initial = table_for(self.get_agent_framework().memory)
                yield from stream_run(initial_log_data, log_queue, result_queue, initial)
            finally:
                detach_handler(handler)

        def do_select(selected_index: gr.SelectData):
            row = selected_index.index[0]
            opportunity = self.get_agent_framework().memory[row]
            self.get_agent_framework().init_agents_as_needed()
            self.get_agent_framework().planner.messenger.alert(opportunity)

        with gr.Blocks(title="The Price is Right", fill_width=True) as ui:
            log_data = gr.State([])
            gr.Markdown('<div style="text-align: center;font-size:24px"><strong>The Price is Right</strong> - Autonomous Agent Framework that hunts for deals</div>')
            gr.Markdown('<div style="text-align: center;font-size:14px">A fine-tuned LLM deployed on Modal and a RAG pipeline with a frontier model collaborate to send push notifications with great online deals.</div>')
            table = gr.Dataframe(
                headers=["Deals found so far", "Price", "Estimate", "Discount", "URL"],
                wrap=True, column_widths=[6, 1, 1, 1, 3], row_count=10, col_count=5, max_height=400,
            )
            with gr.Row():
                with gr.Column(scale=1):
                    logs = gr.HTML()
                with gr.Column(scale=1):
                    gr.Plot(value=get_plot(), show_label=False)
            ui.load(run_with_logging, inputs=[log_data], outputs=[log_data, logs, table])
            timer = gr.Timer(value=self.settings.ui_refresh_seconds, active=True)
            timer.tick(run_with_logging, inputs=[log_data], outputs=[log_data, logs, table])
            table.select(do_select)
        return ui

    def run(self, share: bool = False, inbrowser: bool = True):
        self.build().launch(share=share, inbrowser=inbrowser)
