"""Gradio evaluation dashboard with a pipeline selector."""

from __future__ import annotations

from collections import defaultdict

import gradio as gr
import pandas as pd

from ..config import Settings
from ..llm import LiteLLMClient
from ..pipelines import get_pipeline
from .evaluator import (
    evaluate_all_answers,
    evaluate_all_retrieval,
    summarize_answers,
    summarize_retrieval,
)
from .metrics import load_tests

# Colour thresholds. Retrieval: MRR/nDCG 0-1, coverage %. Answers: 1-5 scale.
THRESHOLDS = {
    "mrr": (0.9, 0.75),
    "ndcg": (0.9, 0.75),
    "coverage": (90.0, 75.0),
    "accuracy": (4.5, 4.0),
    "completeness": (4.5, 4.0),
    "relevance": (4.5, 4.0),
}


def get_color(value: float, metric_type: str) -> str:
    if metric_type not in THRESHOLDS:
        return "black"
    green, amber = THRESHOLDS[metric_type]
    return "green" if value >= green else "orange" if value >= amber else "red"


def format_metric_html(label, value, metric_type, is_percentage=False, score_format=False) -> str:
    color = get_color(value, metric_type)
    text = f"{value:.1f}%" if is_percentage else f"{value:.2f}/5" if score_format else f"{value:.4f}"
    return f"""
    <div style="margin: 10px 0; padding: 15px; background-color: #f5f5f5; border-radius: 8px; border-left: 5px solid {color};">
        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">{label}</div>
        <div style="font-size: 28px; font-weight: bold; color: {color};">{text}</div>
    </div>"""


def _done(count: int) -> str:
    return (
        "<div style='margin-top: 20px; padding: 10px; background-color: #d4edda; border-radius: 5px; "
        f"text-align: center; border: 1px solid #c3e6cb;'><b style='color:#155724'>Evaluation complete: {count} tests</b></div>"
    )


def run_retrieval_evaluation(pipeline_name: str, progress=gr.Progress()):
    settings = Settings.from_env()
    pipeline = get_pipeline(pipeline_name, settings)
    rows, by_cat = [], defaultdict(list)
    for test, result, frac in evaluate_all_retrieval(pipeline, load_tests(settings.tests_file)):
        rows.append((test, result))
        by_cat[test.category].append(result.mrr)
        progress(frac, desc=f"Evaluating test {len(rows)}...")
    s = summarize_retrieval(rows)
    html = (
        "<div>"
        + format_metric_html("Mean Reciprocal Rank (MRR)", s["mrr"], "mrr")
        + format_metric_html("Normalized DCG (nDCG)", s["ndcg"], "ndcg")
        + format_metric_html("Keyword Coverage", s["coverage"], "coverage", is_percentage=True)
        + _done(s["count"])
        + "</div>"
    )
    df = pd.DataFrame([{"Category": c, "Average MRR": sum(v) / len(v)} for c, v in by_cat.items()])
    return html, df


def run_answer_evaluation(pipeline_name: str, progress=gr.Progress()):
    settings = Settings.from_env()
    pipeline = get_pipeline(pipeline_name, settings)
    judge = LiteLLMClient(settings.judge_llm_model)
    rows, by_cat = [], defaultdict(list)
    for test, result, frac in evaluate_all_answers(pipeline, judge, load_tests(settings.tests_file)):
        rows.append((test, result))
        by_cat[test.category].append(result.accuracy)
        progress(frac, desc=f"Evaluating test {len(rows)}...")
    s = summarize_answers(rows)
    html = (
        "<div>"
        + format_metric_html("Accuracy", s["accuracy"], "accuracy", score_format=True)
        + format_metric_html("Completeness", s["completeness"], "completeness", score_format=True)
        + format_metric_html("Relevance", s["relevance"], "relevance", score_format=True)
        + _done(s["count"])
        + "</div>"
    )
    df = pd.DataFrame([{"Category": c, "Average Accuracy": sum(v) / len(v)} for c, v in by_cat.items()])
    return html, df


PLACEHOLDER = "<div style='padding: 20px; text-align: center; color: #999;'>Click 'Run Evaluation' to start</div>"


def build_ui(default_pipeline: str = "basic") -> gr.Blocks:
    with gr.Blocks(title="RAG Evaluation Dashboard", theme=gr.themes.Soft()) as app:
        gr.Markdown("# RAG Evaluation Dashboard\nEvaluate retrieval and answer quality for the Insurellm RAG system")
        pipeline = gr.Radio(["basic", "pro"], value=default_pipeline, label="Pipeline")

        gr.Markdown("## Retrieval Evaluation")
        r_btn = gr.Button("Run Evaluation", variant="primary")
        with gr.Row():
            r_html = gr.HTML(PLACEHOLDER)
            r_plot = gr.BarPlot(x="Category", y="Average MRR", title="Average MRR by Category", y_lim=[0, 1], height=400)

        gr.Markdown("## Answer Evaluation (LLM-as-judge)")
        a_btn = gr.Button("Run Evaluation", variant="primary")
        with gr.Row():
            a_html = gr.HTML(PLACEHOLDER)
            a_plot = gr.BarPlot(x="Category", y="Average Accuracy", title="Average Accuracy by Category", y_lim=[1, 5], height=400)

        r_btn.click(run_retrieval_evaluation, inputs=pipeline, outputs=[r_html, r_plot])
        a_btn.click(run_answer_evaluation, inputs=pipeline, outputs=[a_html, a_plot])
    return app


def main(pipeline: str = "basic", **launch_kwargs) -> None:
    build_ui(pipeline).launch(**launch_kwargs)
