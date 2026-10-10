from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
import threading
import traceback
from typing import Any
import uuid

import dash
from dash import Dash, Input, Output, State, dcc, html, no_update
import dash_ag_grid as dag
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Ensure local eval modules resolve regardless of launch working directory.
EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from common import (
    call_evaluate,
    call_health,
    call_upload_file,
    load_cases_from_path,
    parse_cases_text,
    parse_upload_contents,
    write_json_report,
    write_jsonl,
)
from judge import (
    DEFAULT_JUDGE_BASE_URL,
    DEFAULT_JUDGE_MODEL,
    JudgeConfig,
    make_judge_config,
    api_error_judge_result,
    disabled_judge_result,
    evaluate_with_api,
)
from adapters import AdapterProfile, HttpRagAdapter, load_profile
from runner import iter_evaluations


DEFAULT_DATASET_PATH = Path("app/src/debug/assets/eval_cases.json")
DEFAULT_CASES = load_cases_from_path(DEFAULT_DATASET_PATH) if DEFAULT_DATASET_PATH.exists() else []
SETUP_MODEL_CATALOG_PATH = (
    EVAL_DIR.parents[1]
    / "app/src/main/java/io/brite/medrag/models/data/SetupModelCatalog.kt"
)


def load_android_llm_options() -> list[dict[str, str]]:
    options = [{"label": "Auto-select LiteRT-LM", "value": ""}]

    try:
        catalog = SETUP_MODEL_CATALOG_PATH.read_text(encoding="utf-8")
    except OSError:
        return options

    for model_id, model_name in re.findall(
        r'modelId\s*=\s*"([^"]+)".*?modelName\s*=\s*"([^"]+)"',
        catalog,
        flags=re.DOTALL,
    ):
        options.append({"label": model_name, "value": model_id})

    return options


ANDROID_LLM_OPTIONS = load_android_llm_options()
RUN_LOG_LIMIT = 250

RUN_LOCK = threading.Lock()
EVALUATE_TIMEOUT_SECONDS = 180
RUN_STATE: dict[str, Any] = {
    "job_id": None,
    "status": "idle",
    "rows": [],
    "logs": [],
    "summary": "",
    "live_status": "Idle",
    "completed_cases": 0,
    "total_cases": 0,
    "output_path": None,
}

# ── Design tokens ──────────────────────────────────────────────────────────────
BG          = "#030609"
SURFACE     = "#080c14"
SURFACE_HI  = "#0d1220"
SURFACE_HH  = "#121a2c"
BORDER      = "#182236"
BORDER_HI   = "#253555"
ACCENT      = "#3b82f6"
ACCENT_LO   = "#60a5fa"
ACCENT_DIM  = "#1e3a5f"
SUCCESS     = "#10b981"
DANGER      = "#f43f5e"
WARN        = "#f59e0b"
PURPLE      = "#a78bfa"
TEXT        = "#f0f4ff"
TEXT_MUTED  = "#475569"
TEXT_DIM    = "#94a3b8"
FONT        = "'Bricolage Grotesque', 'Segoe UI', Helvetica, Arial, sans-serif"
MONO        = "'JetBrains Mono', 'Fira Code', Consolas, monospace"

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=FONT, color=TEXT_DIM, size=11),
    title_font=dict(family=FONT, color=TEXT, size=12),
    margin=dict(l=8, r=8, t=10, b=8),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER, tickfont=dict(size=10)),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER, tickfont=dict(size=10)),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, font=dict(size=11)),
    hoverlabel=dict(bgcolor=SURFACE_HH, bordercolor=BORDER_HI, font=dict(family=MONO, size=11)),
)

CUSTOM_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,300;12..96,400;12..96,500;12..96,600;12..96,700;12..96,800&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}
body {{
    background: {BG} !important;
    background-image: radial-gradient(circle, {BORDER}66 1px, transparent 1px) !important;
    background-size: 28px 28px !important;
    color: {TEXT} !important;
    font-family: {FONT} !important;
    margin: 0;
    -webkit-font-smoothing: antialiased;
}}

::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {SURFACE}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER_HI}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {ACCENT}; }}

/* AG Grid dark override */
.ag-theme-alpine-dark {{
    --ag-background-color: {SURFACE} !important;
    --ag-odd-row-background-color: {SURFACE_HI} !important;
    --ag-header-background-color: {SURFACE_HH} !important;
    --ag-border-color: {BORDER} !important;
    --ag-row-border-color: {BORDER} !important;
    --ag-foreground-color: {TEXT} !important;
    --ag-header-foreground-color: {TEXT_DIM} !important;
    --ag-secondary-foreground-color: {TEXT_MUTED} !important;
    --ag-row-hover-color: {SURFACE_HH} !important;
    --ag-selected-row-background-color: {ACCENT_DIM}55 !important;
    --ag-range-selection-border-color: {ACCENT} !important;
    --ag-input-focus-border-color: {ACCENT} !important;
    --ag-checkbox-checked-color: {ACCENT} !important;
    --ag-font-family: {FONT} !important;
    --ag-font-size: 12px !important;
    --ag-cell-horizontal-padding: 14px !important;
    --ag-row-height: 38px !important;
    --ag-header-height: 40px !important;
}}
.ag-theme-alpine-dark .ag-root-wrapper {{
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    background: {SURFACE} !important;
}}
.ag-theme-alpine-dark .ag-header {{
    border-bottom: 1px solid {BORDER_HI} !important;
}}
.ag-theme-alpine-dark .ag-header-cell-label {{
    font-size: 10px !important;
    font-weight: 700 !important;
    color: {TEXT_MUTED} !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}
.ag-theme-alpine-dark .ag-cell {{
    font-family: {MONO} !important;
    font-size: 12px !important;
    color: {TEXT_DIM} !important;
}}
.ag-theme-alpine-dark .ag-paging-panel {{
    background: {SURFACE_HI} !important;
    border-top: 1px solid {BORDER} !important;
    color: {TEXT_MUTED} !important;
    font-size: 11px !important;
    font-family: {MONO} !important;
}}
.ag-theme-alpine-dark .ag-row-selected {{
    background: {ACCENT_DIM}44 !important;
}}
.ag-theme-alpine-dark .ag-row:hover {{
    background: {SURFACE_HH} !important;
}}

/* Inputs */
input[type=text], input[type=number], input[type=password] {{
    background: {SURFACE_HI} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    color: {TEXT} !important;
    color-scheme: dark !important;
    font-family: {MONO} !important;
    font-size: 13px !important;
    padding: 8px 12px !important;
    outline: none !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}}
input[type=number] {{
    appearance: textfield !important;
    -moz-appearance: textfield !important;
}}
input[type=number]::-webkit-inner-spin-button,
input[type=number]::-webkit-outer-spin-button {{
    -webkit-appearance: none !important;
    appearance: none !important;
    margin: 0 !important;
}}
input[type=text]:focus, input[type=number]:focus, input[type=password]:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 3px {ACCENT}22 !important;
}}
/* Dash dropdown */
.dark-dropdown .Select-control,
.dark-dropdown .Select-menu-outer,
.dark-dropdown .Select-menu,
.dark-dropdown .Select-placeholder,
.dark-dropdown .Select-value,
.dark-dropdown .Select-input,
.dark-dropdown .Select-input > input {{
    background: {SURFACE_HI} !important;
    color: {TEXT} !important;
    border-color: {BORDER} !important;
    font-family: {FONT} !important;
    font-size: 11px !important;
}}
.dark-dropdown .Select-control {{
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    min-height: 36px !important;
    box-shadow: none !important;
}}
.dark-dropdown.is-focused .Select-control,
.dark-dropdown .Select-control:hover {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 3px {ACCENT}22 !important;
}}
.dark-dropdown .Select-value-label,
.dark-dropdown .Select-placeholder {{
    color: {TEXT_DIM} !important;
    line-height: 34px !important;
}}
.dark-dropdown .Select-arrow-zone {{
    background: {SURFACE_HI} !important;
}}
.dark-dropdown .Select-arrow {{
    border-top-color: {TEXT_DIM} !important;
}}
.dark-dropdown .Select-menu-outer {{
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    margin-top: 4px !important;
    overflow: hidden !important;
}}
.dark-dropdown .VirtualizedSelectOption,
.dark-dropdown .Select-option {{
    background: {SURFACE_HI} !important;
    color: {TEXT_DIM} !important;
    font-family: {FONT} !important;
    font-size: 11px !important;
}}
.dark-dropdown .VirtualizedSelectFocusedOption,
.dark-dropdown .Select-option.is-focused {{
    background: {SURFACE_HH} !important;
    color: {TEXT} !important;
}}
.dark-dropdown .Select-option.is-selected {{
    background: {ACCENT_DIM} !important;
    color: {TEXT} !important;
}}
#android-llm-model,
#android-llm-model .Select,
#android-llm-model .Select-control,
#android-llm-model .Select-menu-outer,
#android-llm-model .Select-menu,
#android-llm-model .Select-placeholder,
#android-llm-model .Select-value,
#android-llm-model .Select-input,
#android-llm-model .Select-input > input,
#android-llm-model [class*="control"],
#android-llm-model [class*="menu"],
#android-llm-model [class*="singleValue"],
#android-llm-model [class*="placeholder"],
#android-llm-model [class*="Input"] {{
    background-color: {SURFACE_HI} !important;
    color: {TEXT} !important;
    border-color: {BORDER} !important;
}}
#android-llm-model .Select-control,
#android-llm-model [class*="control"] {{
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    min-height: 36px !important;
}}
#android-llm-model .Select-value-label,
#android-llm-model .Select-placeholder,
#android-llm-model [class*="singleValue"],
#android-llm-model [class*="placeholder"] {{
    color: {TEXT_DIM} !important;
}}
#android-llm-model .Select-option,
#android-llm-model .VirtualizedSelectOption,
#android-llm-model [class*="option"] {{
    background-color: {SURFACE_HI} !important;
    color: {TEXT_DIM} !important;
}}
#android-llm-model .Select-option.is-focused,
#android-llm-model .VirtualizedSelectFocusedOption,
#android-llm-model [class*="option"]:hover {{
    background-color: {SURFACE_HH} !important;
    color: {TEXT} !important;
}}

/* KPI top accent line */
.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, {ACCENT} 0%, {ACCENT_LO} 100%);
    border-radius: 12px 12px 0 0;
    opacity: 0.8;
}}
.kpi-card.kpi-green::before {{ background: linear-gradient(90deg, {SUCCESS} 0%, #34d399 100%); }}
.kpi-card.kpi-red::before   {{ background: linear-gradient(90deg, {DANGER} 0%, #fb7185 100%); }}
.kpi-card.kpi-amber::before {{ background: linear-gradient(90deg, {WARN} 0%, #fbbf24 100%); }}
.kpi-card.kpi-purple::before {{ background: linear-gradient(90deg, {PURPLE} 0%, #c4b5fd 100%); }}

/* Run button */
@keyframes glow-pulse {{
    0%, 100% {{ box-shadow: 0 0 16px {ACCENT}55; }}
    50%       {{ box-shadow: 0 0 36px {ACCENT}88, 0 0 70px {ACCENT}22; }}
}}
#run-eval {{ animation: glow-pulse 3s ease-in-out infinite; transition: all 0.15s ease !important; }}
#run-eval:hover {{ filter: brightness(1.18) !important; transform: translateY(-1px) !important; box-shadow: 0 6px 24px {ACCENT}55 !important; }}
#run-eval:active {{ transform: translateY(0) !important; filter: brightness(0.92) !important; }}
#run-eval:disabled {{
    animation: none !important;
    cursor: wait !important;
    opacity: 0.72 !important;
    filter: saturate(0.75) !important;
    transform: none !important;
    box-shadow: none !important;
}}

/* Upload pill hover */
.upload-pill:hover {{
    border-color: {ACCENT} !important;
    color: {TEXT} !important;
    background: {SURFACE_HH} !important;
}}

/* Checkbox */
input[type=checkbox] {{ accent-color: {ACCENT} !important; }}
.dash-checklist label,
#use-api-judge label {{
    color: {TEXT_DIM} !important;
    font-family: {FONT} !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    line-height: 1.4 !important;
    cursor: pointer !important;
}}
#failure-only label {{
    color: {TEXT} !important;
    font-family: {FONT} !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    line-height: 1.4 !important;
    cursor: pointer !important;
}}

/* Markdown detail */
#case-detail h3 {{ color: {TEXT} !important; margin: 0 0 8px; font-size: 15px; }}
#case-detail h4 {{ color: {TEXT_DIM} !important; margin: 16px 0 6px; font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; }}
#case-detail p, #case-detail li {{ color: {TEXT_DIM} !important; line-height: 1.75; font-size: 13px; }}
#case-detail code {{ background: {SURFACE_HH} !important; color: {ACCENT_LO} !important; padding: 2px 6px; border-radius: 4px; font-family: {MONO}; font-size: 12px; }}
#case-detail blockquote {{ border-left: 3px solid {ACCENT} !important; padding-left: 14px; color: {TEXT_MUTED} !important; font-style: italic; margin: 10px 0; font-size: 12px; }}
#case-detail strong {{ color: {TEXT} !important; }}
#case-detail a {{ color: {ACCENT_LO} !important; }}
"""


# ── UI helpers ────────────────────────────────────────────────────────────────

def _dot(color: str, glow: bool = False) -> html.Span:
    shadow = f"0 0 6px {color}cc" if glow else "none"
    return html.Span(style={
        "display": "inline-block",
        "width": "7px", "height": "7px",
        "borderRadius": "50%",
        "background": color,
        "marginRight": "7px",
        "verticalAlign": "middle",
        "flexShrink": "0",
        "boxShadow": shadow,
    })


def _kpi_card(card_id: str, label: str, icon: str = "", variant: str = "") -> html.Div:
    cls = f"kpi-card kpi-{variant}" if variant else "kpi-card"
    return html.Div(
        [
            html.Div(
                [
                    html.Span(icon, style={"marginRight": "7px", "fontSize": "13px"}),
                    html.Span(label, style={
                        "fontSize": "10px", "fontWeight": "700",
                        "letterSpacing": "0.1em", "textTransform": "uppercase",
                        "color": TEXT_MUTED,
                    }),
                ],
                style={"display": "flex", "alignItems": "center", "marginBottom": "14px"},
            ),
            html.Div("—", id=card_id, style={
                "fontSize": "30px", "fontWeight": "700",
                "fontFamily": MONO, "color": TEXT,
                "letterSpacing": "-0.03em", "lineHeight": "1",
            }),
        ],
        className=cls,
        style={
            "background": f"linear-gradient(150deg, {SURFACE_HH} 0%, {SURFACE_HI} 100%)",
            "border": f"1px solid {BORDER}",
            "borderRadius": "12px",
            "padding": "18px 20px 22px",
            "position": "relative",
            "overflow": "hidden",
        },
    )


def _field(label: str, control: Any, width: str | None = None) -> html.Div:
    style: dict = {"flex": "1"} if not width else {"width": width, "flexShrink": "0"}
    return html.Div([
        html.Div(label, style={
            "fontSize": "10px", "fontWeight": "700",
            "letterSpacing": "0.1em", "textTransform": "uppercase",
            "color": TEXT_MUTED, "marginBottom": "6px",
        }),
        control,
    ], style=style)


def _upload(upload_id: str, label: str) -> dcc.Upload:
    return dcc.Upload(
        id=upload_id,
        children=html.Div(
            ["↑  ", label],
            className="upload-pill",
            style={
                "background": SURFACE_HI,
                "border": f"1px solid {BORDER}",
                "borderRadius": "8px",
                "color": TEXT_DIM,
                "padding": "8px 16px",
                "fontSize": "12px",
                "fontWeight": "500",
                "cursor": "pointer",
                "display": "inline-block",
                "transition": "all 0.2s",
                "whiteSpace": "nowrap",
                "userSelect": "none",
            },
        ),
        multiple=False,
    )


def _hex_to_rgba(color: str, alpha: float) -> str:
    value = color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected a 6-digit hex color, got: {color}")
    red = int(value[0:2], 16)
    green = int(value[2:4], 16)
    blue = int(value[4:6], 16)
    bounded_alpha = max(0.0, min(1.0, float(alpha)))
    return f"rgba({red}, {green}, {blue}, {bounded_alpha:.3f})"


def _chart_card(title: str, graph_id: str) -> html.Div:
    return html.Div(
        [
            html.Div(title, style={
                "fontSize": "10px", "fontWeight": "700",
                "letterSpacing": "0.1em", "textTransform": "uppercase",
                "color": TEXT_MUTED, "padding": "14px 16px 0",
            }),
            dcc.Graph(id=graph_id, style={"height": "240px"},
                      config={"displayModeBar": False}),
        ],
        style={
            "flex": "1",
            "background": f"linear-gradient(150deg, {SURFACE_HH} 0%, {SURFACE_HI} 100%)",
            "border": f"1px solid {BORDER}",
            "borderRadius": "12px",
            "overflow": "hidden",
        },
    )


def _run_button_children(icon: str, label: str) -> list[Any]:
    return [html.Span(icon, style={"marginRight": "8px", "fontSize": "10px"}), label]


# ── App ────────────────────────────────────────────────────────────────────────

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "BRITE MedRAG Eval"

app.index_string = f"""<!DOCTYPE html>
<html>
<head>
    {{%metas%}}
    <title>{{%title%}}</title>
    {{%favicon%}}
    {{%css%}}
    <style>{CUSTOM_CSS}</style>
</head>
<body>
    {{%app_entry%}}
    <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
</body>
</html>"""

app.layout = html.Div(
    [
        # ── Header ───────────────────────────────────────────────────────────
        html.Div(
            [
                html.Div([
                    html.Div([
                        html.Span("BRITE", style={
                            "background": ACCENT_DIM,
                            "border": f"1px solid {BORDER_HI}",
                            "borderRadius": "4px",
                            "padding": "2px 8px",
                            "fontSize": "10px",
                            "fontWeight": "700",
                            "letterSpacing": "0.14em",
                            "color": ACCENT_LO,
                            "textTransform": "uppercase",
                            "fontFamily": MONO,
                            "marginRight": "8px",
                        }),
                        html.Span("/", style={"color": BORDER_HI, "fontSize": "16px", "marginRight": "8px"}),
                        html.Span("MedRAG · Evaluation Suite", style={
                            "fontFamily": MONO,
                            "fontSize": "11px",
                            "color": TEXT_MUTED,
                            "letterSpacing": "0.04em",
                        }),
                    ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
                    html.Div("Evaluation Dashboard", style={
                        "fontSize": "26px", "fontWeight": "800", "color": TEXT,
                        "letterSpacing": "-0.03em", "lineHeight": "1",
                    }),
                ]),
                html.Div([
                    html.Div([
                        _dot(SUCCESS, glow=True),
                        html.Span(
                            "Local evaluator",
                            style={"fontSize": "11px", "color": TEXT_DIM, "fontFamily": MONO},
                        ),
                    ], style={
                        "display": "flex", "alignItems": "center",
                        "background": SURFACE_HI,
                        "border": f"1px solid {BORDER}",
                        "borderRadius": "6px",
                        "padding": "6px 12px",
                        "marginBottom": "7px",
                    }),
                    html.Span(
                        "Python-controlled  ·  Android and HTTP RAG evaluation",
                        style={
                            "fontSize": "10px", "color": TEXT_MUTED, "fontFamily": MONO,
                            "background": SURFACE_HH,
                            "border": f"1px solid {BORDER}",
                            "borderRadius": "5px",
                            "padding": "3px 9px",
                            "letterSpacing": "0.04em",
                        },
                    ),
                ], style={"display": "flex", "flexDirection": "column", "alignItems": "flex-end"}),
            ],
            style={
                "display": "flex", "justifyContent": "space-between", "alignItems": "center",
                "borderBottom": f"1px solid {BORDER}",
                "paddingBottom": "20px", "marginBottom": "24px",
            },
        ),

        # ── Config ───────────────────────────────────────────────────────────
        html.Div(
            [
                _field("Application API Base URL",
                       dcc.Input(id="api-base", type="text",
                                 value="http://127.0.0.1:9000", style={"width": "100%"})),
                _field("Application Adapter",
                       dcc.Dropdown(id="adapter-kind", options=[
                           {"label": "BRITE Android (existing app)", "value": "brite"},
                           {"label": "Generic Android / HTTP API", "value": "generic"},
                           {"label": "Uploaded API profile", "value": "custom"},
                       ], value="brite", clearable=False, className="dark-dropdown"), width="260px"),
                _field("Android LLM",
                       dcc.Dropdown(
                           id="android-llm-model",
                           options=ANDROID_LLM_OPTIONS,
                           value="",
                           clearable=False,
                           className="dark-dropdown",
                           style={"width": "100%"},
                       ), width="300px"),
                _field("Model ID Override (other apps)",
                       dcc.Input(id="external-model-id", type="text", value="",
                                 placeholder="Application default", style={"width": "100%"}), width="260px"),
                _field("Judge Domain Override",
                       dcc.Input(id="judge-domain", type="text", value="",
                                 placeholder="Adapter default", style={"width": "100%"}), width="240px"),
                _field("Top K Override",
                       dcc.Input(id="top-k", type="number", min=1, step=1,
                                 value=5, style={"width": "100%"}), width="148px"),
                _field("Max Questions",
                       dcc.Input(id="max-questions", type="number", min=1, step=1,
                                 value=None, placeholder="all", style={"width": "100%"}), width="148px"),
                _field("Use API Judge",
                       dcc.Checklist(
                           id="use-api-judge",
                           options=[{"label": " Enabled", "value": "enabled"}],
                           value=["enabled"],
                           style={"color": TEXT_DIM, "fontSize": "12px", "fontWeight": "500"},
                           inputStyle={"accentColor": ACCENT, "marginRight": "6px"},
                       ), width="160px"),
                _field("Retrieval Only",
                       dcc.Checklist(
                           id="retrieval-only",
                           options=[{"label": " Enabled", "value": "enabled"}],
                           value=[],
                           style={"color": TEXT_DIM, "fontSize": "12px", "fontWeight": "500"},
                           inputStyle={"accentColor": ACCENT, "marginRight": "6px"},
                       ), width="160px"),
                _field("Judge API Base URL",
                       dcc.Input(id="judge-base-url", type="text",
                                 value=os.environ.get("RAG_JUDGE_BASE_URL") or DEFAULT_JUDGE_BASE_URL, style={"width": "100%"}), width="250px"),
                _field("Judge Model",
                       dcc.Input(id="judge-model", type="text",
                                 value=os.environ.get("RAG_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL,
                                 placeholder="Model ID from your provider", style={"width": "100%"}), width="250px"),
                _field("Judge API Key",
                       dcc.Input(id="judge-api-key", type="password", value="",
                                 placeholder="Or set RAG_JUDGE_API_KEY", autoComplete="off",
                                 persistence=False, style={"width": "100%"}), width="270px"),
            ],
            style={"display": "flex", "gap": "12px", "marginBottom": "14px", "flexWrap": "wrap"},
        ),

        html.P("API judging sends questions, reference answers, generated answers, and retrieved evidence to your selected provider. The key stays in memory and is excluded from reports.",
               style={"color": TEXT_MUTED, "fontSize": "12px", "lineHeight": "1.6", "marginBottom": "16px"}),

        # ── Actions ──────────────────────────────────────────────────────────
        html.Div([
            _upload("adapter-profile-upload", "Upload API Profile (JSON)"),
            html.Span(id="adapter-profile-label", children="Optional: map another app's endpoints and fields",
                      style={"fontSize": "11px", "color": TEXT_DIM, "marginLeft": "12px"}),
        ], style={"marginBottom": "14px"}),
        html.Div(
            [
                html.Div([
                    _upload("cases-upload", "Upload Questions (JSON/JSONL)"),
                    html.Div(
                        id="cases-upload-label",
                        children=[
                            _dot(SUCCESS if DEFAULT_CASES else TEXT_MUTED, glow=bool(DEFAULT_CASES)),
                            f"{len(DEFAULT_CASES)} cases loaded" if DEFAULT_CASES else "No dataset loaded",
                        ],
                        style={"fontSize": "11px", "marginTop": "6px",
                               "color": SUCCESS if DEFAULT_CASES else TEXT_MUTED,
                               "fontFamily": MONO, "display": "flex", "alignItems": "center"},
                    ),
                ]),
                html.Div([
                    _upload("corpus-upload", "Upload Source Corpus"),
                    html.Div(
                        id="corpus-upload-label",
                        children=[_dot(TEXT_MUTED), "Android default corpus"],
                        style={"fontSize": "11px", "marginTop": "6px", "color": TEXT_MUTED,
                               "fontFamily": MONO, "display": "flex", "alignItems": "center"},
                    ),
                ]),
                html.Div(style={"flex": "1"}),
                html.Button(
                    _run_button_children("▶", "Run Evaluation"),
                    id="run-eval", n_clicks=0,
                    style={
                        "background": f"linear-gradient(135deg, {ACCENT} 0%, {ACCENT_DIM} 100%)",
                        "border": f"1px solid {ACCENT_LO}33",
                        "borderRadius": "9px",
                        "color": "#fff",
                        "padding": "10px 26px",
                        "fontSize": "13px",
                        "fontWeight": "600",
                        "cursor": "pointer",
                        "letterSpacing": "0.02em",
                        "whiteSpace": "nowrap",
                        "alignSelf": "flex-start",
                        "fontFamily": FONT,
                    },
                ),
            ],
            style={"display": "flex", "gap": "16px", "alignItems": "flex-start",
                   "marginBottom": "8px"},
        ),

        # ── Status line ──────────────────────────────────────────────────────
        dcc.Loading(
            html.Div(
                [
                    html.Div(id="run-live-status", style={
                        "fontSize": "11px", "fontFamily": MONO, "color": WARN,
                        "minHeight": "20px", "letterSpacing": "0.01em",
                    }),
                    html.Div(id="run-status", style={
                        "fontSize": "11px", "fontFamily": MONO, "color": ACCENT_LO,
                        "minHeight": "20px", "letterSpacing": "0.01em",
                    }),
                ],
                style={"marginBottom": "22px"},
            ),
            color=ACCENT_LO,
            type="circle",
        ),

        html.Div(
            [
                html.Div("Run Log", style={
                    "fontSize": "10px", "fontWeight": "700",
                    "letterSpacing": "0.1em", "textTransform": "uppercase",
                    "color": TEXT_MUTED, "marginBottom": "10px",
                }),
                html.Pre(
                    id="run-log",
                    children="No evaluation run yet.",
                    style={
                        "margin": "0",
                        "padding": "14px 16px",
                        "minHeight": "140px",
                        "maxHeight": "220px",
                        "overflowY": "auto",
                        "whiteSpace": "pre-wrap",
                        "wordBreak": "break-word",
                        "background": SURFACE,
                        "border": f"1px solid {BORDER}",
                        "borderRadius": "10px",
                        "fontFamily": MONO,
                        "fontSize": "11px",
                        "lineHeight": "1.7",
                        "color": TEXT_DIM,
                    },
                ),
            ],
            style={"marginBottom": "16px"},
        ),

        # ── KPI row ──────────────────────────────────────────────────────────
        html.Div(
            [
                _kpi_card("kpi-total",       "Total Cases",       "📋", "blue"),
                _kpi_card("kpi-pass-rate",   "Pass Rate",         "✓",  "green"),
                _kpi_card("kpi-avg-score",   "Avg Semantic Score", "◎",  "blue"),
                _kpi_card("kpi-avg-latency", "Avg Total Latency", "⚡", "amber"),
                _kpi_card("kpi-avg-tps",     "Avg Tokens/s",      "⟳",  "purple"),
                _kpi_card("kpi-avg-ttft",    "Avg TTFT",          "⏱",  "amber"),
                _kpi_card("kpi-avg-output",  "Avg Output Tokens", "✎",  "blue"),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(4, minmax(0, 1fr))",
                "gap": "10px", "marginBottom": "16px",
            },
        ),

        # ── Charts ───────────────────────────────────────────────────────────
        html.Div(
            [
                _chart_card("Semantic Score Distribution", "score-chart"),
                _chart_card("Latency by Category", "latency-chart"),
            ],
            style={"display": "flex", "gap": "10px", "marginBottom": "16px"},
        ),

        # ── Filter ───────────────────────────────────────────────────────────
        html.Div(
            dcc.Checklist(
                id="failure-only",
                options=[{"label": "  Show failures only", "value": "failures"}],
                value=[],
                style={"color": TEXT, "fontSize": "12px", "fontWeight": "600",
                       "fontFamily": FONT, "cursor": "pointer"},
                inputStyle={"accentColor": ACCENT, "marginRight": "6px"},
            ),
            style={"marginBottom": "10px"},
        ),

        # ── Grid ─────────────────────────────────────────────────────────────
        dag.AgGrid(
            id="results-grid",
            columnDefs=[
                {"field": "case_id",   "headerName": "Case ID",   "minWidth": 130, "flex": 1},
                {"field": "category",  "headerName": "Category",  "minWidth": 110, "flex": 1},
                {"field": "pass",      "headerName": "Pass",      "minWidth": 80,
                 "cellStyle": {"function":
                                   "params.value===true?{'color':'#22c55e','fontWeight':'700'}:"
                                   "params.value===false?{'color':'#f43f5e','fontWeight':'700'}:{}"}},
                {"field": "semantic_score", "headerName": "Semantic", "minWidth": 105,
                 "valueFormatter": {"function": "params.value!=null?params.value.toFixed(3):'—'"},
                 "cellStyle": {"function":
                                   "params.value>=0.75?{'color':'#22c55e'}:"
                                   "params.value>=0.5?{'color':'#f59e0b'}:{'color':'#f43f5e'}"}},
                {"field": "semantic_judge_status",  "headerName": "Judge",       "minWidth": 105},
                {"field": "retrieval_verdict",      "headerName": "Ret Verdict", "minWidth": 120},
                {"field": "retrieval_hit_at_5",     "headerName": "Hit@5",       "minWidth": 85},
                {"field": "retrieval_precision_at_5", "headerName": "P@5",       "minWidth": 85,
                 "valueFormatter": {"function": "params.value!=null?params.value.toFixed(3):'—'"}},
                {"field": "retrieval_soft_precision_at_5", "headerName": "Soft P@5", "minWidth": 100,
                 "valueFormatter": {"function": "params.value!=null?params.value.toFixed(3):'—'"}},
                {"field": "retrieval_mrr_at_5",     "headerName": "MRR@5",       "minWidth": 95,
                 "valueFormatter": {"function": "params.value!=null?params.value.toFixed(3):'—'"}},
                {"field": "retrieval_ndcg_at_5",    "headerName": "nDCG@5",      "minWidth": 95,
                 "valueFormatter": {"function": "params.value!=null?params.value.toFixed(3):'—'"}},
                {"field": "retrieval_distractor_count", "headerName": "Distractors", "minWidth": 110},
                {"field": "answer_score",           "headerName": "Answer",      "minWidth": 95,
                 "valueFormatter": {"function": "params.value!=null?params.value.toFixed(3):'—'"}},
                {"field": "answer_verdict",         "headerName": "Ans Verdict", "minWidth": 120},
                {"field": "dangerous_claims",       "headerName": "Dangerous",   "minWidth": 160, "flex": 1,
                 "valueFormatter": {"function": "Array.isArray(params.value)?params.value.join('; '):(params.value||'—')"}},
                {"field": "missing_facts",          "headerName": "Missing",     "minWidth": 180, "flex": 1,
                 "valueFormatter": {"function": "Array.isArray(params.value)?params.value.join('; '):(params.value||'—')"}},
                {"field": "unsupported_claims",     "headerName": "Unsupported", "minWidth": 180, "flex": 1,
                 "valueFormatter": {"function": "Array.isArray(params.value)?params.value.join('; '):(params.value||'—')"}},
                {"field": "retrieval_confidence",   "headerName": "Runtime Ret", "minWidth": 110},
                {"field": "vector_candidate_count", "headerName": "Vector",      "minWidth": 90},
                {"field": "bm25_candidate_count",   "headerName": "BM25",        "minWidth": 90},
                {"field": "fused_candidate_count",  "headerName": "RRF",         "minWidth": 90},
                {"field": "expanded_result_count",  "headerName": "Expanded",    "minWidth": 95},
                {"field": "top_fused_score",        "headerName": "Top RRF",     "minWidth": 95,
                 "valueFormatter": {"function": "params.value!=null?params.value.toFixed(3):'—'"}},
                {"field": "top_final_score",        "headerName": "Top Final",   "minWidth": 95,
                 "valueFormatter": {"function": "params.value!=null?params.value.toFixed(3):'—'"}},
                {"field": "total_latency_ms",     "headerName": "Total ms",      "minWidth": 100},
                {"field": "tokens_per_second",    "headerName": "t/s",           "minWidth": 90,
                 "valueFormatter": {"function": "params.value!=null?params.value.toFixed(1):'—'"}},
                {"field": "time_to_first_token_ms", "headerName": "TTFT ms",     "minWidth": 95,
                 "valueFormatter": {"function": "params.value!=null?params.value.toFixed(0):'—'"}},
                {"field": "error_message",        "headerName": "Error",         "minWidth": 160},
            ],
            rowData=[],
            defaultColDef={"sortable": True, "filter": True, "resizable": True},
            dashGridOptions={
                "pagination": True, "paginationPageSize": 20,
                "rowSelection": "single", "animateRows": True,
            },
            className="ag-theme-alpine-dark",
            style={"height": "400px", "width": "100%"},
        ),

        # ── Case detail ──────────────────────────────────────────────────────
        html.Div(
            [
                html.Div(
                    [
                        html.Span("◈  ", style={"color": ACCENT, "fontSize": "12px"}),
                        html.Span("Case Detail", style={
                            "fontSize": "10px", "fontWeight": "700",
                            "letterSpacing": "0.12em", "textTransform": "uppercase",
                            "color": TEXT_MUTED,
                        }),
                    ],
                    style={
                        "display": "flex", "alignItems": "center",
                        "marginBottom": "14px", "paddingBottom": "14px",
                        "borderBottom": f"1px solid {BORDER}",
                    },
                ),
                dcc.Markdown(
                    id="case-detail",
                    children="_Select a row to inspect question, answer, judge output, and retrieved chunks._",
                    style={"color": TEXT_DIM, "fontSize": "13px", "lineHeight": "1.75"},
                ),
            ],
            style={
                "marginTop": "14px",
                "background": f"linear-gradient(150deg, {SURFACE_HH} 0%, {SURFACE_HI} 100%)",
                "border": f"1px solid {BORDER}",
                "borderRadius": "12px",
                "padding": "18px 22px 24px",
            },
        ),

        dcc.Store(id="cases-store",   data=DEFAULT_CASES),
        dcc.Store(id="corpus-store",  data={}),
        dcc.Store(id="adapter-profile-store", data=None),
        dcc.Store(id="results-store", data=[]),
        dcc.Store(id="run-control", data={"request_id": 0}),
        dcc.Interval(id="run-poll", interval=1000, n_intervals=0),
    ],
    style={
        "background": BG,
        "minHeight": "100vh",
        "padding": "26px 30px 48px",
        "fontFamily": FONT,
        "color": TEXT,
        "maxWidth": "1800px",
        "margin": "0 auto",
    },
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("adapter-profile-store", "data"),
    Output("adapter-profile-label", "children"),
    Output("adapter-kind", "value"),
    Input("adapter-profile-upload", "contents"),
    State("adapter-profile-upload", "filename"),
    prevent_initial_call=True,
)
def on_adapter_profile_upload(contents: str | None, filename: str | None):
    if not contents or not filename:
        return no_update, no_update, no_update
    try:
        data = json.loads(parse_upload_contents(contents, filename).decode("utf-8"))
        profile = AdapterProfile.from_dict(data)
        return data, f"Profile loaded: {profile.name}", "custom"
    except Exception as exc:
        # Do not silently run an older profile after a rejected upload.
        return {"_error": str(exc)}, f"Profile rejected: {exc}", "custom"


def make_application_adapter(api_base, adapter_kind="brite", adapter_profile=None):
    if adapter_kind == "custom":
        if not adapter_profile or adapter_profile.get("_error"):
            raise ValueError("Upload a valid API profile before starting a custom-adapter run.")
        profile = AdapterProfile.from_dict(adapter_profile)
    else:
        profile = load_profile(adapter=adapter_kind)
    return HttpRagAdapter(api_base, profile, health_call=call_health,
                          evaluate_call=call_evaluate, upload_call=call_upload_file)

@app.callback(
    Output("cases-store", "data"),
    Output("cases-upload-label", "children"),
    Input("cases-upload", "contents"),
    State("cases-upload", "filename"),
    prevent_initial_call=True,
)
def on_cases_upload(contents: str | None, filename: str | None):
    if not contents or not filename:
        return no_update, no_update
    try:
        raw_bytes = parse_upload_contents(contents, filename)
        cases = parse_cases_text(raw_bytes.decode("utf-8", errors="replace"), filename)
        if not cases:
            raise ValueError("No valid cases found.")
        return cases, [_dot(SUCCESS, True), f"{len(cases)} cases from {filename}"]
    except Exception as exc:  # noqa: BLE001
        return no_update, [_dot(DANGER), f"Upload failed: {exc}"]


@app.callback(
    Output("corpus-store", "data"),
    Output("corpus-upload-label", "children"),
    Input("corpus-upload", "contents"),
    State("corpus-upload", "filename"),
    prevent_initial_call=True,
)
def on_corpus_upload(contents: str | None, filename: str | None):
    if not contents or not filename:
        return no_update, no_update
    try:
        raw_bytes = parse_upload_contents(contents, filename)
        if not raw_bytes:
            raise ValueError("Uploaded file is empty.")
        upload_dir = Path("tools/eval/tmp/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_name = _safe_upload_filename(filename)
        local_path = upload_dir / f"{uuid.uuid4().hex}_{safe_name}"
        local_path.write_bytes(raw_bytes)
        return {
            "filename": filename,
            "path": str(local_path),
            "size_bytes": len(raw_bytes),
        }, [_dot(SUCCESS, True), f"Raw file ready: {filename} ({len(raw_bytes):,} bytes)"]
    except Exception as exc:  # noqa: BLE001
        return no_update, [_dot(DANGER), f"Corpus failed: {exc}"]


@app.callback(
    Output("run-control", "data"),
    Input("run-eval", "n_clicks"),
    State("api-base", "value"),
    State("android-llm-model", "value"),
    State("top-k", "value"),
    State("max-questions", "value"),
    State("use-api-judge", "value"),
    State("retrieval-only", "value"),
    State("judge-base-url", "value"),
    State("judge-model", "value"),
    State("cases-store", "data"),
    State("corpus-store", "data"),
    State("adapter-kind", "value"),
    State("adapter-profile-store", "data"),
    State("external-model-id", "value"),
    State("judge-domain", "value"),
    State("judge-api-key", "value"),
    prevent_initial_call=True,
)
def start_evaluation(
        n_clicks: int,
        api_base: str,
        llm_model_id: str | None,
        top_k: int | None,
        max_questions: int | None,
        use_judge_flags: list[str] | None,
        retrieval_only_flags: list[str] | None,
        judge_base_url: str | None,
        judge_model: str | None,
        cases: list[dict[str, Any]] | None,
        corpus_docs: dict[str, Any] | None,
        adapter_kind: str = "brite",
        adapter_profile: dict[str, Any] | None = None,
        external_model_id: str | None = None,
        judge_domain: str | None = None,
        judge_api_key: str | None = None,
):
    if not n_clicks:
        return no_update

    active_snapshot = _snapshot_run_state()
    if active_snapshot["status"] == "running":
        _append_run_log(active_snapshot["job_id"], "Ignored a new start request because a run is already in progress.")
        return {"request_id": n_clicks, "job_id": active_snapshot["job_id"]}

    if not cases:
        _publish_run_error("No cases loaded.")
        return {"request_id": n_clicks, "job_id": None}
    if not api_base:
        _publish_run_error("API base URL is required.")
        return {"request_id": n_clicks, "job_id": None}
    try:
        make_application_adapter(api_base, adapter_kind, adapter_profile)
    except ValueError as exc:
        _publish_run_error(str(exc))
        return {"request_id": n_clicks, "job_id": None}
    llm_model_id = external_model_id or (llm_model_id if adapter_kind == "brite" else None)

    total_loaded_cases = len(cases)
    if max_questions is None:
        selected_cases = cases
    else:
        max_questions = int(max_questions)
        if max_questions < 1:
            _publish_run_error("Max Questions must be at least 1.")
            return {"request_id": n_clicks, "job_id": None}
        selected_cases = cases[:max_questions]

    job_id = uuid.uuid4().hex[:8]
    retrieval_only = "enabled" in (retrieval_only_flags or [])
    use_judge = "enabled" in (use_judge_flags or [])
    use_judge_effective = use_judge and not retrieval_only
    try:
        judge_config = make_judge_config(judge_base_url, judge_model, judge_api_key) if use_judge_effective else None
    except ValueError as exc:
        _publish_run_error(str(exc))
        return {"request_id": n_clicks, "job_id": None}
    _initialize_run_state(
        job_id=job_id,
        total_cases=len(selected_cases),
        live_status=f"Queued {len(selected_cases)} case(s) for evaluation.",
        summary=(
            f"Preparing run against {api_base}  ·  "
            f"selected_cases={len(selected_cases)}/{total_loaded_cases}"
        ),
    )
    _append_run_log(
        job_id,
        (
            f"[start] api_base={api_base} top_k={top_k or 'default'} "
            f"llm_model_id={llm_model_id} "
            f"selected_cases={len(selected_cases)}/{total_loaded_cases} corpus_file={bool(corpus_docs)} "
            f"retrieval_only={'enabled' if retrieval_only else 'disabled'} "
            f"api_judge={'enabled' if use_judge_effective else 'disabled'}"
        ),
    )

    worker = threading.Thread(
        target=_run_evaluation_job,
        args=(
            job_id,
            api_base,
            llm_model_id,
            top_k,
            use_judge_effective,
            retrieval_only,
            judge_config,
            selected_cases,
            corpus_docs or {},
            total_loaded_cases,
            adapter_kind,
            adapter_profile,
            judge_domain,
        ),
        daemon=True,
    )
    worker.start()
    return {"request_id": n_clicks, "job_id": job_id}


@app.callback(
    Output("results-store", "data"),
    Output("run-live-status", "children"),
    Output("run-status", "children"),
    Output("run-log", "children"),
    Output("run-eval", "disabled"),
    Output("run-eval", "children"),
    Input("run-control", "data"),
    Input("run-poll", "n_intervals"),
)
def refresh_run_state(_: dict[str, Any] | None, __: int):
    snapshot = _snapshot_run_state()
    status = snapshot["status"]
    if status == "running":
        live_status = [
            _dot(WARN, True),
            f"Running {snapshot['completed_cases']}/{snapshot['total_cases']} case(s)...",
        ]
        button_disabled = True
        button_children = _run_button_children("⋯", f"Running {snapshot['completed_cases']}/{snapshot['total_cases']}")
    elif status == "completed":
        live_status = [_dot(SUCCESS, True), f"Completed {snapshot['completed_cases']}/{snapshot['total_cases']} case(s)."]
        button_disabled = False
        button_children = _run_button_children("▶", "Run Evaluation")
    elif status == "failed":
        live_status = [_dot(DANGER, True), snapshot["live_status"] or "Run failed."]
        button_disabled = False
        button_children = _run_button_children("▶", "Run Evaluation")
    else:
        live_status = [_dot(TEXT_MUTED), "Idle"]
        button_disabled = False
        button_children = _run_button_children("▶", "Run Evaluation")

    log_text = "\n".join(snapshot["logs"]) if snapshot["logs"] else "No evaluation run yet."
    summary = snapshot["summary"] or "Ready."
    return snapshot["rows"], live_status, summary, log_text, button_disabled, button_children


@app.callback(
    Output("kpi-total",       "children"),
    Output("kpi-pass-rate",   "children"),
    Output("kpi-avg-score",   "children"),
    Output("kpi-avg-latency", "children"),
    Output("kpi-avg-tps",     "children"),
    Output("kpi-avg-ttft",    "children"),
    Output("kpi-avg-output",  "children"),
    Output("score-chart",     "figure"),
    Output("latency-chart",   "figure"),
    Output("results-grid",    "rowData"),
    Input("results-store",    "data"),
    Input("failure-only",     "value"),
)
def render_metrics(rows: list[dict[str, Any]] | None, failure_only_flags: list[str]):
    def _empty() -> go.Figure:
        fig = go.Figure()
        fig.update_layout(**PLOT_LAYOUT)
        fig.add_annotation(
            text="No data — run evaluation to populate",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color=TEXT_MUTED, size=12, family=FONT),
        )
        return fig

    if not rows:
        return "—", "—", "—", "—", "—", "—", "—", _empty(), _empty(), []

    frame = pd.DataFrame(rows)
    frame["pass"] = frame.get("pass", pd.Series([None] * len(frame))).map(_normalize_pass_value)
    for column in [
        "semantic_score",
        "answer_score",
        "retrieval_ndcg_at_5",
        "total_latency_ms",
        "tokens_per_second",
        "time_to_first_token_ms",
        "tokens_predicted",
    ]:
        if column not in frame:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    total       = len(frame)
    scored_pass = frame["pass"].dropna()
    pass_rate   = scored_pass.mean() if not scored_pass.empty else pd.NA
    avg_score   = frame["semantic_score"].mean()
    avg_latency = frame["total_latency_ms"].fillna(0).mean()
    avg_tps     = frame["tokens_per_second"].mean()
    avg_ttft    = frame["time_to_first_token_ms"].mean()
    avg_output  = frame["tokens_predicted"].mean()

    score_frame = frame.dropna(subset=["semantic_score"]).copy()
    if score_frame.empty:
        score_fig = _empty()
    else:
        score_frame["pass_label"] = score_frame["pass"].map({True: "pass", False: "fail"}).fillna("unscored")
        score_fig = px.histogram(
            score_frame, x="semantic_score", color="pass_label", nbins=20, barmode="overlay",
            color_discrete_map={"pass": SUCCESS, "fail": DANGER, "unscored": TEXT_MUTED},
        )
        score_fig.update_traces(opacity=0.82, marker_line_width=0)
        score_fig.update_layout(**PLOT_LAYOUT, showlegend=True, legend_title_text="",
                                xaxis_title="Semantic Score", yaxis_title="Count")

    lat_frame = frame.copy()
    lat_frame["category"] = lat_frame["category"].fillna("uncategorized")
    latency_fig = px.box(
        lat_frame, x="category", y="total_latency_ms", points="all",
        color_discrete_sequence=[ACCENT],
    )
    latency_fig.update_traces(
        marker=dict(color=ACCENT_LO, size=4, opacity=0.7),
        line_color=ACCENT, fillcolor=_hex_to_rgba(ACCENT, 0.13),
    )
    latency_fig.update_layout(**PLOT_LAYOUT, xaxis_title="Category", yaxis_title="Latency (ms)")

    filtered = frame if "failures" not in (failure_only_flags or []) else frame[frame["pass"] != True]  # noqa: E712
    return (
        f"{total}",
        _format_percent_metric(pass_rate),
        _format_numeric_metric(avg_score, 3, ""),
        f"{avg_latency:.0f} ms",
        _format_numeric_metric(avg_tps, 1, " t/s"),
        _format_numeric_metric(avg_ttft, 0, " ms"),
        _format_numeric_metric(avg_output, 0, ""),
        score_fig, latency_fig, _records_for_dash(filtered),
    )


@app.callback(
    Output("case-detail", "children"),
    Input("results-grid", "selectedRows"),
)
def show_case_detail(selected_rows: list[dict[str, Any]] | None):
    if not selected_rows:
        return "_Select a case row to inspect details._"
    row = selected_rows[0]
    verdict_icon = "✅" if row.get("pass") is True else "❌" if row.get("pass") is False else "•"
    return "\n".join([
        f"### {verdict_icon} `{row.get('case_id')}`",
        f"**Category:** `{row.get('category') or 'n/a'}`  ·  "
        f"**Semantic Score:** `{row.get('semantic_score')}`  ·  "
        f"**Judge:** `{row.get('semantic_judge_status') or 'n/a'}`  ·  "
        f"**Runtime Retrieval:** `{row.get('retrieval_confidence') or 'n/a'}`",
        f"> {row.get('answer_explanation') or row.get('semantic_judge_error') or ''}",
        "", "#### ❓ Question", row.get("question", ""),
        "", "#### 📖 Ground Truth", row.get("ground_truth_answer", ""),
        "", "#### 🤖 Generated Answer", row.get("generated_answer", ""),
        "", "#### Retrieval Verdict", f"`{row.get('retrieval_verdict') or 'n/a'}`",
        "", "#### Retrieval Metrics", _semantic_retrieval_metrics_markdown(row),
        "", "#### Needed Facts", _list_to_markdown(row.get("needed_facts", [])),
        "", "#### Found Facts", _list_to_markdown(row.get("found_facts", [])),
        "", "#### Retrieval Missing Facts", _list_to_markdown(row.get("retrieval_missing_facts", [])),
        "", "#### Distractor Chunks", _distractor_chunks_markdown(row.get("distractor_chunks", [])),
        "", "#### Answer Verdict",
        f"`{row.get('answer_verdict') or 'n/a'}`  ·  **Answer Score:** `{row.get('answer_score')}`",
        "", "#### Answer Missing Facts", _list_to_markdown(row.get("answer_missing_facts", row.get("missing_facts", []))),
        "", "#### Unsupported Claims", _list_to_markdown(row.get("unsupported_claims", [])),
        "", "#### Dangerous Claims", _list_to_markdown(row.get("dangerous_claims", [])),
        "", "#### 📈 Runtime Metrics", _runtime_metrics_markdown(row),
        "", "#### 🧭 Retrieval Diagnostics", _retrieval_diagnostics_markdown(row),
        "", "#### 📚 Retrieved Chunks",
        _retrieved_chunks_markdown(row),
    ])


# ── Internal helpers ───────────────────────────────────────────────────────────

def _judge_case(
    case: dict[str, Any],
    response: dict[str, Any],
    use_judge: bool,
    judge_config: JudgeConfig | None,
    domain: str = "medical first-aid",
) -> dict[str, Any]:
    error = (response.get("error") or {}) if isinstance(response, dict) else {}
    if error:
        return api_error_judge_result(error.get("message", "RAG API error."))
    if not use_judge:
        return disabled_judge_result()
    return evaluate_with_api(
        question=case.get("question", ""),
        ground_truth=case.get("ground_truth_answer", ""),
        generated_answer=response.get("generated_answer", "") or "",
        retrieved_chunks=response.get("retrieved_chunks") or [],
        config=judge_config or make_judge_config(),
        domain=domain,
    )


def _build_row(case, request_payload, response, judge) -> dict[str, Any]:
    error   = response.get("error") or {}
    runtime = response.get("runtime") or {}
    decoding = response.get("decoding_metrics") or {}
    retrieval_diag = runtime.get("retrieval_diagnostics") or {}
    retrieval_judge = judge.get("retrieval") or {}
    retrieval_metrics = retrieval_judge.get("metrics") or {}
    answer_judge = judge.get("answer") or {}
    distractor_chunks = retrieval_judge.get("distractor_chunks") or []
    judge_status = judge.get("judge_status")
    final_chunk_count = retrieval_diag.get("deduped_final_count")
    return {
        "case_id": case.get("case_id"), "category": case.get("category"),
        "question": case.get("question"), "ground_truth_answer": case.get("ground_truth_answer"),
        "generated_answer": response.get("generated_answer") or "",
        "retrieved_chunks": response.get("retrieved_chunks") or [],
        "retrieval_timing_ms": response.get("retrieval_timing_ms"),
        "generation_timing_ms": response.get("generation_timing_ms"),
        "total_latency_ms": response.get("total_latency_ms"),
        "retrieval_confidence": runtime.get("retrieval_confidence"),
        "query_top_k": retrieval_diag.get("query_top_k"),
        "vector_candidate_count": retrieval_diag.get("vector_candidate_count"),
        "bm25_candidate_count": retrieval_diag.get("bm25_candidate_count"),
        "fused_candidate_count": retrieval_diag.get("fused_candidate_count"),
        "deduped_final_count": final_chunk_count,
        "expanded_result_count": retrieval_diag.get("expanded_result_count"),
        "top_fused_score": retrieval_diag.get("top_fused_score"),
        "top_final_score": retrieval_diag.get("top_final_score"),
        "reranker_candidate_count": retrieval_diag.get("reranker_candidate_count"),
        "reranker_applied": retrieval_diag.get("reranker_applied"),
        "tokens_per_second": decoding.get("tokensPerSecond"),
        "time_to_first_token_ms": decoding.get("timeToFirstTokenMs"),
        "tokens_evaluated": decoding.get("tokensEvaluated"),
        "tokens_predicted": decoding.get("tokensPredicted"),
        "total_tokens": _sum_metrics(decoding.get("tokensEvaluated"), decoding.get("tokensPredicted")),
        "model_size_mb": decoding.get("modelSizeMB"),
        "context_size_mb": decoding.get("contextSizeMB"),
        "memory_usage_percent": decoding.get("memoryUsagePercent"),
        "context_tokens_used": decoding.get("contextTokensUsed"),
        "context_tokens_max": decoding.get("contextTokensMax"),
        "context_usage_percent": decoding.get("contextUsagePercent"),
        "semantic_judge_status": judge_status,
        "semantic_judge_error": judge.get("error_message"),
        "pass": judge.get("pass"),
        "semantic_score": judge.get("semantic_score"),
        "needed_facts": judge.get("needed_facts") or [],
        "retrieval_verdict": retrieval_judge.get("verdict"),
        "chunk_relevance": retrieval_judge.get("chunk_relevance") or [],
        "found_facts": retrieval_judge.get("found_facts") or [],
        "retrieval_missing_facts": retrieval_judge.get("missing_facts") or [],
        "distractor_chunks": distractor_chunks,
        "retrieval_distractor_count": retrieval_metrics.get("retrieval_distractor_count") if judge_status == "ok" else None,
        "retrieval_hit_at_5": retrieval_metrics.get("retrieval_hit_at_5"),
        "retrieval_precision_at_5": retrieval_metrics.get("retrieval_precision_at_5"),
        "retrieval_soft_precision_at_5": retrieval_metrics.get("retrieval_soft_precision_at_5"),
        "retrieval_mrr_at_5": retrieval_metrics.get("retrieval_mrr_at_5"),
        "retrieval_ndcg_at_5": retrieval_metrics.get("retrieval_ndcg_at_5"),
        "answer_score": answer_judge.get("score"),
        "answer_verdict": answer_judge.get("verdict"),
        "answer_missing_facts": answer_judge.get("missing_facts") or [],
        "missing_facts": answer_judge.get("missing_facts") or [],
        "unsupported_claims": answer_judge.get("unsupported_claims") or [],
        "dangerous_claims": answer_judge.get("dangerous_claims") or [],
        "answer_explanation": answer_judge.get("explanation") or "",
        "error_code": error.get("code"),
        "android_error_message": error.get("message"),
        "error_message": error.get("message") or judge.get("error_message"),
        "raw_request": json.dumps(request_payload, ensure_ascii=False),
        "raw_response": json.dumps(response, ensure_ascii=False),
        "raw_judge": json.dumps(judge, ensure_ascii=False),
    }


def _list_to_markdown(items: list[str]) -> str:
    return "_None_" if not items else "\n".join(f"- {i}" for i in items)


def _semantic_retrieval_metrics_markdown(row: dict[str, Any]) -> str:
    metric_lines = []
    _append_metric_line(metric_lines, "Hit@5", row.get("retrieval_hit_at_5"), decimals=0)
    _append_metric_line(metric_lines, "Precision@5", row.get("retrieval_precision_at_5"), decimals=3)
    _append_metric_line(metric_lines, "Soft Precision@5", row.get("retrieval_soft_precision_at_5"), decimals=3)
    _append_metric_line(metric_lines, "MRR@5", row.get("retrieval_mrr_at_5"), decimals=3)
    _append_metric_line(metric_lines, "nDCG@5", row.get("retrieval_ndcg_at_5"), decimals=3)
    _append_metric_line(metric_lines, "Distractor Count", row.get("retrieval_distractor_count"), decimals=0)
    return "_No semantic retrieval metrics available._" if not metric_lines else "\n".join(f"- {line}" for line in metric_lines)


def _distractor_chunks_markdown(items: list[dict[str, Any]]) -> str:
    if not items:
        return "_None_"
    lines = []
    for item in items:
        if not isinstance(item, dict):
            continue
        lines.append(f"- **Rank {item.get('rank')}:** {item.get('reason') or ''}")
    return "\n".join(lines) if lines else "_None_"


def _retrieved_chunks_markdown(row: dict[str, Any]) -> str:
    chunks = row.get("retrieved_chunks", []) or []
    if not chunks:
        return "_No chunks returned._"

    relevance_by_rank: dict[int, dict[str, Any]] = {}
    for item in row.get("chunk_relevance", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            relevance_by_rank[int(item.get("rank"))] = item
        except (TypeError, ValueError):
            continue
    lines = []
    for rank, chunk in enumerate(chunks, 1):
        if not isinstance(chunk, dict):
            continue
        relevance = relevance_by_rank.get(rank, {})
        source = chunk.get("sourceName") or chunk.get("source_name") or chunk.get("source") or "unknown"
        raw_snippet = (
            chunk.get("contentSnippet")
            or chunk.get("content_snippet")
            or chunk.get("snippet")
            or chunk.get("content")
            or ""
        )
        snippet = raw_snippet.strip() if isinstance(raw_snippet, str) else str(raw_snippet)
        lines.append(
            f"{rank}. **{source}** "
            f"(relevance={relevance.get('relevance', 'n/a')}, score={chunk.get('score')})\n\n"
            f"**Reason:** {relevance.get('reason') or 'n/a'}\n\n"
            f"{snippet}"
        )
    return "\n\n".join(lines)


def _snapshot_run_state() -> dict[str, Any]:
    with RUN_LOCK:
        return {
            "job_id": RUN_STATE["job_id"],
            "status": RUN_STATE["status"],
            "rows": list(RUN_STATE["rows"]),
            "logs": list(RUN_STATE["logs"]),
            "summary": RUN_STATE["summary"],
            "live_status": RUN_STATE["live_status"],
            "completed_cases": RUN_STATE["completed_cases"],
            "total_cases": RUN_STATE["total_cases"],
            "output_path": RUN_STATE["output_path"],
        }


def _update_run_state(job_id: str | None = None, **updates: Any) -> bool:
    with RUN_LOCK:
        if job_id is not None and RUN_STATE["job_id"] != job_id:
            return False
        RUN_STATE.update(updates)
        return True


def _publish_run_error(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    with RUN_LOCK:
        RUN_STATE.update({
            "job_id": None,
            "status": "failed",
            "rows": [],
            "logs": [f"[{timestamp}] [error] {message}"],
            "summary": message,
            "live_status": message,
            "completed_cases": 0,
            "total_cases": 0,
            "output_path": None,
        })


def _initialize_run_state(job_id: str, total_cases: int, live_status: str, summary: str) -> None:
    with RUN_LOCK:
        RUN_STATE.update({
            "job_id": job_id,
            "status": "running",
            "rows": [],
            "logs": [],
            "summary": summary,
            "live_status": live_status,
            "completed_cases": 0,
            "total_cases": total_cases,
            "output_path": None,
        })


def _append_run_log(job_id: str | None, message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    with RUN_LOCK:
        if job_id is not None and RUN_STATE["job_id"] != job_id:
            return
        logs = list(RUN_STATE["logs"])
        logs.append(f"[{timestamp}] {message}")
        RUN_STATE["logs"] = logs[-RUN_LOG_LIMIT:]


def _default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("tools/eval/output") / f"eval_run_{timestamp}.jsonl"


def _safe_upload_filename(filename: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in Path(filename).name)
    return safe[:120] or "uploaded_file"


def _run_evaluation_job(
        job_id: str,
        api_base: str,
        llm_model_id: str | None,
        top_k: int | None,
        use_judge: bool,
        retrieval_only: bool,
        judge_config: JudgeConfig | None,
        cases: list[dict[str, Any]],
        corpus_docs: dict[str, Any] | None,
        total_loaded_cases: int,
        adapter_kind: str = "brite",
        adapter_profile: dict[str, Any] | None = None,
        judge_domain: str | None = None,
) -> None:
    try:
        adapter = make_application_adapter(api_base, adapter_kind, adapter_profile)
        health_path = adapter.profile.health_path
        _append_run_log(job_id, f"[adapter] {adapter.profile.name}")
        _append_run_log(job_id, f"[health] probing {api_base.rstrip('/')}{health_path}" if health_path else "[health] skipped by adapter")
        health = adapter.check_health()
        runtime_ready = bool(health.get("runtime_ready"))
        _append_run_log(
            job_id,
            (
                f"[health] status={health.get('status')} runtime_ready={runtime_ready} "
                f"litert_lm_model_loaded={health.get('litert_lm_model_loaded')} "
                f"embedding_initialized={health.get('embedding_initialized')}"
            ),
        )

        if corpus_docs:
            _append_run_log(job_id, "[upload] streaming raw corpus file to application")

            local_path = corpus_docs.get("path")
            filename = corpus_docs.get("filename") or "uploaded_file"
            if not local_path:
                raise RuntimeError("Corpus upload is missing a local file path.")

            temp_path = Path(local_path)
            if not temp_path.exists():
                raise RuntimeError(f"Corpus upload file is missing: {temp_path}")

            upload_response = adapter.upload_corpus(temp_path)
            upload_error = upload_response.get("error") if isinstance(upload_response, dict) else None
            if upload_error:
                raise RuntimeError(f"Corpus upload failed: {upload_error}")

            _append_run_log(
                job_id,
                (
                    f"[upload] rag_id={upload_response.get('rag_id')} "
                    f"name={upload_response.get('name')} "
                    f"node_count={upload_response.get('node_count')}"
                ),
            )

        result_rows: list[dict[str, Any]] = []
        output_rows: list[dict[str, Any]] = []
        output_path = _default_output_path()
        json_report_path = output_path.with_suffix(".json")
        report_metadata = {
            "adapter": adapter.profile.name,
            "judge_domain": judge_domain or adapter.profile.judge_domain,
            "api_base": api_base,
            "llm_model_id": llm_model_id,
            "top_k_override": top_k,
            "selected_cases": len(cases),
            "total_loaded_cases": total_loaded_cases,
            "corpus_uploaded": bool(corpus_docs),
            "judge_enabled": use_judge and not retrieval_only,
            "retrieval_only": retrieval_only,
            **(judge_config.report_metadata() if judge_config else {}),
        }
        _append_run_log(job_id, f"[output] json={json_report_path} jsonl={output_path}")

        def judge_case(case, response):
            return _judge_case(case, response, use_judge and not retrieval_only,
                               judge_config,
                               domain=judge_domain or adapter.profile.judge_domain)

        runs = iter_evaluations(adapter, cases, judge_case, top_k=top_k,
                               model_id=llm_model_id, retrieval_only=retrieval_only,
                               timeout=EVALUATE_TIMEOUT_SECONDS,
                               on_case_start=lambda index, case: _append_run_log(
                                   job_id, f"[case {index}/{len(cases)}] starting case={case['case_id']}"))
        for index, output_row in enumerate(runs, start=1):
            case = output_row["case"]
            payload, api_response, judge = output_row["request"], output_row["response"], output_row["judge"]
            error = api_response.get("error") or {}
            if error.get("code") == "request_timeout":
                _append_run_log(job_id, f"[case {index}/{len(cases)}] timeout case={case['case_id']}; recording error row and continuing")
            row = _build_row(case, payload, api_response, judge)
            result_rows.append(row)
            output_rows.append(output_row)
            write_jsonl(output_path, output_rows)
            write_json_report(json_report_path, output_rows, metadata=report_metadata)

            pass_count = sum(1 for item in result_rows if item.get("pass"))
            avg_score = _mean_numeric(item.get("semantic_score") for item in result_rows)
            _append_run_log(job_id, _case_log_line(index, len(cases), row))
            if not _update_run_state(
                    job_id,
                    rows=list(result_rows),
                    completed_cases=index,
                    live_status=f"Completed {index}/{len(cases)} case(s).",
                    summary=(
                            f"Running against {api_base}  ·  selected_cases={len(cases)}/{total_loaded_cases}  ·  "
                            f"pass_rate={_format_pass_rate(pass_count, result_rows)}  ·  avg_semantic={_format_average(avg_score)}"
                    ),
            ):
                return

        write_jsonl(output_path, output_rows)
        write_json_report(json_report_path, output_rows, metadata=report_metadata)
        avg_score = _mean_numeric(item.get("semantic_score") for item in result_rows)
        pass_count = sum(1 for item in result_rows if item.get("pass"))
        _append_run_log(
            job_id,
            (
                f"[done] saved_json={json_report_path} saved_jsonl={output_path} "
                f"pass_rate={_format_pass_rate(pass_count, result_rows)} "
                f"avg_semantic={_format_average(avg_score)}"
            ),
        )
        _update_run_state(
            job_id,
            status="completed",
            rows=list(result_rows),
            completed_cases=len(result_rows),
            live_status=f"Completed {len(result_rows)}/{len(cases)} case(s).",
            summary=(
                f"Saved {len(result_rows)} case(s) to {json_report_path}  ·  "
                f"jsonl={output_path}  ·  "
                f"pass_rate={_format_pass_rate(pass_count, result_rows)}  ·  avg_semantic={_format_average(avg_score)}"
            ),
            output_path=str(json_report_path),
        )
    except Exception as exc:  # noqa: BLE001
        _append_run_log(job_id, f"[error] {exc}")
        _append_run_log(job_id, traceback.format_exc(limit=5).strip())
        _update_run_state(
            job_id,
            status="failed",
            live_status=str(exc),
            summary=f"Run failed: {exc}",
        )


def _case_log_line(index: int, total: int, row: dict[str, Any]) -> str:
    error_code = row.get("error_code") or "none"
    return (
        f"[case {index}/{total}] case={row.get('case_id')} pass={row.get('pass')} "
        f"semantic={_log_metric_value(row.get('semantic_score'), 3)} "
        f"retrieval_verdict={row.get('retrieval_verdict') or 'n/a'} "
        f"answer_verdict={row.get('answer_verdict') or 'n/a'} "
        f"ndcg5={_log_metric_value(row.get('retrieval_ndcg_at_5'), 3)} "
        f"answer_score={_log_metric_value(row.get('answer_score'), 3)} "
        f"runtime_retrieval={row.get('retrieval_confidence') or 'n/a'} "
        f"latency_ms={row.get('total_latency_ms') or 'n/a'} "
        f"vector={row.get('vector_candidate_count') or 0} "
        f"bm25={row.get('bm25_candidate_count') or 0} "
        f"rrf={row.get('fused_candidate_count') or 0} "
        f"expanded={row.get('expanded_result_count') or 0} "
        f"final={row.get('deduped_final_count') or 0} "
        f"reranker={row.get('reranker_applied') if row.get('reranker_applied') is not None else 'n/a'} "
        f"top_rrf={_log_metric_value(row.get('top_fused_score'), 3)} "
        f"top_final={_log_metric_value(row.get('top_final_score'), 3)} "
        f"tps={_log_metric_value(row.get('tokens_per_second'), 1)} "
        f"ttft_ms={_log_metric_value(row.get('time_to_first_token_ms'), 0)} "
        f"judge={row.get('semantic_judge_status') or 'n/a'} "
        f"error={error_code}"
    )


def _log_metric_value(value: Any, decimals: int) -> str:
    if _is_missing(value):
        return "n/a"
    return f"{float(value):.{decimals}f}"


def _format_numeric_metric(value: Any, decimals: int, suffix: str) -> str:
    if _is_missing(value):
        return "—"
    return f"{float(value):.{decimals}f}{suffix}"


def _format_percent_metric(value: Any) -> str:
    if _is_missing(value):
        return "—"
    return f"{float(value):.1%}"


def _normalize_pass_value(value: Any) -> bool | None:
    if value is True:
        return True
    if value is False:
        return False
    return None


def _mean_numeric(values: Any) -> float | None:
    numbers: list[float] = []
    for value in values:
        if _is_missing(value):
            continue
        try:
            numbers.append(float(value))
        except (TypeError, ValueError):
            continue
    return None if not numbers else sum(numbers) / len(numbers)


def _format_average(value: Any) -> str:
    if _is_missing(value):
        return "n/a"
    return f"{float(value):.3f}"


def _format_pass_rate(pass_count: int, rows: list[dict[str, Any]]) -> str:
    scored_count = sum(1 for row in rows if row.get("pass") is not None)
    if scored_count == 0:
        return "n/a"
    return f"{pass_count / scored_count:.1%}"


def _records_for_dash(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        records.append({key: None if _is_missing(value) else value for key, value in record.items()})
    return records


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, (bool, str, list, dict)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _runtime_metrics_markdown(row: dict[str, Any]) -> str:
    metric_lines = []
    _append_metric_line(metric_lines, "Total Tokens", row.get("total_tokens"), decimals=0)
    _append_metric_line(metric_lines, "Prompt Tokens", row.get("tokens_evaluated"), decimals=0)
    _append_metric_line(metric_lines, "Generated Tokens", row.get("tokens_predicted"), decimals=0)
    _append_metric_line(metric_lines, "Speed", row.get("tokens_per_second"), " t/s", 1)
    _append_metric_line(metric_lines, "Time to First Token", row.get("time_to_first_token_ms"), " ms", 0)
    _append_metric_line(metric_lines, "Model Size", row.get("model_size_mb"), " MB", 0)
    _append_metric_line(metric_lines, "Context Size", row.get("context_size_mb"), " MB", 0)
    _append_metric_line(metric_lines, "Memory Usage", row.get("memory_usage_percent"), "%", 1)
    _append_metric_line(metric_lines, "Context Tokens Used", row.get("context_tokens_used"), decimals=0)
    _append_metric_line(metric_lines, "Context Tokens Max", row.get("context_tokens_max"), decimals=0)
    _append_metric_line(metric_lines, "Context Usage", row.get("context_usage_percent"), "%", 1)
    return "_No generation metrics returned._" if not metric_lines else "\n".join(f"- {line}" for line in metric_lines)


def _retrieval_diagnostics_markdown(row: dict[str, Any]) -> str:
    metric_lines = []
    _append_metric_line(metric_lines, "Query Top K", row.get("query_top_k"), decimals=0)
    _append_metric_line(metric_lines, "Vector Candidates", row.get("vector_candidate_count"), decimals=0)
    _append_metric_line(metric_lines, "BM25 Candidates", row.get("bm25_candidate_count"), decimals=0)
    _append_metric_line(metric_lines, "RRF Candidates", row.get("fused_candidate_count"), decimals=0)
    _append_metric_line(metric_lines, "Expanded Results", row.get("expanded_result_count"), decimals=0)
    _append_metric_line(metric_lines, "Final Chunks", row.get("deduped_final_count"), decimals=0)
    _append_metric_line(metric_lines, "Reranker Candidates", row.get("reranker_candidate_count"), decimals=0)
    if row.get("reranker_applied") is not None:
        metric_lines.append(f"Reranker Applied: `{bool(row.get('reranker_applied'))}`")
    _append_metric_line(metric_lines, "Top RRF Score", row.get("top_fused_score"), decimals=3)
    _append_metric_line(metric_lines, "Top Final Score", row.get("top_final_score"), decimals=3)
    return "_No retrieval diagnostics returned._" if not metric_lines else "\n".join(f"- {line}" for line in metric_lines)


def _append_metric_line(lines: list[str], label: str, value: Any, suffix: str = "", decimals: int = 0) -> None:
    if _is_missing(value):
        return
    numeric_value = float(value)
    if numeric_value < 0:
        return
    lines.append(f"**{label}:** {numeric_value:.{decimals}f}{suffix}")


def _sum_metrics(left: Any, right: Any) -> int | None:
    if left is None and right is None:
        return None
    return int(left or 0) + int(right or 0)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=False)
