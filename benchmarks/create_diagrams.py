"""Generate benchmark charts as SVG in the turboquant-wasm aesthetic.

Reads JSON files from ./results/ and writes:
  ../docs/arm_speed_st.svg, ../docs/arm_speed_mt.svg
  ../docs/x86_speed_st.svg, ../docs/x86_speed_mt.svg
  ../docs/recall_d1536.svg, ../docs/recall_d3072.svg, ../docs/recall_glove.svg
  ../docs/compression.svg
  ../docs/stack.svg, query_paths.svg, recall_delta.svg,
  speed_grid.svg, selectivity.svg, migration.svg
"""

import json
import math
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")

FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
C = {
    "title": "#0f172a",
    "subtitle": "#475569",
    "label": "#0f172a",
    "secondary": "#475569",
    "tick": "#64748b",
    "axis": "#334155",
    "grid": "#e5e7eb",
    "baseline": "#94a3b8",
    "tq": "#635bff",
    "tq_stroke": "#4338ca",
    "tq_text": "#4338ca",
    "faiss": "#9aa7b6",
    "fp32": "#9aa7b6",
    "four_bit": "#1d4ed8",
    "two_bit": "#635bff",
    "tq_2": "#635bff",
    "tq_4": "#0f766e",
    "faiss_2": "#9aa7b6",
    "faiss_4": "#64748b",
    "turbo": "#635bff",
    "graph": "#15803d",
}


def xe(s):
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def nice_ceil(value):
    if value <= 1:
        return 1
    exponent = math.floor(math.log10(value))
    fraction = value / 10 ** exponent
    if fraction <= 1:
        nf = 1
    elif fraction <= 1.5:
        nf = 1.5
    elif fraction <= 2:
        nf = 2
    elif fraction <= 5:
        nf = 5
    else:
        nf = 10
    return nf * 10 ** exponent


def style_block():
    return (
        f'<style>\n'
        f'  .title {{ font: 700 20px {FONT}; fill: {C["title"]}; }}\n'
        f'  .subtitle {{ font: 400 12px {FONT}; fill: {C["subtitle"]}; }}\n'
        f'  .panel {{ font: 700 14px {FONT}; fill: {C["title"]}; }}\n'
        f'  .label {{ font: 600 12px {FONT}; fill: {C["label"]}; }}\n'
        f'  .secondary {{ font: 400 11px {FONT}; fill: {C["secondary"]}; }}\n'
        f'  .tick {{ font: 400 11px {FONT}; fill: {C["tick"]}; }}\n'
        f'  .value {{ font: 700 11px {FONT}; fill: {C["label"]}; }}\n'
        f'  .value-accent {{ font: 700 11px {FONT}; fill: {C["tq_text"]}; }}\n'
        f'  .axis {{ font: 600 12px {FONT}; fill: {C["axis"]}; }}\n'
        f'  .legend {{ font: 600 12px {FONT}; fill: {C["label"]}; }}\n'
        f'</style>'
    )


def grid_lines(px, py, pw, ph, y_lo, y_hi, fmt, step_count=5):
    parts = []
    for i in range(step_count + 1):
        v = y_lo + (y_hi - y_lo) * i / step_count
        y = py + ph - (v - y_lo) / (y_hi - y_lo) * ph
        parts.append(
            f'<line x1="{px}" y1="{y:.1f}" x2="{px + pw}" y2="{y:.1f}" stroke="{C["grid"]}" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{px - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{xe(fmt(v))}</text>'
        )
    parts.append(
        f'<line x1="{px}" y1="{py + ph:.1f}" x2="{px + pw}" y2="{py + ph:.1f}" stroke="{C["baseline"]}" stroke-width="1.5" />'
    )
    return "\n".join(parts)


def paired_panel(px, py, pw, ph, panel_title, groups, tick_fmt, value_fmt, y_max):
    parts = [grid_lines(px, py, pw, ph, 0, y_max, tick_fmt)]
    parts.append(f'<text x="{px}" y="{py - 14}" class="panel">{xe(panel_title)}</text>')
    n = len(groups)
    band = pw / n
    bar_w = min(44, band * 0.32)
    gap = 6
    for i, g in enumerate(groups):
        cx = px + band * i + band / 2
        tq_x = cx - bar_w - gap / 2
        faiss_x = cx + gap / 2
        tq_h = (g["tq"] / y_max) * ph
        faiss_h = (g["faiss"] / y_max) * ph
        tq_y = py + ph - tq_h
        faiss_y = py + ph - faiss_h
        label_y = py + ph + 22
        parts.append(
            f'<rect x="{tq_x:.1f}" y="{tq_y:.1f}" width="{bar_w}" height="{tq_h:.1f}" rx="6" '
            f'fill="{C["tq"]}" stroke="{C["tq_stroke"]}" stroke-width="1.5" />'
        )
        parts.append(
            f'<rect x="{faiss_x:.1f}" y="{faiss_y:.1f}" width="{bar_w}" height="{faiss_h:.1f}" rx="6" fill="{C["faiss"]}" />'
        )
        parts.append(
            f'<text x="{tq_x + bar_w/2:.1f}" y="{tq_y - 6:.1f}" text-anchor="middle" class="value-accent">{xe(value_fmt(g["tq"]))}</text>'
        )
        parts.append(
            f'<text x="{faiss_x + bar_w/2:.1f}" y="{faiss_y - 6:.1f}" text-anchor="middle" class="value">{xe(value_fmt(g["faiss"]))}</text>'
        )
        primary, _, secondary = g["label"].partition("|")
        parts.append(f'<text x="{cx:.1f}" y="{label_y}" text-anchor="middle" class="label">{xe(primary)}</text>')
        if secondary:
            parts.append(f'<text x="{cx:.1f}" y="{label_y + 15}" text-anchor="middle" class="secondary">{xe(secondary)}</text>')
    return "\n".join(parts)


def legend_tq_faiss(x, y):
    parts = [
        f'<rect x="{x}" y="{y - 10}" width="14" height="14" rx="3" fill="{C["tq"]}" stroke="{C["tq_stroke"]}" stroke-width="1.5" />',
        f'<text x="{x + 22}" y="{y + 1}" class="legend" style="fill: {C["tq_text"]};">TurboQuant</text>',
        f'<rect x="{x + 140}" y="{y - 10}" width="14" height="14" rx="3" fill="{C["faiss"]}" />',
        f'<text x="{x + 162}" y="{y + 1}" class="legend">FAISS</text>',
    ]
    return "\n".join(parts)


def load_json(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def speed_panels(arch):
    panels = {"st": [], "mt": []}
    for dim in (1536, 3072):
        for bw in (2, 4):
            for th in ("st", "mt"):
                entry = load_json(f"speed_d{dim}_{bw}bit_{arch}_{th}.json")
                panels[th].append(
                    {
                        "label": f"d={dim}|{bw}-bit",
                        "tq": entry["tq_ms_per_query"],
                        "faiss": entry["faiss_ms_per_query"],
                    }
                )
    return panels


def write_speed_panel(arch, hw_label, thread_key, thread_label, tick_fmt, value_fmt, filename):
    panels = speed_panels(arch)
    width, height = 900, 460
    margin = {"top": 82, "right": 32, "bottom": 108, "left": 84}
    pw = width - margin["left"] - margin["right"]
    ph = height - margin["top"] - margin["bottom"]
    px = margin["left"]
    py = margin["top"]

    y_max = nice_ceil(max(max(g["tq"], g["faiss"]) for g in panels[thread_key]) * 1.22)

    parts = [
        paired_panel(
            px, py, pw, ph, thread_label, panels[thread_key],
            tick_fmt=tick_fmt,
            value_fmt=value_fmt,
            y_max=y_max,
        ),
        f'<text x="26" y="{py + ph/2}" transform="rotate(-90, 26, {py + ph/2})" class="axis">ms / query</text>',
        legend_tq_faiss(margin["left"], height - 26),
    ]
    body = "\n".join(parts)

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Search Latency — {xe(hw_label)} — {xe(thread_label)}">
  {style_block()}
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{margin["left"]}" y="32" class="title">Search Latency — {xe(hw_label)} — {xe(thread_label)}</text>
  <text x="{margin["left"]}" y="52" class="subtitle">100K vectors, 1K queries, k=64, median of 5 runs</text>
  {body}
</svg>
"""
    out = os.path.join(DOCS_DIR, filename)
    with open(out, "w") as f:
        f.write(svg)
    print(f"wrote {out}")


def line_panel(px, py, pw, ph, panel_title, series, x_values, x_labels, y_lo, y_hi):
    parts = [
        grid_lines(px, py, pw, ph, y_lo, y_hi, lambda v: f"{v:.2f}"),
        f'<text x="{px}" y="{py - 14}" class="panel">{xe(panel_title)}</text>',
    ]
    x_min = math.log2(x_values[0])
    x_max = math.log2(x_values[-1])

    def xpx(v):
        return px + (math.log2(v) - x_min) / (x_max - x_min) * pw

    def ypx(v):
        return py + ph - (v - y_lo) / (y_hi - y_lo) * ph

    for v, lbl in zip(x_values, x_labels):
        parts.append(
            f'<text x="{xpx(v):.1f}" y="{py + ph + 20}" text-anchor="middle" class="label">{xe(lbl)}</text>'
        )

    for s in series:
        color = s["color"]
        dash = ' stroke-dasharray="6 4"' if s.get("dashed") else ""
        points = [(xpx(x), ypx(y)) for x, y in zip(x_values, s["values"])]
        path = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.25"{dash} />'
        )
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}" />')

    return "\n".join(parts)


def write_recall_panel(dim_key, dim_label, filename, y_lo=0.85):
    width, height = 900, 460
    margin = {"top": 82, "right": 32, "bottom": 108, "left": 84}
    pw = width - margin["left"] - margin["right"]
    ph = height - margin["top"] - margin["bottom"]
    px = margin["left"]
    py = margin["top"]

    x_values = [1, 2, 4, 8, 16, 32, 64]
    x_labels = ["1", "2", "4", "8", "16", "32", "64"]

    # Draw FAISS lines first (background), then TurboQuant on top — emphasises
    # the TQ series when lines overlap or cross at high-K.
    faiss_series = []
    tq_series = []
    for bw_key, bw_label in [("2bit", "2-bit"), ("4bit", "4-bit")]:
        data = load_json(f"recall_{dim_key}_{bw_key}.json")
        tq_vals = [float(data["tq_recalls"][str(k)]) for k in x_values]
        faiss_vals = [float(data["faiss_recalls"][str(k)]) for k in x_values]
        tq_color = C["tq_2"] if bw_key == "2bit" else C["tq_4"]
        faiss_color = C["faiss_2"] if bw_key == "2bit" else C["faiss_4"]
        tq_series.append({"label": f"TQ {bw_label}", "values": tq_vals, "color": tq_color})
        faiss_series.append({"label": f"FAISS {bw_label}", "values": faiss_vals, "color": faiss_color, "dashed": True})
    series = faiss_series + tq_series

    parts = [
        line_panel(px, py, pw, ph, dim_label, series, x_values, x_labels, y_lo, 1.005),
        f'<text x="{px - 62}" y="{py + ph/2}" transform="rotate(-90, {px - 62}, {py + ph/2})" class="axis">recall@1@k</text>',
        f'<text x="{px + pw/2}" y="{py + ph + 48}" text-anchor="middle" class="axis">k</text>',
    ]

    legend_y = height - 26
    lx = margin["left"]
    items = [
        ("TQ 2-bit", C["tq_2"], False),
        ("TQ 4-bit", C["tq_4"], False),
        ("FAISS 2-bit", C["faiss_2"], True),
        ("FAISS 4-bit", C["faiss_4"], True),
    ]
    for i, (lbl, col, dash) in enumerate(items):
        cx = lx + i * 140
        dash_attr = ' stroke-dasharray="6 4"' if dash else ""
        parts.append(
            f'<line x1="{cx}" y1="{legend_y - 2}" x2="{cx + 24}" y2="{legend_y - 2}" stroke="{col}" stroke-width="2.25"{dash_attr} />'
        )
        parts.append(f'<circle cx="{cx + 12}" cy="{legend_y - 2}" r="3.5" fill="{col}" />')
        parts.append(f'<text x="{cx + 32}" y="{legend_y + 1}" class="legend">{xe(lbl)}</text>')

    body = "\n".join(parts)
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Recall — {xe(dim_label)}">
  {style_block()}
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{margin["left"]}" y="32" class="title">Recall — {xe(dim_label)}</text>
  <text x="{margin["left"]}" y="52" class="subtitle">100K vectors, k=64 search. recall@1@k measures how often the true top-1 result appears in the top-k returned.</text>
  {body}
</svg>
"""
    out = os.path.join(DOCS_DIR, filename)
    with open(out, "w") as f:
        f.write(svg)
    print(f"wrote {out}")


def write_compression_chart(filename):
    datasets = [
        ("GloVe|d=200", 76.3, 9.9, 5.1),
        ("OpenAI|d=1536", 585.9, 73.6, 37.0),
        ("OpenAI|d=3072", 1171.9, 146.9, 73.6),
    ]
    width, height = 900, 460
    margin = {"top": 82, "right": 32, "bottom": 108, "left": 84}
    pw = width - margin["left"] - margin["right"]
    ph = height - margin["top"] - margin["bottom"]
    px = margin["left"]
    py = margin["top"]

    y_max = nice_ceil(max(d[1] for d in datasets) * 1.15)

    parts = [grid_lines(px, py, pw, ph, 0, y_max, lambda v: f"{v:.0f}")]

    n = len(datasets)
    band = pw / n
    bar_w = min(56, band * 0.22)
    gap = 10

    for i, (label, fp32, four, two) in enumerate(datasets):
        cx = px + band * i + band / 2
        x_fp = cx - 1.5 * bar_w - gap
        x_4 = cx - 0.5 * bar_w
        x_2 = cx + 0.5 * bar_w + gap

        def draw(xbar, val, color, accent=False):
            h = (val / y_max) * ph
            y = py + ph - h
            stroke = (
                f' stroke="{C["tq_stroke"]}" stroke-width="1.5"' if accent else ""
            )
            value_cls = "value-accent" if accent else "value"
            return "\n".join(
                [
                    f'<rect x="{xbar:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" rx="6" fill="{color}"{stroke} />',
                    f'<text x="{xbar + bar_w/2:.1f}" y="{y - 6:.1f}" text-anchor="middle" class="{value_cls}">{xe(f"{val:.0f}")}</text>',
                ]
            )

        parts.append(draw(x_fp, fp32, C["fp32"]))
        parts.append(draw(x_4, four, C["four_bit"]))
        parts.append(draw(x_2, two, C["two_bit"], accent=True))

        label_y = py + ph + 22
        primary, _, secondary = label.partition("|")
        parts.append(f'<text x="{cx:.1f}" y="{label_y}" text-anchor="middle" class="label">{xe(primary)}</text>')
        if secondary:
            parts.append(f'<text x="{cx:.1f}" y="{label_y + 15}" text-anchor="middle" class="secondary">{xe(secondary)}</text>')

    parts.append(
        f'<text x="26" y="{py + ph/2}" transform="rotate(-90, 26, {py + ph/2})" class="axis">Index size (MB)</text>'
    )

    legend_y = height - 26
    lx = margin["left"]
    items = [
        ("FP32", C["fp32"], False),
        ("4-bit", C["four_bit"], False),
        ("2-bit", C["two_bit"], True),
    ]
    for i, (lbl, col, accent) in enumerate(items):
        lcx = lx + i * 120
        stroke = f' stroke="{C["tq_stroke"]}" stroke-width="1.5"' if accent else ""
        parts.append(f'<rect x="{lcx}" y="{legend_y - 10}" width="14" height="14" rx="3" fill="{col}"{stroke} />')
        parts.append(f'<text x="{lcx + 22}" y="{legend_y + 1}" class="legend">{xe(lbl)}</text>')

    body = "\n".join(parts)
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Index Size — TurboQuant">
  {style_block()}
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{margin["left"]}" y="32" class="title">Index Size — 100K vectors</text>
  <text x="{margin["left"]}" y="52" class="subtitle">TurboQuant packs vectors ~16× smaller than FP32 at 2-bit with comparable recall</text>
  {body}
</svg>
"""
    out = os.path.join(DOCS_DIR, filename)
    with open(out, "w") as f:
        f.write(svg)
    print(f"wrote {out}")


def arrow_defs(marker_id="arr", color="#525252"):
    return (
        f'<defs><marker id="{marker_id}" markerWidth="8" markerHeight="8" '
        f'refX="7" refY="4" orient="auto">'
        f'<path d="M0,0 L8,4 L0,8 z" fill="{color}"/></marker></defs>'
    )


def hline(x1, y, x2, color="#525252", w=1.5, marker_end=None):
    me = f' marker-end="url(#{marker_end})"' if marker_end else ""
    return f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="{w}"{me}/>'


def box(x, y, w, h, fill, stroke, label, sub=None, fs=11, sub_fs=9):
    if fill in (C["turbo"], C["graph"]):
        text_fill, sub_fill = "#ffffff", "#ecfdf5"
    else:
        text_fill, sub_fill = "#171717", "#525252"
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>',
        f'<text x="{x + w/2:.1f}" y="{y + (h/2 - 4 if sub else h/2 + 4):.1f}" '
        f'text-anchor="middle" font-size="{fs}" font-weight="600" fill="{text_fill}">{xe(label)}</text>',
    ]
    if sub:
        parts.append(
            f'<text x="{x + w/2:.1f}" y="{y + h/2 + 14:.1f}" text-anchor="middle" '
            f'font-size="{sub_fs}" fill="{sub_fill}">{xe(sub)}</text>'
        )
    return "\n".join(parts)


def vline(x, y1, y2, color="#525252", w=1.5, marker_end=None):
    me = f' marker-end="url(#{marker_end})"' if marker_end else ""
    return f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{color}" stroke-width="{w}"{me}/>'


def build_stack_diagram():
    W, H = 920, 380
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
        style_block(),
        arrow_defs(),
        f'<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{W/2:.0f}" y="28" text-anchor="middle" font-size="16" font-weight="700" fill="#171717">'
        f'{xe("Where turbo-graph sits on top of turbovec")}</text>',
        f'<text x="{W/2:.0f}" y="48" text-anchor="middle" font-size="11" fill="#737373">'
        f'{xe("Same TurboQuant core; graph layer adds metadata, cache, and rerank")}</text>',
    ]

    col_w, box_w, box_h, gap = 400, 340, 44, 8
    lx, rx = 40, 480
    y0 = 70

    turbovec_layers = [
        (C["turbo"], "#171717", "turbovec", "TurboQuant ANN core"),
        (C["turbo"], "#171717", "allowlist / mask", "in-kernel candidate filter"),
        (C["turbo"], "#171717", "post-filter", "optional metadata pass"),
    ]
    graph_layers = [
        (C["turbo"], "#171717", "turbovec", "same TurboQuant core"),
        (C["graph"], "#15803d", "GraphMemoryIndex", "graph + metadata + cache"),
        (C["graph"], "#15803d", "query assembly", "graph + tags + source + time + candidates"),
        (C["graph"], "#15803d", "rerank + telemetry", "optional second stage"),
    ]

    def draw_stack(x, layers, y_start):
        y = y_start
        els = []
        for i, (fill, stroke, label, sub) in enumerate(layers):
            bx = x + (col_w - box_w) / 2
            els.append(box(bx, y, box_w, box_h, fill, stroke, label, sub))
            if i < len(layers) - 1:
                cx = x + col_w / 2
                els.append(hline(cx - 12, y + box_h + 2, cx + 12, color="#a3a3a3", w=1.2))
            y += box_h + gap + 6
        return els, y

    svg.append(
        f'<text x="{lx + col_w/2:.0f}" y="{y0 - 8}" text-anchor="middle" '
        f'font-size="13" font-weight="700" fill="#171717">{xe("turbovec")}</text>'
    )
    svg.append(
        f'<text x="{rx + col_w/2:.0f}" y="{y0 - 8}" text-anchor="middle" '
        f'font-size="13" font-weight="700" fill="{C["graph"]}">{xe("turbo-graph")}</text>'
    )

    left_els, y_end_l = draw_stack(lx, turbovec_layers, y0)
    right_els, y_end_r = draw_stack(rx, graph_layers, y0)
    svg.extend(left_els)
    svg.extend(right_els)

    mid_y = y0 + (len(turbovec_layers) * (box_h + gap + 6) - gap - 6) / 2 + box_h / 2
    svg.extend(
        [
            hline(lx + col_w - 20, mid_y, rx + 20, color="#737373", w=1.5, marker_end="arr"),
            f'<text x="{W/2:.0f}" y="{mid_y - 10:.0f}" text-anchor="middle" font-size="10" fill="#737373">'
            f'{xe("shared core")}</text>',
        ]
    )

    note_y = max(y_end_l, y_end_r) + 16
    svg.append(
        f'<rect x="40" y="{note_y:.0f}" width="840" height="52" rx="8" fill="#fafafa" stroke="#e5e5e5"/>'
    )
    svg.append(
        f'<text x="60" y="{note_y + 20:.0f}" font-size="11" fill="#525252">'
        f'{xe("turbo-graph does not replace allowlist/mask; it composes graph constraints with kernel filters.")}</text>'
    )
    svg.append(
        f'<text x="60" y="{note_y + 38:.0f}" font-size="11" fill="#525252">'
        f'{xe("Use turbovec alone for pure ANN; add turbo-graph when you need structured memory over vectors.")}</text>'
    )
    svg.append("</svg>")
    return "\n".join(svg)


def build_query_paths_diagram():
    W, H = 1000, 340
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
        style_block(),
        arrow_defs(),
        f'<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{W/2:.0f}" y="28" text-anchor="middle" font-size="16" font-weight="700" fill="#171717">'
        f'{xe("Query path: turbovec vs turbo-graph")}</text>',
    ]

    def pipeline(x, title, steps, note, title_color, panel_w=460):
        bw, bh, gap = 78, 36, 12
        y_title, y0 = 52, 78
        els = [
            f'<text x="{x + panel_w/2:.0f}" y="{y_title}" text-anchor="middle" font-size="13" '
            f'font-weight="700" fill="{title_color}">{xe(title)}</text>',
        ]
        cx = x + (panel_w - (len(steps) * bw + (len(steps) - 1) * gap)) / 2
        for i, (label, fill, stroke) in enumerate(steps):
            els.append(box(cx + i * (bw + gap), y0, bw, bh, fill, stroke, label, fs=10))
            if i < len(steps) - 1:
                x1 = cx + i * (bw + gap) + bw + 2
                x2 = x1 + gap - 4
                els.append(hline(x1, y0 + bh / 2, x2, marker_end="arr"))
        ny = y0 + bh + 28
        els.append(
            f'<rect x="{x + 20:.0f}" y="{ny:.0f}" width="{panel_w - 40:.0f}" height="44" rx="6" '
            f'fill="#fafafa" stroke="#e5e5e5"/>'
        )
        els.append(
            f'<text x="{x + panel_w/2:.0f}" y="{ny + 26:.0f}" text-anchor="middle" font-size="10" fill="#525252">'
            f'{xe(note)}</text>'
        )
        return els

    turbovec_steps = [
        ("query", C["turbo"], "#171717"),
        ("allowlist", C["turbo"], "#171717"),
        ("SIMD scan", C["turbo"], "#171717"),
        ("top-k", C["turbo"], "#171717"),
    ]
    graph_steps = [
        ("query", C["graph"], "#15803d"),
        ("graph view", C["graph"], "#15803d"),
        ("metadata", C["graph"], "#15803d"),
        ("candidates", C["graph"], "#15803d"),
        ("rerank", C["graph"], "#15803d"),
    ]

    svg.extend(
        pipeline(
            30,
            "turbovec",
            turbovec_steps,
            "Filter inside kernel; optional post-filter on metadata",
            "#171717",
            panel_w=440,
        )
    )
    svg.extend(
        pipeline(
            520,
            "turbo-graph",
            graph_steps,
            "Assemble graph + tags + time window; cache hot views",
            C["graph"],
            panel_w=450,
        )
    )
    svg.append("</svg>")
    return "\n".join(svg)


def recall_r1_delta_pp(dataset_key):
    data = load_json(f"recall_{dataset_key}.json")
    turbo = data["tq_recalls"]["1"]
    base = data["faiss_recalls"]["1"]
    return (turbo - base) * 100


def build_recall_delta_chart():
    rows = [
        ("1536 2-bit", recall_r1_delta_pp("d1536_2bit")),
        ("1536 4-bit", recall_r1_delta_pp("d1536_4bit")),
        ("3072 2-bit", recall_r1_delta_pp("d3072_2bit")),
        ("3072 4-bit", recall_r1_delta_pp("d3072_4bit")),
        ("GloVe 2-bit", recall_r1_delta_pp("glove_2bit")),
        ("GloVe 4-bit", recall_r1_delta_pp("glove_4bit")),
    ]
    W, H = 720, 320
    ml, mr, mt, mb = 130, 40, 50, 40
    pw, ph = W - ml - mr, H - mt - mb
    max_abs = max(abs(v) for _, v in rows) * 1.25
    zero_x = ml + pw / 2
    scale = pw / 2 / max_abs

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
        style_block(),
        f'<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{W/2:.0f}" y="28" text-anchor="middle" font-size="16" font-weight="700" fill="#171717">'
        f'{xe("Recall@1 delta: TurboQuant vs FAISS IndexPQ (pp)")}</text>',
        f'<text x="{W/2:.0f}" y="46" text-anchor="middle" font-size="10" fill="#737373">'
        f'{xe("Shared turbo-graph / turbovec core | positive = TurboQuant higher recall")}</text>',
        f'<line x1="{zero_x:.1f}" y1="{mt}" x2="{zero_x:.1f}" y2="{mt + ph}" stroke="#a3a3a3" stroke-width="1"/>',
    ]

    bar_h = ph / len(rows) * 0.55
    row_gap = ph / len(rows)
    for i, (label, val) in enumerate(rows):
        cy = mt + i * row_gap + row_gap / 2
        bw = abs(val) * scale
        if val >= 0:
            x, fill = zero_x, C["graph"]
        else:
            x, fill = zero_x - bw, C["faiss"]
        svg.append(f'<rect x="{x:.1f}" y="{cy - bar_h/2:.1f}" width="{bw:.1f}" height="{bar_h:.1f}" rx="3" fill="{fill}"/>')
        svg.append(
            f'<text x="{ml - 10:.0f}" y="{cy + 4:.0f}" text-anchor="end" font-size="11" fill="#404040">{xe(label)}</text>'
        )
        tx = (x + bw + 6) if val >= 0 else (x - 6)
        anchor = "start" if val >= 0 else "end"
        sign = "+" if val >= 0 else ""
        svg.append(
            f'<text x="{tx:.1f}" y="{cy + 4:.0f}" text-anchor="{anchor}" font-size="10" font-weight="600" fill="#171717">'
            f'{xe(f"{sign}{val:.2f} pp")}</text>'
        )

    svg.append(
        f'<text x="{zero_x:.0f}" y="{H - 12:.0f}" text-anchor="middle" font-size="10" fill="#737373">'
        f'{xe("0")}</text>'
    )
    svg.append("</svg>")
    return "\n".join(svg)


def speed_gain_pct(arch, dataset, bits, mode):
    key = f"speed_{dataset}_{bits}bit_{arch}_{mode}.json"
    data = load_json(key)
    tq = data["tq_ms_per_query"]
    faiss = data["faiss_ms_per_query"]
    return (1 - tq / faiss) * 100


def build_speed_grid_chart():
    configs = [
        ("ARM ST", "arm", "st"),
        ("ARM MT", "arm", "mt"),
        ("x86 ST", "x86", "st"),
        ("x86 MT", "x86", "mt"),
    ]
    datasets = [("1536 2b", "d1536", "2"), ("1536 4b", "d1536", "4"), ("3072 2b", "d3072", "2"), ("3072 4b", "d3072", "4")]
    W, H = 720, 360
    ml, mt, cell_w, cell_h, gap = 100, 70, 120, 52, 10

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
        style_block(),
        f'<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{W/2:.0f}" y="28" text-anchor="middle" font-size="16" font-weight="700" fill="#171717">'
        f'{xe("Speed gain: TurboQuant vs FAISS IndexPQFastScan")}</text>',
        f'<text x="{W/2:.0f}" y="46" text-anchor="middle" font-size="10" fill="#737373">'
        f'{xe("Shared core | green = TurboQuant faster (% latency saved)")}</text>',
    ]

    for j, (col_label, _, _) in enumerate(datasets):
        cx = ml + j * (cell_w + gap) + cell_w / 2
        svg.append(
            f'<text x="{cx:.0f}" y="{mt - 12:.0f}" text-anchor="middle" font-size="10" font-weight="600" fill="#404040">'
            f'{xe(col_label)}</text>'
        )

    for i, (row_label, arch, mode) in enumerate(configs):
        ry = mt + i * (cell_h + gap)
        svg.append(
            f'<text x="{ml - 10:.0f}" y="{ry + cell_h/2 + 4:.0f}" text-anchor="end" font-size="11" font-weight="600" fill="#404040">'
            f'{xe(row_label)}</text>'
        )
        for j, (_, dataset, bits) in enumerate(datasets):
            cx = ml + j * (cell_w + gap)
            gain = speed_gain_pct(arch, dataset, bits, mode)
            fill = "#dcfce7" if gain >= 0 else "#fee2e2"
            stroke = "#15803d" if gain >= 0 else "#b91c1c"
            text_color = "#15803d" if gain >= 0 else "#b91c1c"
            sign = "+" if gain >= 0 else ""
            svg.append(
                f'<rect x="{cx:.0f}" y="{ry:.0f}" width="{cell_w:.0f}" height="{cell_h:.0f}" '
                f'rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1"/>'
            )
            svg.append(
                f'<text x="{cx + cell_w/2:.0f}" y="{ry + cell_h/2 + 5:.0f}" text-anchor="middle" '
                f'font-size="13" font-weight="700" fill="{text_color}">{xe(f"{sign}{gain:.0f}%")}</text>'
            )

    svg.append(
        f'<text x="{W/2:.0f}" y="{H - 14:.0f}" text-anchor="middle" font-size="10" fill="#737373">'
        f'{xe("ARM: 8/8 wins | x86: 2-bit MT only regression")}</text>'
    )
    svg.append("</svg>")
    return "\n".join(svg)


SELECTIVITY_ROWS = [
    ("0.10%", 0.014, 0.007, 0.016),
    ("1.00%", 0.013, 0.007, 0.010),
    ("5.00%", 0.019, 0.009, 0.012),
    ("20.0%", 0.034, 0.015, 0.025),
    ("100%", 0.241, 0.094, 0.053),
]


def simple_value_grid(px, py, pw, ph, y_max, steps=6):
    parts = []
    for i in range(steps + 1):
        v = y_max * i / steps
        y = py + ph - v / y_max * ph
        parts.append(
            f'<line x1="{px}" y1="{y:.1f}" x2="{px + pw}" y2="{y:.1f}" stroke="{C["grid"]}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{px - 8}" y="{y + 4:.1f}" text-anchor="end" class="tick">{xe(f"{v:.2f}")}</text>'
        )
    parts.append(
        f'<line x1="{px}" y1="{py + ph:.1f}" x2="{px + pw}" y2="{py + ph:.1f}" '
        f'stroke="{C["baseline"]}" stroke-width="1.5"/>'
    )
    parts.append(
        f'<text x="{px - 36}" y="{py + ph / 2:.0f}" transform="rotate(-90, {px - 36}, {py + ph / 2:.0f})" '
        f'class="axis">{xe("ms")}</text>'
    )
    return parts


def build_selectivity_chart():
    W, H = 720, 360
    ml, mr, mt, mb = 70, 30, 55, 55
    pw, ph = W - ml - mr, H - mt - mb
    n_groups = len(SELECTIVITY_ROWS)
    n_series = 3
    group_w = pw / n_groups
    bar_w = group_w / (n_series + 1)
    y_max = 0.28

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
        style_block(),
        f'<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{W/2:.0f}" y="28" text-anchor="middle" font-size="16" font-weight="700" fill="#171717">'
        f'{xe("graph_view_bench: latency by selectivity (ms)")}</text>',
        f'<text x="{W/2:.0f}" y="46" text-anchor="middle" font-size="10" fill="#737373">'
        f'{xe("Lower is better | 1M vectors, M3 Max")}</text>',
    ]
    svg.extend(simple_value_grid(ml, mt, pw, ph, y_max, steps=6))

    colors = [("#171717", "bool filter"), ("#525252", "slot_mask"), ("#15803d", "graph_view")]
    for gi, (label, b, s, g) in enumerate(SELECTIVITY_ROWS):
        gx = ml + gi * group_w + group_w / 2
        for si, val in enumerate([b, s, g]):
            bx = ml + gi * group_w + (si + 0.5) * bar_w
            bh = val / y_max * ph
            svg.append(
                f'<rect x="{bx:.1f}" y="{mt + ph - bh:.1f}" width="{bar_w * 0.85:.1f}" height="{bh:.1f}" '
                f'rx="2" fill="{colors[si][0]}"/>'
            )
        svg.append(
            f'<text x="{gx:.0f}" y="{mt + ph + 22:.0f}" text-anchor="middle" font-size="10" fill="#404040">'
            f'{xe(label)}</text>'
        )

    leg_y = H - 28
    for i, (color, name) in enumerate(colors):
        lx = ml + i * 180
        svg.append(f'<rect x="{lx:.0f}" y="{leg_y:.0f}" width="12" height="12" rx="2" fill="{color}"/>')
        svg.append(
            f'<text x="{lx + 18:.0f}" y="{leg_y + 10:.0f}" font-size="10" fill="#404040">{xe(name)}</text>'
        )
    svg.append("</svg>")
    return "\n".join(svg)


def build_migration_flowchart():
    W, H = 820, 490
    cx = W / 2
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
        style_block(),
        arrow_defs(),
        f'<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{cx:.0f}" y="28" text-anchor="middle" font-size="16" font-weight="700" fill="#171717">'
        f'{xe("Migration decision flow")}</text>',
    ]

    def diamond(cx, cy, w, h, text):
        pts = f"{cx},{cy - h/2} {cx + w/2},{cy} {cx},{cy + h/2} {cx - w/2},{cy}"
        return (
            f'<polygon points="{pts}" fill="#fafafa" stroke="#525252" stroke-width="1.2"/>'
            f'<text x="{cx:.0f}" y="{cy + 4:.0f}" text-anchor="middle" font-size="10" fill="#171717">'
            f'{xe(text)}</text>'
        )

    nodes = []
    y = 52
    nodes.append(box(cx - 130, y, 260, 40, "#fafafa", "#525252", "Need ANN vector search?", fs=11))
    y += 52
    nodes.append(vline(cx, y - 10, y + 8, marker_end="arr"))
    y += 18
    nodes.append(diamond(cx, y + 26, 300, 56, "Graph / tags / time constraints?"))
    branch_y = y + 58
    split_y = branch_y + 28
    nodes.append(vline(cx, branch_y, split_y))
    y = split_y

    left_cx, right_cx = 210, 610
    nodes.append(hline(left_cx, y, cx, marker_end="arr"))
    nodes.append(hline(cx, y, right_cx, marker_end="arr"))
    nodes.append(f'<text x="{left_cx + 28:.0f}" y="{y - 8:.0f}" font-size="10" fill="#737373">{xe("no")}</text>')
    nodes.append(f'<text x="{right_cx - 36:.0f}" y="{y - 8:.0f}" font-size="10" fill="#737373">{xe("yes")}</text>')

    y_box = y + 18
    nodes.append(vline(left_cx, y, y_box, marker_end="arr"))
    nodes.append(vline(right_cx, y, y_box, marker_end="arr"))
    nodes.append(box(110, y_box, 200, 48, C["turbo"], "#171717", "Stay on turbovec", "pure ANN + allowlist/mask"))
    nodes.append(box(510, y_box, 200, 48, C["graph"], "#15803d", "Adopt turbo-graph", "GraphMemoryIndex layer"))

    y2 = y_box + 58
    nodes.append(vline(610, y_box + 48, y2, marker_end="arr"))
    nodes.append(diamond(610, y2 + 26, 250, 52, "Need cache, rerank, or telemetry?"))
    y3 = y2 + 58
    nodes.append(vline(610, y3, y3 + 16, marker_end="arr"))
    nodes.append(box(510, y3 + 22, 200, 44, C["graph"], "#15803d", "Full migration", "graph index + cache + rerank"))

    foot_y = H - 66
    nodes.append(f'<rect x="40" y="{foot_y:.0f}" width="740" height="58" rx="8" fill="#fafafa" stroke="#e5e5e5"/>')
    nodes.append(
        f'<text x="60" y="{foot_y + 22:.0f}" font-size="10" fill="#525252">'
        f'{xe("Checklist: graph edges, tag/source/time metadata, candidate assembly, cache, rerank, telemetry")}</text>'
    )
    nodes.append(
        f'<text x="60" y="{foot_y + 40:.0f}" font-size="10" fill="#525252">'
        f'{xe("Shared TurboQuant core is identical; migration adds the graph memory layer on top.")}</text>'
    )

    svg.extend(nodes)
    svg.append("</svg>")
    return "\n".join(svg)


def write_readme_diagrams():
    diagrams = [
        ("stack.svg", build_stack_diagram),
        ("query_paths.svg", build_query_paths_diagram),
        ("recall_delta.svg", build_recall_delta_chart),
        ("speed_grid.svg", build_speed_grid_chart),
        ("selectivity.svg", build_selectivity_chart),
        ("migration.svg", build_migration_flowchart),
    ]
    for name, builder in diagrams:
        out = os.path.join(DOCS_DIR, name)
        with open(out, "w") as f:
            f.write(builder())
        print(f"wrote {out}")


if __name__ == "__main__":
    os.makedirs(DOCS_DIR, exist_ok=True)
    write_speed_panel("arm", "ARM (Apple M3 Max)", "st", "Single-threaded",
                      tick_fmt=lambda v: f"{v:.1f}", value_fmt=lambda v: f"{v:.2f}",
                      filename="arm_speed_st.svg")
    write_speed_panel("arm", "ARM (Apple M3 Max)", "mt", "Multi-threaded",
                      tick_fmt=lambda v: f"{v:.2f}", value_fmt=lambda v: f"{v:.3f}",
                      filename="arm_speed_mt.svg")
    write_speed_panel("x86", "x86 (Intel Sapphire Rapids, 8 vCPUs)", "st", "Single-threaded",
                      tick_fmt=lambda v: f"{v:.1f}", value_fmt=lambda v: f"{v:.2f}",
                      filename="x86_speed_st.svg")
    write_speed_panel("x86", "x86 (Intel Sapphire Rapids, 8 vCPUs)", "mt", "Multi-threaded",
                      tick_fmt=lambda v: f"{v:.2f}", value_fmt=lambda v: f"{v:.3f}",
                      filename="x86_speed_mt.svg")
    write_recall_panel("d1536", "d=1536", "recall_d1536.svg")
    write_recall_panel("d3072", "d=3072", "recall_d3072.svg")
    write_recall_panel("glove", "GloVe d=200", "recall_glove.svg", y_lo=0.4)
    write_compression_chart("compression.svg")
    write_readme_diagrams()
