#!/usr/bin/env python3
"""Interactive per-request BF16/FP8 scatter plots, plus graph-only PDF export."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess

import numpy as np
import matplotlib

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RESULT = 'profiling/results/bf16-fp8-vast-reactome-c4-100-2026-11-14'
DEFAULT_REF = 'origin/profiling/bf16-fp8-request-records'
INPUTS = {
    'input_tokens': 'Input tokens (pre-dispatch)',
    'requested_output_token_limit': 'Requested output-token limit',
    'running_requests_at_dispatch': 'Running requests (pre-dispatch sample)',
    'waiting_requests_at_dispatch': 'Waiting requests (pre-dispatch sample)',
}
OUTPUTS = {
    'e2e_latency_ms': 'End-to-end latency (s)',
    'ttft_ms': 'Time to first generated token (s)',
    'output_tokens': 'Actual output tokens',
    'dispatch_timestamp_utc': 'Dispatch time (s since first dispatch in this run)',
    'completion_timestamp_utc': 'Completion time (s since first dispatch in this run)',
}
COLORS = {'bf16': '#2166AC', 'fp8': '#D6604D'}


def epoch(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


def load_data(args):
    """Explicit paths override local result folders, then the selected Git revision."""
    data, sources = {}, {}
    commit = None
    for precision in COLORS:
        explicit = getattr(args, precision)
        local = args.results_dir / precision / 'measurement/canonical-requests.jsonl'
        if explicit or local.is_file():
            path = explicit or local
            text = path.read_text()
            sources[precision] = str(path.resolve())
        else:
            if commit is None:
                commit = subprocess.check_output(['git', 'rev-parse', args.ref], cwd=REPO, text=True).strip()
            relative = f'{RESULT}/{precision}/measurement/canonical-requests.jsonl'
            text = subprocess.check_output(['git', 'show', f'{commit}:{relative}'], cwd=REPO, text=True)
            sources[precision] = f'{commit}:{relative}'
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        ids = [(r.get('run_id'), r['request_id']) for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate run/request IDs in {precision}')
        origins = {}
        for row in rows:
            if row.get('dispatch_timestamp_utc'):
                run = row.get('run_id', precision)
                ts = epoch(row['dispatch_timestamp_utc'])
                origins[run] = min(origins.get(run, ts), ts)
        data[precision] = (rows, origins)
    return data, sources


def values(data, precision, x, y):
    rows, origins = data[precision]
    pairs = []
    for row in rows:
        # Missing measurements are omitted pairwise, never replaced with zero.
        if row.get(x) is None or row.get(y) is None:
            continue
        xv, yv = float(row[x]), row[y]
        if y.endswith('_timestamp_utc'):
            origin = origins.get(row.get('run_id', precision))
            if origin is None:
                continue
            yv = epoch(yv) - origin
        else:
            yv = float(yv) / (1000 if y.endswith('_ms') else 1)
        if np.isfinite(xv) and np.isfinite(yv):
            pairs.append((xv, yv))
    arr = np.asarray(pairs, dtype=float).reshape(-1, 2)
    return arr, len(rows)


def draw(ax, data, x, y, selected, colors):
    ax.clear()
    for precision in selected:
        arr, total = values(data, precision, x, y)
        ax.scatter(arr[:, 0], arr[:, 1], s=33, alpha=0.65,
                   color=colors[precision], marker='o' if precision == 'bf16' else '^',
                   edgecolors='none', label=f'{precision.upper()} (n={len(arr)}/{total})')
    ax.set_xlabel(INPUTS[x], labelpad=10)
    ax.set_ylabel(OUTPUTS[y], labelpad=10)
    ax.set_title(f'{OUTPUTS[y].split(" (")[0]} vs {INPUTS[x].split(" (")[0]}', pad=16)
    ax.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    if selected:
        ax.legend(loc='best', frameon=True)
    else:
        ax.text(.5, .5, 'Select a precision to display', ha='center', transform=ax.transAxes)
    ax.ticklabel_format(useOffset=False, style='plain', axis='both')
    ax.margins(x=0.08, y=0.1)


def export_pdf(data, sources, x, y, selected, colors, path):
    # Separate figure ensures the widget panel is never included in the PDF.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(9, 6.5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    draw(ax, data, x, y, selected, colors)
    fig.subplots_adjust(left=.14, right=.97, top=.88, bottom=.2)
    note = 'Per-request observations; no fitted trend. Counts show finite pairs / all attempts.'
    if y.endswith('_timestamp_utc'):
        note += '\nTimestamps are relative to each run’s first dispatch; they are not request runtimes.'
    fig.text(.14, .055, note, fontsize=8, color='#444444')
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format='pdf', metadata={'Title': ax.get_title(), 'Creator': 'request_scatter.py'})
    path.with_suffix('.json').write_text(json.dumps({
        'x': x, 'y': y, 'precisions': selected, 'colors': colors, 'sources': sources,
        'timestamp_mode': 'seconds_since_first_dispatch_per_run',
        'counts': {p: {'plotted':len(values(data,p,x,y)[0]),'attempts':len(data[p][0])} for p in selected}
    }, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=REPO / RESULT)
    parser.add_argument('--bf16', type=Path, help='Explicit BF16 canonical JSONL')
    parser.add_argument('--fp8', type=Path, help='Explicit FP8 canonical JSONL')
    parser.add_argument('--ref', default=DEFAULT_REF, help='Local Git ref fallback; no fetch or checkout performed')
    parser.add_argument('--x', choices=INPUTS, default='input_tokens')
    parser.add_argument('--y', choices=OUTPUTS, default='e2e_latency_ms')
    parser.add_argument('--precisions', nargs='+', choices=COLORS, default=list(COLORS))
    parser.add_argument('--bf16-color', default=COLORS['bf16'])
    parser.add_argument('--fp8-color', default=COLORS['fp8'])
    parser.add_argument('--output-dir', type=Path, default=HERE / 'exports')
    parser.add_argument('--export-all', action='store_true', help='Save all 20 graph-only PDFs without opening a window')
    args = parser.parse_args()
    if args.export_all:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import is_color_like
    from matplotlib.widgets import RadioButtons, CheckButtons, Button, TextBox
    colors = {'bf16': args.bf16_color, 'fp8': args.fp8_color}
    if not all(is_color_like(c) for c in colors.values()):
        parser.error('Colors must be Matplotlib color names or hex strings.')
    data, sources = load_data(args)
    selected = list(dict.fromkeys(args.precisions))
    if args.export_all:
        for x in INPUTS:
            for y in OUTPUTS:
                export_pdf(data, sources, x, y, selected, colors, args.output_dir / f'{y}__vs__{x}.pdf')
        print(f'Exported 20 PDFs and their source metadata to {args.output_dir.resolve()}')
        return
    state = {'x': args.x, 'y': args.y}
    fig = plt.figure(figsize=(15, 8))
    fig.canvas.manager.set_window_title('Serving request scatter explorer')
    ax = fig.add_axes([.085, .16, .57, .75])
    fig.text(.70, .955, 'Plot controls', fontsize=15, weight='bold')
    fig.text(.70, .90, 'Precisions', fontsize=10, weight='bold')
    checks = CheckButtons(fig.add_axes([.70, .79, .27, .10]), ['BF16', 'FP8'], [p in selected for p in COLORS])
    fig.text(.70, .765, 'X axis', fontsize=10, weight='bold')
    xs = RadioButtons(fig.add_axes([.70, .56, .28, .195]), ['Input tokens','Output-token limit','Running requests','Waiting requests'], active=list(INPUTS).index(args.x))
    fig.text(.70, .535, 'Y axis', fontsize=10, weight='bold')
    ys = RadioButtons(fig.add_axes([.70, .29, .28, .235]), ['End-to-end latency','TTFT','Actual output tokens','Dispatch time (relative)','Completion time (relative)'], active=list(OUTPUTS).index(args.y))
    bcolor = TextBox(fig.add_axes([.79, .225, .17, .04]), 'BF16 color ', initial=colors['bf16'])
    fcolor = TextBox(fig.add_axes([.79, .17, .17, .04]), 'FP8 color ', initial=colors['fp8'])
    export = Button(fig.add_axes([.70, .095, .28, .055]), 'Export graph PDF')
    status = fig.text(.70, .025, 'Colors: names or #RRGGBB; press Enter.\nPDFs save to the selected output directory.', fontsize=8)
    fig.text(.085, .035, 'All attempts with finite values are shown; missing values are not zeros.\nTimestamp axes show run-relative time, not latency. Constant input values form vertical stacks.', fontsize=9)
    def redraw():
        draw(ax, data, state['x'], state['y'], selected, colors)
        fig.canvas.draw_idle()
    def choose_x(label):
        state['x'] = list(INPUTS)[['Input tokens','Output-token limit','Running requests','Waiting requests'].index(label)]
        redraw()
    def choose_y(label):
        state['y'] = list(OUTPUTS)[['End-to-end latency','TTFT','Actual output tokens','Dispatch time (relative)','Completion time (relative)'].index(label)]
        redraw()
    def toggle(label):
        selected[:] = [p for p, visible in zip(COLORS, checks.get_status()) if visible]
        redraw()
    def color_change(precision, color):
        if is_color_like(color):
            colors[precision] = color
            status.set_text('Color updated. Export saves only the graph.')
            redraw()
        else:
            status.set_text('Invalid color. Try blue or #2166AC.')
            fig.canvas.draw_idle()
    def save(event):
        if not selected:
            status.set_text('Select at least one precision before exporting.')
        else:
            name = f'{state["y"]}__vs__{state["x"]}__{"-".join(selected)}.pdf'
            path = args.output_dir / name
            try:
                export_pdf(data, sources, state['x'], state['y'], selected, colors, path)
                status.set_text('PDF saved in:\n'+str(args.output_dir.resolve()))
                print(f'Saved {path.resolve()}')
            except Exception as exc:
                status.set_text(f'Export failed: {exc}')
        fig.canvas.draw_idle()
    checks.on_clicked(toggle)
    xs.on_clicked(choose_x)
    ys.on_clicked(choose_y)
    bcolor.on_submit(lambda c: color_change('bf16', c))
    fcolor.on_submit(lambda c: color_change('fp8', c))
    export.on_clicked(save)
    redraw()
    plt.show()

if __name__ == '__main__':
    main()
