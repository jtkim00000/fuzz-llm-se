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
RESULT = 'profiling/results/vast-full-bf16-fp8-2026-11-15'
DEFAULT_REF = 'origin/profiling/bf16-fp8-request-records'
INPUTS = {'input_tokens': 'Input tokens (pre-dispatch)'}
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


class PlotData(dict):
    """Caches are scoped to this loaded dataset, not persisted across input changes."""
    def __init__(self):
        super().__init__()
        self.pairs = {}
        self.fits = {}


def load_data(args):
    """Explicit paths override local result folders, then the selected Git revision."""
    data, sources = PlotData(), {}
    commit = None
    for precision in COLORS:
        explicit = getattr(args, precision)
        rows = []
        sources[precision] = []
        paths = [(None, explicit)] if explicit else [(c, args.results_dir / precision / c / 'canonical-requests.jsonl') for c in [f'c{c}-r{r}' for c in (1, 4) for r in (1, 2, 3)]]
        for block, path in paths:
            if path.is_file():
                text = path.read_text()
                sources[precision].append(str(path.resolve()))
            elif explicit or args.results_dir.resolve() != (REPO / RESULT).resolve():
                raise FileNotFoundError(path)
            else:
                if commit is None:
                    commit = subprocess.check_output(['git', 'rev-parse', args.ref], cwd=REPO, text=True).strip()
                relative = f'{RESULT}/{precision}/{block}/canonical-requests.jsonl'
                text = subprocess.check_output(['git', 'show', f'{commit}:{relative}'], cwd=REPO, text=True)
                sources[precision].append(f'{commit}:{relative}')
            for line in text.splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if block:
                    row['_concurrency'] = int(block.split('-')[0][1:])
                elif 'concurrency' in row:
                    row['_concurrency'] = int(row['concurrency'])
                else:
                    import re
                    match = re.search(r'(?:^|[-_])c(1|4)(?:[-_:]|$)', row.get('run_id', ''))
                    if not match:
                        raise ValueError('Explicit input needs concurrency or a run_id containing c1/c4')
                    row['_concurrency'] = int(match.group(1))
                rows.append(row)
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


def select_data(data, concurrency, output_limit):
    subset = PlotData()
    subset.selection = {'concurrency': concurrency, 'output_limit': output_limit}
    for precision, (rows, origins) in data.items():
        subset[precision] = ([r for r in rows
            if (concurrency == 'all' or r['_concurrency'] == int(concurrency))
            and (output_limit == 'all' or r['requested_output_token_limit'] == int(output_limit))], origins)
    return subset


def values(data, precision, x, y):
    key = (precision, x, y)
    if isinstance(data, PlotData) and key in data.pairs:
        return data.pairs[key]
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
    result = (arr, len(rows))
    if isinstance(data, PlotData):
        data.pairs[key] = result
    return result


LAYERS = ['bf16_scatter', 'bf16_regression', 'fp8_scatter', 'fp8_regression']
LAYER_LABELS = ['BF16 scatter', 'BF16 regression', 'FP8 scatter', 'FP8 regression']


def linear_fit(arr):
    if len(arr) < 3 or len(np.unique(arr[:, 0])) < 2:
        return {'available': False, 'reason': 'Need 3 points and 2 distinct X values', 'n': len(arr)}
    x, y = arr[:, 0], arr[:, 1]
    center = x.mean()
    slope, centered_intercept = np.linalg.lstsq(
        np.column_stack([x - center, np.ones(len(x))]), y, rcond=None)[0]
    predicted = slope * (x - center) + centered_intercept
    residual = np.sum((y - predicted) ** 2)
    total = np.sum((y - y.mean()) ** 2)
    return {'available': True, 'n': len(arr), 'slope': float(slope),
            'intercept': float(centered_intercept - slope * center),
            'r_squared': float(1 - residual / total) if total > 0 else None,
            'rmse': float(np.sqrt(residual / len(x))),
            'x_center': float(center), 'y_at_center': float(centered_intercept)}


def trimmed_fit(arr, percent=0):
    """One-pass trimming by absolute residual from the full-data OLS fit."""
    if not np.isfinite(percent) or not 0 <= percent < 100:
        raise ValueError('Outlier percentage must be finite and in [0, 100).')
    baseline = linear_fit(arr)
    excluded = np.array([], dtype=int)
    if baseline['available'] and percent > 0:
        predicted = baseline['slope'] * (arr[:, 0] - baseline['x_center']) + baseline['y_at_center']
        count = int(np.floor(len(arr) * percent / 100))
        # Stable ties are resolved by original finite-pair row order.
        excluded = np.argsort(-np.abs(arr[:, 1] - predicted), kind='stable')[:count]
    retained = np.ones(len(arr), dtype=bool)
    retained[excluded] = False
    fit = linear_fit(arr[retained])
    fit.update(trim_percent=percent, total_pairs=len(arr), excluded_count=len(excluded),
               excluded_finite_pair_indices=excluded.tolist(), baseline=baseline)
    if fit['available']:
        fit['domain'] = [float(arr[retained, 0].min()), float(arr[retained, 0].max())]
    return fit


def draw(ax, data, x, y, layers, colors, trim_percent=0):
    from matplotlib.transforms import Bbox
    if not hasattr(ax, '_request_artists'):
        ax._request_artists = {}
        for precision in COLORS:
            scatter = ax.scatter([], [], s=33, alpha=0.65,
                                 marker='o' if precision == 'bf16' else '^', edgecolors='none',
                                 label=f'{precision.upper()} scatter')
            line, = ax.plot([], [], linestyle='--', linewidth=2,
                            label=f'{precision.upper()} regression')
            ax._request_artists[precision] = (scatter, line)
    ax.dataLim = Bbox.null()
    statistics = {}
    handles = []
    for precision in COLORS:
        arr, total = values(data, precision, x, y)
        key = (precision, x, y, float(trim_percent))
        if isinstance(data, PlotData) and key in data.fits:
            fit = data.fits[key]
        else:
            fit = trimmed_fit(arr, trim_percent)
            if isinstance(data, PlotData):
                if len(data.fits) >= 256:
                    data.fits.clear()
                data.fits[key] = fit
        statistics[precision] = dict(fit, attempts=total)
        keep = np.ones(len(arr), dtype=bool)
        keep[fit['excluded_finite_pair_indices']] = False
        retained = arr[keep]
        scatter, line = ax._request_artists[precision]
        scatter.set_offsets(retained)
        scatter.set_color(colors[precision])
        scatter.set_visible(precision + '_scatter' in layers)
        line.set_color(colors[precision])
        line.set_visible(precision + '_regression' in layers and fit['available'])
        if scatter.get_visible():
            handles.append(scatter)
            if len(retained):
                ax.update_datalim(retained)
        if line.get_visible():
            domain = np.array(fit['domain'])
            curve = fit['slope'] * (domain - fit['x_center']) + fit['y_at_center']
            line.set_data(domain, curve)
            ax.update_datalim(np.column_stack([domain, curve]))
            handles.append(line)
    ax.set_autoscale_on(True)
    ax.autoscale_view()
    ax.set_xlabel(INPUTS[x], labelpad=10)
    ax.set_ylabel(OUTPUTS[y], labelpad=10)
    condition = getattr(data, 'selection', {})
    subtitle = f"Concurrency: {condition.get('concurrency', 'all')} | Output limit: {condition.get('output_limit', 'all')}"
    ax.set_title(f'{OUTPUTS[y].split(" (")[0]} vs input tokens\n{subtitle}', pad=12)
    ax.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    if handles:
        ax.legend(handles=handles, loc='upper right', frameon=True)
    ax.ticklabel_format(useOffset=False, style='plain', axis='both')
    ax.margins(x=0.08, y=0.1)
    return statistics


def export_pdf(data, sources, x, y, layers, colors, path, limits=None, trim_percent=0):
    # A separate figure contains only the graph, never controls or fit statistics.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(9, 6.5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    statistics = draw(ax, data, x, y, layers, colors, trim_percent)
    if limits:
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
    fig.subplots_adjust(left=.14, right=.97, top=.88, bottom=.17)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format='pdf', metadata={'Title': ax.get_title(), 'Creator': 'request_scatter.py'})
    path.with_suffix('.json').write_text(json.dumps({
        'x': x, 'y': y, 'selection': getattr(data, 'selection', {}), 'visible_layers': layers, 'colors': colors, 'sources': sources,
        'timestamp_mode': 'seconds_since_first_dispatch_per_run',
        'regression_function': 'linear', 'statistics': statistics,
        'scatter_excludes_trimmed_points': True,
        'trim_percent': trim_percent, 'trim_method': 'one-pass largest absolute full-data OLS residuals; floor(n*p/100)',
        'x_limits': list(ax.get_xlim()), 'y_limits': list(ax.get_ylim())
    }, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=REPO / RESULT)
    parser.add_argument('--bf16', type=Path, help='Explicit BF16 canonical JSONL')
    parser.add_argument('--fp8', type=Path, help='Explicit FP8 canonical JSONL')
    parser.add_argument('--ref', default=DEFAULT_REF, help='Local Git ref fallback; no fetch or checkout performed')
    parser.add_argument('--y', choices=OUTPUTS, default='e2e_latency_ms')
    parser.add_argument('--precisions', nargs='+', choices=COLORS, default=list(COLORS))
    parser.add_argument('--bf16-color', default=COLORS['bf16'])
    parser.add_argument('--fp8-color', default=COLORS['fp8'])
    parser.add_argument('--output-dir', type=Path, default=HERE / 'exports')
    parser.add_argument('--export-all', action='store_true', help='Save all 5 input-token graph PDFs without opening a window')
    parser.add_argument('--regression', action='store_true', help='Show separate descriptive linear fits')
    parser.add_argument('--layers', nargs='+', choices=LAYERS, help='Explicit independent layers; overrides --precisions/--regression')
    parser.add_argument('--trim-percent', type=float, default=0, help='Exclude this percent of largest baseline residuals from each fit only')
    parser.add_argument('--concurrency', choices=['1', '4', 'all'], default='1')
    parser.add_argument('--output-limit', choices=['32', '256', 'all'], default='256')
    args = parser.parse_args()
    if not np.isfinite(args.trim_percent) or not 0 <= args.trim_percent < 100:
        parser.error('--trim-percent must be finite and in [0, 100).')
    if args.export_all:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import is_color_like
    from matplotlib.widgets import RadioButtons, CheckButtons, Button, TextBox
    colors = {'bf16': args.bf16_color, 'fp8': args.fp8_color}
    if not all(is_color_like(c) for c in colors.values()):
        parser.error('Colors must be Matplotlib color names or hex strings.')
    all_data, sources = load_data(args)
    data = select_data(all_data, args.concurrency, args.output_limit)
    layers = list(dict.fromkeys(args.layers)) if args.layers else [
        layer for p in dict.fromkeys(args.precisions)
        for layer in ([p + '_scatter', p + '_regression'] if args.regression else [p + '_scatter'])]
    if args.export_all:
        for x in INPUTS:
            for y in OUTPUTS:
                export_pdf(data, sources, x, y, layers, colors, args.output_dir / f'{y}__vs__{x}__c{args.concurrency}__o{args.output_limit}.pdf', trim_percent=args.trim_percent)
        print(f'Exported 5 PDFs and their source metadata to {args.output_dir.resolve()}')
        return
    state = {'x': 'input_tokens', 'y': args.y, 'trim_percent': args.trim_percent}
    fig = plt.figure(figsize=(15, 10))
    fig.canvas.manager.set_window_title('Serving request scatter explorer')
    concurrency_box = TextBox(fig.add_axes([.22, .95, .10, .035]), 'Concurrency ', initial=args.concurrency)
    output_box = TextBox(fig.add_axes([.49, .95, .10, .035]), 'Output limit ', initial=args.output_limit)
    fig.text(.085, .925, 'Filters: concurrency 1 / 4 / all; output limit 32 / 256 / all. Press Enter.', fontsize=8)
    ax = fig.add_axes([.085, .30, .57, .55])
    fig.text(.70, .955, 'Plot controls', fontsize=15, weight='bold')
    fig.text(.70, .915, 'Visible layers', fontsize=10, weight='bold')
    checks = CheckButtons(fig.add_axes([.70, .78, .28, .125]), LAYER_LABELS, [layer in layers for layer in LAYERS])
    fig.text(.70, .76, 'X axis: Input tokens (fixed)', fontsize=10, weight='bold')
    fig.text(.70, .72, 'Y axis', fontsize=10, weight='bold')
    ys = RadioButtons(fig.add_axes([.70, .535, .28, .175]), ['End-to-end latency','TTFT','Actual output tokens','Dispatch time (relative)','Completion time (relative)'], active=list(OUTPUTS).index(args.y))
    bcolor = TextBox(fig.add_axes([.79, .48, .17, .035]), 'BF16 color ', initial=colors['bf16'])
    fcolor = TextBox(fig.add_axes([.79, .425, .17, .035]), 'FP8 color ', initial=colors['fp8'])
    fig.text(.70, .385, 'Linear fit statistics: y = slope*x + intercept', fontsize=10, weight='bold')
    stats_text = fig.text(.70, .365, '', va='top', fontsize=9, linespacing=1.25)
    export = Button(fig.add_axes([.70, .045, .28, .05]), 'Export graph PDF')
    status = fig.text(.70, .008, 'Colors: names or #RRGGBB; press Enter.\nPDFs save to the selected output directory.', fontsize=8)
    fig.text(.085, .035, 'Scatter and fits show retained observations; missing values are not zeros.\nTimestamp axes show run-relative time, not latency. Excluded points remain in the source data.', fontsize=9)
    trim_box = TextBox(fig.add_axes([.27, .205, .10, .03]), 'Exclude outliers (%) ', initial=str(args.trim_percent))
    fig.text(.39, .215, 'Largest baseline residuals; hidden from scatter and fit.', fontsize=8)
    fields = [TextBox(fig.add_axes(rect), label, initial='0') for rect, label in [
        ([.15, .16, .16, .035], 'X min '), ([.46, .16, .16, .035], 'X max '),
        ([.15, .105, .16, .035], 'Y min '), ([.46, .105, .16, .035], 'Y max ')]]
    auto = Button(fig.add_axes([.51, .055, .14, .035]), 'Auto axes')
    def sync_fields():
        for field, value in zip(fields, (*ax.get_xlim(), *ax.get_ylim())):
            field.eventson = False
            field.drawon = False
            field.set_val(format(value, '.10g'))
            field.drawon = True
            field.eventson = True
    def apply_limits(_):
        try:
            nums = [float(field.text) for field in fields]
            if not all(np.isfinite(nums)) or nums[0] >= nums[1] or nums[2] >= nums[3]:
                raise ValueError('Use finite limits with min < max.')
            ax.set_xlim(nums[:2])
            ax.set_ylim(nums[2:])
            status.set_text('Axis limits applied. Press Enter to apply typed values.')
        except ValueError:
            status.set_text('Invalid limits: enter numbers with min < max.\nThe graph keeps its previous range.')
        fig.canvas.draw_idle()
    def redraw(reset=False):
        limits = (ax.get_xlim(), ax.get_ylim())
        stats = draw(ax, data, state['x'], state['y'], layers, colors, state['trim_percent'])
        sections = []
        for precision, fit in stats.items():
            heading = f'{precision.upper()} — retained {fit["n"]}/{fit["total_pairs"]}; excluded {fit["excluded_count"]}'
            if fit['available']:
                r2 = f'{fit["r_squared"]:.5g}' if fit['r_squared'] is not None else 'undefined'
                sections.append(heading + f'\nSlope: {fit["slope"]:.6g}  Intercept: {fit["intercept"]:.6g}'
                                + f'\nRetained R²: {r2}  RMSE: {fit["rmse"]:.5g}'
                                + f'\nFull-data R²: {fit["baseline"]["r_squared"] if fit["baseline"]["r_squared"] is None else format(fit["baseline"]["r_squared"], ".5g")}')
            else:
                sections.append(heading + '\n' + fit['reason'])
        stats_text.set_text('\n\n'.join(sections) + '\n\nR² after exclusion is exploratory, not validation.\nExcluded points hidden; zoom does not refit.')
        if not reset:
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
        sync_fields()
        fig.canvas.draw_idle()
    def apply_filters(_):
        nonlocal data
        c, o = concurrency_box.text.strip(), output_box.text.strip()
        if c not in ('1', '4', 'all') or o not in ('32', '256', 'all'):
            status.set_text('Use concurrency 1/4/all and output limit 32/256/all.')
            fig.canvas.draw_idle()
            return
        data = select_data(all_data, c, o)
        status.set_text('Filters applied. Mixed conditions can obscure trends.' if 'all' in (c,o) else 'Matched condition selected; 36 requests per precision across three repetitions.')
        redraw(reset=True)
    concurrency_box.on_submit(apply_filters)
    output_box.on_submit(apply_filters)
    def choose_y(label):
        state['y'] = list(OUTPUTS)[['End-to-end latency','TTFT','Actual output tokens','Dispatch time (relative)','Completion time (relative)'].index(label)]
        redraw(reset=True)
    def toggle(label):
        layers[:] = [layer for layer, visible in zip(LAYERS, checks.get_status()) if visible]
        redraw(reset=True)
    def color_change(precision, color):
        if is_color_like(color):
            colors[precision] = color
            status.set_text('Color updated. Export saves only the graph.')
            redraw()
        else:
            status.set_text('Invalid color. Try blue or #2166AC.')
            fig.canvas.draw_idle()
    def save(event):
        if not layers:
            status.set_text('Select at least one layer before exporting.')
        else:
            name = f'{state["y"]}__vs__{state["x"]}__c{data.selection["concurrency"]}__o{data.selection["output_limit"]}__{"-".join(layers)}.pdf'
            path = args.output_dir / name
            try:
                export_pdf(data, sources, state['x'], state['y'], layers, colors, path,
                           limits=(ax.get_xlim(), ax.get_ylim()), trim_percent=state['trim_percent'])
                status.set_text('PDF saved in:\n'+str(args.output_dir.resolve()))
                print(f'Saved {path.resolve()}')
            except Exception as exc:
                status.set_text(f'Export failed: {exc}')
        fig.canvas.draw_idle()
    def update_trim(text):
        try:
            percent = float(text)
            if not np.isfinite(percent) or not 0 <= percent < 100:
                raise ValueError()
            if percent == state['trim_percent']:
                return
            state['trim_percent'] = percent
            status.set_text('Fit updated; exclusion is per precision.\nOriginal observations remain unchanged.')
            redraw()
        except ValueError:
            status.set_text('Enter an exclusion percentage from 0 to less than 100.')
            fig.canvas.draw_idle()
    trim_box.on_submit(update_trim)
    for field in fields:
        field.on_submit(apply_limits)
    auto.on_clicked(lambda event: redraw(reset=True))
    checks.on_clicked(toggle)
    ys.on_clicked(choose_y)
    bcolor.on_submit(lambda c: color_change('bf16', c))
    fcolor.on_submit(lambda c: color_change('fp8', c))
    export.on_clicked(save)
    redraw(reset=True)
    plt.show()

if __name__ == '__main__':
    main()
