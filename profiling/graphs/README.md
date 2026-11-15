# Input-token scatter viewer

From the repository root:

```sh
.venv-aiperf/bin/python profiling/graphs/request_scatter.py
```

X is fixed to **pre-dispatch input tokens**. Choose one of five Y variables:
end-to-end latency, TTFT, actual output tokens, dispatch time, or completion time.
Four independent checkboxes control BF16 scatter/regression and FP8
scatter/regression. Edit color names or hex strings and press Enter.

## Controls

- **Exclude outliers (%)**: fit a baseline line separately for each precision,
  rank absolute vertical residuals, remove the largest `floor(n * percent / 100)`,
  then refit once. Excluded points disappear from BOTH scatter and regression,
  including exported PDFs. Zero restores all finite observations. This does not
  delete or modify source data. Ties follow finite-pair source row order.
- **X min/max, Y min/max**: enter finite numeric bounds with min < max and press
  Enter. **Auto axes** chooses bounds for the currently visible retained data.
  Changing Y or layer visibility resets bounds; colors and exclusion edits preserve
  the current bounds. Click Auto axes after exclusion if you want a tighter view.
- Side panel: retained/excluded counts, slope, intercept, retained-set R²/RMSE and
  full-data R². Linear fits need at least three retained points and two distinct X
  values. Scatter visibility and zoom do not affect the observations used to fit.
- **Export graph PDF**: saves only the graph and simple layer legend, with current
  visibility, exclusion, colors and bounds. Statistics stay out of the PDF and are
  retained in its JSON sidecar, along with source references and exclusion indices.

Exclusion is exploratory outcome-based selection, not improved predictive accuracy.
Retain and report the full-data baseline, exclusion method and percentage when
sharing a trimmed analysis. Slow requests may reflect real serving behavior.

## Data sources and command-line options

Default data: `profiling/results/vast-full-bf16-fp8-2026-11-15`, with
144 requests per precision (72 in c1 and 72 in c4).
If absent from the checkout, the script reads the committed records directly from
`origin/profiling/bf16-fp8-request-records` without checkout or network access.
The resolved commit is saved for Git-sourced exports. Local files take precedence.

```sh
.venv-aiperf/bin/python profiling/graphs/request_scatter.py \
  --bf16 /path/to/bf16/canonical-requests.jsonl \
  --fp8 /path/to/fp8/canonical-requests.jsonl \
  --layers bf16_scatter bf16_regression fp8_regression \
  --trim-percent 5 --output-dir /path/to/plots
```

`--results-dir` accepts a parent containing `bf16/c1-r1/canonical-requests.jsonl` through `bf16/c1-r3/canonical-requests.jsonl`,
and `bf16/c4-r1/canonical-requests.jsonl` through `bf16/c4-r3/canonical-requests.jsonl` and the corresponding FP8 paths. Explicit JSONL
inputs need a concurrency field or a run_id containing c1/c4. `--y` selects a canonical output field. `--regression`
enables linear lines alongside scatter; explicit `--layers` overrides it.
`--bf16-color` and `--fp8-color` accept color names/hex. There is no X-variable
selector or `--x` option now.

```sh
.venv-aiperf/bin/python profiling/graphs/request_scatter.py --export-all --regression
```

Batch export now generates **five** input-token plots, one per output. Previously
created 20-plot exports are left untouched; old files for other X variables are
historical. Repeated exports to the same filename overwrite that file.

## Timing, interpretation and responsiveness

Latency and TTFT are seconds; source milliseconds are converted. Dispatch and
completion times are seconds since each run's first dispatch, not runtimes.
Missing/nonfinite pairs are omitted rather than changed to zero. The current
input range is narrow and these single-variable fits do not establish causality.

Parsed arrays and fits are cached per loaded dataset. Plot artists are reused,
axis text synchronization avoids redundant immediate renders, and the legend uses
a fixed position. This reduces repeated computation and redraw overhead. Restart
or reload the script to use modified input files; there is no live file watcher.

Dependencies: Python 3.10+, NumPy, Matplotlib (see requirements.txt). The existing
`.venv-aiperf` environment supports this script. Interactive operation requires a
desktop Matplotlib backend; batch export uses Agg. Original profiling results are
never modified by the viewer.

## Full campaign filters

The default view is **concurrency 1, output limit 256**, with 36 points per
precision. Above the graph, type concurrency `1`, `4`, or `all`, and output limit
`32`, `256`, or `all`, then press Enter. Every combination is available, including
all 144 points per precision. Compare matching conditions first; pooling output
lengths/concurrencies can obscure input-length relationships. The plot title and
PDF sidecar record the active selection; filenames include it.

Command-line equivalents: `--concurrency 1 --output-limit 256`. Batch export generates
five output plots for that selection, not every condition. Filters reset axis
ranges; all earlier color, layer, trimming and PDF controls remain available.

The full campaign has 36 points per precision for each concurrency/output condition,
covering four length targets, three examples and three repetitions (nine observations
per length cell). Repetitions are pooled by these filters, and timestamp origins remain
separate for each run. Prefix caching remains enabled and unverified; this is not a
cold-cache or quality benchmark. Trimming defaults to zero.

The update was checked against branch commit `1d102e0`: 288 successful records,
matched prompt traces and per-ID token counts, exact 32/256 outputs, complete ledger
and raw-record joins, and pre-dispatch telemetry linkage. Maximum sample age was
approximately 244 ms. At c1/output 256, untrimmed linear TTFT R² was 0.9969 (BF16)
and 0.9993 (FP8); E2E R² was 0.9711 and 0.9859. These are in-sample descriptions,
not held-out prediction scores. Loaded c4 relationships are substantially weaker.

## Original dataset viewer (separate script)

```sh
.venv-aiperf/bin/python profiling/graphs/request_scatter_original.py
```

Restores the first working viewer from commit `15e6548`, with four selectable
input axes, five output axes, BF16/FP8 visibility and color controls, and graph-only
PDF export. It loads the historical 100-request-per-precision dataset now stored
under `deprecated-direct-node-bf16-fp8-vast-reactome-c4-100-2026-11-14`, using the
same Git fallback as before. Exports go into `exports-original`, separate from the
new campaign. `--export-all` generates all 20 input/output combinations.

The original viewer includes independent BF16/FP8 regression toggles and fits spanning the visible graph,
and slope/intercept/R² below the graph. X/Y range sliders change the visible limits
without recalculating fits; Auto axes restores the default bounds. PDF exports
preserve those bounds and the regression setting, excluding controls and fit statistics.
There are no axis text fields or outlier trimming. The current `request_scatter.py`
remains unchanged. The old dataset
has more variable natural output lengths, but a NARROWER input-token range
(5,889–6,200), a fixed 128-token cap, and concurrency 4. The new campaign controls
output length and varies inputs more broadly. Comparing these shows workload
characteristics, not a controlled old-versus-new speedup. The branch marks the old
data deprecated for active training; this viewer uses it only for historical analysis.

The original viewer’s **Legend** checkbox shows or hides the color guide without
changing the axes or fitted lines. PDF exports preserve this setting.
