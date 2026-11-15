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

Default data: matched 100-request BF16/FP8 result folder under `profiling/results`.
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

`--results-dir` accepts a parent containing `bf16/measurement/canonical-requests.jsonl`
and the corresponding FP8 path. `--y` selects a canonical output field. `--regression`
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
