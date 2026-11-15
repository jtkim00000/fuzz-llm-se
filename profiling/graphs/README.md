# Interactive request scatter plots

Launch from the repository root using the existing environment:

```sh
.venv-aiperf/bin/python profiling/graphs/request_scatter.py
```

The right panel lets you select either or both precisions, choose one of four input axes and five output axes, and edit colors (name or hex, then press Enter). Click **Export graph PDF** to save only the graph, without the control panel, in `profiling/graphs/exports`. The exported filename identifies the axes and selected precisions; repeating the same export overwrites that file. The terminal prints its full path.

Twenty graph-only PDFs are already generated in `exports`, covering every input/output combination with both precisions. Each has a JSON sidecar recording its source, units through field names/labels, selected precisions, colors and point counts.

## Inputs and reproducibility

The default source is the matched 100-request BF16/FP8 result folder. If that folder is absent from your current checkout, the script reads the canonical records directly from the locally fetched `origin/profiling/bf16-fp8-request-records` Git ref. It does not fetch, switch branches, modify results or launch inference jobs. Git-sourced exports record the resolved commit. An existing local result folder takes precedence.

You can replace both inputs explicitly:

```sh
.venv-aiperf/bin/python profiling/graphs/request_scatter.py \
  --bf16 /path/to/bf16/canonical-requests.jsonl \
  --fp8 /path/to/fp8/canonical-requests.jsonl \
  --bf16-color '#2166AC' --fp8-color '#D6604D' \
  --precisions bf16 fp8 --output-dir /path/to/plots
```

Use `--precisions bf16` or `--precisions fp8` to initially display just one. The checkboxes can change that after launch. `--x` and `--y` accept canonical field names; see `--help`. `--results-dir` accepts a directory containing `bf16/measurement/canonical-requests.jsonl` and `fp8/measurement/canonical-requests.jsonl`.

To regenerate all 20 PDFs without opening the UI:

```sh
.venv-aiperf/bin/python profiling/graphs/request_scatter.py --export-all
```

## Interpretation

- Every finite X/Y pair is plotted, without aggregation, fitting or jitter. Legend counts show plotted rows / total attempts. Missing values are omitted pairwise, never replaced with zero. Both precision files currently contain 100 successful attempts each.
- Latency and TTFT are converted from milliseconds to seconds.
- Dispatch and completion timestamps are converted to seconds since the first dispatch **within each run**. Completion time uses that same dispatch origin, preserving duration. These axes show scheduling/progress, not per-request runtime; select end-to-end latency for runtime. Absolute UTC timestamps remain in source files.
- Running/waiting counts are pre-dispatch samples, not instantaneous measurements. Sample ages can reach about a second in these runs.
- All output limits equal 128, so those plots are vertical stacks. Integer load counts also overlap. Transparency and different marker shapes help, but coincident points can obscure one another.
- The input range is narrow (5,889–6,200 tokens). These graphs do not establish broad context-length scaling or causality.
- Precision labels identify deployed configurations; attention backends and physical workers differ. Quality checks passed only 3% in each run and require investigation separately.

Dependencies: Python 3.10+, NumPy and Matplotlib. The existing `.venv-aiperf` was used here (NumPy 2.4.6, Matplotlib 3.11.2). For a separate environment install `requirements.txt`. Interactive use requires a working desktop Matplotlib backend; `--export-all` uses the noninteractive Agg backend.
