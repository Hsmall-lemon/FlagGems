# Reproduce Hygon aten::to.dtype device results

Run on an idle gfx936 DCU with the FlagOS development image
`20261103-hygon-optimization` and DTK 26.04. The production backend discovers
`hipcc` on PATH and caches the compiled packed conversion in a writable cache.
The checkout must contain the production dispatch being measured.

```bash
work=$(mktemp -d "${TMPDIR:-/tmp}/dtype-production.XXXXXXXX")
export FLAGOS_DEV_CACHE="$work/cache"
scripts=docs/hygon-optimization/to_dtype/reproduce-production-cast
bash "$scripts/run.sh" "$PWD" "$work/model" 0 model
bash "$scripts/run.sh" "$PWD" "$work/scale" 0 scale
python3 "$scripts/summarize.py" "$work"
```

The third argument selects a physical card. Each suite restricts visibility to
one card. The runner retains the application logs and timing summaries and
removes the raw profiler output after successful analysis.

The FG worker calls public `torch.ops.aten.to.dtype` with the regular
FlagGems registration enabled. The DAS reference uses the development image's
boxing conversion on the same device input. The report checks the profiled
kernel names for the ten tuned cases.

Each provider is compared independently with CPU ATen on the same seeded
input. The test checks exact values, non-NaN floating-point bits, NaN positions,
dtype, stride, device, and aliasing. The 35 inputs and their storage geometry are
in [cases.json](cases.json).

The score uses device-side hipprof timings. Model cases measure 25 ranges of
20 calls per group, including kernel and device-copy durations in each range.
Scaling cases warm up for 100 calls and measure five ranges of 1,000 calls per
group, using the mean paired kernel-duration ratio. The lowest of three rotated
groups is compared with the 90% target. Rows where DAS fails correctness retain
their timing data and carry a separate status.

[results.csv](results.csv) records every workbook row, both providers'
correctness, their device times in microseconds, the profiled FG kernel, and
the three DAS/FG percentages. Zero-device-work rows 776 and 1068 return the
input tensor. DAS returns a different stride from CPU ATen for rows 772, 773,
and 1059–1064. Their measured ratios are included in the CSV.
