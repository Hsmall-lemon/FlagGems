> Acceptance withdrawn (2026-11-09). PR remains draft; see [current review status](REPORT.md). The following is historical documentation.

# Hygon aten::mm device optimization

Scope: the 94 `aten::mm` spreadsheet cases assigned to 万 with status
`平均性能<90%`: 68 FP32 and 26 FP64 cases. Shape, dtype, strides and storage
offset are recorded in `cases.json`; `all-mm-cases.json` also contains the
27 mm cases outside the tuning scope for regression checks.

Target: DAS kernel time / optimized kernel time >= 0.90 for every target case.
The final results distinguish the historical spreadsheet DAS time from a
fresh DAS run. Timings exclude host submission and synchronization overhead.

Measurements use native DTK hipprof HIPOPS `DurationNs`. Each synchronized
HIPTX range encloses 10 mm calls; all kernels belonging to one call are
summed, including split-K reduction and input transpose. The reported time
is the median across five ranges. Correlation indices associate kernels with
ranges because host and device trace clocks can differ. `analyze_trace.py`
can merge the multiple JSON trace parts produced by hipprof.

## Environment

- Node: a01r3n18, gfx936; physical cards 0 and 1 only.
- DEV: `/work1/share/scnethpc2601/flagos-dev/20261103-hygon-optimization`.
- Use writable caches under `/public/home/wzx226226/.cache/flagos-dev/`.
- Set HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES to the same physical card.
  Unset ROCR_VISIBLE_DEVICES to avoid filtering the device list twice.
- Import torch_fl before torch and FlagGems, with the checkout's `src` first
  on PYTHONPATH. DAS measurements route only mm to the official boxing
  provider. Candidate measurements call the checkout's flag_gems.mm directly.

FP64 arithmetic retains FP64 operands and accumulators. The gfx936 MMAC
assembly path is guarded by the device architecture. FP32 uses IEEE dot
precision. Offline launch choices depend only on dtype, shape and strides;
there is no runtime tuning or dispatch through the DAS implementation.

Accuracy is checked against CPU FP64 matmul. The per-element bound is
`abs(error) <= atol + rtol * abs(reference)`: FP64 rtol=atol=1e-10;
FP32 rtol=1e-4, atol=2e-4. Additional regression tests check empty dimensions,
zero K, nonzero offsets, two non-unit strides and FP64 mantissa bits that
would be lost in an FP32 conversion.

The public-entry measurements are in [REPORT.md](REPORT.md),
[results.csv](results.csv), and [results.json](results.json). The report
lists the accuracy and performance counts for both DAS baselines, and
explicitly identifies any cases below the target. Final acceptance passes
94/94 accuracy checks, 94/94 spreadsheet targets and 94/94 paired DAS
targets. The minimum paired performance is 91.80%.

The corrected queue gate executes dependent opaque assembly for about
5.9 ms before each ROI. It is excluded from the mm kernel sums. Profiler
audits verify that it survives compilation. Earlier results are preserved
in `sequential-reference/` and `ineffective-gate-reference/`; the latter
used an old wait loop that the compiler eliminated. Host launch gaps are
excluded from the duration sums, although some gaps remain within ROIs.

The development scripts and complete native profiler traces are retained at
`/public/home/wzx226226/算子优化/`. To reproduce a profile from that directory:

```bash
bash run_profile.sh 0 verify-fp32 --queued --mode paired --dtype float32 --output verify-fp32.jsonl
python3 mm_pair_summarize.py verify-fp32
python3 mm_queue_audit.py verify-fp32
bash run_profile.sh 1 verify-fp64 --queued --mode paired --dtype float64 --output verify-fp64.jsonl
python3 mm_pair_summarize.py verify-fp64
python3 mm_queue_audit.py verify-fp64
```

Use a new output name for every run, since measurement JSONL files are append-only.
`validation.json` records the final checks; `validation/` contains the second-seed
regression records and the final entry-point checks. The selected tuning source
for each shape is recorded in `tuning-selection.json`. Selection uses only
native device timings; the five-range public-entry results are the acceptance
measurements, independent of the candidate minima.
