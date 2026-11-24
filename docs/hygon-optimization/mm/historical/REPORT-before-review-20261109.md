> SUPERSEDED historical report. See ../REPORT.md for current review status.

# aten::mm device performance results

The public FlagGems entry passes accuracy for 94/94 target cases. 94/94 reach at least 90% of the spreadsheet DAS baseline; 94/94 reach at least 90% of DAS measured immediately before FlagGems in the same process.

Performance means `DAS device time / optimized device time`. Higher is better. All constituent kernels are included; host submission and synchronization time are excluded.

| Dtype | Cases | Spreadsheet target passed | Paired DAS target passed |
|---|---:|---:|---:|
| torch.float32 | 68 | 68 | 68 |
| torch.float64 | 26 | 26 | 26 |

Raw per-case timings, five-range samples, kernel names, exact layouts, and selected launches are recorded in `results.json`; `results.csv` is suitable for spreadsheet analysis.

FP32 profile: `paired-fp32-gated-r1` on physical card 0. FP64 profile: `paired-fp64-gated-r2` on physical card 1. Native profiler traces remain under `/public/home/wzx226226/算子优化/profiles/`.

Both providers submit ten mm calls behind the same GPU gate. The gate is launched before each annotated range and excluded from every reported duration. Only the sum of the mm constituent HIPOPS durations is compared. The earlier sequential submission results remain in `sequential-reference/`; their device durations can differ because submission gaps affect GPU execution state even when those gaps are excluded from the kernel sum. The earlier ineffective gate results remain in `ineffective-gate-reference/`: the compiler eliminated that old wait loop. The corrected gate is verified by profiler durations above 1 ms. Some ROI launch gaps remain; they are excluded from HIPOPS sums, and each result uses the median of five ranges. These conditions must not be mixed.

Final regression: 120/120 FP32 and FP64 spreadsheet cases pass a second seed (100003). The ATen dispatcher, direct entry, and experimental matrix installer pass FP32/FP64 and BF16 fallback checks. The focused GPU suite passes 17 tests (3 deselected), source inventory passes 8 tests, and the four required static checks and pre-commit checks pass. See `validation.json`.

A fresh process with `mm = flaggems` witnesses exactly one optimized mm export call for each FP32 and FP64 ATen mm invocation. See `validation/mm-route-witness.json`.

Source hashes in `source_sha256` and `validation/remote-source-sha256.txt` describe the raw measured files. Most used CRLF; Git stores LF text. `validation/source-representations.json` verifies all 15 sources are byte-for-byte identical to the committed Git blobs after CRLF-to-LF normalization, and `committed_source_sha256_lf` records the portable committed-file hashes. No executable code changed.
