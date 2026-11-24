# aten::mm device performance results

Optimization is in progress. The table below is a legacy diagnostic run whose wait loop was eliminated by the compiler. It is not final validation for the current rawpipe source. A corrected wait has been checked at approximately 5.9 ms; row 25 passes the real public entry at 217.65 us. Rows 15, 19 and 20 still require optimization.

The public FlagGems entry passes accuracy for 94/94 target cases. 94/94 reach at least 90% of the spreadsheet DAS baseline; 90/94 reach at least 90% of DAS measured immediately before FlagGems in the same process.

Performance means `DAS device time / optimized device time`. Higher is better. All constituent kernels are included; host submission and synchronization time are excluded.

| Dtype | Cases | Spreadsheet target passed | Paired DAS target passed |
|---|---:|---:|---:|
| torch.float32 | 68 | 68 | 68 |
| torch.float64 | 26 | 26 | 22 |

Cases below the paired DAS target:

| Spreadsheet row | Shape M,N,K | Optimized us | Paired DAS us | Performance |
|---:|---|---:|---:|---:|
| 15 | 192,6528,89 | 24.9440 | 18.2720 | 73.25% |
| 19 | 2208,1280,64 | 34.5600 | 29.5360 | 85.46% |
| 20 | 2208,512,64 | 16.3200 | 14.2880 | 87.55% |
| 25 | 24576,256,368 | 244.7519 | 207.2639 | 84.68% |

Raw per-case timings, five-range samples, kernel names, exact layouts, and selected launches are recorded in `results.json`; `results.csv` is suitable for spreadsheet analysis.

FP32 profile: `paired-fp32-queued-r1` on physical card 0. FP64 profile: `paired-fp64-queued-r1` on physical card 1. Native profiler traces remain under `/public/home/wzx226226/算子优化/profiles/`.

Both providers submit ten mm calls behind the same GPU gate. The gate is launched before each annotated range and excluded from every reported duration. Only the sum of the mm constituent HIPOPS durations is compared. The earlier sequential submission results remain in `sequential-reference/`; their device durations can differ because submission gaps affect GPU execution state even when those gaps are excluded from the kernel sum. These two conditions must not be mixed.
