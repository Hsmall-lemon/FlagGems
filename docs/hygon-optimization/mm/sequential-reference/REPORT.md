# aten::mm device performance results

The public FlagGems entry passes accuracy for 94/94 target cases. 94/94 reach at least 90% of the spreadsheet DAS baseline; 88/94 reach at least 90% of DAS measured immediately before FlagGems in the same process.

Performance means `DAS device time / optimized device time`. Higher is better. All constituent kernels are included; host submission and synchronization time are excluded.

| Dtype | Cases | Spreadsheet target passed | Paired DAS target passed |
|---|---:|---:|---:|
| torch.float32 | 68 | 68 | 67 |
| torch.float64 | 26 | 26 | 21 |

Cases below the paired DAS target:

| Spreadsheet row | Shape M,N,K | Optimized us | Paired DAS us | Performance |
|---:|---|---:|---:|---:|
| 4 | 1344,128,256 | 17.9201 | 15.1840 | 84.73% |
| 15 | 192,6528,89 | 24.3520 | 18.2880 | 75.10% |
| 16 | 192,768,89 | 9.5360 | 8.3040 | 87.08% |
| 19 | 2208,1280,64 | 38.8000 | 29.5040 | 76.04% |
| 25 | 24576,256,368 | 244.6722 | 206.4322 | 84.37% |
| 61 | 1024,256,217 | 12.7200 | 11.0240 | 86.67% |

Raw per-case timings, five-range samples, kernel names, exact layouts, and selected launches are recorded in `results.json`; `results.csv` is suitable for spreadsheet analysis.

FP32 profile: `paired-fp32-r1` on physical card 0. FP64 profile: `paired-fp64-r2` on physical card 1. Native profiler traces remain under `/public/home/wzx226226/算子优化/profiles/`.

Final regression: 120/120 FP32 and FP64 spreadsheet cases pass a second seed (100003). The ATen dispatcher, direct entry, and experimental matrix installer pass FP32/FP64 and BF16 fallback checks. The focused GPU suite passes 15 tests (3 deselected), source inventory passes 8 tests, and the four required static checks and pre-commit checks pass. See `validation.json`.
