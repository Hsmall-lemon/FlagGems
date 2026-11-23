# Reproduce Hygon FP32 aten::mm acceptance

The suite covers 116 FP32 cases from the model workloads above 5% and scale extension worksheets. [cases.json](cases.json) records the input geometry. [fp32-acceptance-frozen.csv](fp32-acceptance-frozen.csv) maps each case to its worksheet row and tolerance.

## Tolerances

FG and DAS are compared independently against CPU FP32 with identical inputs. Every output element must satisfy `abs(actual - reference) <= atol + 1.3e-6 * abs(reference)` on all five seeds (20261109, 20261210, 20261311, 20261412 and 20261513). The worker reconstructs seeded FP32 tensors using the recorded shapes, strides, storage offsets and aliases. The first seed supplies the timed inputs.

| atol | rtol | Cases | Model worksheet rows | Scale extension worksheet rows |
|---:|---:|---:|---|---|
| 1e-5 | 1.3e-6 | 91 | 56, 57, 58, 59, 62, 64, 65, 66, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 115, 116, 118, 119, 121, 122, 124, 126, 129, 130, 131, 132, 133, 134, 136, 137, 138, 139, 140, 141, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153 | 401, 404, 406, 408, 409, 411, 412, 413, 415, 417, 419, 424, 425, 427, 429, 430, 431, 436, 438, 439, 440, 442, 444, 447, 448, 450, 454, 457, 465, 467, 468, 474, 475, 490, 491, 492, 498 |
| 2e-5 | 1.3e-6 | 5 | 114 | 466, 472, 478, 485 |
| 5e-5 | 1.3e-6 | 15 | 61, 63, 117, 120, 123, 125, 127, 128, 135, 142 | 452, 462, 469, 477, 497 |
| 1e-4 | 1.3e-6 | 5 | 60 | 479, 480, 487, 495 |

## Run the suite

From the PR checkout on a reserved compute node with eight idle gfx936 DCUs, set up a writable run directory and cache.

```bash
export DEV=/work1/share/scnethpc2601/flagos-dev/20261103-hygon-optimization
export MM_RUN_ROOT="${TMPDIR:-/tmp}/flaggems-mm-repro-${USER}"
export FLAGOS_DEV_CACHE="$MM_RUN_ROOT/cache"
bash docs/hygon-optimization/mm/reproduce-fp32-acceptance/reproduce.sh "$PWD"
```

The launcher verifies the checkout against [provider.json](provider.json) and writes one log per card. The worker uses public aten::mm with FlagGems enabled for FG and the DTK boxing configuration for DAS. Each profiler database is removed after its timing records are extracted.

Device time sums kernel and copy durations inside each hipprof timing range and divides by five calls. Each case uses the median across three block medians, with four timing ranges per block. The performance ratio is DAS device time divided by FG device time. The summary marks 90% as the performance target.

In fp32-acceptance-rerun.csv, the rerun_* columns hold the new run's results. Each FG or DAS pass count is out of five seeds. The ratio is a fraction (0.90 means 90%). A joint pass requires five passing FG seeds and a ratio of at least 0.90.

## Final results

Eight gfx936 DCUs on f13r2n12, FlagOS development image 20261103-hygon-optimization, DTK 26.04, Torch-FL with PyTorch 2.10.0+cpu, and Triton 3.6.0.

| atol | Cases | FG accuracy | DAS accuracy | FG accuracy and at least 90% of DAS |
|---|---:|---:|---:|---:|
| 1e-5 | 91 | 91 | 74 | 91 |
| 2e-5 | 5 | 5 | 4 | 5 |
| 5e-5 | 15 | 15 | 8 | 11 |
| 1e-4 | 5 | 5 | 0 | 2 |
| **Total** | **116** | **116** | **86** | **109** |

The final run covered 66 model cases and 50 scale extension cases. The lowest DAS / FG ratio was 51.02% for scale extension row 479. The full measurements are in [results.csv](results.csv).

### Included cases below 90%

| Worksheet | Row | atol | FG seeds | DAS seeds | FG device time (us) | DAS device time (us) | DAS / FG |
|---|---:|---:|---:|---:|---:|---:|---:|
| Model above 5% | 125 | 5e-05 | 5/5 | 5/5 | 12.40 | 11.09 | 89.42% |
| Model above 5% | 135 | 5e-05 | 5/5 | 0/5 | 24.83 | 22.11 | 89.05% |
| Scale extension | 469 | 5e-05 | 5/5 | 5/5 | 9.76 | 7.97 | 81.64% |
| Scale extension | 477 | 5e-05 | 5/5 | 1/5 | 93.81 | 83.78 | 89.31% |
| Scale extension | 479 | 0.0001 | 5/5 | 0/5 | 95.58 | 48.77 | 51.02% |
| Scale extension | 480 | 0.0001 | 5/5 | 0/5 | 78.13 | 68.91 | 88.20% |
| Scale extension | 495 | 0.0001 | 5/5 | 2/5 | 135.17 | 116.56 | 86.23% |

Scale extension row 479 used atol=1e-4 and rtol=1.3e-6. FG passed all five seeds. Its 51.02% DAS / FG ratio fell below the separate 60% performance target.
