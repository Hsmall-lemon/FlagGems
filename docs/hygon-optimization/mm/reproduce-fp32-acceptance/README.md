# Reproduce Hygon FP32 aten::mm acceptance

The suite covers 115 FP32 cases from the model workloads above 5% and scale extension worksheets. [cases.json](cases.json) records the input geometry. [fp32-acceptance-frozen.csv](fp32-acceptance-frozen.csv) maps each case to its worksheet row and tolerance.

## Tolerances

FG and DAS are compared independently against CPU FP32 with identical inputs. Every output element must satisfy `abs(actual - reference) <= atol + 1.3e-6 * abs(reference)` on all five seeds (20261109, 20261210, 20261311, 20261412 and 20261513). The worker reconstructs seeded FP32 tensors using the recorded shapes, strides, storage offsets and aliases. The first seed supplies the timed inputs.

| atol | Cases | Model worksheet rows | Scale extension worksheet rows |
|---|---:|---|---|
| 1e-5 | 91 | See the per-case CSV | See the per-case CSV |
| 2e-5 | 5 | 114 | 466, 472, 478, 485 |
| 5e-5 | 15 | 61, 63, 117, 120, 123, 125, 127, 128, 135, 142 | 452, 462, 469, 477, 497 |
| 1e-4 | 4 | 60 | 480, 487, 495 |

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
| 5e-5 | 15 | 15 | 8 | 14 |
| 1e-4 | 4 | 4 | 0 | 2 |
| **Total** | **115** | **115** | **86** | **112** |

The final run covered 66 model cases and 49 scale extension cases. The lowest DAS / FG ratio was 79.08% for scale extension row 469. The full measurements are in [results.csv](results.csv).

### Included cases below 90%

| Worksheet | Row | atol | FG seeds | DAS seeds | FG device time (us) | DAS device time (us) | DAS / FG |
|---|---:|---:|---:|---:|---:|---:|---:|
| Scale extension | 469 | 5e-05 | 5/5 | 5/5 | 9.02 | 7.14 | 79.08% |
| Scale extension | 480 | 0.0001 | 5/5 | 0/5 | 77.12 | 68.00 | 88.17% |
| Scale extension | 495 | 0.0001 | 5/5 | 2/5 | 135.87 | 120.86 | 88.95% |

Scale extension row 479 is outside this PR's dispatch scope. Its previous full run passed FG accuracy on five seeds and measured 50.31% of DAS at atol=1e-4, below its 60% target.
