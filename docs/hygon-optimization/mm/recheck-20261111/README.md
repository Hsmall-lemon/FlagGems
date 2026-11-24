# Final candidate check, 2026-11-11

The candidate passes the declared **reconstructed metadata** acceptance scope:
197 MM workbook rows (98 model workloads and 99 scale extensions), with the
previous 179 records and added controls preserved. The minimum workbook
DAS/new FG ratio is **90.1874%**. Device latency is measured independently on
one GPU per process; simultaneous GPU throughput is never used as latency.

The workbook SHA256 is
`d2596052c66d7145af47b35fda45198d1f213b69825aeb91d10112dfd8871597`.
Both requested sheets, `大于5%的模型工况` and `规模扩展工况`, were reread and
all 197 MM rows matched exactly, including dtype, strides and storage offsets.

The original reviewer operands are unavailable. This package does not claim
original-tensor replay, reproduction of the 100% FP64 failure, a compiler/source
root cause, or whole-model acceptance. The candidate bypasses the reported
`mm_mmac_peel` path: none of the 175 selected configurations uses it. The
historical failure reports and failed measurements remain in the repository.
PR #2 stays Draft for external review.

## Fresh checks partitioned across three devices

| Input | DAS (us) | New public FG (us) | DAS/new FG | DAS / new A/A drift |
|---|---:|---:|---:|---:|
| FP64 `[1344,256] @ [256,128]`, B offset 0 | 13.80594 | 14.49230 | 95.2640% | 0.2620% / 0.0227% |
| Same FP64 values, B offset 131072 | 13.81456 | 14.47446 | 95.4410% | 0.0579% / 0.0822% |
| FP32 `[60025,128] @ [128,128]`, transposed B | 80.31526 | 72.85644 | 110.2377% | 1.1136% / 0.0517% |
| BF16 `[32,64] @ [64,32]` | 5.33578 | 4.83875 | 110.2719% | 0.6152% / 0.6036% |

FP64 ran on a01r3n13; FP32 and BF16 ran on separate GPUs 0 and 1 on f11r2n17.
Each process saw one device and recorded its PCI identity and imported source
hashes. Native HIP DAS shared the exact physical input storage, logical values,
sizes, strides and offset used by new FG. The new path was called through public
`torch.mm` with `use_gems`, with a configuration-selection witness. Each capture
has three A/A blocks of five ROIs per provider. Both DAS and new FG require drift
at most 5% and numerical checks under unchanged tolerances.

48 distinct stress cases passed across the three categories: four random seeds,
two offsets, basis inputs, low mantissa bits, dyadic scaling and cancellation.
References used CPU FP64 accumulation, rounded to BF16 for BF16 output. The
separate current-source pytest runs passed **97 + 313 = 410** numerical tests.

The first FP32 capture failed DAS stability (29.095% drift) and is retained in
full. A second capture prewarmed all three providers and their complete output
pools, retaining the 1 GiB output budget; its entire three-block result passed.
The first FP64 launch with an incorrect ROCR device ordinal saw no device and
produced no numerical/performance observations. It is not counted as a pass.

Old public FG was also captured. Its BF16 A/A drift was 28.893%, so this package
does not claim an accepted BF16 speedup against old FG. DAS/new acceptance is
computed from the independently stable DAS and new FG groups.

## Evidence and verification

`device-observations.json.gz` contains every fresh ROI, all stress records,
source fingerprints, pytest logs, native archive hashes, and every kernel/copy
DurationNs in the 225 ROIs, including the failed first FP32 run. Both the native
SQLite and official profiler JSON analyzers agreed. The native databases were
compressed, hash checked and independently reread on CPU before packaging;
their 4 compressed archives are backed up outside Git.

The metric sums **all device kernels and copies**, including irregular one-time
allocator initialization copies inside an ROI, divided by actual calls. Host
submission and the queue gate are excluded. Copy counts may be fractional per
call; no copy duration is discarded to force an integer count.

Run from any directory with CPU-only Python 3.9 or later:

```sh
python docs/hygon-optimization/mm/recheck-20261111/verify_evidence.py
```

The verifier checks all 197 workbook observations, all 179 historical records,
exact current source hashes, old route/kernel preservation, the two changed
K=1 keys against their newer accepted workbook captures, the 410 numerical test
records, all 225 fresh ROI sums, three-block medians, both A/A guards and the
90% target. The prior verifiers still print their historical submission hold;
this new package is an additional candidate check, not a rewrite of history.
The evidence combines independently qualified captures and preserved routes;
it is not one uniform final run of every record on every GPU.

The captured harnesses are stored as `.py.txt` to preserve their exact source.
`baseline119` and `historical-source-proof.json` allow the previous dispatcher,
configurations and shared device kernels to be checked without a GPU.
