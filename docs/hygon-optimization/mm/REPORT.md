# aten::mm: candidate passes reconstructed-metadata acceptance


## 2026-11-11 final candidate check

The candidate passes all **197 MM rows** in the two requested workbook sheets
and preserves the previously qualified **179 records**, including added controls.
The minimum workbook DAS/new FG ratio is **90.1874%**. Each measurement uses
one isolated physical GPU; host overhead is excluded.

Fresh work was partitioned across three devices. FP64 `[1344,256] @ [256,128]`
passes at B offsets 0 and 131072, with identical logical values and DAS/new
**95.2640% / 95.4410%**. The FP32 transposed-B case passes at **110.2377%** and
the BF16 control at **110.2719%**. Both DAS and new FG pass the 5% A/A guard.
The first unstable FP32 capture is retained; the accepted capture prewarms all
providers before all three timed blocks. No observations are dropped.

The current source passes **410 pytest cases** plus **48 distinct precision
stress cases**. No selected configuration uses the reported `mm_mmac_peel` path.
All device kernels and copies, including allocator initialization copies, count
in the metric. Two independent analyzers and a fresh CPU read of all 225 native
ROIs agree. The source and complete workbook mapping are checked against the
candidate, and earlier evidence is used only where routes and device kernels
are preserved or replaced by a newer accepted capture.

See [final candidate evidence and limitations](recheck-20261111/README.md) and
run `python docs/hygon-optimization/mm/recheck-20261111/verify_evidence.py`.

**Scope limitation:** original reviewer tensors were not supplied and were not
replayed. The candidate bypasses the reported path and passes reconstructed
metadata and regression tests; it does not claim reproduction or root-cause
resolution of the original report, or whole-model acceptance. PR #2 stays Draft
for external review. The dated historical holds and failures below are preserved
as historical records; the candidate check above does not rewrite them.

## Historical evidence (superseded candidate status)

## 2026-11-10 workbook scope extension: metadata reconstruction checks passed

All 197 workbook MM rows pass the current reconstructed-input numerical, device
performance, and A/A checks: 98 model-workload rows and 99 scale-extension rows.
The minimum DAS/new FG ratio is 90.19%. Measurements sum all device kernels
and copies; host submission and queue gates are excluded. Each process sees one
physical GPU. Both providers require A/A drift <=5%.

The final FP32 `[905,1159] @ [1159,1159]` input now uses an IEEE FP32
global/LDS/MMAC pipeline. Physical GPU 0 records 68.92984 / 74.63149 us
(DAS/new FG = 92.36%); physical GPU 1 records 69.02212 / 74.55875 us (92.57%).
Both captures pass numerical and stability checks independently. Two-GPU
aggregate throughput is never used as an operator measurement.

The v16 candidate retains 175 exact launch keys. Relative to v14, two FP32
configurations differ; the other 173 configurations and all earlier kernel
sources are unchanged. Relative to v15, only the strided FP32 launch config
changes: K tile 16 and unroll 4. The dispatcher and all kernels are byte-identical.
The two v14 K=1 FP64 streaming-store revisions remain qualified. The candidate
passes 410 numerical tests on each physical card, including 40 new native FP32
precision and boundary regressions. Four required static operator checks, full pre-commit, and explicit checks
of all new files pass.

The corrected strided FP32 `[10114,104] @ [104,207]` input passes on both cards:

- Physical GPU 0: DAS 23.53874 us / new FG 24.75749 us = 95.08%; DAS A/A 0.84%, new A/A 0.29%.
- Physical GPU 1: DAS 23.63375 us / new FG 24.71250 us = 95.63%; DAS A/A 0.35%, new A/A 0.41%.

The prior 89.27% cross-card failure remains in the evidence.

Portable evidence covers all rows and preserves historical failures:
`extended-20261110/verify_evidence.py` recomputes medians, A/A drift, per-ROI
kernel/copy sums, source hashes, and route preservation using only the CPU.
Canonical source comparisons use LF newlines; raw capture hashes remain recorded.

The exact original reviewer tensors remain unavailable. Reconstruction tests do
not establish original-input replay or whole-model acceptance. PR #2 remains
Draft and the combined submission gate remains false. The historical sections
below retain their original scope and limitations.

<!-- current-submission-r4n15 -->
Current known-scope review: [179 records, compact observations and independent verifier](correction-20261109/submission-r4n15/README.md).
Overall submission remains blocked; the original MM review failures are not closed.
The expanded reconstructed-input performance checks above are complete; original-operand acceptance remains unverified.
The historical review and separately recorded phases follow below.

PR #2 is a draft. The previous claim that all target cases were accepted is
withdrawn. Historical traces and test counts do not establish current correctness
or performance acceptance. The review feedback below is supplied by the user;
it is not presented as a new measurement by this checkout.

| Input | Review result | Acceptance |
|---|---|---|
| FP32 [60025,128] x [128,128], transposed B | Old FG, new FG and DAS each fail 1/7,683,200 elements | Correctness unresolved; performance not accepted |
| FP64 [1344,256] x [256,128], B offset 131072 | New FG fails 172,032/172,032 outputs; old FG and DAS pass; old 82.08 us | Severe new correctness regression; blocks merge |
| Same FP64 logical inputs, B offset 0 | New FG still fails every output; old 81.92 us | Offset is not the sole cause; blocks merge |
| FP64 [192,128] x [128,128] | 44.64 -> 8.16 us; DAS/new 105.88% | Review reports accuracy and stability pass |
| FP64 [282624,15] x [15,3] | 754.80 -> 44.08 us; DAS/new 103.81% | Review reports accuracy and stability pass |
| FP32 [28852,1] x [1,32] | 14.56 -> 4.00 us; DAS/new 174.00% | Review reports accuracy and stability pass |
| FP32 [6432,3] x [3,9] | 8.32 -> 2.72 us; DAS/new 147.06% | Review reports accuracy and stability pass |
| BF16 [32,64] x [64,32], fallback control | Accuracy passes; A/A max drift 25.64%; 11.36 -> 10.88 us diagnostic | Performance not accepted |

The FP64 failure is associated with the `mm_mmac_peel` path and the current
compiler stack. A source-only or compiler-only root cause has not been proved.
The reported failure was reproduced by the reviewer on two nodes. Passing a
different input or environment does not resolve that report. Do not relax
accuracy tolerances to convert a failure into acceptance.

The current correction replaces all three selected `peel` configurations with
the existing explicit-layout `swizzle` kernel, pipelined loads, prefetch 8 and
two warps. It bypasses the reported compiler-sensitive path without asserting
a source-only or compiler-only root cause. The original failure has not been
reproduced in this isolated environment, and is NOT marked resolved. The exact
reviewer input, test entry point and environment still need to be matched.

The second correction tunes only spreadsheet rows 20 and 32. Row 20 uses a
swapped explicit-layout swizzle launch with four warps; row 32 uses the existing
pipelined two-warp swizzle configuration. Both shapes are added to the
multi-seed, offset and low-mantissa-bit regressions. On a01r3n13 card 0 this
source passes **33 focused tests and 30 public-route checks**. The 19 targeted
spreadsheet cases pass both generated seeds (38 accuracy records). With a new,
separately recorded 200-call/ROI protocol, rows 20 and 32 reach **91.49% and
103.52%** DAS/new, with both sides below the 5% A/A guard. Overall, **11/19**
targeted groups pass this protocol; rows 9, 81, 82, 101, 107, 108, 111 and 114
remain unstable. These results do not establish acceptance of all 94 targets.
The separate BF16 fallback control also fails DAS stability and is diagnostic.

See [second-correction status](correction-20261109/phase2-long/status.json),
[all targeted samples](correction-20261109/phase2-long/device-samples.json),
[source identity](correction-20261109/phase2-long/source-proof.json) and
[original spreadsheet mapping](correction-20261109/phase2-long/spreadsheet-scope-audit.json).
The original FP32 and FP64 review failures remain unresolved. A separate
a01r4n15 run uses physical cards 0 and 1 in independent processes; every
operator itself runs on one card. The completed results are recorded below.
Formatting, full pre-commit and the required MM static checks pass.

On a01r4n15, the independent PR #2 source passes 33 focused tests and 30 public
route checks. Full-scope generated accuracy covers all 120 FP32/FP64 cases and
two seeds: **240/240 records pass** for DAS and new FG under the unchanged
recorded tolerances. The full 200-call/ROI performance run passes both A/A and
the 90% target for **44/94** groups; 50 fail stability. Those failures remain
in the evidence. A separately recorded 1024-call/ROI protocol passes all 22
preselected focus cases, and then all 36 uncovered unstable cases split across
the two cards. In total, **94/94 target groups have a passing device-performance
record**, using 36 groups from the 200-call protocol and 58 from the 1024-call
protocol. This is coverage across different burst lengths and cards, not one
uniform full-scope run. Source hashes match the same production implementation.
Generated inputs still do not close the original reviewer failures, and the PR
remains a draft.

See [complete cross-node classification](correction-20261109/crossnode-r4n15/status.json),
[retained full 200-call results](correction-20261109/crossnode-r4n15/device-samples-full-200.json),
[focused 1024-call results](correction-20261109/crossnode-r4n15/device-samples-focused-1024.json),
[card-0 remaining results](correction-20261109/crossnode-r4n15/device-samples-remaining-1024-card0.json),
[card-1 remaining results](correction-20261109/crossnode-r4n15/device-samples-remaining-1024-card1.json)
and [all measured source proofs](correction-20261109/crossnode-r4n15/source-proof.json).

The first-correction source on a01r3n13 visible card 0 passed 25 focused tests.
Eighteen public `torch.mm` checks cover three seeds and B offsets 0/131072 for
each former peel case. Every call witnesses one FG export and one explicit-layout
kernel launch, and checks every output against CPU FP64 at atol=rtol=1e-10.
The original 42-check diagnostic did not reproduce the old peel failure either;
its passing result is not evidence that the reviewer report was incorrect.

The expanded scope run checks all 120 FP32/FP64 spreadsheet cases with seeds
100003 and 880301. All 240 case/seed records pass for DAS and the new FG export
under the recorded tolerances (FP64 atol=rtol=1e-10; FP32 atol=2e-4, rtol=1e-4).
These generated inputs do not resolve the reviewer's shared FP32 1-element failure.

In that first-correction run, only **75/94** target groups passed both the 90% paired DAS threshold and the
newly declared 5% A/A guard in the single-compute-unit gate protocol. Rows 20
and 32 remain below 90%; 17 further rows fail stability for FG and/or DAS. They
are diagnostic, not accepted speedups. The three replacement configurations
separately measured DAS/new at 96.00%, 102.18% and 107.17%, with FG A/A drift
below 0.3%, in the confirmation run. Conditions differ from the user review
table; these measurements must not be combined with it.

See [first-correction status](correction-20261109/revalidation-status.json),
[device samples](correction-20261109/device-samples-single-cu.json),
[public route checks](correction-20261109/public-route-checks.json) and
[first-correction source proof](correction-20261109/source-proof.json). The scope performance run
applies both PR source patches; the focused tests and public-route checks also
run against PR #2 independently. The PR remains a draft while the original
failure report and outstanding performance/stability cases remain unresolved.

For all new acceptance, require full value/layout checks before performance,
actual kernel/route evidence, matching valid DAS semantics, same-device paired
measurements and independent A/A blocks. Sum all constituent device kernel
durations and exclude host and gate time. Unstable timings remain diagnostic.
The identity/no-kernel case must not be assigned a speedup ratio.

Original reports are preserved in `historical/`. Existing raw result and
validation documents are historical snapshots, explicitly marked as superseded.
Source/hash/AST checks prove source identity only, not runtime correctness.

## Added regression inputs included in the device-performance target

The added scope includes 54 seed/layout records: thirty FP64 seed/offset
regressions, eight exact low-mantissa-bit inputs, four both-strided inputs,
six empty-dimension records and six BF16 small-matrix seed/offset controls.
The final source adds a 32x32 BF16 tiled launch and two exact both-strided
reduction configurations. It passes 39 focused tests. All 94 prior exact
configurations and eleven device-source files remain unchanged.

Current paired coverage passes **50/50 device-work records**, with
minimum DAS/new **90.71%**. Four M/N-empty
outputs launch no device work; the two K=0 nonempty outputs are measured.
Coverage combines thirty initial-source FP64 seed/offset records with sixteen
final-source layout/BF16/empty records and eight final-source low-bit records.
It is not a single uniform final-source run. The initial four strided failures
and case 930037 DAS A/A failure remain in the evidence. Original reviewer
failures remain unmatched and the PR stays Draft.

See [case-by-case current and retained results](correction-20261109/expanded-r4n15/status.json),
[unchanged original launches](correction-20261109/expanded-r4n15/unchanged-original-launches.json)
and [actual measured source proof](correction-20261109/expanded-r4n15/source-proof.json).
