# aten::to.dtype / _to_copy: current device acceptance

## 2026-11-11: complete dtype scope and independent old-path comparisons

The [current evidence and CPU verifier](recheck-20261111/README.md) accept all
46 inputs from the two requested workbook sheets and 13 additional controls:
56 device-work inputs reach DAS/new FG >=90%, with a minimum of 96.57%; three
identity/empty inputs launch no work. Correctness, exact output layout and
both providers' A/A drift <=5% pass. Only device kernels and copies are timed.

The original ten dtype inputs pass independently on three physical cards.
The final launch change affects six workbook inputs, all rechecked on each
card with seven old/new/reference paths. Each card also passes the complete
106-test numerical suite after the final change. The other 53 input/control
routes are preserved exactly, with source restoration and metadata checks.

The FP32-to-BF16 168200-element input is about 0.8%-1.4% faster than old public
FG in the three final captures. The 37120-element input remains about
0.3%-0.5% slower; no speedup is claimed for it. Both pass the DAS target.
Routing-only and old-generic comparisons remain separate, and invalid old
public/boxing output strides remain diagnostic failures. There is no claim
that every new cast is faster than every previous path.

This is dtype acceptance for reconstructed workbook metadata and added
controls. It does not assert recovery of original model operands or close
the separate MM review. PR #3 remains a draft for external review.

## Historical 2026-11-10 workbook scope extension

The [extended evidence and CPU verifier](extended-20261110/README.md) cover
all 46 dtype/copy workbook rows and 13 additional controls: 56 device-work
records pass correctness, both A/A checks and DAS/new >=90%; three records
have no device work. The latest minimum is 96.57%. All 106 numerical tests
pass. Original model operands have not been replayed, and the combined
submission remains held by unresolved MM work. Historical measurements and
failed or unstable captures are retained below and in the evidence archive.

The two required tabs of `待优化设备耗时的算子1.xlsx` contain 10 `to.dtype`
rows and 36 related `_to_copy` rows. All 10 `to.dtype` inputs match previous
metadata records; the [36 new copy/cast metadata regressions](extended-cases-20261110.json)
include the original owner and worksheet row. They cover FP32 -> BOOL,
BF16 -> FP32, FP32 -> BF16 and INT64 -> FP32, including larger sizes.

The new FP32 -> BOOL checks exposed incorrect values in the generic route.
An explicit nonzero comparison and additional dense conversion paths
are verified in the extended evidence above. The older evidence below describes its recorded
source revision; it does not validate the expanded dense route. Correctness,
DAS/new >= 0.90 and both A/A checks must pass before submission.

<!-- current-submission-r4n15 -->
Earlier known-scope review: [23 records, compact observations and independent verifier](correction-20261109/submission-r4n15/README.md).
That combined submission was held by the separate original MM review.
The additional workload rows pass reconstructed-input verification above; this does not assert original model replay.
The historical review and separately recorded phases follow below.

PR #3 is a draft. The previous blanket acceptance and claim that the new cast
is uniformly faster are withdrawn. The following is user-supplied review data,
not a new measurement by this checkout. All times are device kernel times.

| Excel row / conversion | Review result | Acceptance |
|---|---|---|
| 1711 BF16 -> FP32 [1,145,2560] | 3.84 -> 3.04 us diagnostic; A/A max drift 6.25% | Performance not accepted |
| 1712 BF16 -> FP32 [1,4,145,256], stride [148480,256,1024,1] | New FG preserves layout; old public FG and DAS do not; new 2.56 us diagnostic; A/A fails | Layout fix reported; performance not accepted |
| 1713 BF16 -> FP32 [1,8,145,256], stride [296960,256,2048,1] | New FG preserves layout; old public FG and DAS do not; new 2.88 us diagnostic; A/A fails | Layout fix reported; performance not accepted |
| 1714 BF16 -> FP32 [2560] | 3.04 -> 2.40 us diagnostic; A/A max drift 6.46% | Performance not accepted |
| 1715 BF16 -> FP32 [256] | 2.5595 -> 2.32 us; 1.10x; DAS/new 110.34%; generic path | Review reports accuracy and stability pass |
| 1716 BF16 -> BF16 scalar [] | Returns input; no device kernel | Identity; speedup and DAS ratio undefined |
| 1717 BOOL -> INT32 [1,145] | 7.76 -> 3.04 us diagnostic; A/A max drift 7.23% | Performance not accepted |
| 1718 FP32 -> BF16 [1,145,256] | 2.64 -> 3.52 us diagnostic; new cast_dense; A/A max drift 6.18% | Performance not accepted |
| 1719 FP32 -> BF16 [1,8,145,145] | 3.84 -> 4.00 us; old/new 0.96x; DAS/new 92.00% | Accuracy/stability pass; DAS target met, but 4.17% slower than old public FG; 1.80x vs old generic |
| 1720 INT64 -> FP32 [1,145,1] | 7.60 -> 3.04 us; 2.50x; DAS/new 194.74%; old generic also 3.04 us | Accuracy/stability pass; benefit mainly from routing |

Keep old public FG, old generic direct, routing-only, new specialized cast,
and DAS measurements separate. A routing gain is not a new-kernel gain.
For the two permuted inputs, a public baseline that changes output strides
is not semantically equivalent for layout acceptance. A low-level
`_to_copy.out` boxing reference with explicit preserved strides is an additional
reference and must not be relabeled as the same public baseline.

The standalone PR #3 source passes 33 focused tests on a01r3n13 visible card 0.
Fresh measurements distinguish six providers, including an explicit recreation
of the original three-field `_to_copy` registration, the unchanged generic
kernel, a routing-only change, the optimized public path and a separate
valid-layout boxing reference. Native kernel names confirm the executed paths.
Only the two FP32-to-BF16 spreadsheet conversions execute `cast_dense`.

The earlier single-compute-unit gate comparison checked all ten exact layouts
with three A/A blocks, five ranges per block and twenty calls per range. New
FG value/dtype/stride/alias checks pass. However, row 1715 measures 5.456 us
against the valid-layout boxing reference's 4.808 us (88.12%), despite a stable
new-path A/A result. This was a kernel-only proxy comparison: the later trace
audit found an additional device copy in that proxy, so 88.12% does not express
its complete device cost or the same public API baseline. Row 1720 failed the newly declared 5% A/A guard (14.89%
drift). The two permuted old public outputs still have incorrect strides.
There is no blanket acceptance and no new dtype kernel change in this correction.

A separately recorded 200-call/ROI comparison preserves all old failures and
retains both kernel and device-copy components. The explicit-layout boxing
`_to_copy.out` reference performs one conversion kernel and one DeviceToDevice
copy per nonidentity call; it remains an additional reference. Neither the
native public nor FG public measurements contain in-range device copies.
All ten new FG value/dtype/stride/alias checks pass. New public A/A still fails
for rows 1712, 1713, 1717, 1718 and 1720. Against the native public baseline,
only rows 1714 and 1719 pass both sides' A/A guard and the 90% threshold in this
run. Rows 1712 and 1713 still have invalid native public output strides; row
1716 remains a no-kernel identity. Do not infer a uniform new-kernel speedup.

See [long-sampling status](correction-20261109/phase2-long/status.json),
[kernel and copy samples](correction-20261109/phase2-long/device-samples-six-providers.json),
[source identity](correction-20261109/phase2-long/source-proof.json) and
[spreadsheet mapping](correction-20261109/phase2-long/spreadsheet-scope-audit.json).
All ten dtype cases are present in the original spreadsheet, rows 1711-1720,
assigned to the requested owner and originally marked unmeasured.

On a01r4n15 physical card 1 (PCI 0000:36:00.0), a separately declared
1024-call/ROI comparison passes all ten new FG correctness and A/A checks;
maximum new-path drift is 2.62%. The seven nonidentity conversions with valid
native public outputs pass both providers' A/A guard and reach **99.25%-101.75%**
native DAS/new. The two permuted native outputs still have incorrect strides,
so they remain outside equivalent-public-baseline acceptance despite the new
FG outputs being correct and stable. Row 1716 remains an identity. This run
uses one GPU per operator, with MM work concurrently on physical card 0.
It changes no production kernel and does not replace the retained user review
or failed 200-call measurements. There is still no uniform kernel-speedup claim.

See [cross-node classification](correction-20261109/crossnode-r4n15/status.json),
[every 1024-call sample](correction-20261109/crossnode-r4n15/device-samples-1024.json),
[retained 200-call comparison](correction-20261109/crossnode-r4n15/device-samples-200-six-providers.json)
and [runtime/source proof](correction-20261109/crossnode-r4n15/runtime.json).

See [provider definitions](correction-20261109/provider-definitions.json),
[earlier kernel-only samples](correction-20261109/device-samples-six-providers.json),
[first-correction status](correction-20261109/revalidation-status.json) and
[source proof](correction-20261109/source-proof.json). Timing conditions differ
from the user's review table; preserve both records and do not combine their
absolute timings. Failed stability checks must not produce an accepted speedup.

Original reports are preserved in `historical/`. Existing raw result and
validation documents are historical snapshots, explicitly marked as superseded.
Source/hash/AST checks prove source identity only, not runtime correctness.

## Added regression cases on a01r4n15

The user has included added regression inputs in the same DAS/new >=90% goal.
All 13 dtype regression inputs (six views in two memory formats, plus FP32/BF16
rounding and special values) pass new public and new export correctness and
the unchanged 5% A/A guard at 1024 calls/ROI. The ten inputs with device work
and correct native public outputs reach **99.91%-200.49%** DAS/new. Two empty
views launch no device work. The preserve-format transpose has incorrect native
public DAS output strides; it remains an acceptance gap, with no valid public
ratio. No dtype production code changed for this extension.

See [all added cases and limitations](correction-20261109/expanded-r4n15/status.json),
[all device samples and components](correction-20261109/expanded-r4n15/device-samples.json)
and [measured source proof](correction-20261109/expanded-r4n15/source-proof.json).

## Explicit native conversion reference for invalid public layouts

A separate valid device reference allocates the required output strides on each
call and invokes boxing `aten.copy_.default(output, input)`. The trace proves
exactly one native conversion kernel and no D2D copy. All value/dtype/stride/alias
checks pass for rows 1712/1713 and added case 920001. Against this reference,
new public FG reaches **99.74%, 99.18%, and 100.17%**, respectively; both
providers pass the unchanged A/A guard. This is a device conversion comparison
through a different ATen entry point, not the same public `Tensor.to` baseline.
The public DAS layout failures remain recorded and are not declared fixed.

See [reference definition and all three results](correction-20261109/explicit-native-layout-r4n15/status.json),
[every device sample](correction-20261109/explicit-native-layout-r4n15/device-samples.json)
and [source/runtime identity](correction-20261109/explicit-native-layout-r4n15/runtime.json).
