# Hygon FlashAttention: DAS185 acceptance incomplete

**The requested task is not accepted.** The user specified
`flash_attn==2.8.3+das185.dtk2604.torch2100.2608132155.g9febea`, which remains
the DEV image default. The agent switched the comparison baseline to DAS
2.8.4 without user authorization. The previous completed-task claim and
promotion of PR #5 to ready for review are withdrawn; PR #5 is a draft.

The DAS185 comparison still has unresolved full-LSE precision failures.
No valid >=90% acceptance against the requested baseline has been established.
These failures must be investigated under the unchanged precision contract;
a different native version or historical timing cannot replace that target.

The measurements below are preserved **only as an alternative DAS 2.8.4
experiment**, not fulfillment of the user's requested task. They use four
deterministic worksheet metadata reconstructions. Original model operand
payload and the external review's frozen gate have not been replayed.
See [acceptance-status.json](acceptance-status.json) and
[workbook-scope.md](workbook-scope.md).

## Alternative DAS 2.8.4 experiment: not requested acceptance

Durations are microseconds per call, summed across **all** device kernels.
Card 3's profile finishes before card 4's profile starts; these are separate
single-card results. Host overhead, compilation, allocation and transfers are
excluded. Every arm uses 100 warmups and five rounds of 1000 timed calls.

| Physical card | Worksheet case, BF16 Q/K/V | DAS wheel us | DAS ATen us | FlagGems us | Mean per-round DAS/FG | Ratio of mean times |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 3 | Model row 645: `[8,16,2049,128]` | 1611.276 | 1611.305 | 1741.769 | 92.51% | 92.51% |
| 3 | Expansion row 2: `[2,16,3,128]` | 7.230 | 7.227 | 4.801 | 150.74% | 150.55% |
| 3 | Expansion row 3: `[2,16,23,128]` | 8.250 | 8.247 | 5.745 | 143.63% | 143.56% |
| 3 | Expansion row 4: `[2,16,362,128]` | 42.239 | 41.790 | 43.767 | 95.48% | 95.48% |
| 4 | Model row 645: `[8,16,2049,128]` | 1611.516 | 1611.452 | 1742.202 | 92.49% | 92.50% |
| 4 | Expansion row 2: `[2,16,3,128]` | 7.257 | 7.253 | 4.921 | 147.55% | 147.40% |
| 4 | Expansion row 3: `[2,16,23,128]` | 8.265 | 8.265 | 5.740 | 144.03% | 143.98% |
| 4 | Expansion row 4: `[2,16,362,128]` | 42.260 | 41.821 | 43.811 | 95.46% | 95.46% |

The alternative experiment tested both ratio definitions against 90%;
this is not the requested DAS185 acceptance. Each round uses the faster eligible
native entry for its denominator. The model case includes its norm/guard,
attention and conditional stable-recomputation kernels (three kernels per
call). Each expansion case has one kernel per call, including the N=362
conditional slow path inside its HIP kernel. The analyzer rejects missing
kernels or any missing/duplicate range among the 60 timing ranges per card.

## Implementation

The four shape/stride combinations dispatch through the Hygon backend;
other layouts retain the existing attention implementation. Q/K/V pointer
offsets and independent/shared backing stores are preserved. The captured
scale `0.08838834764831843` and the adjacent Python default expression both
select the appropriate specialization.

The model path handles nonfinite norm-derived shifts and rejects large V
from its single-BF16-probability fast path. Its device-side guard selects a
stable online-softmax recomputation. N=3 and N=23 use small stable Triton
tiles. N=362 uses a pipelined gfx936 MMAC kernel: probability and value
conversion use FP16, while denominator and LSE stay FP32. Moving the next-V
checks/conversion after the current PV MMAC preserves global-load overlap.
An input-dependent bound checks probability rounding, value magnitude and
the exponent range. Rejected blocks recompute with online softmax and three
BF16 probability components in the same kernel. No operand values or
reconstruction seeds are hardcoded into the production dispatch.

## Precision and stability

The unchanged common tolerances are output `atol=0.002, rtol=0.02`, and LSE
`atol=rtol=2e-5`. CPU FP32 SDPA supplies the output reference; CPU FP32
matmul/logsumexp supplies LSE. The native wheel, native ATen and repository
ATen entry all pass **every** output/LSE element with zero mismatches and
zero nonfinite values for all four timed fixtures on both cards. Torch-FL
requires a temporary PrivateUse1 binding in this image; the reproduction
scripts explicitly bind and exercise that exact ATen entry.

Each card also passes 33 GPU regression tests and 34 stability fixtures,
three repeats each, with full output/LSE checks and bitwise repeatability.
The stability fixtures cover three seeds and both direct-BF16 and
FP32-to-BF16 generation for every shape, plus ten N=362 boundary patterns.
The latter include large Q/K, zero Q with huge K, huge Q with zero K,
large/signed/tiny V, large scores with V=1e30, negative scores and changing
scores. Three additional model boundary patterns pass on both cards. Native
large-V boundary output fails the precision gate and is explicitly
ineligible for a performance comparison; no such timing is used here.

Four operator static checks and 14 CPU acceptance-gate regression checks
pass. Focused pre-commit hooks pass. The all-files invocation passed after
an unrelated baseline import grouping had been formatted; that unrelated
file was restored and is excluded from this change. The real DEV
clang-format 18 dry-run passes for the measured C++ bytes; the Windows
clang-format wrapper is not used as C++ validation evidence.

## Native identity and withdrawn claims

The alternative experiment's denominator is SourceFind's official DAS nightly
`flash_attn==2.8.4+dtk2604.torch2100.2609292239.g74de14`, selected from an
isolated user directory. Both its extension and matching library are
identified by SHA256 in each gate. Its wheel URL, size and SHA256 are in
`workbook-verification/native-package.json`. No shared image modification
is required for this comparison.

The installed `2.8.3+das185.dtk2604.torch2100.2608132155.g9febea` remains
numerically ineligible. This work does not repair or qualify DAS185, and
does not reuse its historical timing or the author's timing as a denominator.
The prior 94.57% claim remains withdrawn. The previous 95.30%/95.39% results
refer to source with reproduced edge failures and do not qualify this source.
Older files in `node15-verification/`, `validation.json`, and
`source-equivalence.json` are historical evidence with their original scope;
`workbook-verification/` records the current-source alternative experiment;
**it does not qualify the requested DAS185 acceptance target**.

## Alternative experiment evidence and reproduction

`workbook-verification/inventory.json` indexes source, reproduction-tooling
and evidence hashes. Each card has a fresh gate/run ID, complete kernel
summary, labeled compressed HIPOPS/HIPTX subset, GPU-test log, edge checks
and stability receipts. The filtered trace records every original split
trace file's SHA256, size and event count. Complete raw trace parts remain
compressed on the node; the committed subset intentionally omits host API
events. The final source hashes match the Git blobs byte for byte.

Verify the archive without a GPU:

```bash
python docs/hygon-optimization/flash_attention/reproduce/replay_workbook_bundle.py
```

On the allocated a01r4n15 node, run the two cards **sequentially**, each with
a fresh output directory (the second command starts after the first exits):

```bash
bash docs/hygon-optimization/flash_attention/reproduce/run_workbook.sh \
  /path/to/FlagGems /path/to/unpacked/native/package 3 /path/to/new-card3
bash docs/hygon-optimization/flash_attention/reproduce/run_workbook.sh \
  /path/to/FlagGems /path/to/unpacked/native/package 4 /path/to/new-card4
```

The wrapper uses the validated DEV environment and masks the physical card
before entering it. It runs boundary checks and GPU tests, the full common
gate and device profile, archives/analyzes all trace parts, compresses raw
traces, then checks multi-fixture stability. The node/card and native-library
identities must match the receipts. Original payload/external frozen-gate
acceptance remains explicitly unverified in the summaries.
