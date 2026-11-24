# Hygon FlashAttention: four workbook reconstructions pass the authorized Release target

On 2026-11-10, physical cards 3 and 4 of `a01r4n15` each passed all four
worksheet cases against the user-selected SourceFind **Release**
`flash_attn==2.8.4+dtk2610.torch2130.2610211904.g0f1a5a` (package ID 13419).
The lowest individual-round native/FlagGems performance ratio is **98.38%**.
Both cards pass the unchanged full output/LSE gate, 33 GPU regression tests,
and 34 full-tensor stability fixtures repeated three times per card.

These inputs reconstruct the workbook metadata. The model case preserves the
reviewed shared backing and Q/K/V offsets 0/2048/4096. The original model
operand payload and the external review's frozen input/gate were not supplied;
this report does not claim to have replayed the earlier 23.6% failure case.
Input provenance and exact layouts are in [workbook-scope.md](workbook-scope.md).

## Device performance

Durations below are microseconds per call. Only GPU kernel durations are
summed, including all helpers and conditional recomputation. Host execution,
allocation, transfers and compilation are outside the timed kernel samples.
Card 3 finished before card 4 started. Each arm has 100 warmups followed by
five rounds of 1000 calls, with alternating arm order. Ratios use the faster
of the qualified native extension and native ATen entries. Every round,
the ratio of mean durations, and the ratio of median durations exceed 90%.

| Card | Workbook case | Native extension us | Native ATen us | FlagGems us | Median DAS/FG | Lowest round DAS/FG |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 3 | model_state | 2162.252 | 2162.328 | 1741.865 | 124.13% | 124.12% |
| 3 | expanded_small | 8.612 | 8.605 | 5.433 | 158.39% | 146.79% |
| 3 | expanded_medium | 9.562 | 9.559 | 6.390 | 149.59% | 143.88% |
| 3 | expanded_large | 43.597 | 43.119 | 43.786 | 98.48% | 98.38% |
| 4 | model_state | 2162.446 | 2162.461 | 1741.980 | 124.14% | 124.12% |
| 4 | expanded_small | 8.625 | 8.628 | 5.491 | 157.09% | 143.16% |
| 4 | expanded_medium | 9.579 | 9.579 | 6.280 | 152.51% | 146.20% |
| 4 | expanded_large | 43.615 | 43.109 | 43.759 | 98.51% | 98.50% |

The model uses three kernels per call: norms, HIP attention, and conditional
stable recomputation. Each expansion case uses one kernel per call. The
profiler records all device kernels; the independent verifier checks event
counts and recomputes every duration directly from the trace data.

## Common precision and regression gate

Before measuring each case, extension, native ATen, and FlagGems each run
three times in the **same process** on the same byte-identical backing stores.
All output and LSE elements are compared with CPU FP32 references. The output
contract remains `atol=0.002, rtol=0.02`; LSE remains `atol=rtol=2e-5`.
Every timed arm has zero mismatches, zero nonfinite values and bitwise stable
repeats. A failed native result cannot provide a timing denominator.

After timing, both cards independently pass the repository's 33 STATE tests.
Each also passes 34 full-tensor fixtures with three repeats: three seeds and
two generation methods for every shape, plus ten N=362 boundary patterns.
The latter cover large scores, zero Q with huge K, huge Q with zero K,
large/signed/tiny V, large scores with V=1e30, negative and changing scores.
The regression receipts and original logs are included in the bundle.

## Implementation and execution environment

The Hygon specialization covers exactly four shape/stride combinations.
The model uses a bounded fixed-shift fast path with stable online-softmax
recomputation. N=3/23 use small stable Triton tiles. N=362 uses a gfx936 MMAC
kernel with FP16 probability/value conversion, an input-dependent error
bound, and a stable three-component BF16 fallback in the same kernel.
Production dispatch contains no input payload or seed constants.

The current compatibility change explicitly sets `enable_fp_fusion=False`
on the four Triton launches. The installed DTK clang-18 rejects the optional
`-fp-fuse=fast` backend switch. No host Torch computation was introduced.

The test environment is isolated inside the existing Apptainer rootfs:

| Component | Measured configuration |
| --- | --- |
| Rootfs | `20261103-hygon-optimization/images/flagos-dev-20261103.rootfs` |
| Python | 3.12.16 |
| Torch | `2.13.0+dtk2610.2610261432.gefb4e1`, native HIP |
| FlashAttention | `2.8.4+dtk2610.torch2130.2610211904.g0f1a5a` |
| FlagGems Triton frontend | `3.6.0+dtk2604.torch2100.2610090934.g328354`, cp312 |
| Triton compiler configuration | `AMDGCN_USE_BUFFER_OPS=0`, fusion disabled on STATE launches |
| HIP compiler/device libraries | existing clang-18 and DTK 26.04 |
| Runtime libraries | isolated official DTK 26.10; node HSA runtime |
| Node driver | `6.3.31-V1.5.0a` |
| Account / allocation | `wzx226226` / existing Slurm job 1071075 |

The default Triton 3.7 frontend in the native-only candidate environment
requires a newer LLVM text format than the available compiler accepts. A
separate Triton 3.6 directory is selected explicitly for the comparison.
Buffer operations are disabled through the frontend's supported environment
option because its newer resource intrinsic is incompatible with clang-18.
FlashAttention and Torch remain the specified binaries. The shared rootfs,
shared default packages and node driver were not changed. No compiler
source or native library was patched to make these results pass.

The wheel SHA256 is
`4ae3186173b36f792bc9c425c6f24e562ae078402d4c09934d18657e263c3c38`.
The verifier pins the extension and matching library separately:
`f130f8a659a91b67a9776f8cd2a2746d645db6e20697ff4e16bd25dc12e91663`
and `69060dc997524cdc803a5bd691d698ae6e97a125cea767d37b941f536787f00b`.

## Native functionality outside the timed fixtures

The specified Release passes all four target fixtures and the tested ordinary
forward, causal/window/GQA/packed/varlen, backward, KV-cache and corrected
dropout combinations. Broader testing also reproduces failures with complete
Q=K=V pointer aliasing on `[1,3,2,64]`, zero/negative scale on `[2,129,4,128]`,
a BF16 scale=0.25 element, and model V multiplied by 100. These inputs supply
no performance denominator. The installed driver/runtime combination is
recorded; these observations do not establish a library source-code cause.
Details and raw checks are in
[the native functionality snapshot](release2610-verification/native-functional-report.md).
This report qualifies the four timed reconstructions, not every API input.

## Evidence and reproduction

[device-acceptance.json](release2610-verification/device-acceptance.json)
contains independently recomputed ratios.
[inventory.json](release2610-verification/inventory.json) records bundle hashes.
The bundle retains every GPU kernel event from all 120 final profiles, raw
trace hashes, full precision/stability receipts, environment identity and
unmodified measurement-script snapshots. Full original traces remain under
`/public/home/wzx226226/flash_attention_node15_20261109/native_candidates/13419/benchmark36-finalsource-gpu{3,4}`.
Committed projections omit host events and preserve every kernel event.

Verify the committed evidence using a CPU-only interpreter:

```bash
python docs/hygon-optimization/flash_attention/reproduce/verify_release2610.py
```

For a new run, use the existing isolated candidate environment and a fresh
output directory on each authorized physical card, sequentially:

```bash
bash docs/hygon-optimization/flash_attention/reproduce/run_release2610.sh \
  /path/to/FlagGems /path/to/native_candidates/13419 3 /path/to/new-card3
bash docs/hygon-optimization/flash_attention/reproduce/run_release2610.sh \
  /path/to/FlagGems /path/to/native_candidates/13419 4 /path/to/new-card4
```

The earlier DAS185 LSE failure and unauthorized DTK2604 2.8.4 measurements
remain historical records. Their completion claims remain withdrawn. The
current denominator is the DTK2610 Release explicitly selected by the user;
no historical native timing is used. The previous status is archived in
[prior-status-before-authorized-release.md](release2610-verification/prior-status-before-authorized-release.md).

## Submission checks

The final source bytes were rerun after formatting; both performance and
regression receipts pin the same source hashes as this PR. All four operator
metadata checks pass. The C++ source passes clang-format 18 in the real
Apptainer environment, and the reproduction shell passes `bash -n`.

`pre-commit run --all-files` was executed. Its only remaining failure is
an existing isort change to `_hygon/experimental/__init__.py`, outside this
operator's scope; that unrelated edit was restored. The focused checks on
every changed PR file pass. Logs are retained in the bundle's `checks/`.
