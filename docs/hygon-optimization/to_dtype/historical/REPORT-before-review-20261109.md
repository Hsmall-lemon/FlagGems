> SUPERSEDED historical report. See ../REPORT.md for current review status.

# Hygon aten::to.dtype device performance

All ten exact spreadsheet inputs pass value, dtype, output-stride and alias checks on both physical cards 0 and 1. Nine cases perform a conversion and all nine exceed 90% of both DAS references on each card. The remaining scalar already has the requested dtype: it returns the same tensor, launches no kernel, and has no defined performance ratio.

Minimum same-operator performance: 99.19%. Minimum paired boxing cast performance: 122.09%. All 33 focused GPU tests pass.

| Excel row | Input dtype | Target dtype | Shape | GPU 0 us | GPU 0 DAS/FG | GPU 1 us | GPU 1 DAS/FG |
|---:|---|---|---|---:|---:|---:|---:|
| 1711 | torch.bfloat16 | torch.float32 | [1, 145, 2560] | 3.8080 | 107.56% | 3.9680 | 102.02% |
| 1712 | torch.bfloat16 | torch.float32 | [1, 4, 145, 256] | 3.7920 | 293.25% | 3.9840 | 273.90% |
| 1713 | torch.bfloat16 | torch.float32 | [1, 8, 145, 256] | 3.8240 | 296.65% | 4.0000 | 278.40% |
| 1714 | torch.bfloat16 | torch.float32 | [2560] | 3.7600 | 108.94% | 3.9840 | 100.40% |
| 1715 | torch.bfloat16 | torch.float32 | [256] | 3.7920 | 108.02% | 3.9680 | 99.19% |
| 1716 | torch.bfloat16 | torch.bfloat16 | [] | 0.0000 | identity | 0.0000 | identity |
| 1717 | torch.bool | torch.int32 | [1, 145] | 3.7920 | 176.79% | 3.9680 | 172.58% |
| 1718 | torch.float32 | torch.bfloat16 | [1, 145, 256] | 3.7760 | 108.05% | 3.8238 | 105.03% |
| 1719 | torch.float32 | torch.bfloat16 | [1, 8, 145, 145] | 3.7920 | 108.02% | 3.8240 | 105.86% |
| 1720 | torch.int64 | torch.float32 | [1, 145, 1] | 3.7920 | 169.20% | 3.8240 | 166.53% |

The captured spreadsheet currently marks these cases unmeasured and does not provide DAS device timings. Its H100 timing column is not used as DAS. Full input metadata and spreadsheet provenance are in `cases.json` and `spreadsheet-source.json`.

The primary same-operator reference invokes `aten::to.dtype` through the pinned Torch-FL C++ implementation, without a FlagGems context. It is profiled separately on the same physical card and seed. An additional same-process boxing `aten::_to_copy.out` reference uses the CPU-preserved output strides. The C++ reference densifies the two permuted BF16 inputs, changing their output strides; all its copy and cast kernels are included. FlagGems correctness is always checked against CPU, including the original preserved strides.

Each timing is the median of five HIPTX ranges, each containing ten calls. All constituent HIPOPS DurationNs are summed; DurationNs is cross-checked against EndNs-BeginNs. Host gaps, events and synchronization are excluded. A verified gate of more than 5 ms precedes each range and is excluded. Launch gaps may remain and can affect GPU execution state. Raw traces remain under `/public/home/wzx226226/算子优化/profiles/`.

The pinned Torch-FL `_to_copy` wrapper directly calls its C++ implementation and bypasses configuration routing. The `_to_copy` registration therefore adds PrivateUse1 only for Hygon with the renamed `flagos` backend. Every public converted case witnesses execution of the live FlagGems export, and its trace contains a Triton kernel. A dense FP32-to-BF16 kernel uses physical element order with matching output strides. Other inputs use the existing generic implementation. There is no runtime autotuning or DAS delegation in the optimized path.

The first correctly routed generic run took 6.176 us for row 1719; the final GPU 0 result is 3.792 us. Rounding, subnormals, signed zeros, infinities, NaNs, storage offsets, empty tensors, dense permutations and layouts with holes are covered by the GPU tests.

Final formatting and lint checks: passed (all tracked files and the explicit changed/new file set). See `validation.json` and `validation/` for source proof, samples, audits and logs.

The measured kernel hash matches the committed kernel. The final formatter reflowed the test file after the 33-test run; its raw hash therefore differs. The exact measured test is preserved in `validation/measured-test-source.py.txt` and matches the original recorded SHA256. `validation/source-representations.json` verifies its complete executable AST is identical to the committed test. `committed_source_sha256_lf` records the committed source hashes.
