# Hygon optimization integration branch

This branch consolidates locally developed FlagGems optimizations without
turning every historical experiment into a default backend implementation.
It starts from fork master `161356a17187565a063e8ee2fc8e06a222c9800e` and
merges official master `f26de5a8d279ca02c1908c581899fadae2bf2ed0` before
adding the optimization sources. The fork default branch itself is unchanged.
The fork's existing FP32 Flash Attention commit is also included with its
original authorship. No private benchmark logs, tensors, credentials, scheduler
scripts, or shared-environment configuration are published here.

## Source organization and status

| Component | Location | Status carried forward |
| --- | --- | --- |
| FP32 Flash Attention | `src/flag_gems/runtime/backend/_hygon/ops/_flash_attention_forward.py` | Existing fork patch, with its benchmark and tests; selected from `release/r6-20261026-fp32-fa` |
| Stable sort | `src/flag_gems/runtime/backend/_hygon/experimental/sort.py` | Selected Oct 29 device candidate; full API confirmation was ineligible |
| Logical reductions | `under90/f01.py` | Selected row reduction and empty/dtype repairs |
| index_add | `under90/f02.py` | Selected copy/validate/scatter, repeated indices and alpha handling |
| nonzero / masked_select | `under90/f03.py` | Selected compaction and empty-input handling |
| masked_fill | `under90/f04.py` | Selected contiguous specialization with delegation |
| ones / ones_like / linspace / floor / nan_to_num | `under90/f05.py` | Latest second-round factories/numeric source; known Torch-FL layout gap retained |
| cos / sin / atan2 | `under90/f06.py` | Selected trigonometric source and guarded delegation |
| gelu / tanh / neg | `under90/f07.py` | Latest special-value repairs and selected activation kernels |
| rsqrt | `under90/f08.py` | Latest subnormal handling and alignment repair |
| randperm | `under90/f09.py` | Latest selected bounded algorithm; serialized eager only, n <= 16384 |
| mm / bmm / addmm | `src/flag_gems/runtime/backend/_hygon/experimental/matrix.py` | Top20 candidate 2; 33 primary + 18 semantic records / 76 subchecks passed in the pinned environment |
| index.Tensor integration | `src/flag_gems/runtime/backend/_hygon/experimental/index.py` | Process-local Torch-FL/CUDA registration; existing gather algorithm unchanged |
| Earlier sum, SiLU, exp, clamp, layer norm, softmax, pooling and other revisions | `history/MANIFEST.json` and `history/sources/` | Historical source only; includes rejected, superseded and model-regression-blocked work |

`under90/*` paths in the table are relative to
`src/flag_gems/runtime/backend/_hygon/experimental/`. The exact file hashes and original source
hashes are in `SELECTED_SOURCES.json`; the runtime integrity map is
`src/flag_gems/runtime/backend/_hygon/experimental/SOURCES.json`.

The historical manifest contains every implementation blob identified across
the locally available optimization archive commits, plus the matrix revisions.
Its selection rule is explicit. Duplicate blobs are stored once; every original
relative artifact path and provenance commit is retained. All source entries
from the 15-schema historical operator registry are accounted for. The archive
is a source recovery aid, not an importable set of independently qualified
implementations. Rejected revisions stay rejected. It does not claim to cover
unavailable experiments or every unrelated branch in the fork.

The pinned index provider and fork master have identical source text after
newline normalization. Consequently this branch does not duplicate or rewrite
the upstream index kernel to represent routing work as a new algorithm.
The old `hip-flag-gems` branch contains a much older general HIP development
history; merging that divergent branch wholesale is outside this consolidation.

## Explicit process-local use

Nothing in the experimental package installs on import. Installation must
happen once in a fresh process before `enable`, `only_enable`, `use_gems`, model
execution or dispatcher registration. Normal FlagGems initialization is unchanged.

For the matrix family on Hygon:

```python
import torch
import flag_gems
from flag_gems.runtime.backend._hygon.experimental import install

receipt = install(groups=("matrix",))
with flag_gems.use_gems():
    a = torch.randn((303, 320), device=flag_gems.device)
    b = torch.randn((320, 320), device=flag_gems.device)
    c = torch.mm(a, b)
```

The `under90` and `index` groups require the Torch-FL private backend named
`flagos`. Initialize the matching `torch_fl` environment first. For example:

```python
import torch_fl  # initialize the qualified integration before torch/FlagGems
import torch
import flag_gems
from flag_gems.runtime.backend._hygon.experimental import install

receipt = install(groups=("sort", "under90", "matrix"), serialized_eager=True)
```

`serialized_eager=True` is an explicit acknowledgement of the randperm contract;
it does not make concurrent calls, CUDA graphs, compilation or training safe.
The original module's own bounds and capture checks remain in place. The order
of combined installation is sort, under90, matrix, index. Sources are verified
before import; a failed installation requires restarting the process.

`index` is a separate compatibility group. Its original Torch-FL experiment
also selected vendor `aminmax` in the caller's route configuration to avoid a
noncontiguous-stride defect. The loader does not change any route files, bypass
bounds checks, or guarantee that an arbitrary Torch-FL release has that fix.
Do not enable this group without validating the caller's routing contract.

## What the existing measurements establish

These are historical fixed-release results, not a hardware qualification of
this master-based branch:

- Under90 round 2: 58/69 witnessed FG cases met all four device-pair thresholds;
  11 did not. Three unwitnessed repeat_interleave cases remained excluded.
  The original extended suite was 144/146 because of two existing Torch-FL
  ones_like memory_format forwarding failures. Formal G stayed baseline.
- Sort/randperm follow-up: device improvements did not pass the formal full API
  promotion gate. Randperm's supported use remained bounded serialized eager.
- Top20 matrix: 24/33 cases met every one of four paired DAS/FG kernel ratios
  >= 0.9; nine misses were retained. Eight AB/BA confirmations against the FG
  baseline passed with median weighted API speedup 1.1657463x. This is neither
  full API parity with DAS nor an end-to-end model speedup.
- Top20 index: 18/18 signatures met all four device-pair thresholds and 34
  semantic records passed. This qualified routing to an existing implementation;
  it did not introduce a new gather algorithm. Bounds/D2H/synchronization costs
  were preserved and reported separately from device kernels.

The archived hardware environment used Torch 2.10, Torch-FL 2.10
`5699bb03`, FlagGems 5.4, FlagTree 0.7 + HCU 3.6, and gfx936. Candidate code in
this branch has been formatted and source-checked locally. Its executable AST
is equivalent to the archived source, excluding docstrings and comments.
No GPU retest, new performance claim, deployment, or upstream PR is implied.

## 底层运行逻辑与原理分析

The verified experimental chain was public torch API -> ATen overload ->
Torch-FL PrivateUse1 routing/boxing -> the selected FlagGems callable ->
Triton/FlagTree Hygon code generation -> HIP stream/device kernels -> output
transfer and CPU-reference comparison. Index kept the existing C++ bounds
checks and related min/max and D2H work. These operator experiments did not
establish Dynamo/Inductor whole-model fusion or training coverage.

The new opt-in loader imports the retained kernel modules, patches process-local
FlagGems roots, then refreshes both registrar views by callable identity while
preserving schema metadata. This last step prevents an unchanged registrar cache
from silently routing back to old callables. The original standalone candidate
modules and normal upstream backend remain separate; no route configuration or
shared environment is modified.

## Local checks and hardware smoke

The inventory/installer test has no torch, Triton or GPU dependency:

```bash
python tests/test_hygon_optimization_inventory.py -v
```

On an appropriately prepared Hygon device, the new small regression smoke is
explicitly gated and supplemental to the original full contracts:

```bash
FLAGGEMS_HYGON_GROUP=matrix pytest -q tests/test_hygon_optimization_device.py
# separate fresh processes for each of: sort, under90, index
```

These new smoke checks have not been executed on hardware. Before proposing an
upstream kernel patch, port the narrow supported domain into the Hygon backend,
keep a valid fallback, validate all relevant semantics and actual ATen routing,
and repeat paired performance measurements on the resulting source revision.

## Preparing official contributions

Keep this as the long-lived personal integration branch. Prepare a separate
short branch from current upstream master for each official contribution;
cherry-pick the relevant family commits and required loader/test dependencies,
then turn the opt-in implementation into a narrow backend patch. Do not open a
single PR that activates every historical candidate.

Recommended order: FP32 attention; matrix kernels and edge fixes; individual
under90 families; stable sort; bounded randperm only with its RNG/capture contract.
Torch-FL index registration and memory_format forwarding belong to integration
work, rather than an invented FlagGems kernel change. Historical layer norm,
softmax and model-regression-blocked sources require new isolation and evidence
before an upstream performance PR.

## Consolidation validation receipt

See `LOCAL_VALIDATION.json` for the reproducible local check status and commit
index. Eight inventory/installer tests, selected-source Black/flake8 checks,
new-bootstrap/test isort checks, package-data parsing, and upstream export/operator
metadata checks passed. The required `pre-commit run --all-files` was attempted
but stopped while installing its isolated environment because PyPI returned
TLS SSLEOF errors for setuptools. It is recorded as blocked, not passed.
The retained provider import order is intentional and checked against archived
source ASTs; immutable history blobs are excluded from automatic whitespace fixes.
