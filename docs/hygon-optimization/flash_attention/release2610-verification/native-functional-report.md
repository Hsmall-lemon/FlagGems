> Native functionality snapshot taken before the final device-performance run.
> Its "performance not tested" statement describes that earlier phase.
> Current performance acceptance is recorded in [the main report](../REPORT.md).
> The additional native boundary failures below remain applicable.

# FlashAttention 2.8.4 DTK 26.10 Release 功能核查

2026-11-10，账号 `wzx226226`，节点 `a01r4n15`，物理 GPU 3、4。

**结论：指定 Release 已下载，并在原 Apptainer rootfs 中的隔离环境运行。四个工作簿重建工况全部通过，但扩展功能和边界输入仍存在可复现的失败，不能声明整个库功能正常，也未验收此 Release 下的 DAS 90% 性能。**

## 安装与运行环境

| 项目 | 实际配置 |
| --- | --- |
| FlashAttention | `2.8.4+dtk2610.torch2130.2610211904.g0f1a5a`，SourceFind **Release**，包 ID 13419 |
| Python | 3.12.16；该 wheel 只提供 cp312 构建 |
| Torch | `2.13.0+dtk2610.2610261432.gefb4e1`，原生 HIP 构建 |
| Triton | `3.7.0+dtk2610.torch2130.2610212057.g024953` |
| DTK | 官网 DTK 26.10 Ubuntu 22.04 x86_64 运行库 |
| HSA | 节点驱动提供的 `/opt/hyhal/lib/libhsa-runtime64.so.1.11.0` |
| 节点内核驱动 | `6.3.31-V1.5.0a`；官方推荐的 6.4.2-V2.1.0 未验证 |
| MPI | 超算现有 OpenMPI 5.0.3，包含新 Torch 所需的 `MPIX_Query_rocm_support` |
| Apptainer rootfs | `/work1/share/scnethpc2601/flagos-dev/20261103-hygon-optimization/images/flagos-dev-20261103.rootfs` |
| 隔离目录 C | `/public/home/wzx226226/flash_attention_node15_20261109/native_candidates/13419` |

没有修改共享 rootfs、默认 Python/Torch/FlashAttention、节点驱动或算子生产源码。此环境使用私有 `apptainer-run`，沿用原启动脚本并增加只读 MPI 依赖绑定；`native-entry.sh` 设置隔离解释器和 DTK 运行库路径。

原 Python 3.10 环境不能直接安装 cp312 wheel。包还在公开 API 导入时使用了未声明的 `pytest`，已安装该依赖。SDK 中附带的 `.hyhal` 备用 HSA 在当前节点识别不到设备；改用节点已安装驱动的 HSA 后，两卡均识别为一个可用设备，扩展可加载。

所有结论仅适用于本报告记录的驱动、容器和依赖组合，未将失败归因为已经证实的库源码缺陷。

## 四个工作簿工况

输入来自 `待优化设备耗时的算子1.xlsx` 的确定性重建。复用原 Python 3.10 环境导出的 backing 原始字节，在新环境保留 shape、stride、storage offset 与 backing 共享关系。它们不是原模型实际 payload 或外部评审方未提供的冻结输入。

工作簿 SHA256：`d2596052c66d7145af47b35fda45198d1f213b69825aeb91d10112dfd8871597`。

| 工况 | BF16 Q/K/V 的 BHSD shape | GPU 3 | GPU 4 | LSE 最大绝对误差 |
| --- | --- | --- | --- | --- |
| 大于5%的模型工况，645 行 | `[8,16,2049,128]` | 通过 | 通过 | `1.90735e-6` |
| 规模扩展工况，2 行 | `[2,16,3,128]` | 通过 | 通过 | `4.76837e-7` |
| 规模扩展工况，3 行 | `[2,16,23,128]` | 通过 | 通过 | `4.76837e-7` |
| 规模扩展工况，4 行 | `[2,16,362,128]` | 通过 | 通过 | `9.53674e-7` |

每卡每工况分别测试 direct extension、原生 ATen、公开 API 三条路径，每条重复 3 次。所有 output/LSE 全量检查超差数为 0，结果逐位稳定。模型工况每次检查 33,570,816 个 output 元素及 262,272 个 LSE 元素。

门槛保持为：output `atol=0.002, rtol=0.02`；LSE `atol=rtol=2e-5`，判断使用绝对与相对容差之和。模型 backing SHA256：`44ebfeb326f5a257e890668d149dd0e28a715052ab3d7560d14726f39107fc04`。

这次重建模型输入没有再出现 LSE 超差。由于外部评审方的原始输入未提供，不能据此声称已重放并解决其 23.6% 超差案例。

## 其他功能与失败范围

| 测试范围 | 两卡结果 |
| --- | --- |
| BF16/FP16 普通前向，长度 3/23/127/128/129/137/257，D32/64/128/256 | 已测组合通过 |
| causal、局部 window、cross attention、GQA | 已测组合通过 |
| QKV-packed、KV-packed、varlen | 已测组合通过 |
| 普通/causal/GQA backward | output 与梯度检查通过；梯度门槛单列为 atol 0.01、rtol 0.03 |
| KV cache 单步 | 通过 |
| dropout 0.125 | 修正测试调用后，output/LSE 和恢复 GPU RNG 后的重复性通过 |
| 大模型 zero-Q/large-K、large-Q/zero-K | 通过 |
| Q/K/V 完全别名的小形状 | **失败**，见下文 |
| scale 为 0 或负值 | **失败**，返回非有限值 |
| BF16 scale=0.25 | **1/132,096 个 output 元素超差**；LSE 通过 |
| 模型输入 V 放大 100 倍 | **994,756/33,570,816 个 output 元素超差，约 2.963%**；LSE 通过 |

完全别名的最小复现：BSHD shape `[1,3,2,64]`，`q=k=v` 指向完全相同的数据，BF16 与 FP16 均失败。direct extension、ATen、公开 API 三条路径均复现；三次调用有不稳定情况，输入本身未被修改。将相同数值分别复制成三个独立 tensor 后，同形状三条路径均通过。这个条件不同于工作簿模型工况中共享 backing、offset 分别为 0/2048/4096 的三个 view。

scale=0 和 scale=-0.08838834764831843 的失败形状是 BSHD `[2,129,4,128]`。BF16、FP16 的三条路径均出现全量非有限 output/LSE。BF16 scale=0.25 的单元素超差也在三条路径及两卡复现。

dropout 初测把无梯度输入返回的空 S_dmask 当作四维 mask，导致两个测试脚本异常。后续改为 `requires_grad=True`，并只检查有效 97×97 区域的 dropout mask，不比较未定义 padding。两卡两种 dtype 的数值、output/LSE 重复性、有效 mask 重复性均通过，观测保留比例为 0.873977，目标为 0.875。初次脚本异常和对 padding 的错误比较不计为已确认的库失败。

小形状 CPU FP32 参考还通过独立 NumPy FP64 交叉检查，避免将参考实现错误当作 native 失败。

## 复现

只在授权节点/卡和仍有效的 Slurm allocation 内运行；本次使用作业 1071075，没有取消该作业。以下命令运行完整功能测试；原始证据保留在已有目录，应在新结果目录运行：

```bash
C=/public/home/wzx226226/flash_attention_node15_20261109/native_candidates/13419
mkdir -p "$C/recheck-gpu3"
cd "$C/recheck-gpu3"
srun --input=none --jobid=1071075 --overlap -N1 -n1 -w a01r4n15 \
  env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=3 CUDA_VISIBLE_DEVICES=3 \
  FLAGOS_DEV_CACHE=/public/home/wzx226226/.cache/flagos-dev/20261103-flash-node15-gpu3 \
  bash "$C/apptainer-run" -- bash "$C/native-entry.sh" "$C/functional.py"
```

将 GPU 与 cache 改为 4 可复测另一张卡。`focused.py` 和 `final_checks.py` 分别重放失败范围和最终别名/dropout 核查；同样应在新目录运行。原始 functional.py 保留最初 dropout 测试错误，以便审计，dropout 最终结论依据 final_checks.py。

## 下载校验与证据

FlashAttention wheel：`flash_attn-2.8.4+dtk2610.torch2130.2610211904.g0f1a5a-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl`，310,490,106 字节。

SHA256：`4ae3186173b36f792bc9c425c6f24e562ae078402d4c09934d18657e263c3c38`。

来源：[SourceFind Release 查询](https://pypi.sourcefind.cn/pypi/search?q=flash-attn&exact=1)、[DTK 26.10 官方下载](https://download.sourcefind.cn:65024/1/main/DTK-26.10/Ubuntu22.04)。DTK 压缩包已通过官方 MD5 `adbd7076f1c5d2c1dbe6d61e85c6e5f9`，运行库解压后删除压缩包，释放 3,653,732,295 字节。候选环境约 22 GB，完成后 `/public` 可用约 57 GB。

每卡运行完整初测 65 项、失败追测 46 项、最终核查 14 项。保留初测和追测的全部失败记录，没有用追测覆盖旧证据。没有测量 host 开销或为此 Release 计算性能比值。

- [机器可读汇总](native-functional-summary.json)
- [Apptainer 身份、实际包版本、动态库及 smoke 结果](evidence/runtime-probe.json)
- [GPU 3 初测](evidence/functional-gpu3/functional.json)、[GPU 4 初测](evidence/functional-gpu4/functional.json)
- [GPU 3 追测](evidence/focused-gpu3/focused.json)、[GPU 4 追测](evidence/focused-gpu4/focused.json)
- [GPU 3 最终别名/dropout 核查](evidence/final-checks-gpu3/final-checks.json)、[GPU 4 最终核查](evidence/final-checks-gpu4/final-checks.json)
- [依赖锁定记录](evidence/pip-freeze.txt)、[安装日志](evidence/setup.log)、[DTK 来源、校验与运行库清单](evidence/dtk2610-runtime-manifest.json)
