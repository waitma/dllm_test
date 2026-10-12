# 单次 forward 的 attention bias 复用原型

更新：2026-09-30 20:23:59 UTC。状态：**CPU 等价验证通过；独立原型，未部署，未进行 GPU 显存验证**。

本轮为当前 R12–R19 100 步诊断结束后的下一版实现准备。没有改动正在使用的 [research_next/model.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/research_next/model.py)、[run_immune_architecture.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/run_immune_architecture.py)、旧 runner、release snapshot 或看护进程，没有 submit/enqueue。

## 问题与证据边界

现有 `ArchitectureModel.denoise()` 每层执行一次 `base_bias.masked_fill()`，生成一个独立 `[B,H,L,L]` 张量，并把它传给 activation checkpoint。8 层 global attention 使用相同 mask，却构造 8 份 bias。

现有 GPU profile 显示，在 microbatch8、L2048、FP32 参数/Adam 与 BF16 autocast 下，旧 R00 峰值为 11.195548 GiB，新 R12 为 20.457307 GiB。参见 [R00 测量](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/profiles/r00_a4_control/bs8/measurement.json)、[R12 测量](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/profiles/r12_architecture_control/bs8/measurement.json)。这是原实现的已有测量，**不是本原型的显存结果**；两版还有其他实现差异，不能把峰值差全部归因于一行代码。

本轮 CPU 操作计数确证了重复 `masked_fill` 构造和不同 storage 的存在，并验证复用后计算等价；它不测 GPU allocator 峰值，也不预测优化后的推荐 microbatch。

## 原型与建议改动

独立文件：[bias_cache.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/bias_cache.py)。`bind_bias_cache(model)` 只将 `denoise_with_bias_cache` 绑定到一个明确传入的临时模型实例，未修改 `ArchitectureModel` 类，导入模块也不自动绑定任何模型。

建议下一版在原 `denoise` 中作如下局部改动：

1. 在层循环前创建局部 `biases = {}`，按当前规则区分 `('global',)`、`('chain',)`、`('local', radius)`；R17 前半层是 chain，R18 偶数层是 local，其余为 global。
2. 每种 mask 第一次出现时构造 bias，后续层复用同一张量。global 且全部 attention 位有效时，直接复用 `base_bias`。
3. 有 padding 或链/local 限制时，继续调用原 `layer_allowed()`，以非原地 `masked_fill()` 生成该类 bias。padding key、padding query 的兜底有效 key、prefix 与链隔离的规则均原样保留。
4. 不 `detach`、不使用 `masked_fill_`、不改共享张量；共享 bias 的各层梯度仍回传并累加到 `chain_bias`。缓存只存在于这次 `denoise` 调用的局部变量中，不放到模型/层属性、全局对象或后续 forward。
5. bias 与 `first_values` 继续作为 checkpoint 函数的显式参数。跨层复用 bias，不改变 R19 的第一层 value 残差生命周期。

原型保留了 `ArchitectureBlock._cast_attn_bias()` 的行为；在 BF16 autocast 下，它仍可能逐层产生 cast 后的 bias。当前优化仅复用传入 block/checkpoint 的原 dtype bias，没有宣称消除所有 `[B,H,L,L]` 分配。另外，检测 `valid.all()` 在 GPU 上可能增加同步，实际吞吐影响需下一版 GPU profile 测量。

## 分配数量与 CPU 实测

令单个 FP32 bias 的逻辑大小为 `S = B × H × L² × 4 bytes`。以下针对 8 层：

| 类型 | Padding | 原 masked_fill 次数 | 原型次数 | 原 block bias 独立 storage | 原型独立 storage | 原型构造 base+masked bias 数 |
|---|---|---:|---:|---:|---:|---:|
| R12/R13/R14/R15/R16/R19，全部 global | 无 | 8 | 0 | 8 | 1 | 1 |
| 同上 | 有 | 8 | 1 | 8 | 1 | 2 |
| R17，chain/global | 无 | 8 | 1 | 8 | 2 | 2 |
| R17，chain/global | 有 | 8 | 2 | 8 | 2 | 3 |
| R18，local/global | 无 | 8 | 1 | 8 | 2 | 2 |
| R18，local/global | 有 | 8 | 2 | 8 | 2 | 3 |

原实现每次 forward 构造 1 个 base bias 加 8 个 masked bias，即 9 个逻辑结果张量。这个计数不等于峰值时同时存活的张量数。以生产示例 B8/H12/L2048 代入，`S=1.5 GiB`；这些数字只能推导逻辑构造量，**不能据此宣布 GPU 峰值会下降若干 GiB**。casting、SDPA、autograd、checkpoint 重算、其他激活、allocator 复用都会影响峰值。

证据：[counts.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_architecture_bias_cache_cpu_20260930/counts.json)。在 `pllm` / PyTorch `2.8.0+cu126` 中用 CPU、d32/L8/H4、whole-layer checkpoint 测量 8 variants × 两种 padding，共 16 个实例对。计数器实际拦截指定 `[B,H,L,L]` 形状的 `aten.masked_fill.Scalar`，同时记录各 block 收到的 bias storage 地址；没有根据预设公式直接伪造计数。

| CPU 输入 | 每个 bias 大小 | 原 masked_fill 输出字节合计 | global 原型 | chain/local 原型 |
|---|---:|---:|---:|---:|
| 无 padding，B2/H4/L52 | 86,528 bytes | 692,224 bytes | 0 bytes | 86,528 bytes |
| 有 padding，B2/H4/L60 | 115,200 bytes | 921,600 bytes | 115,200 bytes | 230,400 bytes |

16 对输出的 logits 与 hidden 全部逐元素一致。统计中的 `masked_fill_output_bytes_total` 是实际调用返回张量的逻辑字节累计，不是进程 RSS 或 allocator 峰值；为避免地址回收复用混淆，计数 hook 保持 bias 引用，因此这个工具本身不能当作完整内存基准。计数阶段没有 backward；梯度和 checkpoint 重算由下面的独立测试验证。

重现命令（先激活 `pllm`）：

```bash
USE_TF=0 python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/bias_cache.py --output /tmp/bias-cache-counts.json
```

## 等价测试

测试文件：[test_architecture_bias_cache.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_architecture_bias_cache.py)。

```bash
USE_TF=0 python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_architecture_bias_cache.py -q
```

结果：**82 passed，14.56 秒**。2,016 条 warning 来自既有底层 CPU autocast API 的弃用提示，没有测试失败。

- 全部 8 variants，FP32 和 CPU BF16 autocast，有/无 padding：logits、hidden、loss 均严格逐元素一致（atol=rtol=0）。主要功能用 d32/L4/H4；人工 antigen 长 50 AA，实际跨过 R18 半径 32 的局部窗口边界。`chain_bias` 使用非零且非均匀初值，避免只验证全零偏置的特殊情形。
- 每个参数比较梯度存在性、有限性与数值：`atol=2e-6, rtol=1e-5`；同时确认 `chain_bias` 存在非零梯度。浮点加法聚合顺序可能改变，因此梯度不宣称逐位相等。
- 全部 8 variants、两种 dtype：whole-layer checkpoint 下先保留两个不同 shape/批大小的 forward，再联合 backward，对比原实现全部参数梯度；覆盖第二视图覆盖第一视图状态、延迟重算和 R19 value residual 的风险。
- 全部 8 variants：跨 A→B→A 调用、不同 shape、更新 `chain_bias` 后再次调用，均与未缓存模型一致；模型属性没有新增缓存张量，state_dict key 不变。
- 全部 8 variants：单独记录与同批 padding 的有效 logits/hidden 一致；非法全 padding 样本、label 非法 feature 仍按原接口拒绝。
- 16 个 L8 用例断言实际 masked_fill 次数和独立 storage 数与上表一致。另验证 prototype/test 均不进入 live fingerprints，实例绑定未修改模型类。

CPU 计数运行前后的完整 live source fingerprints 一致，证据 JSON 已记录完整映射。关键 SHA256：

| 文件 | SHA256 |
|---|---|
| prototype | `5ddce88a2688eaf1bc3ed1769c7361c8432bde019bb6df409230154835e2fb95` |
| prototype tests | `2874cf7afe5e737adb29cab63ee15a0459a08de919138761fbcecd0f0003562d` |
| 未修改的 research_next/model.py | `c7c8f9e8c69e665204cb0eb404993e927eedbf77473cbbfff022e3b79ae576fa` |
| 未修改的新 runner | `79b75c741a8b27976254fc0188f5e925b22e1ba77bff99527cfb82f2ef770dac` |

## 下一版进入条件

等当前冻结的 100 步诊断结束后，由主线程将经过 review 的局部改动纳入新 release；本原型不自动部署。新源码必须重新 profile，不能复用本轮 source SHA 对应的 profile COMPLETE。下一版需在生产 A100、同配置/精度/样本顺序/噪声种子下实测峰值、吞吐和训练数值，再决定 microbatch。若需要比较架构效果，各候选和 R12 应统一使用修复后的实现；本次偏置复用属于实现成本优化，不作为新的生物学架构增益。

没有下载权重、产生 checkpoint 或创建后台进程。新增持久结果只有源码、测试、本文档和小型计数 JSON；共享项目账本由主线程同步。
