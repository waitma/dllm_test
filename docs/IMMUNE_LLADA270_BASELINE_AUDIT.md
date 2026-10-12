# LLaDA270 与当前 A4 基线核实

核实日期：2026-09-30 UTC。只读源码、配置和 checkpoint 元数据；未改模型、runner、queue 或已有 snapshot，未提交任务。本文不把人工诊断结果当作生物学或下游性能证据。

## 结论

项目旧名称 **LLaDA270M** 对应 **8 层、宽度 768、12 个 attention heads、12 个 KV heads、head_dim 64、SwiGLU 中间宽度 3072**。当前 A4 的 Transformer 深宽与它相同。两者总参数差异主要来自词表，不能将旧 270M 理解为一个更深或更宽的主干。

真实旧 no-ESMC checkpoint 的参数总数为 **269,759,232**；其中输入词嵌入和独立输出 head 合计 **194,248,704**，约占 72%。当前 A4 使用 50-token 词表，总参数 **75,601,176**。缩减的是大部分未使用文本词表行，不是 Transformer 层数或宽度。

最直接的配置依据是 [旧 no-ESMC YAML 第 88 行](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/protein_llada270m_noesmc_diffusion_v2_immune_v6_2node4gpu_1m.yml:88)，构造方式见 [scratch decoder 加载器](/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py:785)。后者从 LLaDA-8B 借用配置和词表，覆盖深宽并随机初始化，没有加载 8B 模型权重。

## 实物与参数分解

检查了两个独立训练步数的保存产物：

- [checkpoint-150000 的 fusion_config](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_llada270m_noesmc_diffusion_v2_immune_v6_2node4gpu_1m/checkpoint-150000/fusion_config.json)。
- [checkpoint-182000 的 fusion_config](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_llada270m_noesmc_diffusion_v2_immune_v6_2node4gpu_1m/checkpoint-182000/fusion_config.json)。

分别读取各目录 `model.safetensors` 的 8-byte 头长度和 9,112-byte JSON header，没有加载参数数据。两者均为 75 个参数张量；所有张量名都属于 decoder，没有 ESMC encoder 或条件 projection。输入和输出矩阵均为 `[126464, 768]`；每层 Q/K/V/O 为 `[768, 768]`，SwiGLU 的三个矩阵为两份 `[3072, 768]` 和一份 `[768, 3072]`。

| 模块 | 旧 no-ESMC 270M | 当前 A4/R12 control |
|---|---:|---:|
| 8 个 Transformer blocks | 75,509,760 | 75,509,760 |
| 最后 RMSNorm | 768 | 768 |
| 输入 embedding | 97,124,352 | 38,400 |
| 独立输出 head | 97,124,352 | 38,400 |
| role embedding | 0 | 9,216 |
| scope embedding | 0 | 4,608 |
| same/different-chain attention bias | 0 | 24 |
| **总参数** | **269,759,232** | **75,601,176** |

当前列通过 CPU 构造 `ImmuneModel(ImmuneConfig())` 并对 `named_parameters()` 求和独立核实，未建立 optimizer 或 checkpoint。R12 仅包裹原层，不新增参数。

两份旧 checkpoint 的 header SHA256 都是 `1d6f953b6ef91ece7cb66eaa4802fa583e3ebce83d2f6abb849c9fea56cfa47b`；两份 `fusion_config.json` SHA256 都是 `3ee72a5eae1af325884dbb71a566c258b5e4d77f37a7755e4788a051e69cbc3d`。**这些 hash 证明的是结构/配置一致，不是完整权重相同**；未计算或声称完整权重 SHA 一致。

旧 `fusion_config` 同时确认：`residue_cond_mode=token`、`condition_hidden_size=0`、`vocab_size=126464`、`fixed_receptor_lengths=true`、`predict_eos=true`、`decoder_chain_eos_token_id=126389`、`diffusion_all_chains=false`。no-ESMC 路径确实不含 encoder。

## 词表与基础层配置

旧加载器保留 LLaDA 的 **126,464 行** padded embedding 和输出空间。原 tokenizer 的已有行与追加的 residue/grammar token 没有超出该容量时，不会缩小或重建这两张矩阵，见 [避免缩小词嵌入的逻辑](/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py:1126)。

当前本机可读的原始配置位于 [LLaDA-8B 本地 cache config](/vepfs-mlp2/c20250601/251105016/conda/cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Base/snapshots/0f2787f2d87eac5eed8a087d5ecd24277e6255b2/config.json)。它确认 `block_type=llama`、`activation_type=silu`、RMSNorm、`rope_theta=500000`、无 attention/MLP bias、无 dropout、untied embedding/head。旧加载器把原始 8B 的宽度/深度/heads/MLP 覆盖为 768/8/12/3072，并把 KV heads 同步为 12。

当前 [ImmuneConfig 与底层 ModelConfig](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/v3/model.py:26) 仍采用同一种 LLaDA block、SwiGLU、RMSNorm、RoPE500000 和 untied head，但选择 [GrammarTokenizer](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/grammar.py:129)：33 个基础 token 加 17 个 grammar token，共 50。角色 12 类、scope 6 类分别通过独立 embedding 编码，不需要扩充 token 词表。

[LegacyVocabulary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/v3/vocabulary.py:12) 只在 `ablation=A0` 启用 126,464 行词表；其 [映射契约](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/v3/legacy_vocabulary.json) 保存 46 个已用 token ID。A4 不能仅通过改数字就兼容旧 token ID。

## 当前实现与旧训练不等价的部分

| 项目 | 旧 no-ESMC v2 YAML / wrapper | 当前 A4 + research_next |
|---|---|---|
| Transformer 深宽 | 768 / 8 / 12 / 3072 | 相同 |
| 词表 | 126,464 行，旧 LLaDA token IDs | 50 token，小词表 IDs |
| 输入特征 | token embedding；不使用 ESMC 条件 | token + role + scope embedding |
| 位置 | 旧 wrapper 不传自定义 position_ids，使用拼接序列位置 | 显式链内 position_ids，链间 attention bias |
| target 长度 | heavy/light 固定 167/135 decoder slots，AA 后重复 EOS targets | full/fragment 容量 256/64；真实观测 AA 加一个 `<protd>`，后续槽位不监督 |
| 单次输入上限 | YAML max_length=1024 | 当前 production max_length=2048 |
| target token 限制 | 固定 canvas 输出允许 AA/chain EOS，其他 logits 被屏蔽 | target 输出允许 AA/`<protd>` |
| 数据 | `immune_v6_binding_only` | 计划入口为 `immune_v7_mhc_sequence_v1`，真实训练必须通过 readiness certificates |
| 分布式 | 2 节点 × 4 GPU FSDP | 8 卡容器内每个实验独立单卡 |
| 每实验 global batch | 4 × 8 accumulation × 8 GPUs = 256 | measured micro × accumulation = 256 |
| 采样 | YAML source_balanced_training=True | 默认 natural；仅 equal_source 变体使用相应 source sampler |
| 日程 | 1,000,000 horizon，warmup 2,000 | 当前 pilot horizon 10,000，warmup100，stop最多5,000 |

旧位置与 target masking 证据：[旧 `_denoise` 调用 decoder](/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_fusion_model.py:835)。当前位置与 target 特征契约：[A4 model](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/v3/model.py:70)、[A4 layout](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/v3/layout.py:165)。训练日程证据：[旧 YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/protein_llada270m_noesmc_diffusion_v2_immune_v6_2node4gpu_1m.yml:98)、[当前 runner 的 train](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/run_immune_research.py:346)。

不能把 A0 简单称为旧 v2 checkpoint 的完全复现：它虽具有相同参数量和旧词表，但当前 v3 A0 的训练 softmax 不应用 A4 的 allowed-target mask，而旧 v2 固定 canvas 的 wrapper 会屏蔽 AA/EOS 之外的 target logits。旧 v2 与 A4 的真实数据、布局、位置、采样和日程也不同，总 loss 不宜直接排名。

## “基于270”的两种明确口径

**保留项目既有 LLaDA 主干，并继续蛋白小词表。** 当前 A4 已满足主干继承关系，不能因总参数75.6M而称其主干更小。后续机制探索应保留 768/8/12/3072 的匹配 control，并把“75.6M 总参数、与旧270M相同 Transformer 主干”写清楚。这是避免重新引入194M文本词表参数的默认解释，最终以用户确认的口径为准。

**需要约270M的有效蛋白模型总参数。** 这是新尺度实验，需要显式改变深宽，而不是称为恢复旧尺寸。按现有无 bias、untied、50-token A4 的结构公式：

`P = L × (4d² + 3df + 2d) + d + 2Vd + (12 + 6)d + 2h`

| 数学候选 | d / L / heads / MLP | 总参数 |
|---|---|---:|
| 增宽与加深 | 1024 / 16 / 16 / 4096 | 268,590,112 |
| 保持宽度、主要加深 | 768 / 28 / 12 / 3072 | 264,375,576 |

这些只是参数公式计算，未构造、profile、训练或批准为新基线。两者相对75.6M control会同时改变容量与优化条件；应有同尺寸无新增机制的 control，并分别比较训练 token/step 预算与实测计算成本，不能把尺寸收益归因于新模块。

若确实要求保留旧126,464行词表以比较精确legacy270规模，需将词表选择与 A4 layout 分开配置，做合法 token-ID 映射、保存/恢复词表契约及正确 allowed-target mask；**不能直接把 research_next 的 A4 改成 A0**，因为其基类要求 A4，前向依赖 role/scope/chain bias。旧 checkpoint 参数名、词表 ID 和 wrapper 也不同，不能 `strict=False` 静默加载当作无损继承。

## 适配与 profile 必须一起变更的接口

1. 把明确尺寸/词表/layout 规格同时传入 model、collator、profile child 和 train；不要只改一个模型构造位置。当前 [config_for](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/run_immune_research.py:113)、[profile 验证](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/run_immune_research.py:136)、[新 runner profile 摘要](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/run_immune_architecture.py:83) 和 [晋级助手 production_config](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/enqueue_immune_architecture.py:50) 都假设同一默认 `ImmuneConfig(A4)`，新尺度须一致修改并独立 release。
2. 检查深度相关机制预算：[R17](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/research_next/model.py:93) 是前一半层链隔离，扩到16层会自动变成8层隔离；R18交替 local/global 的层数也随深度变化。要记录绝对层数与比例，不能视为完全相同的架构消融。
3. 保存/恢复需锁定具体 config、vocabulary、variant options、source SHA、mask protocol 和噪声 RNG 规则。旧270、现75.6和新大尺度模型不能共享 profile目录或续训 optimizer；扩大step只在相同训练horizon等provenance下恢复。
4. 重新测每个新规格/机制的显存，保持真实最大输入长度、capacity、whole-layer checkpoint、FP32参数/Adam状态与BF16 autocast、实际目标词表/CE以及同一GPU。每个 OOM 档必须退出独立子进程；通过内存证据才选 microbatch，并令 accumulation 保持 global256。
5. 当前 synthetic 模式最多100步、train最多5000步。更长且更广的人工验证应使用明确的人工数据协议与独立输出，不得把人工500步叫成真实语料训练；真实数据分支保留 `assert_ready` 先行。

当前只读 profile 证据：[R12 profile](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/profiles/r12_architecture_control/summary.json) 的 d768/L8、50-token A4 在 A100-SXM4-80GB 上推荐 micro16，测得 peak allocated40.0208GiB；[R13](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/profiles/r13_head_output_gate/summary.json) 为40.1154GiB，[R15](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/profiles/r15_layerscale/summary.json) 为40.0209GiB。**这些数据仅适用于保存的旧规格与 source hash，不支持推断新270有效参数模型的batch或吞吐。**

本次仅完成基线核实。没有新增模型实验、初步真实训练、queue任务或 checkpoint。
