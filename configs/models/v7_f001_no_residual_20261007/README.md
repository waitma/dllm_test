# V7 f001：去掉残差缩放

本版本从正式 f011/50082 的模型配置派生，复用现有 FactorialModel 的 f001；不是新的消融因子命名。只删除 options.residual_scale=0.1，ArchitectureBlock 默认使用1.0，attention/FFN均恢复普通残差。保留局部半径32与全局层交替、A4、768/8/12/3072、50词表、角色/范围嵌入、链内RoPE、可训练chain bias、256/64容量及2048上限。参数量仍为75,601,176。

配置：/vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_f001_no_residual_20261007/research_config.json。版本身份：/vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_f001_no_residual_20261007/version.json。这是模型配置，不是checkpoint目录或可直接提交的平台YAML。未启动训练。

在激活pllm且位于仓库根目录后，可用既有模型类构造：

```python
import json
from pathlib import Path
from dllm.pipelines.immune_llada.v3.model import ImmuneConfig
from dllm.pipelines.immune_llada.research_refined.factorial import FactorialModel
settings = json.loads(Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_f001_no_residual_20261007/research_config.json').read_text())
model = FactorialModel(ImmuneConfig(**settings['base']), settings['variant'])
```

正式入口run_v7_formal.py已经支持 --variant f001；新的实际训练仍需要独立输出目录、源码快照/manifest及原数据审核。当前工作区where修复和SDPA策略均继承，默认auto；如果做残差效果对照，两个版本必须统一后端和其余训练设置。未把此配置接入旧冻结任务或改写旧f011。

配置默认用于从零构造模型。不能仅修改旧checkpoint的variant标签后宣称原生续训：0.1→1.0会立即改变相同权重的前向函数；加载旧权重作初始化需另行明确登记。

校验目录：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_f001_no_residual_20261007。只保留JSON/日志，无测试权重。
