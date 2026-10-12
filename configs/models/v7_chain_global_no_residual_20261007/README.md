# V7 链内全注意力 / 全局交替，无残差缩放

用户第二项变更接续第一项：使用既有 FactorialModel 的 cv_segment_global。此处segment就是A4中由chain_ids划分的链块，不是任意长度窗口。无需复制注意力实现或改写历史f001/f011定义。

8层按1起编号：1/3/5/7层同链所有有效位置可直接互相关注；2/4/6/8层全局有效位置可互相关注。移除radius32，不改变每链位置编码。同链层排除其他链和grammar前缀作为桥梁，padding不作为key；padding query的安全回退不用于有效位置输出。保留目标容量内原本attended的MASK余槽，不按隐藏真实长度裁掉这些位置。全局层已经融合的信息可以进入后续同链层，不宣称整网始终链间独立。

attention/FFN残差系数均1，保持第一项改动。参数75,601,176，A4/8层/768维/12头/FFN3072/full256/fragment64/max2048等其余模型设置不变。where修复和SDPA auto策略继承；此为稠密mask语义变化，不保证提速。

配置 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_chain_global_no_residual_20261007/research_config.json；版本身份 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_chain_global_no_residual_20261007/version.json。该配置接替上一阶段 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_f001_no_residual_20261007/research_config.json 作为本次新版本选择，旧配置保留可追溯性。模型从零构造，没有把50082旧权重或优化器当作兼容原生续训。

在激活pllm、仓库根目录中用现有类构造：

```python
import json
from pathlib import Path
from dllm.pipelines.immune_llada.v3.model import ImmuneConfig
from dllm.pipelines.immune_llada.research_refined.factorial import FactorialModel
s = json.loads(Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_chain_global_no_residual_20261007/research_config.json').read_text())
model = FactorialModel(ImmuneConfig(**s['base']), s['variant'])
```

正式入口已经支持 --variant cv_segment_global；配置不是可直接提交的YAML或checkpoint。新训练需独立源码封版、输出目录及既有数据门禁。本次未提交/启动/重启任务。完整模型CPU检查和已有梯度隔离/重计算/保存恢复回归记录放在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_chain_global_no_residual_20261007，只保留日志与JSON，测试临时权重自动清理。
