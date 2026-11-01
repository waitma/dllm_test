# 三种子完整架构确认

更新时间：2026-11-01T05:45:36Z UTC；完成9/9；decision=not_yet_stable。

A0原架构、B0=A5全局注意力、B1=全局＋可学习残差。配对seed42/43/44，各随机初始化3epoch/3828步；共同micro32/global512，原V8形式与固定10%数据。
候选来自既有screen筛选；所有九组完成后才对confirm留出终点作配对统计。训练过程中只用screen曲线观察，不按confirm结果选checkpoint。原抗体特异性五折评估一并保留，属于辅助验证。
主要门槛：三种子增益全同向，5000次seed→来源内受体cluster配对bootstrap，三对模型×两终点共六项比较使用Bonferroni(99.1667%双侧区间)，区间下界>0；生成平均相对CE增益≥2%并通过原解码质量guard≥2/3；表征平均AUROC增益≥0.005且平均AUPRC不回退。只宣告通过的方向，不声称全面提升。

| 变体 | seed | 状态 | step | confirm整链AAR | confirm CDR AAR | confirm AUROC | confirm AUPRC |
|---|---:|---|---:|---:|---:|---:|---:|
| A0 | 42 | completed | 3828 | 0.1096 | 0.8042 | 0.8898 | 0.8022 |
| B0 | 42 | completed | 3828 | 0.3486 | 0.8040 | 0.8843 | 0.7318 |
| B1 | 42 | completed | 3828 | 0.3340 | 0.8061 | 0.8807 | 0.7327 |
| A0 | 43 | completed | 3828 | 0.0301 | 0.8048 | 0.8925 | 0.8260 |
| B0 | 43 | completed | 3828 | 0.3436 | 0.8012 | 0.9126 | 0.7828 |
| B1 | 43 | completed | 3828 | 0.3509 | 0.8039 | 0.8871 | 0.7841 |
| A0 | 44 | completed | 3828 | 0.1737 | 0.8017 | 0.8625 | 0.7328 |
| B0 | 44 | completed | 3828 | 0.3454 | 0.8021 | 0.8716 | 0.7712 |
| B1 | 44 | completed | 3828 | 0.3622 | 0.8037 | 0.8670 | 0.7401 |

![完整screen训练/验证曲线](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/confirmation_20261031/curves.png)

[逐步原始CSV](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/confirmation_20261031/all_curves.csv) · [终点汇总](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/confirmation_20261031/summary.csv) · [冻结统计判定](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/confirmation_20261031/decision.json)

尚未完成全部配对不作稳定性结论；门槛不通过则目标仍未完成，保留负结果并继续设计下一轮，不降低阈值、不挑最好seed。三种子与小样本受体聚类限制保留；不是同源独立或普适下游能力证明。
训练GPU小时不含评估和checkpoint保存；模型按实验最新/最佳生成/最佳表征及epoch末节点保存完整恢复状态。

## 终点统计

```json
{
  "status": "complete",
  "repetitions": 5000,
  "familywise_alpha": 0.05,
  "comparisons": 6,
  "interval": "99.1667% two-sided paired hierarchical percentile interval; Bonferroni over 3 pairs x 2 endpoints",
  "resampling": "training seeds, then exact receptor clusters within each source; paired observations",
  "pairs": {
    "B0_vs_A0": {
      "generation": {
        "mean_gain": 0.08552894034525689,
        "sample_sd": 0.12091203298795576,
        "per_seed_gain": [
          0.09946187285749285,
          0.19887093083673257,
          -0.04174598265845472
        ],
        "adjusted_ci": [
          -0.04399713217576945,
          0.20158070341393722
        ],
        "all_three_seeds_improve": false,
        "decoder_guard_passes": 1,
        "stable_improvement": false
      },
      "representation": {
        "mean_gain": 0.007893139040680008,
        "sample_sd": 0.012793750762419866,
        "per_seed_gain": [
          -0.0054644808743170015,
          0.02003642987249543,
          0.009107468123861595
        ],
        "adjusted_ci": [
          -0.03333462103263492,
          0.050580418870566614
        ],
        "all_three_seeds_improve": false,
        "mean_auprc_gain": -0.02505537641201631,
        "stable_improvement": false
      }
    },
    "B1_vs_A0": {
      "generation": {
        "mean_gain": 0.08854932917883214,
        "sample_sd": 0.11232874805148635,
        "per_seed_gain": [
          0.09946595346214625,
          0.19502121012423723,
          -0.028839176049887068
        ],
        "adjusted_ci": [
          -0.03190530630771728,
          0.19815742589086904
        ],
        "all_three_seeds_improve": false,
        "decoder_guard_passes": 2,
        "stable_improvement": false
      },
      "representation": {
        "mean_gain": -0.0033394049787492386,
        "sample_sd": 0.0070741834837468355,
        "per_seed_gain": [
          -0.009107468123861595,
          -0.0054644808743168904,
          0.00455373406193077
        ],
        "adjusted_ci": [
          -0.04141548948958332,
          0.03326624872387736
        ],
        "all_three_seeds_improve": false,
        "mean_auprc_gain": -0.03470114483528867,
        "stable_improvement": false
      }
    },
    "B1_vs_B0": {
      "generation": {
        "mean_gain": 0.00203219172382984,
        "sample_sd": 0.00909726823900887,
        "per_seed_gain": [
          0.00028352060767400226,
          -0.006063807238505696,
          0.011876861802321214
        ],
        "adjusted_ci": [
          -0.007492041235139404,
          0.012873703415219163
        ],
        "all_three_seeds_improve": false,
        "decoder_guard_passes": 1,
        "stable_improvement": false
      },
      "representation": {
        "mean_gain": -0.011232544019429247,
        "sample_sd": 0.012365155952893891,
        "per_seed_gain": [
          -0.0036429872495445936,
          -0.025500910746812322,
          -0.004553734061930825
        ],
        "adjusted_ci": [
          -0.045625114849823886,
          0.01828039340783749
        ],
        "all_three_seeds_improve": false,
        "mean_auprc_gain": -0.00964576842327236,
        "stable_improvement": false
      }
    }
  },
  "limitation": "Three seeds; fixed small protected probes; exact receptor clustering is not homology-independent validation.",
  "accepted_vs_original": {},
  "decision": "not_yet_stable",
  "next_action": "Goal remains unfinished: inspect failure directions and preregister a new architecture batch; do not lower thresholds or claim success.",
  "observations_sha256": "a28cf365680bce930007d49a9f70cb48cdaf829af24160b13cc7ce46bade398e"
}
```
