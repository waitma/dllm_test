# 架构实验工作池监督记录

最后观测：2026-10-03 22:17:54 UTC。

范围：8 GPU 工作池全部方法与步数。每个 job_id 单独留档，20 步与 100 步等不同实例不互相覆盖。首轮完成标记仅指 R00–R11 各自最早的 20 步诊断。

监督只读取证据，不改变活跃训练源码、不取消或重提平台任务、不自动加入新实验。后台监听持续落盘，当前对话中的监督 agent 负责解释与排错。

状态计数：{&quot;pending&quot;: 3, &quot;running&quot;: 0, &quot;succeeded&quot;: 93, &quot;failed&quot;: 41}。全部首轮结束：True；全部当前实验结束：False。

指纹校验以 release_map.json 中每个 variant 的 source_root 为准；无映射时使用当前仓库。旧源码快照和旧 profile 应一同保留。

mode=train 只是入口模式。kind=synthetic_generalization_diagnostic 仍为人工字符串诊断，不代表真实语料训练；kind 未落盘时不推断数据性质。CUDA operator诊断单列cases；D01 Trainer runtime诊断单列实际3个optimizer windows，training_started指人工训练，corpus_training_started=false。

| job_id | 方法 | 模式 | kind | 状态 | 完成/目标 step | 指标层级 | micro-bs | 峰值 GiB | loss | AA acc | end acc | source SHA | 关注项 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1790824088674261408\_d01\_pilot\_runtime\_gpu\_r20\_v1\_9af03156 | d01\_pilot\_runtime\_gpu\_r20\_v1 | profile-synthetic | trainer\_gpu\_runtime\_functional\_diagnostic | failed | ?/3 optimizer windows | artificial Trainer windows | — | — | — | — | — | matched (526 files) | trainer\_diagnostic\_execution\_failed |
| 1790827241091679008\_d01\_pilot\_runtime\_gpu\_r20\_v1\_ea0bba4a | d01\_pilot\_runtime\_gpu\_r20\_v1 | profile-synthetic | trainer\_gpu\_runtime\_functional\_diagnostic | failed | ?/3 optimizer windows | artificial Trainer windows | — | — | — | — | — | matched (526 files) | trainer\_diagnostic\_execution\_failed |
| 1790831680607223314\_d01\_pilot\_runtime\_gpu\_r20\_v1\_48f59959 | d01\_pilot\_runtime\_gpu\_r20\_v1 | profile-synthetic | trainer\_gpu\_runtime\_functional\_diagnostic | succeeded | 3/3 optimizer windows | artificial Trainer windows | — | — | — | — | — | matched (526 files) |  |
| 1790906045287527921\_d02\_pilot\_seed\_gpu\_r20\_seed42\_v2\_786cccfe | d02\_pilot\_seed\_gpu\_r20\_seed42\_v2 | profile-synthetic | unknown | succeeded | 0/1 | mask100 | — | — | — | — | — | matched (532 files) | inconsistent\_completion |
| 1790906045298475453\_d02\_pilot\_seed\_gpu\_r20\_seed43\_v2\_a9a6be7e | d02\_pilot\_seed\_gpu\_r20\_seed43\_v2 | profile-synthetic | unknown | succeeded | 0/1 | mask100 | — | — | — | — | — | matched (532 files) | inconsistent\_completion |
| 1790906045305519301\_d02\_pilot\_seed\_gpu\_r21\_seed42\_v2\_82caee86 | d02\_pilot\_seed\_gpu\_r21\_seed42\_v2 | profile-synthetic | unknown | succeeded | 0/1 | mask100 | — | — | — | — | — | matched (534 files) | inconsistent\_completion |
| 1790906045312184986\_d02\_pilot\_seed\_gpu\_r21\_seed43\_v2\_1e0b6229 | d02\_pilot\_seed\_gpu\_r21\_seed43\_v2 | profile-synthetic | unknown | failed | 0/1 | mask100 | — | — | — | — | — | pending (0 files) | runner\_failure |
| 1790918753525278168\_d02\_pilot\_seed\_gpu\_r21\_seed43\_v2\_82aa7438 | d02\_pilot\_seed\_gpu\_r21\_seed43\_v2 | profile-synthetic | unknown | succeeded | 0/1 | mask100 | — | — | — | — | — | mismatch (534 files) | source\_changed, inconsistent\_completion |
| 1790906045319006048\_d02\_pilot\_seed\_gpu\_r24\_seed42\_v2\_57a0529c | d02\_pilot\_seed\_gpu\_r24\_seed42\_v2 | profile-synthetic | unknown | succeeded | 0/1 | mask100 | — | — | — | — | — | matched (534 files) | inconsistent\_completion |
| 1790906045325720427\_d02\_pilot\_seed\_gpu\_r24\_seed43\_v2\_32829b7f | d02\_pilot\_seed\_gpu\_r24\_seed43\_v2 | profile-synthetic | unknown | succeeded | 0/1 | mask100 | — | — | — | — | — | matched (534 files) | inconsistent\_completion |
| 1790906045332754556\_d02\_pilot\_seed\_gpu\_r26\_seed42\_v2\_83205e87 | d02\_pilot\_seed\_gpu\_r26\_seed42\_v2 | profile-synthetic | unknown | succeeded | 0/1 | mask100 | — | — | — | — | — | matched (534 files) | inconsistent\_completion |
| 1790906045339673599\_d02\_pilot\_seed\_gpu\_r26\_seed43\_v2\_19e36eda | d02\_pilot\_seed\_gpu\_r26\_seed43\_v2 | profile-synthetic | unknown | succeeded | 0/1 | mask100 | — | — | — | — | — | matched (534 files) | inconsistent\_completion |
| 1790910822468354321\_d03\_r21\_seed43\_tiny\_gpu\_observation\_v1\_58e84271 | d03\_r21\_seed43\_tiny\_gpu\_observation\_v1 | profile-synthetic | unknown | failed | 0/1 | mask100 | — | — | — | — | — | pending (0 files) | runner\_failure |
| 1790911529399018156\_d03\_r21\_seed43\_tiny\_gpu\_observation\_v1\_6acb659f | d03\_r21\_seed43\_tiny\_gpu\_observation\_v1 | profile-synthetic | unknown | failed | 0/1 | mask100 | — | — | — | — | — | pending (0 files) | runner\_failure |
| 1790912303689399341\_d03\_r21\_seed43\_tiny\_gpu\_observation\_v1\_324e2b1c | d03\_r21\_seed43\_tiny\_gpu\_observation\_v1 | profile-synthetic | unknown | succeeded | 0/1 | mask100 | — | — | — | — | — | pending (0 files) | inconsistent\_completion |
| 1790914034411779826\_d03\_r21\_seed43\_tiny\_gpu\_observation\_v1\_e950d2a0 | d03\_r21\_seed43\_tiny\_gpu\_observation\_v1 | profile-synthetic | unknown | succeeded | 0/1 | mask100 | — | — | — | — | — | pending (0 files) | inconsistent\_completion |
| 1790915494600058674\_d03\_r21\_seed43\_tiny\_gpu\_observation\_v1\_5d224d8a | d03\_r21\_seed43\_tiny\_gpu\_observation\_v1 | profile-synthetic | unknown | succeeded | 0/1 | mask100 | — | — | — | — | — | pending (0 files) | inconsistent\_completion |
| 1790917182885479822\_d03\_r21\_seed43\_tiny\_gpu\_observation\_v1\_6c5bc4a4 | d03\_r21\_seed43\_tiny\_gpu\_observation\_v1 | profile-synthetic | unknown | succeeded | 0/1 | mask100 | — | — | — | — | — | pending (0 files) | inconsistent\_completion |
| 1790923015794991437\_d04\_r21\_CE\_learning\_v1\_4dc72fa7 | d04\_r21\_CE\_learning\_v1 | profile-synthetic | synthetic\_learning\_transport | succeeded | 0/25 | mask100 | — | — | — | — | — | pending (0 files) | inconsistent\_completion |
| 1790924220633707073\_d04\_r21\_CE\_learning\_v1\_dd5651b2 | d04\_r21\_CE\_learning\_v1 | train | synthetic\_learning\_transport | succeeded | 0/500 | trainer latest | — | — | — | — | — | pending (0 files) | inconsistent\_completion |
| 1790926923146311111\_d05\_r20\_seed42\_learning\_v1\_cc8db6a6 | d05\_r20\_seed42\_learning\_v1 | profile-synthetic | synthetic\_learning\_transport | succeeded | 0/25 | mask100 | — | — | — | — | — | pending (0 files) | inconsistent\_completion |
| 1790928092530413132\_d05\_r20\_seed42\_learning\_v1\_590a927e | d05\_r20\_seed42\_learning\_v1 | train | synthetic\_learning\_transport | succeeded | 0/500 | trainer latest | — | — | — | — | — | pending (0 files) | inconsistent\_completion |
| 1790926923157371090\_d05\_r20\_seed43\_learning\_v1\_4b31a0c1 | d05\_r20\_seed43\_learning\_v1 | profile-synthetic | synthetic\_learning\_transport | succeeded | 0/25 | mask100 | — | — | — | — | — | pending (0 files) | inconsistent\_completion |
| 1790928092539965903\_d05\_r20\_seed43\_learning\_v1\_582ef7c9 | d05\_r20\_seed43\_learning\_v1 | train | synthetic\_learning\_transport | succeeded | 0/500 | trainer latest | — | — | — | — | — | pending (0 files) | inconsistent\_completion |
| 1790926923165197398\_d05\_r21\_seed42\_learning\_v1\_12e96433 | d05\_r21\_seed42\_learning\_v1 | profile-synthetic | synthetic\_learning\_transport | succeeded | 0/25 | mask100 | — | — | — | — | — | pending (0 files) | inconsistent\_completion |
| 1790928092548359830\_d05\_r21\_seed42\_learning\_v1\_c91f2623 | d05\_r21\_seed42\_learning\_v1 | train | synthetic\_learning\_transport | succeeded | 0/500 | trainer latest | — | — | — | — | — | pending (0 files) | inconsistent\_completion |
| 1790926923172990505\_d05\_r24\_seed42\_learning\_v1\_85f994fa | d05\_r24\_seed42\_learning\_v1 | profile-synthetic | synthetic\_learning\_transport | succeeded | 0/25 | mask100 | — | — | — | — | — | pending (0 files) | inconsistent\_completion |
| 1790928092555818254\_d05\_r24\_seed42\_learning\_v1\_36c7b5d6 | d05\_r24\_seed42\_learning\_v1 | train | synthetic\_learning\_transport | succeeded | 0/500 | trainer latest | — | — | — | — | — | pending (0 files) | inconsistent\_completion |
| 1790926923180466754\_d05\_r24\_seed43\_learning\_v1\_47ed9473 | d05\_r24\_seed43\_learning\_v1 | profile-synthetic | synthetic\_learning\_transport | succeeded | 0/25 | mask100 | — | — | — | — | — | pending (0 files) | inconsistent\_completion |
| 1790928092563094796\_d05\_r24\_seed43\_learning\_v1\_89ccffc5 | d05\_r24\_seed43\_learning\_v1 | train | synthetic\_learning\_transport | succeeded | 0/500 | trainer latest | — | — | — | — | — | pending (0 files) | inconsistent\_completion |
| 1790926923188130232\_d05\_r26\_seed42\_learning\_v1\_5c83be2b | d05\_r26\_seed42\_learning\_v1 | profile-synthetic | synthetic\_learning\_transport | succeeded | 0/25 | mask100 | — | — | — | — | — | pending (0 files) | inconsistent\_completion |
| 1790928092570644812\_d05\_r26\_seed42\_learning\_v1\_15eb88d3 | d05\_r26\_seed42\_learning\_v1 | train | synthetic\_learning\_transport | succeeded | 0/500 | trainer latest | — | — | — | — | — | pending (0 files) | inconsistent\_completion |
| 1790926923199830052\_d05\_r26\_seed43\_learning\_v1\_1b590a01 | d05\_r26\_seed43\_learning\_v1 | profile-synthetic | synthetic\_learning\_transport | succeeded | 0/25 | mask100 | — | — | — | — | — | pending (0 files) | inconsistent\_completion |
| 1790928092580810482\_d05\_r26\_seed43\_learning\_v1\_ea041411 | d05\_r26\_seed43\_learning\_v1 | train | synthetic\_learning\_transport | succeeded | 0/500 | trainer latest | — | — | — | — | — | pending (0 files) | inconsistent\_completion |
| 1790944282201965405\_g01\_r21\_world8\_member\_0\_64116e94 | g01\_r21\_world8\_member\_0 | profile-synthetic | unknown | failed | 0/6 | mask100 | — | — | — | — | — | pending (0 files) | runner\_failure |
| 1790944282201974433\_g01\_r21\_world8\_member\_1\_2bce92c7 | g01\_r21\_world8\_member\_1 | profile-synthetic | unknown | failed | 0/6 | mask100 | — | — | — | — | — | pending (0 files) | runner\_failure |
| 1790944282201980741\_g01\_r21\_world8\_member\_2\_29d5bdcf | g01\_r21\_world8\_member\_2 | profile-synthetic | unknown | failed | 0/6 | mask100 | — | — | — | — | — | pending (0 files) | runner\_failure |
| 1790944282201984472\_g01\_r21\_world8\_member\_3\_5371c8df | g01\_r21\_world8\_member\_3 | profile-synthetic | unknown | failed | 0/6 | mask100 | — | — | — | — | — | pending (0 files) | runner\_failure |
| 1790944282201987921\_g01\_r21\_world8\_member\_4\_b0826020 | g01\_r21\_world8\_member\_4 | profile-synthetic | unknown | failed | 0/6 | mask100 | — | — | — | — | — | pending (0 files) | runner\_failure |
| 1790944282201990944\_g01\_r21\_world8\_member\_5\_2c7fe185 | g01\_r21\_world8\_member\_5 | profile-synthetic | unknown | failed | 0/6 | mask100 | — | — | — | — | — | pending (0 files) | runner\_failure |
| 1790944282201993846\_g01\_r21\_world8\_member\_6\_fb50cd24 | g01\_r21\_world8\_member\_6 | profile-synthetic | unknown | failed | 0/6 | mask100 | — | — | — | — | — | pending (0 files) | runner\_failure |
| 1790944282201997144\_g01\_r21\_world8\_member\_7\_bf02438a | g01\_r21\_world8\_member\_7 | profile-synthetic | unknown | failed | 0/6 | mask100 | — | — | — | — | — | pending (0 files) | runner\_failure |
| 1790948366803107556\_g02\_r21\_world8\_member\_0\_04be8e50 | g02\_r21\_world8\_member\_0 | profile-synthetic | unknown | failed | 0/6 | mask100 | — | — | — | — | — | pending (0 files) | runner\_failure |
| 1790948366803116904\_g02\_r21\_world8\_member\_1\_54a65390 | g02\_r21\_world8\_member\_1 | profile-synthetic | unknown | failed | 0/6 | mask100 | — | — | — | — | — | pending (0 files) | runner\_failure |
| 1790948366803122484\_g02\_r21\_world8\_member\_2\_55767f54 | g02\_r21\_world8\_member\_2 | profile-synthetic | unknown | failed | 0/6 | mask100 | — | — | — | — | — | pending (0 files) | runner\_failure |
| 1790948366803125661\_g02\_r21\_world8\_member\_3\_97773210 | g02\_r21\_world8\_member\_3 | profile-synthetic | unknown | failed | 0/6 | mask100 | — | — | — | — | — | pending (0 files) | runner\_failure |
| 1790948366803128800\_g02\_r21\_world8\_member\_4\_3a3aa296 | g02\_r21\_world8\_member\_4 | profile-synthetic | unknown | failed | 0/6 | mask100 | — | — | — | — | — | pending (0 files) | runner\_failure |
| 1790948366803131655\_g02\_r21\_world8\_member\_5\_b5922d3c | g02\_r21\_world8\_member\_5 | profile-synthetic | unknown | failed | 0/6 | mask100 | — | — | — | — | — | pending (0 files) | runner\_failure |
| 1790948366803134323\_g02\_r21\_world8\_member\_6\_9cc6aa7a | g02\_r21\_world8\_member\_6 | profile-synthetic | unknown | failed | 0/6 | mask100 | — | — | — | — | — | pending (0 files) | runner\_failure |
| 1790948366803137277\_g02\_r21\_world8\_member\_7\_e2a05b6f | g02\_r21\_world8\_member\_7 | profile-synthetic | unknown | failed | 0/6 | mask100 | — | — | — | — | — | pending (0 files) | runner\_failure |
| 1790953672752444849\_g03\_r21\_world8\_member\_0\_674b9eed | g03\_r21\_world8\_member\_0 | profile-synthetic | unknown | failed | 0/6 | mask100 | — | — | — | — | — | pending (0 files) | runner\_failure |
| 1790953672752451387\_g03\_r21\_world8\_member\_1\_e6edbcbb | g03\_r21\_world8\_member\_1 | profile-synthetic | unknown | failed | 0/6 | mask100 | — | — | — | — | — | pending (0 files) | runner\_failure |
| 1790953672752455089\_g03\_r21\_world8\_member\_2\_e4c41241 | g03\_r21\_world8\_member\_2 | profile-synthetic | unknown | failed | 0/6 | mask100 | — | — | — | — | — | pending (0 files) | runner\_failure |
| 1790953672752458546\_g03\_r21\_world8\_member\_3\_f4bae8db | g03\_r21\_world8\_member\_3 | profile-synthetic | unknown | failed | 0/6 | mask100 | — | — | — | — | — | pending (0 files) | runner\_failure |
| 1790953672752461949\_g03\_r21\_world8\_member\_4\_0d9e8e48 | g03\_r21\_world8\_member\_4 | profile-synthetic | unknown | failed | 0/6 | mask100 | — | — | — | — | — | pending (0 files) | runner\_failure |
| 1790953672752465889\_g03\_r21\_world8\_member\_5\_7e6275aa | g03\_r21\_world8\_member\_5 | profile-synthetic | unknown | failed | 0/6 | mask100 | — | — | — | — | — | pending (0 files) | runner\_failure |
| 1790953672752469183\_g03\_r21\_world8\_member\_6\_6fcb1ff2 | g03\_r21\_world8\_member\_6 | profile-synthetic | unknown | failed | 0/6 | mask100 | — | — | — | — | — | pending (0 files) | runner\_failure |
| 1790953672752472296\_g03\_r21\_world8\_member\_7\_21a90bb3 | g03\_r21\_world8\_member\_7 | profile-synthetic | unknown | failed | 0/6 | mask100 | — | — | — | — | — | pending (0 files) | runner\_failure |
| 1790968470149719891\_p01\_v7\_r20\_seed42\_838aa052 | p01\_v7\_r20\_seed42 | train | exploratory\_architecture\_reconstruction | succeeded | 3125/3125 | trainer latest | — | — | — | — | — | mismatch (46 files) | source\_changed |
| 1790968636135679016\_p01\_v7\_r20\_seed43\_abf75de1 | p01\_v7\_r20\_seed43 | train | exploratory\_architecture\_reconstruction | succeeded | 3125/3125 | trainer latest | — | — | — | — | — | mismatch (46 files) | source\_changed |
| 1790968470161653699\_p01\_v7\_r21\_seed42\_ccb7e981 | p01\_v7\_r21\_seed42 | train | exploratory\_architecture\_reconstruction | succeeded | 3125/3125 | trainer latest | — | — | — | — | — | mismatch (46 files) | source\_changed |
| 1790968636145868351\_p01\_v7\_r21\_seed43\_3f1b2997 | p01\_v7\_r21\_seed43 | train | exploratory\_architecture\_reconstruction | succeeded | 3125/3125 | trainer latest | — | — | — | — | — | mismatch (46 files) | source\_changed |
| 1790968636111438133\_p01\_v7\_r24\_seed42\_b0d0a2be | p01\_v7\_r24\_seed42 | train | exploratory\_architecture\_reconstruction | succeeded | 3125/3125 | trainer latest | — | — | — | — | — | mismatch (46 files) | source\_changed |
| 1790968636157009559\_p01\_v7\_r24\_seed43\_a9336889 | p01\_v7\_r24\_seed43 | train | exploratory\_architecture\_reconstruction | succeeded | 3125/3125 | trainer latest | — | — | — | — | — | mismatch (46 files) | source\_changed |
| 1790968636125595682\_p01\_v7\_r26\_seed42\_be605dad | p01\_v7\_r26\_seed42 | train | exploratory\_architecture\_reconstruction | succeeded | 3125/3125 | trainer latest | — | — | — | — | — | mismatch (46 files) | source\_changed |
| 1790968636168787261\_p01\_v7\_r26\_seed43\_6a3a3518 | p01\_v7\_r26\_seed43 | train | exploratory\_architecture\_reconstruction | succeeded | 3125/3125 | trainer latest | — | — | — | — | — | mismatch (46 files) | source\_changed |
| 1790974728135266312\_p02\_v7\_cv\_late\_fusion\_seed42\_7b904983 | p02\_v7\_cv\_late\_fusion\_seed42 | train | exploratory\_architecture\_reconstruction | succeeded | 3750/3750 | trainer latest | — | — | — | — | — | mismatch (49 files) | source\_changed |
| 1790974728176231879\_p02\_v7\_cv\_late\_fusion\_seed43\_a68e3b52 | p02\_v7\_cv\_late\_fusion\_seed43 | train | exploratory\_architecture\_reconstruction | succeeded | 3750/3750 | trainer latest | — | — | — | — | — | mismatch (49 files) | source\_changed |
| 1790974728125458819\_p02\_v7\_cv\_segment\_global\_seed42\_fb1caf55 | p02\_v7\_cv\_segment\_global\_seed42 | train | exploratory\_architecture\_reconstruction | succeeded | 3750/3750 | trainer latest | — | — | — | — | — | mismatch (49 files) | source\_changed |
| 1790974728166373677\_p02\_v7\_cv\_segment\_global\_seed43\_8938fe7e | p02\_v7\_cv\_segment\_global\_seed43 | train | exploratory\_architecture\_reconstruction | succeeded | 3750/3750 | trainer latest | — | — | — | — | — | mismatch (49 files) | source\_changed |
| 1790974728104139450\_p02\_v7\_f000\_seed42\_9a84aad6 | p02\_v7\_f000\_seed42 | train | exploratory\_architecture\_reconstruction | succeeded | 3750/3750 | trainer latest | — | — | — | — | — | mismatch (49 files) | source\_changed |
| 1790974728147557201\_p02\_v7\_f000\_seed43\_e4b3bc10 | p02\_v7\_f000\_seed43 | train | exploratory\_architecture\_reconstruction | succeeded | 3750/3750 | trainer latest | — | — | — | — | — | mismatch (49 files) | source\_changed |
| 1790974728116054138\_p02\_v7\_f001\_seed42\_e0e1afbe | p02\_v7\_f001\_seed42 | train | exploratory\_architecture\_reconstruction | succeeded | 3750/3750 | trainer latest | — | — | — | — | — | mismatch (49 files) | source\_changed |
| 1790974728156871579\_p02\_v7\_f001\_seed43\_e810d85a | p02\_v7\_f001\_seed43 | train | exploratory\_architecture\_reconstruction | succeeded | 3750/3750 | trainer latest | — | — | — | — | — | mismatch (49 files) | source\_changed |
| 1790981433321562451\_p04\_v7\_capacity\_febb28e3 | p04\_v7\_capacity | profile-synthetic | synthetic\_whole\_model\_capacity\_sweep | failed | 0/72 | mask100 | — | — | — | — | — | pending (0 files) | runner\_failure |
| 1790995583011921799\_p04\_v7\_cv\_late\_fusion\_seed42\_905230bb | p04\_v7\_cv\_late\_fusion\_seed42 | train | exploratory\_architecture\_reconstruction | failed | 0/3750 | trainer latest | — | — | — | — | — | mismatch (49 files) | source\_changed, runner\_failure |
| 1790995583011948104\_p04\_v7\_cv\_late\_fusion\_seed43\_e268efa5 | p04\_v7\_cv\_late\_fusion\_seed43 | train | unknown | pending | 0/3750 | trainer latest | — | — | — | — | — | pending (0 files) |  |
| 1790995583011879919\_p04\_v7\_cv\_segment\_global\_seed42\_ffdd2332 | p04\_v7\_cv\_segment\_global\_seed42 | train | exploratory\_architecture\_reconstruction | failed | 0/3750 | trainer latest | — | — | — | — | — | mismatch (49 files) | source\_changed, runner\_failure |
| 1790995583011900971\_p04\_v7\_cv\_segment\_global\_seed43\_d5fe067c | p04\_v7\_cv\_segment\_global\_seed43 | train | exploratory\_architecture\_reconstruction | failed | 0/3750 | trainer latest | — | — | — | — | — | mismatch (49 files) | source\_changed, runner\_failure |
| 1790995583011457554\_p04\_v7\_f000\_seed42\_539537fe | p04\_v7\_f000\_seed42 | train | exploratory\_architecture\_reconstruction | failed | 0/3750 | trainer latest | — | — | — | — | — | mismatch (49 files) | source\_changed, runner\_failure |
| 1790995583011557084\_p04\_v7\_f000\_seed43\_34d8ec5b | p04\_v7\_f000\_seed43 | train | exploratory\_architecture\_reconstruction | succeeded | 3750/3750 | trainer latest | — | — | — | — | — | mismatch (49 files) | source\_changed |
| 1790995583011630260\_p04\_v7\_f001\_seed42\_40c762d3 | p04\_v7\_f001\_seed42 | train | exploratory\_architecture\_reconstruction | succeeded | 3750/3750 | trainer latest | — | — | — | — | — | mismatch (49 files) | source\_changed |
| 1790995583011651525\_p04\_v7\_f001\_seed43\_6374caab | p04\_v7\_f001\_seed43 | train | exploratory\_architecture\_reconstruction | succeeded | 3750/3750 | trainer latest | — | — | — | — | — | mismatch (49 files) | source\_changed |
| 1790995583011584963\_p04\_v7\_f010\_seed42\_93cb62b2 | p04\_v7\_f010\_seed42 | train | exploratory\_architecture\_reconstruction | failed | 0/3750 | trainer latest | — | — | — | — | — | mismatch (49 files) | source\_changed, runner\_failure |
| 1790995583011607766\_p04\_v7\_f010\_seed43\_5b45614b | p04\_v7\_f010\_seed43 | train | exploratory\_architecture\_reconstruction | failed | 0/3750 | trainer latest | — | — | — | — | — | mismatch (49 files) | source\_changed, runner\_failure |
| 1790995583011673811\_p04\_v7\_f011\_seed42\_4863ec8f | p04\_v7\_f011\_seed42 | train | exploratory\_architecture\_reconstruction | succeeded | 3750/3750 | trainer latest | — | — | — | — | — | mismatch (49 files) | source\_changed |
| 1790995583011694915\_p04\_v7\_f011\_seed43\_3ff5b7c7 | p04\_v7\_f011\_seed43 | train | exploratory\_architecture\_reconstruction | succeeded | 3750/3750 | trainer latest | — | — | — | — | — | mismatch (49 files) | source\_changed |
| 1790995583011717074\_p04\_v7\_f100\_seed42\_ca6382b3 | p04\_v7\_f100\_seed42 | train | exploratory\_architecture\_reconstruction | succeeded | 3750/3750 | trainer latest | — | — | — | — | — | mismatch (49 files) | source\_changed |
| 1790995583011737505\_p04\_v7\_f100\_seed43\_87f53303 | p04\_v7\_f100\_seed43 | train | exploratory\_architecture\_reconstruction | succeeded | 3750/3750 | trainer latest | — | — | — | — | — | mismatch (49 files) | source\_changed |
| 1790995583011757747\_p04\_v7\_f101\_seed42\_850e3092 | p04\_v7\_f101\_seed42 | train | exploratory\_architecture\_reconstruction | succeeded | 3750/3750 | trainer latest | — | — | — | — | — | mismatch (49 files) | source\_changed |
| 1790995583011778233\_p04\_v7\_f101\_seed43\_e8a9c4d4 | p04\_v7\_f101\_seed43 | train | exploratory\_architecture\_reconstruction | failed | 0/3750 | trainer latest | — | — | — | — | — | mismatch (49 files) | source\_changed, runner\_failure |
| 1790995583011798231\_p04\_v7\_f110\_seed42\_adafd438 | p04\_v7\_f110\_seed42 | train | exploratory\_architecture\_reconstruction | failed | 0/3750 | trainer latest | — | — | — | — | — | mismatch (49 files) | source\_changed, runner\_failure |
| 1790995583011818395\_p04\_v7\_f110\_seed43\_2f6da59d | p04\_v7\_f110\_seed43 | train | exploratory\_architecture\_reconstruction | failed | 0/3750 | trainer latest | — | — | — | — | — | mismatch (49 files) | source\_changed, runner\_failure |
| 1790995583011837977\_p04\_v7\_f111\_seed42\_0f7dc638 | p04\_v7\_f111\_seed42 | train | exploratory\_architecture\_reconstruction | failed | 0/3750 | trainer latest | — | — | — | — | — | mismatch (49 files) | source\_changed, runner\_failure |
| 1790995583011859514\_p04\_v7\_f111\_seed43\_0a175776 | p04\_v7\_f111\_seed43 | train | exploratory\_architecture\_reconstruction | failed | 0/3750 | trainer latest | — | — | — | — | — | mismatch (49 files) | source\_changed, runner\_failure |
| 1790995583011969884\_p04\_v7\_fixed\_gate\_control\_seed42\_f6421658 | p04\_v7\_fixed\_gate\_control\_seed42 | train | unknown | pending | 0/3750 | trainer latest | — | — | — | — | — | pending (0 files) |  |
| 1790995583011991824\_p04\_v7\_fixed\_gate\_control\_seed43\_7ec4120b | p04\_v7\_fixed\_gate\_control\_seed43 | train | unknown | pending | 0/3750 | trainer latest | — | — | — | — | — | pending (0 files) |  |
| 1790992416352344471\_p05\_v7\_capacity\_v2\_a4854542 | p05\_v7\_capacity\_v2 | profile-synthetic | synthetic\_whole\_model\_capacity\_sweep | succeeded | 0/36 | mask100 | — | — | — | — | — | pending (0 files) | inconsistent\_completion |
| 1790796212564610986\_r00\_a4\_control\_754dd5e6 | r00\_a4\_control | profile-synthetic | synthetic\_functional\_diagnostic | succeeded | 20/20 | mask100 | 32 | 41.9954 | 2.06152 | 0.195122 | 0 | matched (32 files) | end\_not\_learned\_yet |
| 1790796212567802264\_r01\_span\_mask\_26105bc0 | r01\_span\_mask | profile-synthetic | synthetic\_functional\_diagnostic | succeeded | 20/20 | mask100 | 32 | 41.9954 | 2.03641 | 0.219512 | 0 | matched (32 files) | end\_not\_learned\_yet |
| 1790796212571024553\_r02\_chain\_completion\_499888e5 | r02\_chain\_completion | profile-synthetic | synthetic\_functional\_diagnostic | succeeded | 20/20 | mask100 | 32 | 41.9954 | 2.17229 | 0.280488 | 0 | matched (32 files) | end\_not\_learned\_yet |
| 1790796212574033832\_r03\_chain\_balanced\_loss\_0693e278 | r03\_chain\_balanced\_loss | profile-synthetic | synthetic\_functional\_diagnostic | succeeded | 20/20 | mask100 | 32 | 41.9954 | 2.53545 | 0.182927 | 0 | matched (32 files) | end\_not\_learned\_yet |
| 1790796212577263807\_r04\_qk\_norm\_32f94214 | r04\_qk\_norm | profile-synthetic | synthetic\_functional\_diagnostic | succeeded | 20/20 | mask100 | 32 | 42.746 | 2.08672 | 0.195122 | 0 | matched (32 files) | end\_not\_learned\_yet |
| 1790796212580387587\_r05\_register4\_3a525f6b | r05\_register4 | profile-synthetic | synthetic\_functional\_diagnostic | succeeded | 20/20 | mask100 | 32 | 42.1515 | 2.05223 | 0.195122 | 0 | matched (32 files) | end\_not\_learned\_yet |
| 1790796212583598406\_r06\_local\_conv7\_78cd3d2d | r06\_local\_conv7 | profile-synthetic | synthetic\_functional\_diagnostic | succeeded | 20/20 | mask100 | 32 | 43.3084 | 2.06022 | 0.195122 | 0 | matched (32 files) | end\_not\_learned\_yet |
| 1790796212586628536\_r07\_mask\_ratio\_film\_7688a166 | r07\_mask\_ratio\_film | profile-synthetic | synthetic\_functional\_diagnostic | succeeded | 20/20 | mask100 | 32 | 42.1843 | 2.01476 | 0.170732 | 0 | matched (32 files) | end\_not\_learned\_yet |
| 1790796212589837668\_r08\_vicreg\_a1da0f51 | r08\_vicreg | profile-synthetic | synthetic\_functional\_diagnostic | succeeded | 20/20 | mask100 | 32 | 51.1397 | 2.04524 | 0.195122 | 0 | matched (32 files) | end\_not\_learned\_yet |
| 1790796212592865084\_r09\_equal\_source\_fa295088 | r09\_equal\_source | profile-synthetic | synthetic\_functional\_diagnostic | succeeded | 20/20 | mask100 | 32 | 41.9954 | 1.95289 | 0.243902 | 0 | matched (32 files) | end\_not\_learned\_yet |
| 1790796212596100361\_r10\_role\_pair\_bias\_10d5f5b5 | r10\_role\_pair\_bias | profile-synthetic | synthetic\_functional\_diagnostic | succeeded | 20/20 | mask100 | 32 | 41.9954 | 2.0622 | 0.195122 | 0 | matched (32 files) | end\_not\_learned\_yet |
| 1790796212599373446\_r11\_span\_qknorm\_register\_2a307772 | r11\_span\_qknorm\_register | profile-synthetic | synthetic\_functional\_diagnostic | succeeded | 20/20 | mask100 | 32 | 42.9036 | 1.97122 | 0.195122 | 0 | matched (32 files) | end\_not\_learned\_yet |
| 1790799044322727901\_r12\_architecture\_control\_37b2b106 | r12\_architecture\_control | profile-synthetic | synthetic\_functional\_diagnostic | succeeded | 100/100 | mask100 | 16 | 40.0208 | 0.993626 | 0.487805 | 1 | matched (35 files) |  |
| 1790799044331991663\_r13\_head\_output\_gate\_72718882 | r13\_head\_output\_gate | profile-synthetic | synthetic\_functional\_diagnostic | succeeded | 100/100 | mask100 | 16 | 40.1154 | 1.17101 | 0.52439 | 0.636364 | matched (35 files) |  |
| 1790799044337931939\_r14\_fixed\_half\_attention\_8643a4ca | r14\_fixed\_half\_attention | profile-synthetic | synthetic\_functional\_diagnostic | succeeded | 100/100 | mask100 | 16 | 40.0208 | 1.11068 | 0.47561 | 1 | matched (35 files) |  |
| 1790799044341624231\_r15\_layerscale\_d0115c63 | r15\_layerscale | profile-synthetic | synthetic\_functional\_diagnostic | succeeded | 100/100 | mask100 | 16 | 40.0209 | 0.901605 | 0.512195 | 0.818182 | matched (35 files) |  |
| 1790799044345097609\_r16\_fixed\_residual\_scale\_6111de03 | r16\_fixed\_residual\_scale | profile-synthetic | synthetic\_functional\_diagnostic | succeeded | 100/100 | mask100 | 16 | 40.0208 | 1.14674 | 0.487805 | 1 | matched (35 files) |  |
| 1790799044348827183\_r17\_chain\_late\_fusion\_2a3a70e9 | r17\_chain\_late\_fusion | profile-synthetic | synthetic\_functional\_diagnostic | succeeded | 100/100 | mask100 | 16 | 40.0208 | 1.02391 | 0.512195 | 0.818182 | matched (35 files) |  |
| 1790799044352237695\_r18\_chain\_local\_global\_a0a72b83 | r18\_chain\_local\_global | profile-synthetic | synthetic\_functional\_diagnostic | succeeded | 100/100 | mask100 | 16 | 40.0208 | 0.990162 | 0.54878 | 0.545455 | matched (35 files) |  |
| 1790799044355869324\_r19\_value\_residual\_3bcfd14a | r19\_value\_residual | profile-synthetic | synthetic\_functional\_diagnostic | succeeded | 100/100 | mask100 | 16 | 40.0677 | 0.990644 | 0.52439 | 0.727273 | matched (35 files) |  |
| 1790802160338779408\_r20\_refined\_control\_3b41bffe | r20\_refined\_control | profile-synthetic | synthetic\_generalization\_diagnostic | succeeded | 25/25 | mask100 | 32 | 41.9954 | 2.0203 | 0.265618 | 0 | matched (39 files) | end\_not\_learned\_yet |
| 1790803305045499169\_r20\_refined\_control\_a37dc5ab | r20\_refined\_control | train | synthetic\_generalization\_diagnostic | succeeded | 500/500 | mask100 | 32 | 41.9954 | 0.804176 | 0.688368 | 1 | matched (39 files) |  |
| 1790802160343981887\_r21\_refined\_head\_gate\_80d10a6d | r21\_refined\_head\_gate | profile-synthetic | synthetic\_generalization\_diagnostic | succeeded | 25/25 | mask100 | 32 | 42.1838 | 1.95769 | 0.317174 | 0 | matched (39 files) | end\_not\_learned\_yet |
| 1790803305051081571\_r21\_refined\_head\_gate\_9cc7449e | r21\_refined\_head\_gate | train | synthetic\_generalization\_diagnostic | succeeded | 500/500 | mask100 | 32 | 42.1838 | 0.75063 | 0.708128 | 1 | matched (39 files) |  |
| 1790840619261329817\_r21\_refined\_head\_gate\_c27b5443 | r21\_refined\_head\_gate | profile-synthetic | trainer\_gpu\_runtime\_functional\_diagnostic | succeeded | ?/3 optimizer windows | artificial Trainer windows | — | — | — | — | — | rejected (0 files) | trainer\_diagnostic\_invalid\_envelope, trainer\_diagnostic\_runtime\_identity\_mismatch |
| 1790802160346881056\_r22\_refined\_fixed\_gate\_263516a3 | r22\_refined\_fixed\_gate | profile-synthetic | synthetic\_generalization\_diagnostic | succeeded | 25/25 | mask100 | 32 | 41.9954 | 1.98071 | 0.305587 | 0 | matched (39 files) | end\_not\_learned\_yet |
| 1790803305054490194\_r22\_refined\_fixed\_gate\_72a52fe0 | r22\_refined\_fixed\_gate | train | synthetic\_generalization\_diagnostic | succeeded | 500/500 | mask100 | 32 | 41.9954 | 0.737844 | 0.701747 | 1 | matched (39 files) |  |
| 1790802160350276097\_r23\_refined\_layerscale\_eb29f0d4 | r23\_refined\_layerscale | profile-synthetic | synthetic\_generalization\_diagnostic | succeeded | 25/25 | mask100 | 32 | 41.9955 | 1.97572 | 0.311688 | 0 | matched (39 files) | end\_not\_learned\_yet |
| 1790803305057942502\_r23\_refined\_layerscale\_56d8467a | r23\_refined\_layerscale | train | synthetic\_generalization\_diagnostic | succeeded | 500/500 | mask100 | 32 | 41.9955 | 0.733126 | 0.702642 | 0.969697 | matched (39 files) |  |
| 1790802160353131370\_r24\_refined\_fixed\_scale\_c6e0089d | r24\_refined\_fixed\_scale | profile-synthetic | synthetic\_generalization\_diagnostic | succeeded | 25/25 | mask100 | 32 | 41.9954 | 1.97552 | 0.31236 | 0 | matched (39 files) | end\_not\_learned\_yet |
| 1790803305061375040\_r24\_refined\_fixed\_scale\_4edeef26 | r24\_refined\_fixed\_scale | train | synthetic\_generalization\_diagnostic | succeeded | 500/500 | mask100 | 32 | 41.9954 | 0.738684 | 0.702922 | 1 | matched (39 files) |  |
| 1790840619270196057\_r24\_refined\_fixed\_scale\_e01f3f97 | r24\_refined\_fixed\_scale | profile-synthetic | trainer\_gpu\_runtime\_functional\_diagnostic | succeeded | ?/3 optimizer windows | artificial Trainer windows | — | — | — | — | — | rejected (0 files) | trainer\_diagnostic\_invalid\_envelope, trainer\_diagnostic\_runtime\_identity\_mismatch |
| 1790802160356028347\_r25\_refined\_late\_fusion\_4640596c | r25\_refined\_late\_fusion | profile-synthetic | synthetic\_generalization\_diagnostic | succeeded | 25/25 | mask100 | 32 | 48.1204 | 2.14857 | 0.295231 | 0 | matched (39 files) | end\_not\_learned\_yet |
| 1790803305064798754\_r25\_refined\_late\_fusion\_5cd5e385 | r25\_refined\_late\_fusion | train | synthetic\_generalization\_diagnostic | succeeded | 500/500 | mask100 | 32 | 48.1204 | 0.841605 | 0.665417 | 0.969697 | matched (39 files) |  |
| 1790802160358902166\_r26\_refined\_local\_global\_905c9acd | r26\_refined\_local\_global | profile-synthetic | synthetic\_generalization\_diagnostic | succeeded | 25/25 | mask100 | 32 | 53.9681 | 2.08457 | 0.297302 | 0 | matched (39 files) | end\_not\_learned\_yet |
| 1790803305068196112\_r26\_refined\_local\_global\_5bf50a1c | r26\_refined\_local\_global | train | synthetic\_generalization\_diagnostic | succeeded | 500/500 | mask100 | 32 | 53.9681 | 0.749581 | 0.704098 | 1 | matched (39 files) |  |
| 1790840619276677050\_r26\_refined\_local\_global\_e113e0e8 | r26\_refined\_local\_global | profile-synthetic | trainer\_gpu\_runtime\_functional\_diagnostic | succeeded | ?/3 optimizer windows | artificial Trainer windows | — | — | — | — | — | rejected (0 files) | trainer\_diagnostic\_invalid\_envelope, trainer\_diagnostic\_runtime\_identity\_mismatch |
| 1790802160361769452\_r27\_refined\_value\_residual\_0b46ff76 | r27\_refined\_value\_residual | profile-synthetic | synthetic\_generalization\_diagnostic | succeeded | 25/25 | mask100 | 32 | 42.1829 | 1.9652 | 0.28689 | 0 | matched (39 files) | end\_not\_learned\_yet |
| 1790803305071555086\_r27\_refined\_value\_residual\_52f013e5 | r27\_refined\_value\_residual | train | synthetic\_generalization\_diagnostic | succeeded | 500/500 | mask100 | 32 | 42.1829 | 0.74735 | 0.705889 | 0.969697 | matched (39 files) |  |

### 1790824088674261408\_d01\_pilot\_runtime\_gpu\_r20\_v1\_9af03156：人工 Trainer runtime诊断

执行完成=—；功能GPU检查=—；人工训练开始=—；corpus_training_started=False。transport steps=1仅代表一次调用，进度以实际optimizer windows计。

| optimizer step | microsteps | records | finite gradients | mean micro loss |
| --- | --- | --- | --- | --- |

### 1790827241091679008\_d01\_pilot\_runtime\_gpu\_r20\_v1\_ea0bba4a：人工 Trainer runtime诊断

执行完成=—；功能GPU检查=—；人工训练开始=—；corpus_training_started=False。transport steps=1仅代表一次调用，进度以实际optimizer windows计。

| optimizer step | microsteps | records | finite gradients | mean micro loss |
| --- | --- | --- | --- | --- |

### 1790831680607223314\_d01\_pilot\_runtime\_gpu\_r20\_v1\_48f59959：人工 Trainer runtime诊断

执行完成=True；功能GPU检查=True；人工训练开始=True；corpus_training_started=False。transport steps=1仅代表一次调用，进度以实际optimizer windows计。

| optimizer step | microsteps | records | finite gradients | mean micro loss |
| --- | --- | --- | --- | --- |
| 1 | 8 | 256 | True | 3.45091 |
| 2 | 8 | 256 | True | 3.44851 |
| 3 | 8 | 256 | True | 3.38033 |

### 1790840619261329817\_r21\_refined\_head\_gate\_c27b5443：人工 Trainer runtime诊断

执行完成=—；功能GPU检查=—；人工训练开始=—；corpus_training_started=False。transport steps=1仅代表一次调用，进度以实际optimizer windows计。

| optimizer step | microsteps | records | finite gradients | mean micro loss |
| --- | --- | --- | --- | --- |

### 1790840619270196057\_r24\_refined\_fixed\_scale\_e01f3f97：人工 Trainer runtime诊断

执行完成=—；功能GPU检查=—；人工训练开始=—；corpus_training_started=False。transport steps=1仅代表一次调用，进度以实际optimizer windows计。

| optimizer step | microsteps | records | finite gradients | mean micro loss |
| --- | --- | --- | --- | --- |

### 1790840619276677050\_r26\_refined\_local\_global\_e113e0e8：人工 Trainer runtime诊断

执行完成=—；功能GPU检查=—；人工训练开始=—；corpus_training_started=False。transport steps=1仅代表一次调用，进度以实际optimizer windows计。

| optimizer step | microsteps | records | finite gradients | mean micro loss |
| --- | --- | --- | --- | --- |

### 1790799044322727901\_r12\_architecture\_control\_37b2b106：分层 mask 指标

源码目录：/vepfs-mlp2/c20250601/251105016/project/dllm\_test/output/immune\_research\_releases/r12\_r19\_20260930；完成/目标 step：100/100。

kind=synthetic\_functional\_diagnostic；dataset_kind=—；biological_quality_claim=False。

| 时点 | mask level | 指标 | 值 |
| --- | --- | --- | --- |
| initial | mask25 | mask\_ratio | 0.25 |
| initial | mask25 | selected\_count | 21 |
| initial | mask25 | aa\_count | 18 |
| initial | mask25 | terminal\_count | 3 |
| initial | mask25 | loss | 4.77552 |
| initial | mask25 | aa\_loss | 4.81123 |
| initial | mask25 | terminal\_loss | 4.56123 |
| initial | mask25 | aa\_accuracy | 0 |
| initial | mask25 | terminal\_accuracy | 0 |
| initial | mask50 | mask\_ratio | 0.5 |
| initial | mask50 | selected\_count | 52 |
| initial | mask50 | aa\_count | 48 |
| initial | mask50 | terminal\_count | 4 |
| initial | mask50 | loss | 4.74757 |
| initial | mask50 | aa\_loss | 4.76629 |
| initial | mask50 | terminal\_loss | 4.52301 |
| initial | mask50 | aa\_accuracy | 0 |
| initial | mask50 | terminal\_accuracy | 0 |
| initial | mask75 | mask\_ratio | 0.75 |
| initial | mask75 | selected\_count | 77 |
| initial | mask75 | aa\_count | 69 |
| initial | mask75 | terminal\_count | 8 |
| initial | mask75 | loss | 4.57931 |
| initial | mask75 | aa\_loss | 4.58645 |
| initial | mask75 | terminal\_loss | 4.51771 |
| initial | mask75 | aa\_accuracy | 0 |
| initial | mask75 | terminal\_accuracy | 0 |
| initial | mask100 | mask\_ratio | 1 |
| initial | mask100 | selected\_count | 93 |
| initial | mask100 | aa\_count | 82 |
| initial | mask100 | terminal\_count | 11 |
| initial | mask100 | loss | 4.53657 |
| initial | mask100 | aa\_loss | 4.5374 |
| initial | mask100 | terminal\_loss | 4.53044 |
| initial | mask100 | aa\_accuracy | 0 |
| initial | mask100 | terminal\_accuracy | 0 |
| initial | mask\_mean | loss | 4.65974 |
| initial | mask\_mean | aa\_loss | 4.67534 |
| initial | mask\_mean | terminal\_loss | 4.5331 |
| initial | mask\_mean | aa\_accuracy | 0 |
| initial | mask\_mean | terminal\_accuracy | 0 |
| latest | mask25 | mask\_ratio | 0.25 |
| latest | mask25 | selected\_count | 21 |
| latest | mask25 | aa\_count | 18 |
| latest | mask25 | terminal\_count | 3 |
| latest | mask25 | loss | 0.000645038 |
| latest | mask25 | aa\_loss | 0.000751372 |
| latest | mask25 | terminal\_loss | 7.03329e-06 |
| latest | mask25 | aa\_accuracy | 1 |
| latest | mask25 | terminal\_accuracy | 1 |
| latest | mask50 | mask\_ratio | 0.5 |
| latest | mask50 | selected\_count | 52 |
| latest | mask50 | aa\_count | 48 |
| latest | mask50 | terminal\_count | 4 |
| latest | mask50 | loss | 0.00183177 |
| latest | mask50 | aa\_loss | 0.00179316 |
| latest | mask50 | terminal\_loss | 0.00229517 |
| latest | mask50 | aa\_accuracy | 1 |
| latest | mask50 | terminal\_accuracy | 1 |
| latest | mask75 | mask\_ratio | 0.75 |
| latest | mask75 | selected\_count | 77 |
| latest | mask75 | aa\_count | 69 |
| latest | mask75 | terminal\_count | 8 |
| latest | mask75 | loss | 0.00381923 |
| latest | mask75 | aa\_loss | 0.00336398 |
| latest | mask75 | terminal\_loss | 0.00774576 |
| latest | mask75 | aa\_accuracy | 1 |
| latest | mask75 | terminal\_accuracy | 1 |
| latest | mask100 | mask\_ratio | 1 |
| latest | mask100 | selected\_count | 93 |
| latest | mask100 | aa\_count | 82 |
| latest | mask100 | terminal\_count | 11 |
| latest | mask100 | loss | 0.993626 |
| latest | mask100 | aa\_loss | 1.07118 |
| latest | mask100 | terminal\_loss | 0.415527 |
| latest | mask100 | aa\_accuracy | 0.487805 |
| latest | mask100 | terminal\_accuracy | 1 |
| latest | mask\_mean | loss | 0.24998 |
| latest | mask\_mean | aa\_loss | 0.269271 |
| latest | mask\_mean | terminal\_loss | 0.106394 |
| latest | mask\_mean | aa\_accuracy | 0.871951 |
| latest | mask\_mean | terminal\_accuracy | 1 |

评测曲线（原始 step，四档等权均值）：

| step | mean loss | mean AA acc | mean end acc | gradient norm |
| --- | --- | --- | --- | --- |
| 5 | 3.81119 | 0.0761882 | 0 | 28.9978 |
| 10 | 2.30087 | 0.219792 | 0 | 13.9427 |
| 15 | 1.97744 | 0.258456 | 0 | 3.87809 |
| 20 | 1.8529 | 0.282762 | 0 | 2.90401 |
| 25 | 1.70191 | 0.451542 | 0 | 2.74407 |
| 30 | 1.41137 | 0.473717 | 0.159091 | 3.16833 |
| 35 | 1.00283 | 0.615253 | 0.58428 | 3.7475 |
| 40 | 0.722334 | 0.703611 | 0.923295 | 4.73344 |
| 45 | 0.577462 | 0.779455 | 0.954545 | 11.1695 |
| 50 | 0.363021 | 0.86413 | 0.860795 | 7.54473 |
| 55 | 0.296804 | 0.874426 | 0.954545 | 2.65396 |
| 60 | 0.280637 | 0.877474 | 0.931818 | 6.67092 |
| 65 | 0.375001 | 0.874426 | 0.954545 | 4.10594 |
| 70 | 0.262877 | 0.874426 | 0.954545 | 4.79099 |
| 75 | 0.232176 | 0.880523 | 0.909091 | 2.21028 |
| 80 | 0.250419 | 0.875 | 0.977273 | 16.9197 |
| 85 | 0.264203 | 0.884146 | 0.77178 | 8.30223 |
| 90 | 0.241108 | 0.884146 | 0.909091 | 7.47753 |
| 95 | 0.258632 | 0.884146 | 0.909091 | 3.11511 |
| 100 | 0.24998 | 0.871951 | 1 | 2.28095 |

### 1790799044331991663\_r13\_head\_output\_gate\_72718882：分层 mask 指标

源码目录：/vepfs-mlp2/c20250601/251105016/project/dllm\_test/output/immune\_research\_releases/r12\_r19\_20260930；完成/目标 step：100/100。

kind=synthetic\_functional\_diagnostic；dataset_kind=—；biological_quality_claim=False。

| 时点 | mask level | 指标 | 值 |
| --- | --- | --- | --- |
| initial | mask25 | mask\_ratio | 0.25 |
| initial | mask25 | selected\_count | 21 |
| initial | mask25 | aa\_count | 18 |
| initial | mask25 | terminal\_count | 3 |
| initial | mask25 | loss | 4.86202 |
| initial | mask25 | aa\_loss | 4.97225 |
| initial | mask25 | terminal\_loss | 4.20068 |
| initial | mask25 | aa\_accuracy | 0 |
| initial | mask25 | terminal\_accuracy | 0 |
| initial | mask50 | mask\_ratio | 0.5 |
| initial | mask50 | selected\_count | 52 |
| initial | mask50 | aa\_count | 48 |
| initial | mask50 | terminal\_count | 4 |
| initial | mask50 | loss | 4.83696 |
| initial | mask50 | aa\_loss | 4.8924 |
| initial | mask50 | terminal\_loss | 4.17167 |
| initial | mask50 | aa\_accuracy | 0 |
| initial | mask50 | terminal\_accuracy | 0 |
| initial | mask75 | mask\_ratio | 0.75 |
| initial | mask75 | selected\_count | 77 |
| initial | mask75 | aa\_count | 69 |
| initial | mask75 | terminal\_count | 8 |
| initial | mask75 | loss | 4.72895 |
| initial | mask75 | aa\_loss | 4.79542 |
| initial | mask75 | terminal\_loss | 4.15559 |
| initial | mask75 | aa\_accuracy | 0 |
| initial | mask75 | terminal\_accuracy | 0 |
| initial | mask100 | mask\_ratio | 1 |
| initial | mask100 | selected\_count | 93 |
| initial | mask100 | aa\_count | 82 |
| initial | mask100 | terminal\_count | 11 |
| initial | mask100 | loss | 4.69373 |
| initial | mask100 | aa\_loss | 4.75966 |
| initial | mask100 | terminal\_loss | 4.2022 |
| initial | mask100 | aa\_accuracy | 0 |
| initial | mask100 | terminal\_accuracy | 0 |
| initial | mask\_mean | loss | 4.78041 |
| initial | mask\_mean | aa\_loss | 4.85493 |
| initial | mask\_mean | terminal\_loss | 4.18254 |
| initial | mask\_mean | aa\_accuracy | 0 |
| initial | mask\_mean | terminal\_accuracy | 0 |
| latest | mask25 | mask\_ratio | 0.25 |
| latest | mask25 | selected\_count | 21 |
| latest | mask25 | aa\_count | 18 |
| latest | mask25 | terminal\_count | 3 |
| latest | mask25 | loss | 0.00107016 |
| latest | mask25 | aa\_loss | 0.00124851 |
| latest | mask25 | terminal\_loss | 7.94729e-08 |
| latest | mask25 | aa\_accuracy | 1 |
| latest | mask25 | terminal\_accuracy | 1 |
| latest | mask50 | mask\_ratio | 0.5 |
| latest | mask50 | selected\_count | 52 |
| latest | mask50 | aa\_count | 48 |
| latest | mask50 | terminal\_count | 4 |
| latest | mask50 | loss | 0.000314441 |
| latest | mask50 | aa\_loss | 0.00034057 |
| latest | mask50 | terminal\_loss | 8.94069e-07 |
| latest | mask50 | aa\_accuracy | 1 |
| latest | mask50 | terminal\_accuracy | 1 |
| latest | mask75 | mask\_ratio | 0.75 |
| latest | mask75 | selected\_count | 77 |
| latest | mask75 | aa\_count | 69 |
| latest | mask75 | terminal\_count | 8 |
| latest | mask75 | loss | 0.00064701 |
| latest | mask75 | aa\_loss | 0.00065961 |
| latest | mask75 | terminal\_loss | 0.000538332 |
| latest | mask75 | aa\_accuracy | 1 |
| latest | mask75 | terminal\_accuracy | 1 |
| latest | mask100 | mask\_ratio | 1 |
| latest | mask100 | selected\_count | 93 |
| latest | mask100 | aa\_count | 82 |
| latest | mask100 | terminal\_count | 11 |
| latest | mask100 | loss | 1.17101 |
| latest | mask100 | aa\_loss | 1.24322 |
| latest | mask100 | terminal\_loss | 0.632696 |
| latest | mask100 | aa\_accuracy | 0.52439 |
| latest | mask100 | terminal\_accuracy | 0.636364 |
| latest | mask\_mean | loss | 0.29326 |
| latest | mask\_mean | aa\_loss | 0.311368 |
| latest | mask\_mean | terminal\_loss | 0.158309 |
| latest | mask\_mean | aa\_accuracy | 0.881098 |
| latest | mask\_mean | terminal\_accuracy | 0.909091 |

评测曲线（原始 step，四档等权均值）：

| step | mean loss | mean AA acc | mean end acc | gradient norm |
| --- | --- | --- | --- | --- |
| 5 | 2.79909 | 0.217561 | 0 | 28.0778 |
| 10 | 2.08235 | 0.194295 | 0 | 13.361 |
| 15 | 2.00932 | 0.194295 | 0 | 5.99199 |
| 20 | 1.84774 | 0.264554 | 0 | 3.55317 |
| 25 | 1.65499 | 0.380302 | 0 | 2.93067 |
| 30 | 1.34108 | 0.517179 | 0.244318 | 4.91435 |
| 35 | 0.75726 | 0.676349 | 0.752841 | 6.00571 |
| 40 | 0.5447 | 0.805769 | 0.946023 | 9.29207 |
| 45 | 0.462221 | 0.807384 | 0.860795 | 9.67807 |
| 50 | 0.405833 | 0.823432 | 0.954545 | 20.4255 |
| 55 | 0.369667 | 0.870802 | 0.954545 | 1.99985 |
| 60 | 0.382452 | 0.814056 | 0.954545 | 21.5361 |
| 65 | 0.352721 | 0.875 | 0.923295 | 5.47614 |
| 70 | 0.245368 | 0.884146 | 0.909091 | 2.44149 |
| 75 | 0.250839 | 0.881098 | 0.909091 | 2.90366 |
| 80 | 0.231936 | 0.868328 | 1 | 1.91724 |
| 85 | 0.249082 | 0.890244 | 0.863636 | 3.31271 |
| 90 | 0.285717 | 0.871951 | 1 | 6.18352 |
| 95 | 0.266768 | 0.871692 | 0.886364 | 1.91858 |
| 100 | 0.29326 | 0.881098 | 0.909091 | 2.43638 |

### 1790799044337931939\_r14\_fixed\_half\_attention\_8643a4ca：分层 mask 指标

源码目录：/vepfs-mlp2/c20250601/251105016/project/dllm\_test/output/immune\_research\_releases/r12\_r19\_20260930；完成/目标 step：100/100。

kind=synthetic\_functional\_diagnostic；dataset_kind=—；biological_quality_claim=False。

| 时点 | mask level | 指标 | 值 |
| --- | --- | --- | --- |
| initial | mask25 | mask\_ratio | 0.25 |
| initial | mask25 | selected\_count | 21 |
| initial | mask25 | aa\_count | 18 |
| initial | mask25 | terminal\_count | 3 |
| initial | mask25 | loss | 4.8819 |
| initial | mask25 | aa\_loss | 4.9922 |
| initial | mask25 | terminal\_loss | 4.22006 |
| initial | mask25 | aa\_accuracy | 0 |
| initial | mask25 | terminal\_accuracy | 0 |
| initial | mask50 | mask\_ratio | 0.5 |
| initial | mask50 | selected\_count | 52 |
| initial | mask50 | aa\_count | 48 |
| initial | mask50 | terminal\_count | 4 |
| initial | mask50 | loss | 4.82963 |
| initial | mask50 | aa\_loss | 4.88104 |
| initial | mask50 | terminal\_loss | 4.21266 |
| initial | mask50 | aa\_accuracy | 0 |
| initial | mask50 | terminal\_accuracy | 0 |
| initial | mask75 | mask\_ratio | 0.75 |
| initial | mask75 | selected\_count | 77 |
| initial | mask75 | aa\_count | 69 |
| initial | mask75 | terminal\_count | 8 |
| initial | mask75 | loss | 4.73272 |
| initial | mask75 | aa\_loss | 4.79284 |
| initial | mask75 | terminal\_loss | 4.21417 |
| initial | mask75 | aa\_accuracy | 0 |
| initial | mask75 | terminal\_accuracy | 0 |
| initial | mask100 | mask\_ratio | 1 |
| initial | mask100 | selected\_count | 93 |
| initial | mask100 | aa\_count | 82 |
| initial | mask100 | terminal\_count | 11 |
| initial | mask100 | loss | 4.68678 |
| initial | mask100 | aa\_loss | 4.75118 |
| initial | mask100 | terminal\_loss | 4.20667 |
| initial | mask100 | aa\_accuracy | 0 |
| initial | mask100 | terminal\_accuracy | 0 |
| initial | mask\_mean | loss | 4.78275 |
| initial | mask\_mean | aa\_loss | 4.85432 |
| initial | mask\_mean | terminal\_loss | 4.21339 |
| initial | mask\_mean | aa\_accuracy | 0 |
| initial | mask\_mean | terminal\_accuracy | 0 |
| latest | mask25 | mask\_ratio | 0.25 |
| latest | mask25 | selected\_count | 21 |
| latest | mask25 | aa\_count | 18 |
| latest | mask25 | terminal\_count | 3 |
| latest | mask25 | loss | 0.0112753 |
| latest | mask25 | aa\_loss | 0.0112763 |
| latest | mask25 | terminal\_loss | 0.0112698 |
| latest | mask25 | aa\_accuracy | 1 |
| latest | mask25 | terminal\_accuracy | 1 |
| latest | mask50 | mask\_ratio | 0.5 |
| latest | mask50 | selected\_count | 52 |
| latest | mask50 | aa\_count | 48 |
| latest | mask50 | terminal\_count | 4 |
| latest | mask50 | loss | 0.00318219 |
| latest | mask50 | aa\_loss | 0.0033871 |
| latest | mask50 | terminal\_loss | 0.000723301 |
| latest | mask50 | aa\_accuracy | 1 |
| latest | mask50 | terminal\_accuracy | 1 |
| latest | mask75 | mask\_ratio | 0.75 |
| latest | mask75 | selected\_count | 77 |
| latest | mask75 | aa\_count | 69 |
| latest | mask75 | terminal\_count | 8 |
| latest | mask75 | loss | 0.0915297 |
| latest | mask75 | aa\_loss | 0.102063 |
| latest | mask75 | terminal\_loss | 0.000679585 |
| latest | mask75 | aa\_accuracy | 0.985507 |
| latest | mask75 | terminal\_accuracy | 1 |
| latest | mask100 | mask\_ratio | 1 |
| latest | mask100 | selected\_count | 93 |
| latest | mask100 | aa\_count | 82 |
| latest | mask100 | terminal\_count | 11 |
| latest | mask100 | loss | 1.11068 |
| latest | mask100 | aa\_loss | 1.2043 |
| latest | mask100 | terminal\_loss | 0.412774 |
| latest | mask100 | aa\_accuracy | 0.47561 |
| latest | mask100 | terminal\_accuracy | 1 |
| latest | mask\_mean | loss | 0.304166 |
| latest | mask\_mean | aa\_loss | 0.330256 |
| latest | mask\_mean | terminal\_loss | 0.106362 |
| latest | mask\_mean | aa\_accuracy | 0.865279 |
| latest | mask\_mean | terminal\_accuracy | 1 |

评测曲线（原始 step，四档等权均值）：

| step | mean loss | mean AA acc | mean end acc | gradient norm |
| --- | --- | --- | --- | --- |
| 5 | 3.52054 | 0.0761882 | 0 | 32.0382 |
| 10 | 2.28086 | 0.219792 | 0 | 12.279 |
| 15 | 1.97016 | 0.244098 | 0 | 4.98422 |
| 20 | 1.85185 | 0.258456 | 0 | 3.58176 |
| 25 | 1.65446 | 0.369994 | 0.714962 | 4.27533 |
| 30 | 1.16243 | 0.567362 | 0.791667 | 5.55784 |
| 35 | 0.699756 | 0.705275 | 0.900568 | 2.85056 |
| 40 | 0.606375 | 0.765415 | 0.900568 | 11.7256 |
| 45 | 0.42582 | 0.812047 | 0.931818 | 7.50201 |
| 50 | 0.321407 | 0.86413 | 0.954545 | 12.3579 |
| 55 | 0.339357 | 0.85228 | 0.954545 | 17.8566 |
| 60 | 0.270527 | 0.880523 | 0.877841 | 8.94054 |
| 65 | 0.313613 | 0.856448 | 0.954545 | 18.9436 |
| 70 | 0.255113 | 0.880523 | 0.825758 | 3.11286 |
| 75 | 0.230388 | 0.887195 | 0.855114 | 6.14944 |
| 80 | 0.263049 | 0.878049 | 0.954545 | 2.78761 |
| 85 | 0.221973 | 0.884146 | 0.909091 | 1.4899 |
| 90 | 0.259525 | 0.884146 | 0.909091 | 4.02502 |
| 95 | 0.278737 | 0.883572 | 0.886364 | 5.39532 |
| 100 | 0.304166 | 0.865279 | 1 | 1.34353 |

### 1790799044341624231\_r15\_layerscale\_d0115c63：分层 mask 指标

源码目录：/vepfs-mlp2/c20250601/251105016/project/dllm\_test/output/immune\_research\_releases/r12\_r19\_20260930；完成/目标 step：100/100。

kind=synthetic\_functional\_diagnostic；dataset_kind=—；biological_quality_claim=False。

| 时点 | mask level | 指标 | 值 |
| --- | --- | --- | --- |
| initial | mask25 | mask\_ratio | 0.25 |
| initial | mask25 | selected\_count | 21 |
| initial | mask25 | aa\_count | 18 |
| initial | mask25 | terminal\_count | 3 |
| initial | mask25 | loss | 4.09003 |
| initial | mask25 | aa\_loss | 4.08829 |
| initial | mask25 | terminal\_loss | 4.10047 |
| initial | mask25 | aa\_accuracy | 0.0555556 |
| initial | mask25 | terminal\_accuracy | 0 |
| initial | mask50 | mask\_ratio | 0.5 |
| initial | mask50 | selected\_count | 52 |
| initial | mask50 | aa\_count | 48 |
| initial | mask50 | terminal\_count | 4 |
| initial | mask50 | loss | 4.04874 |
| initial | mask50 | aa\_loss | 4.05141 |
| initial | mask50 | terminal\_loss | 4.01666 |
| initial | mask50 | aa\_accuracy | 0.0416667 |
| initial | mask50 | terminal\_accuracy | 0 |
| initial | mask75 | mask\_ratio | 0.75 |
| initial | mask75 | selected\_count | 77 |
| initial | mask75 | aa\_count | 69 |
| initial | mask75 | terminal\_count | 8 |
| initial | mask75 | loss | 3.8785 |
| initial | mask75 | aa\_loss | 3.86184 |
| initial | mask75 | terminal\_loss | 4.02221 |
| initial | mask75 | aa\_accuracy | 0.0724638 |
| initial | mask75 | terminal\_accuracy | 0 |
| initial | mask100 | mask\_ratio | 1 |
| initial | mask100 | selected\_count | 93 |
| initial | mask100 | aa\_count | 82 |
| initial | mask100 | terminal\_count | 11 |
| initial | mask100 | loss | 3.8607 |
| initial | mask100 | aa\_loss | 3.82439 |
| initial | mask100 | terminal\_loss | 4.13143 |
| initial | mask100 | aa\_accuracy | 0.0731707 |
| initial | mask100 | terminal\_accuracy | 0 |
| initial | mask\_mean | loss | 3.96949 |
| initial | mask\_mean | aa\_loss | 3.95648 |
| initial | mask\_mean | terminal\_loss | 4.06769 |
| initial | mask\_mean | aa\_accuracy | 0.0607142 |
| initial | mask\_mean | terminal\_accuracy | 0 |
| latest | mask25 | mask\_ratio | 0.25 |
| latest | mask25 | selected\_count | 21 |
| latest | mask25 | aa\_count | 18 |
| latest | mask25 | terminal\_count | 3 |
| latest | mask25 | loss | 7.48905e-05 |
| latest | mask25 | aa\_loss | 6.14567e-05 |
| latest | mask25 | terminal\_loss | 0.000155493 |
| latest | mask25 | aa\_accuracy | 1 |
| latest | mask25 | terminal\_accuracy | 1 |
| latest | mask50 | mask\_ratio | 0.5 |
| latest | mask50 | selected\_count | 52 |
| latest | mask50 | aa\_count | 48 |
| latest | mask50 | terminal\_count | 4 |
| latest | mask50 | loss | 4.6803e-05 |
| latest | mask50 | aa\_loss | 4.65139e-05 |
| latest | mask50 | terminal\_loss | 5.02721e-05 |
| latest | mask50 | aa\_accuracy | 1 |
| latest | mask50 | terminal\_accuracy | 1 |
| latest | mask75 | mask\_ratio | 0.75 |
| latest | mask75 | selected\_count | 77 |
| latest | mask75 | aa\_count | 69 |
| latest | mask75 | terminal\_count | 8 |
| latest | mask75 | loss | 9.97642e-05 |
| latest | mask75 | aa\_loss | 0.000104217 |
| latest | mask75 | terminal\_loss | 6.13579e-05 |
| latest | mask75 | aa\_accuracy | 1 |
| latest | mask75 | terminal\_accuracy | 1 |
| latest | mask100 | mask\_ratio | 1 |
| latest | mask100 | selected\_count | 93 |
| latest | mask100 | aa\_count | 82 |
| latest | mask100 | terminal\_count | 11 |
| latest | mask100 | loss | 0.901605 |
| latest | mask100 | aa\_loss | 0.956396 |
| latest | mask100 | terminal\_loss | 0.493164 |
| latest | mask100 | aa\_accuracy | 0.512195 |
| latest | mask100 | terminal\_accuracy | 0.818182 |
| latest | mask\_mean | loss | 0.225457 |
| latest | mask\_mean | aa\_loss | 0.239152 |
| latest | mask\_mean | terminal\_loss | 0.123358 |
| latest | mask\_mean | aa\_accuracy | 0.878049 |
| latest | mask\_mean | terminal\_accuracy | 0.954545 |

评测曲线（原始 step，四档等权均值）：

| step | mean loss | mean AA acc | mean end acc | gradient norm |
| --- | --- | --- | --- | --- |
| 5 | 3.38685 | 0.217561 | 0 | 20.3948 |
| 10 | 2.20351 | 0.163518 | 0.247159 | 12.8214 |
| 15 | 1.95659 | 0.267832 | 0 | 5.68247 |
| 20 | 1.8071 | 0.277553 | 0 | 1.90951 |
| 25 | 1.58773 | 0.410613 | 0.161932 | 3.3378 |
| 30 | 1.04216 | 0.545393 | 0.692235 | 3.57067 |
| 35 | 0.627266 | 0.716581 | 0.923295 | 6.60725 |
| 40 | 0.514821 | 0.786672 | 0.954545 | 4.88701 |
| 45 | 0.430505 | 0.822887 | 0.923295 | 8.01635 |
| 50 | 0.307171 | 0.873851 | 0.931818 | 5.55251 |
| 55 | 0.330417 | 0.845033 | 0.954545 | 8.65295 |
| 60 | 0.257098 | 0.877474 | 0.931818 | 3.59037 |
| 65 | 0.253817 | 0.878049 | 0.954545 | 1.45106 |
| 70 | 0.238023 | 0.884146 | 0.909091 | 2.44231 |
| 75 | 0.246677 | 0.875 | 0.977273 | 7.14312 |
| 80 | 0.239415 | 0.875 | 0.977273 | 1.81834 |
| 85 | 0.335972 | 0.880523 | 0.909091 | 4.96953 |
| 90 | 0.26143 | 0.884146 | 0.909091 | 2.71195 |
| 95 | 0.222554 | 0.884146 | 0.909091 | 1.20387 |
| 100 | 0.225457 | 0.878049 | 0.954545 | 9.35526 |

### 1790799044345097609\_r16\_fixed\_residual\_scale\_6111de03：分层 mask 指标

源码目录：/vepfs-mlp2/c20250601/251105016/project/dllm\_test/output/immune\_research\_releases/r12\_r19\_20260930；完成/目标 step：100/100。

kind=synthetic\_functional\_diagnostic；dataset_kind=—；biological_quality_claim=False。

| 时点 | mask level | 指标 | 值 |
| --- | --- | --- | --- |
| initial | mask25 | mask\_ratio | 0.25 |
| initial | mask25 | selected\_count | 21 |
| initial | mask25 | aa\_count | 18 |
| initial | mask25 | terminal\_count | 3 |
| initial | mask25 | loss | 4.09003 |
| initial | mask25 | aa\_loss | 4.08829 |
| initial | mask25 | terminal\_loss | 4.10047 |
| initial | mask25 | aa\_accuracy | 0.0555556 |
| initial | mask25 | terminal\_accuracy | 0 |
| initial | mask50 | mask\_ratio | 0.5 |
| initial | mask50 | selected\_count | 52 |
| initial | mask50 | aa\_count | 48 |
| initial | mask50 | terminal\_count | 4 |
| initial | mask50 | loss | 4.04874 |
| initial | mask50 | aa\_loss | 4.05141 |
| initial | mask50 | terminal\_loss | 4.01666 |
| initial | mask50 | aa\_accuracy | 0.0416667 |
| initial | mask50 | terminal\_accuracy | 0 |
| initial | mask75 | mask\_ratio | 0.75 |
| initial | mask75 | selected\_count | 77 |
| initial | mask75 | aa\_count | 69 |
| initial | mask75 | terminal\_count | 8 |
| initial | mask75 | loss | 3.8785 |
| initial | mask75 | aa\_loss | 3.86184 |
| initial | mask75 | terminal\_loss | 4.02221 |
| initial | mask75 | aa\_accuracy | 0.0724638 |
| initial | mask75 | terminal\_accuracy | 0 |
| initial | mask100 | mask\_ratio | 1 |
| initial | mask100 | selected\_count | 93 |
| initial | mask100 | aa\_count | 82 |
| initial | mask100 | terminal\_count | 11 |
| initial | mask100 | loss | 3.8607 |
| initial | mask100 | aa\_loss | 3.82439 |
| initial | mask100 | terminal\_loss | 4.13143 |
| initial | mask100 | aa\_accuracy | 0.0731707 |
| initial | mask100 | terminal\_accuracy | 0 |
| initial | mask\_mean | loss | 3.96949 |
| initial | mask\_mean | aa\_loss | 3.95648 |
| initial | mask\_mean | terminal\_loss | 4.06769 |
| initial | mask\_mean | aa\_accuracy | 0.0607142 |
| initial | mask\_mean | terminal\_accuracy | 0 |
| latest | mask25 | mask\_ratio | 0.25 |
| latest | mask25 | selected\_count | 21 |
| latest | mask25 | aa\_count | 18 |
| latest | mask25 | terminal\_count | 3 |
| latest | mask25 | loss | 0.000534517 |
| latest | mask25 | aa\_loss | 0.000595715 |
| latest | mask25 | terminal\_loss | 0.00016733 |
| latest | mask25 | aa\_accuracy | 1 |
| latest | mask25 | terminal\_accuracy | 1 |
| latest | mask50 | mask\_ratio | 0.5 |
| latest | mask50 | selected\_count | 52 |
| latest | mask50 | aa\_count | 48 |
| latest | mask50 | terminal\_count | 4 |
| latest | mask50 | loss | 0.000289467 |
| latest | mask50 | aa\_loss | 0.000292072 |
| latest | mask50 | terminal\_loss | 0.000258208 |
| latest | mask50 | aa\_accuracy | 1 |
| latest | mask50 | terminal\_accuracy | 1 |
| latest | mask75 | mask\_ratio | 0.75 |
| latest | mask75 | selected\_count | 77 |
| latest | mask75 | aa\_count | 69 |
| latest | mask75 | terminal\_count | 8 |
| latest | mask75 | loss | 0.000558281 |
| latest | mask75 | aa\_loss | 0.000506894 |
| latest | mask75 | terminal\_loss | 0.0010015 |
| latest | mask75 | aa\_accuracy | 1 |
| latest | mask75 | terminal\_accuracy | 1 |
| latest | mask100 | mask\_ratio | 1 |
| latest | mask100 | selected\_count | 93 |
| latest | mask100 | aa\_count | 82 |
| latest | mask100 | terminal\_count | 11 |
| latest | mask100 | loss | 1.14674 |
| latest | mask100 | aa\_loss | 1.26948 |
| latest | mask100 | terminal\_loss | 0.23181 |
| latest | mask100 | aa\_accuracy | 0.487805 |
| latest | mask100 | terminal\_accuracy | 1 |
| latest | mask\_mean | loss | 0.287032 |
| latest | mask\_mean | aa\_loss | 0.317718 |
| latest | mask\_mean | terminal\_loss | 0.0583093 |
| latest | mask\_mean | aa\_accuracy | 0.871951 |
| latest | mask\_mean | terminal\_accuracy | 1 |

评测曲线（原始 step，四档等权均值）：

| step | mean loss | mean AA acc | mean end acc | gradient norm |
| --- | --- | --- | --- | --- |
| 5 | 3.38355 | 0.217561 | 0 | 20.3244 |
| 10 | 2.20372 | 0.177407 | 0.247159 | 12.7937 |
| 15 | 1.95627 | 0.267832 | 0 | 5.64957 |
| 20 | 1.80506 | 0.277553 | 0 | 1.899 |
| 25 | 1.58464 | 0.40699 | 0.161932 | 3.29806 |
| 30 | 1.03783 | 0.540185 | 0.692235 | 3.54157 |
| 35 | 0.62294 | 0.729351 | 0.869318 | 7.08084 |
| 40 | 0.543376 | 0.786672 | 0.954545 | 3.94483 |
| 45 | 0.430692 | 0.822887 | 0.923295 | 6.79483 |
| 50 | 0.286476 | 0.877474 | 0.931818 | 3.79272 |
| 55 | 0.31644 | 0.865594 | 0.954545 | 3.16243 |
| 60 | 0.281471 | 0.868643 | 0.909091 | 5.96625 |
| 65 | 0.332905 | 0.864705 | 1 | 6.31336 |
| 70 | 0.327439 | 0.861971 | 0.954545 | 10.4752 |
| 75 | 0.320061 | 0.855614 | 0.909091 | 4.35883 |
| 80 | 0.302935 | 0.875315 | 0.877841 | 20.9735 |
| 85 | 0.305036 | 0.867754 | 0.977273 | 14.5497 |
| 90 | 0.257084 | 0.887195 | 0.855114 | 10.3414 |
| 95 | 0.245646 | 0.887195 | 0.886364 | 1.59997 |
| 100 | 0.287032 | 0.871951 | 1 | 2.31886 |

### 1790799044348827183\_r17\_chain\_late\_fusion\_2a3a70e9：分层 mask 指标

源码目录：/vepfs-mlp2/c20250601/251105016/project/dllm\_test/output/immune\_research\_releases/r12\_r19\_20260930；完成/目标 step：100/100。

kind=synthetic\_functional\_diagnostic；dataset_kind=—；biological_quality_claim=False。

| 时点 | mask level | 指标 | 值 |
| --- | --- | --- | --- |
| initial | mask25 | mask\_ratio | 0.25 |
| initial | mask25 | selected\_count | 21 |
| initial | mask25 | aa\_count | 18 |
| initial | mask25 | terminal\_count | 3 |
| initial | mask25 | loss | 4.65726 |
| initial | mask25 | aa\_loss | 4.70515 |
| initial | mask25 | terminal\_loss | 4.36986 |
| initial | mask25 | aa\_accuracy | 0 |
| initial | mask25 | terminal\_accuracy | 0 |
| initial | mask50 | mask\_ratio | 0.5 |
| initial | mask50 | selected\_count | 52 |
| initial | mask50 | aa\_count | 48 |
| initial | mask50 | terminal\_count | 4 |
| initial | mask50 | loss | 4.63829 |
| initial | mask50 | aa\_loss | 4.64417 |
| initial | mask50 | terminal\_loss | 4.56778 |
| initial | mask50 | aa\_accuracy | 0 |
| initial | mask50 | terminal\_accuracy | 0 |
| initial | mask75 | mask\_ratio | 0.75 |
| initial | mask75 | selected\_count | 77 |
| initial | mask75 | aa\_count | 69 |
| initial | mask75 | terminal\_count | 8 |
| initial | mask75 | loss | 4.56175 |
| initial | mask75 | aa\_loss | 4.54874 |
| initial | mask75 | terminal\_loss | 4.67396 |
| initial | mask75 | aa\_accuracy | 0 |
| initial | mask75 | terminal\_accuracy | 0 |
| initial | mask100 | mask\_ratio | 1 |
| initial | mask100 | selected\_count | 93 |
| initial | mask100 | aa\_count | 82 |
| initial | mask100 | terminal\_count | 11 |
| initial | mask100 | loss | 4.52144 |
| initial | mask100 | aa\_loss | 4.50036 |
| initial | mask100 | terminal\_loss | 4.67854 |
| initial | mask100 | aa\_accuracy | 0 |
| initial | mask100 | terminal\_accuracy | 0 |
| initial | mask\_mean | loss | 4.59468 |
| initial | mask\_mean | aa\_loss | 4.59961 |
| initial | mask\_mean | terminal\_loss | 4.57254 |
| initial | mask\_mean | aa\_accuracy | 0 |
| initial | mask\_mean | terminal\_accuracy | 0 |
| latest | mask25 | mask\_ratio | 0.25 |
| latest | mask25 | selected\_count | 21 |
| latest | mask25 | aa\_count | 18 |
| latest | mask25 | terminal\_count | 3 |
| latest | mask25 | loss | 0.00264621 |
| latest | mask25 | aa\_loss | 0.00299951 |
| latest | mask25 | terminal\_loss | 0.000526428 |
| latest | mask25 | aa\_accuracy | 1 |
| latest | mask25 | terminal\_accuracy | 1 |
| latest | mask50 | mask\_ratio | 0.5 |
| latest | mask50 | selected\_count | 52 |
| latest | mask50 | aa\_count | 48 |
| latest | mask50 | terminal\_count | 4 |
| latest | mask50 | loss | 0.00587384 |
| latest | mask50 | aa\_loss | 0.00581312 |
| latest | mask50 | terminal\_loss | 0.00660247 |
| latest | mask50 | aa\_accuracy | 1 |
| latest | mask50 | terminal\_accuracy | 1 |
| latest | mask75 | mask\_ratio | 0.75 |
| latest | mask75 | selected\_count | 77 |
| latest | mask75 | aa\_count | 69 |
| latest | mask75 | terminal\_count | 8 |
| latest | mask75 | loss | 0.0134234 |
| latest | mask75 | aa\_loss | 0.0129201 |
| latest | mask75 | terminal\_loss | 0.0177642 |
| latest | mask75 | aa\_accuracy | 1 |
| latest | mask75 | terminal\_accuracy | 1 |
| latest | mask100 | mask\_ratio | 1 |
| latest | mask100 | selected\_count | 93 |
| latest | mask100 | aa\_count | 82 |
| latest | mask100 | terminal\_count | 11 |
| latest | mask100 | loss | 1.02391 |
| latest | mask100 | aa\_loss | 1.10298 |
| latest | mask100 | terminal\_loss | 0.434533 |
| latest | mask100 | aa\_accuracy | 0.512195 |
| latest | mask100 | terminal\_accuracy | 0.818182 |
| latest | mask\_mean | loss | 0.261464 |
| latest | mask\_mean | aa\_loss | 0.281177 |
| latest | mask\_mean | terminal\_loss | 0.114856 |
| latest | mask\_mean | aa\_accuracy | 0.878049 |
| latest | mask\_mean | terminal\_accuracy | 0.954545 |

评测曲线（原始 step，四档等权均值）：

| step | mean loss | mean AA acc | mean end acc | gradient norm |
| --- | --- | --- | --- | --- |
| 5 | 3.3609 | 0.156918 | 0.0539773 | 23.7661 |
| 10 | 2.04486 | 0.211312 | 0 | 13.7249 |
| 15 | 1.92205 | 0.211463 | 0 | 4.34867 |
| 20 | 1.68532 | 0.362094 | 0 | 3.59883 |
| 25 | 1.3467 | 0.497261 | 0.153409 | 3.45511 |
| 30 | 0.830085 | 0.597138 | 0.683712 | 4.18329 |
| 35 | 0.535814 | 0.779396 | 0.860795 | 3.54034 |
| 40 | 0.713279 | 0.770594 | 0.860795 | 16.8514 |
| 45 | 0.397033 | 0.804046 | 0.923295 | 4.17339 |
| 50 | 0.422661 | 0.859358 | 0.74053 | 4.18403 |
| 55 | 0.302628 | 0.880523 | 0.815341 | 3.85639 |
| 60 | 0.276846 | 0.855299 | 0.977273 | 2.44861 |
| 65 | 0.296625 | 0.871951 | 1 | 1.67202 |
| 70 | 0.244464 | 0.884146 | 0.909091 | 2.27686 |
| 75 | 0.250706 | 0.878938 | 0.909091 | 6.97155 |
| 80 | 0.274268 | 0.868902 | 1 | 4.1264 |
| 85 | 0.265123 | 0.871377 | 0.954545 | 4.14312 |
| 90 | 0.321849 | 0.887195 | 0.886364 | 3.42687 |
| 95 | 0.311175 | 0.871951 | 1 | 3.75398 |
| 100 | 0.261464 | 0.878049 | 0.954545 | 0.774092 |

### 1790799044352237695\_r18\_chain\_local\_global\_a0a72b83：分层 mask 指标

源码目录：/vepfs-mlp2/c20250601/251105016/project/dllm\_test/output/immune\_research\_releases/r12\_r19\_20260930；完成/目标 step：100/100。

kind=synthetic\_functional\_diagnostic；dataset_kind=—；biological_quality_claim=False。

| 时点 | mask level | 指标 | 值 |
| --- | --- | --- | --- |
| initial | mask25 | mask\_ratio | 0.25 |
| initial | mask25 | selected\_count | 21 |
| initial | mask25 | aa\_count | 18 |
| initial | mask25 | terminal\_count | 3 |
| initial | mask25 | loss | 4.66608 |
| initial | mask25 | aa\_loss | 4.67737 |
| initial | mask25 | terminal\_loss | 4.59838 |
| initial | mask25 | aa\_accuracy | 0 |
| initial | mask25 | terminal\_accuracy | 0 |
| initial | mask50 | mask\_ratio | 0.5 |
| initial | mask50 | selected\_count | 52 |
| initial | mask50 | aa\_count | 48 |
| initial | mask50 | terminal\_count | 4 |
| initial | mask50 | loss | 4.61924 |
| initial | mask50 | aa\_loss | 4.61312 |
| initial | mask50 | terminal\_loss | 4.69264 |
| initial | mask50 | aa\_accuracy | 0 |
| initial | mask50 | terminal\_accuracy | 0 |
| initial | mask75 | mask\_ratio | 0.75 |
| initial | mask75 | selected\_count | 77 |
| initial | mask75 | aa\_count | 69 |
| initial | mask75 | terminal\_count | 8 |
| initial | mask75 | loss | 4.5563 |
| initial | mask75 | aa\_loss | 4.544 |
| initial | mask75 | terminal\_loss | 4.66241 |
| initial | mask75 | aa\_accuracy | 0 |
| initial | mask75 | terminal\_accuracy | 0 |
| initial | mask100 | mask\_ratio | 1 |
| initial | mask100 | selected\_count | 93 |
| initial | mask100 | aa\_count | 82 |
| initial | mask100 | terminal\_count | 11 |
| initial | mask100 | loss | 4.52596 |
| initial | mask100 | aa\_loss | 4.50718 |
| initial | mask100 | terminal\_loss | 4.66591 |
| initial | mask100 | aa\_accuracy | 0 |
| initial | mask100 | terminal\_accuracy | 0 |
| initial | mask\_mean | loss | 4.59189 |
| initial | mask\_mean | aa\_loss | 4.58542 |
| initial | mask\_mean | terminal\_loss | 4.65484 |
| initial | mask\_mean | aa\_accuracy | 0 |
| initial | mask\_mean | terminal\_accuracy | 0 |
| latest | mask25 | mask\_ratio | 0.25 |
| latest | mask25 | selected\_count | 21 |
| latest | mask25 | aa\_count | 18 |
| latest | mask25 | terminal\_count | 3 |
| latest | mask25 | loss | 0.154384 |
| latest | mask25 | aa\_loss | 0.00077304 |
| latest | mask25 | terminal\_loss | 1.07605 |
| latest | mask25 | aa\_accuracy | 1 |
| latest | mask25 | terminal\_accuracy | 0.666667 |
| latest | mask50 | mask\_ratio | 0.5 |
| latest | mask50 | selected\_count | 52 |
| latest | mask50 | aa\_count | 48 |
| latest | mask50 | terminal\_count | 4 |
| latest | mask50 | loss | 0.000875731 |
| latest | mask50 | aa\_loss | 0.000241539 |
| latest | mask50 | terminal\_loss | 0.00848604 |
| latest | mask50 | aa\_accuracy | 1 |
| latest | mask50 | terminal\_accuracy | 1 |
| latest | mask75 | mask\_ratio | 0.75 |
| latest | mask75 | selected\_count | 77 |
| latest | mask75 | aa\_count | 69 |
| latest | mask75 | terminal\_count | 8 |
| latest | mask75 | loss | 0.00171774 |
| latest | mask75 | aa\_loss | 0.0013775 |
| latest | mask75 | terminal\_loss | 0.00465234 |
| latest | mask75 | aa\_accuracy | 1 |
| latest | mask75 | terminal\_accuracy | 1 |
| latest | mask100 | mask\_ratio | 1 |
| latest | mask100 | selected\_count | 93 |
| latest | mask100 | aa\_count | 82 |
| latest | mask100 | terminal\_count | 11 |
| latest | mask100 | loss | 0.990162 |
| latest | mask100 | aa\_loss | 1.02755 |
| latest | mask100 | terminal\_loss | 0.711427 |
| latest | mask100 | aa\_accuracy | 0.54878 |
| latest | mask100 | terminal\_accuracy | 0.545455 |
| latest | mask\_mean | loss | 0.286785 |
| latest | mask\_mean | aa\_loss | 0.257486 |
| latest | mask\_mean | terminal\_loss | 0.450154 |
| latest | mask\_mean | aa\_accuracy | 0.887195 |
| latest | mask\_mean | terminal\_accuracy | 0.80303 |

评测曲线（原始 step，四档等权均值）：

| step | mean loss | mean AA acc | mean end acc | gradient norm |
| --- | --- | --- | --- | --- |
| 5 | 3.51962 | 0.153555 | 0.0539773 | 24.0782 |
| 10 | 2.00036 | 0.374884 | 0 | 13.682 |
| 15 | 1.66172 | 0.416475 | 0 | 4.72134 |
| 20 | 1.48097 | 0.449231 | 0.383523 | 3.1102 |
| 25 | 1.00942 | 0.545364 | 0.916667 | 3.73058 |
| 30 | 0.587444 | 0.744221 | 0.784091 | 5.08878 |
| 35 | 0.444934 | 0.831975 | 0.838068 | 5.51184 |
| 40 | 0.377764 | 0.851676 | 0.977273 | 5.42242 |
| 45 | 0.305023 | 0.849516 | 0.923295 | 5.54533 |
| 50 | 0.331509 | 0.864705 | 1 | 8.66959 |
| 55 | 0.251713 | 0.878049 | 0.923295 | 5.16776 |
| 60 | 0.260288 | 0.867179 | 0.954545 | 6.05356 |
| 65 | 0.331427 | 0.881098 | 0.900568 | 3.00103 |
| 70 | 0.347583 | 0.8769 | 0.886364 | 5.55931 |
| 75 | 0.282989 | 0.884146 | 0.909091 | 3.15252 |
| 80 | 0.239021 | 0.868328 | 1 | 2.38216 |
| 85 | 0.218936 | 0.887195 | 0.886364 | 1.69645 |
| 90 | 0.246537 | 0.890244 | 0.863636 | 0.88134 |
| 95 | 0.243412 | 0.887195 | 0.886364 | 2.18306 |
| 100 | 0.286785 | 0.887195 | 0.80303 | 0.617117 |

### 1790799044355869324\_r19\_value\_residual\_3bcfd14a：分层 mask 指标

源码目录：/vepfs-mlp2/c20250601/251105016/project/dllm\_test/output/immune\_research\_releases/r12\_r19\_20260930；完成/目标 step：100/100。

kind=synthetic\_functional\_diagnostic；dataset_kind=—；biological_quality_claim=False。

| 时点 | mask level | 指标 | 值 |
| --- | --- | --- | --- |
| initial | mask25 | mask\_ratio | 0.25 |
| initial | mask25 | selected\_count | 21 |
| initial | mask25 | aa\_count | 18 |
| initial | mask25 | terminal\_count | 3 |
| initial | mask25 | loss | 4.94061 |
| initial | mask25 | aa\_loss | 4.83391 |
| initial | mask25 | terminal\_loss | 5.58083 |
| initial | mask25 | aa\_accuracy | 0 |
| initial | mask25 | terminal\_accuracy | 0 |
| initial | mask50 | mask\_ratio | 0.5 |
| initial | mask50 | selected\_count | 52 |
| initial | mask50 | aa\_count | 48 |
| initial | mask50 | terminal\_count | 4 |
| initial | mask50 | loss | 4.84541 |
| initial | mask50 | aa\_loss | 4.79251 |
| initial | mask50 | terminal\_loss | 5.48023 |
| initial | mask50 | aa\_accuracy | 0 |
| initial | mask50 | terminal\_accuracy | 0 |
| initial | mask75 | mask\_ratio | 0.75 |
| initial | mask75 | selected\_count | 77 |
| initial | mask75 | aa\_count | 69 |
| initial | mask75 | terminal\_count | 8 |
| initial | mask75 | loss | 4.79829 |
| initial | mask75 | aa\_loss | 4.72149 |
| initial | mask75 | terminal\_loss | 5.46074 |
| initial | mask75 | aa\_accuracy | 0 |
| initial | mask75 | terminal\_accuracy | 0 |
| initial | mask100 | mask\_ratio | 1 |
| initial | mask100 | selected\_count | 93 |
| initial | mask100 | aa\_count | 82 |
| initial | mask100 | terminal\_count | 11 |
| initial | mask100 | loss | 4.74923 |
| initial | mask100 | aa\_loss | 4.66611 |
| initial | mask100 | terminal\_loss | 5.36889 |
| initial | mask100 | aa\_accuracy | 0 |
| initial | mask100 | terminal\_accuracy | 0 |
| initial | mask\_mean | loss | 4.83339 |
| initial | mask\_mean | aa\_loss | 4.7535 |
| initial | mask\_mean | terminal\_loss | 5.47267 |
| initial | mask\_mean | aa\_accuracy | 0 |
| initial | mask\_mean | terminal\_accuracy | 0 |
| latest | mask25 | mask\_ratio | 0.25 |
| latest | mask25 | selected\_count | 21 |
| latest | mask25 | aa\_count | 18 |
| latest | mask25 | terminal\_count | 3 |
| latest | mask25 | loss | 0.00243821 |
| latest | mask25 | aa\_loss | 0.00284439 |
| latest | mask25 | terminal\_loss | 1.11262e-06 |
| latest | mask25 | aa\_accuracy | 1 |
| latest | mask25 | terminal\_accuracy | 1 |
| latest | mask50 | mask\_ratio | 0.5 |
| latest | mask50 | selected\_count | 52 |
| latest | mask50 | aa\_count | 48 |
| latest | mask50 | terminal\_count | 4 |
| latest | mask50 | loss | 0.00143298 |
| latest | mask50 | aa\_loss | 0.00154871 |
| latest | mask50 | terminal\_loss | 4.43142e-05 |
| latest | mask50 | aa\_accuracy | 1 |
| latest | mask50 | terminal\_accuracy | 1 |
| latest | mask75 | mask\_ratio | 0.75 |
| latest | mask75 | selected\_count | 77 |
| latest | mask75 | aa\_count | 69 |
| latest | mask75 | terminal\_count | 8 |
| latest | mask75 | loss | 0.00470804 |
| latest | mask75 | aa\_loss | 0.00422119 |
| latest | mask75 | terminal\_loss | 0.00890714 |
| latest | mask75 | aa\_accuracy | 1 |
| latest | mask75 | terminal\_accuracy | 1 |
| latest | mask100 | mask\_ratio | 1 |
| latest | mask100 | selected\_count | 93 |
| latest | mask100 | aa\_count | 82 |
| latest | mask100 | terminal\_count | 11 |
| latest | mask100 | loss | 0.990644 |
| latest | mask100 | aa\_loss | 1.04397 |
| latest | mask100 | terminal\_loss | 0.593153 |
| latest | mask100 | aa\_accuracy | 0.52439 |
| latest | mask100 | terminal\_accuracy | 0.727273 |
| latest | mask\_mean | loss | 0.249806 |
| latest | mask\_mean | aa\_loss | 0.263145 |
| latest | mask\_mean | terminal\_loss | 0.150526 |
| latest | mask\_mean | aa\_accuracy | 0.881098 |
| latest | mask\_mean | terminal\_accuracy | 0.931818 |

评测曲线（原始 step，四档等权均值）：

| step | mean loss | mean AA acc | mean end acc | gradient norm |
| --- | --- | --- | --- | --- |
| 5 | 2.92108 | 0.0761882 | 0 | 26.9264 |
| 10 | 2.09358 | 0.219792 | 0 | 9.09668 |
| 15 | 1.89547 | 0.314448 | 0 | 3.51171 |
| 20 | 1.75743 | 0.430859 | 0 | 3.03962 |
| 25 | 1.43221 | 0.448371 | 0.258523 | 3.89196 |
| 30 | 0.855009 | 0.672805 | 0.760417 | 4.77857 |
| 35 | 0.499315 | 0.783967 | 0.792614 | 5.29235 |
| 40 | 0.475863 | 0.808998 | 0.923295 | 9.83565 |
| 45 | 0.412399 | 0.841984 | 0.829545 | 7.44913 |
| 50 | 0.356319 | 0.858922 | 0.954545 | 6.1416 |
| 55 | 0.286412 | 0.874426 | 0.954545 | 4.51285 |
| 60 | 0.281914 | 0.884146 | 0.877841 | 1.74571 |
| 65 | 0.315596 | 0.881098 | 0.792614 | 21.3043 |
| 70 | 0.230239 | 0.884146 | 0.909091 | 3.02743 |
| 75 | 0.224222 | 0.884146 | 0.909091 | 1.11499 |
| 80 | 0.256827 | 0.875 | 0.977273 | 1.79663 |
| 85 | 0.229102 | 0.887195 | 0.886364 | 1.47889 |
| 90 | 0.253252 | 0.875 | 0.977273 | 4.40846 |
| 95 | 0.219804 | 0.887195 | 0.886364 | 4.05655 |
| 100 | 0.249806 | 0.881098 | 0.931818 | 6.0097 |

### 1790802160338779408\_r20\_refined\_control\_3b41bffe：分层 mask 指标

源码目录：/vepfs-mlp2/c20250601/251105016/project/dllm\_test/output/immune\_research\_releases/r20\_r27\_20261001；完成/目标 step：25/25。

kind=synthetic\_generalization\_diagnostic；dataset_kind=deterministic\_artificial\_sequences；biological_quality_claim=False。

| 时点 | mask level | 指标 | 值 |
| --- | --- | --- | --- |
| initial | mask25 | mask\_ratio | 0.25 |
| initial | mask25 | selected\_count | 4467 |
| initial | mask25 | aa\_count | 4428 |
| initial | mask25 | terminal\_count | 39 |
| initial | mask25 | loss | 3.83723 |
| initial | mask25 | aa\_loss | 3.83333 |
| initial | mask25 | aa\_accuracy | 0.0440379 |
| initial | mask25 | terminal\_loss | 4.27954 |
| initial | mask25 | terminal\_accuracy | 0 |
| initial | mask50 | mask\_ratio | 0.5 |
| initial | mask50 | selected\_count | 8888 |
| initial | mask50 | aa\_count | 8819 |
| initial | mask50 | terminal\_count | 69 |
| initial | mask50 | loss | 3.95698 |
| initial | mask50 | aa\_loss | 3.95337 |
| initial | mask50 | aa\_accuracy | 0.0481914 |
| initial | mask50 | terminal\_loss | 4.41734 |
| initial | mask50 | terminal\_accuracy | 0 |
| initial | mask75 | mask\_ratio | 0.75 |
| initial | mask75 | selected\_count | 13437 |
| initial | mask75 | aa\_count | 13330 |
| initial | mask75 | terminal\_count | 107 |
| initial | mask75 | loss | 4.0171 |
| initial | mask75 | aa\_loss | 4.01295 |
| initial | mask75 | aa\_accuracy | 0.0588897 |
| initial | mask75 | terminal\_loss | 4.53343 |
| initial | mask75 | terminal\_accuracy | 0 |
| initial | mask100 | mask\_ratio | 1 |
| initial | mask100 | selected\_count | 17996 |
| initial | mask100 | aa\_count | 17864 |
| initial | mask100 | terminal\_count | 132 |
| initial | mask100 | loss | 3.99692 |
| initial | mask100 | aa\_loss | 3.99259 |
| initial | mask100 | aa\_accuracy | 0.033979 |
| initial | mask100 | terminal\_loss | 4.58199 |
| initial | mask100 | terminal\_accuracy | 0 |
| initial | mask\_mean | loss | 3.95206 |
| initial | mask\_mean | aa\_loss | 3.94806 |
| initial | mask\_mean | terminal\_loss | 4.45307 |
| initial | mask\_mean | aa\_accuracy | 0.0462745 |
| initial | mask\_mean | terminal\_accuracy | 0 |
| latest | mask25 | mask\_ratio | 0.25 |
| latest | mask25 | selected\_count | 4467 |
| latest | mask25 | aa\_count | 4428 |
| latest | mask25 | terminal\_count | 39 |
| latest | mask25 | loss | 1.87435 |
| latest | mask25 | aa\_loss | 1.84471 |
| latest | mask25 | aa\_accuracy | 0.312556 |
| latest | mask25 | terminal\_loss | 5.23954 |
| latest | mask25 | terminal\_accuracy | 0 |
| latest | mask50 | mask\_ratio | 0.5 |
| latest | mask50 | selected\_count | 8888 |
| latest | mask50 | aa\_count | 8819 |
| latest | mask50 | terminal\_count | 69 |
| latest | mask50 | loss | 1.87066 |
| latest | mask50 | aa\_loss | 1.84522 |
| latest | mask50 | aa\_accuracy | 0.314095 |
| latest | mask50 | terminal\_loss | 5.12312 |
| latest | mask50 | terminal\_accuracy | 0 |
| latest | mask75 | mask\_ratio | 0.75 |
| latest | mask75 | selected\_count | 13437 |
| latest | mask75 | aa\_count | 13330 |
| latest | mask75 | terminal\_count | 107 |
| latest | mask75 | loss | 1.86597 |
| latest | mask75 | aa\_loss | 1.84061 |
| latest | mask75 | aa\_accuracy | 0.309977 |
| latest | mask75 | terminal\_loss | 5.02579 |
| latest | mask75 | terminal\_accuracy | 0 |
| latest | mask100 | mask\_ratio | 1 |
| latest | mask100 | selected\_count | 17996 |
| latest | mask100 | aa\_count | 17864 |
| latest | mask100 | terminal\_count | 132 |
| latest | mask100 | loss | 2.0203 |
| latest | mask100 | aa\_loss | 2.00126 |
| latest | mask100 | aa\_accuracy | 0.265618 |
| latest | mask100 | terminal\_loss | 4.59695 |
| latest | mask100 | terminal\_accuracy | 0 |
| latest | mask\_mean | loss | 1.90782 |
| latest | mask\_mean | aa\_loss | 1.88295 |
| latest | mask\_mean | terminal\_loss | 4.99635 |
| latest | mask\_mean | aa\_accuracy | 0.300562 |
| latest | mask\_mean | terminal\_accuracy | 0 |

评测曲线（原始 step，四档等权均值）：

| step | mean loss | mean AA acc | mean end acc | gradient norm |
| --- | --- | --- | --- | --- |
| 25 | 1.90782 | 0.300562 | 0 | 3.65246 |

有条件/无条件曲线（每档按token计数合并，再平均四档；缺失分组不推断数值）：

| step | 子集 | mean loss | mean AA | mean end | Full-mask AA | Full-mask end | Full-mask AA count | Full-mask end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | conditional | 3.99673 | 0.043826 | 0 | 0.034688 | 0 | 9744 | 72 |
| initial | unconditional | 3.89862 | 0.0492004 | 0 | 0.0331281 | 0 | 8120 | 60 |
| 25 | conditional | 1.84973 | 0.301396 | 0 | 0.272065 | 0 | 9744 | 72 |
| 25 | unconditional | 1.97741 | 0.299542 | 0 | 0.257882 | 0 | 8120 | 60 |

人工数据 manifest：d40bbbbc1c4294eed843d88bcdd0cf193886de6c2d7e9db9069bbf86e931c022；精确 target sequence 重叠=0；精确 record 重叠=0。

跨 radius32 覆盖：{&quot;possible&quot;: true, &quot;train&quot;: true, &quot;valid&quot;: true}。生成规则共享是诊断设计的一部分，不能当作真实下游去重证书。


按任务/链长分组（全局 level 按 token 计数汇总，下面保留各组计数；不同组不得直接比较未经加权的总 loss）：

| 时点 | group/length | mask level | loss | AA acc | end acc | selected count | AA count | end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | ab\_antigen/long | mask25 | 4.18791 | 0 | 0 | 530 | 526 | 4 |
| initial | ab\_antigen/long | mask50 | 4.46389 | 0 | 0 | 1019 | 1013 | 6 |
| initial | ab\_antigen/long | mask75 | 4.58208 | 0.0274151 | 0 | 1538 | 1532 | 6 |
| initial | ab\_antigen/long | mask100 | 4.47814 | 0.0303922 | 0 | 2048 | 2040 | 8 |
| initial | ab\_antigen/long | mask\_mean | 4.42801 | 0.0144518 | 0 | — | — | — |
| initial | ab\_antigen/medium | mask25 | 4.60088 | 0.0288066 | 0 | 246 | 243 | 3 |
| initial | ab\_antigen/medium | mask50 | 4.58408 | 0.0677291 | 0 | 507 | 502 | 5 |
| initial | ab\_antigen/medium | mask75 | 4.58341 | 0.0620053 | 0 | 765 | 758 | 7 |
| initial | ab\_antigen/medium | mask100 | 4.52137 | 0.0610236 | 0 | 1024 | 1016 | 8 |
| initial | ab\_antigen/medium | mask\_mean | 4.57243 | 0.0548911 | 0 | — | — | — |
| initial | ab\_antigen/short | mask25 | 4.71372 | 0.0588235 | 0 | 38 | 34 | 4 |
| initial | ab\_antigen/short | mask50 | 4.46987 | 0.0795455 | 0 | 93 | 88 | 5 |
| initial | ab\_antigen/short | mask75 | 4.39464 | 0.0785714 | 0 | 145 | 140 | 5 |
| initial | ab\_antigen/short | mask100 | 4.34276 | 0.09375 | 0 | 200 | 192 | 8 |
| initial | ab\_antigen/short | mask\_mean | 4.48025 | 0.0776726 | 0 | — | — | — |
| initial | tcr\_peptide/long | mask25 | 3.61712 | 0.088172 | 0 | 466 | 465 | 1 |
| initial | tcr\_peptide/long | mask50 | 3.7337 | 0.114169 | 0 | 985 | 981 | 4 |
| initial | tcr\_peptide/long | mask75 | 3.79636 | 0.116373 | 0 | 1484 | 1478 | 6 |
| initial | tcr\_peptide/long | mask100 | 3.82355 | 0.0720588 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_peptide/long | mask\_mean | 3.74268 | 0.0976934 | 0 | — | — | — |
| initial | tcr\_peptide/medium | mask25 | 3.81146 | 0.0248963 | 0 | 242 | 241 | 1 |
| initial | tcr\_peptide/medium | mask50 | 3.79279 | 0.0119522 | 0 | 506 | 502 | 4 |
| initial | tcr\_peptide/medium | mask75 | 3.72871 | 0 | 0 | 767 | 760 | 7 |
| initial | tcr\_peptide/medium | mask100 | 3.66509 | 0 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_peptide/medium | mask\_mean | 3.74952 | 0.00921211 | 0 | — | — | — |
| initial | tcr\_peptide/short | mask25 | 3.65752 | 0 | 0 | 51 | 48 | 3 |
| initial | tcr\_peptide/short | mask50 | 3.54852 | 0 | 0 | 110 | 104 | 6 |
| initial | tcr\_peptide/short | mask75 | 3.58623 | 0 | 0 | 144 | 137 | 7 |
| initial | tcr\_peptide/short | mask100 | 3.62608 | 0 | 0 | 200 | 192 | 8 |
| initial | tcr\_peptide/short | mask\_mean | 3.60459 | 0 | 0 | — | — | — |
| initial | tcr\_pmhc/long | mask25 | 3.25602 | 0.0391791 | 0 | 540 | 536 | 4 |
| initial | tcr\_pmhc/long | mask50 | 3.48187 | 0.0357834 | 0 | 1038 | 1034 | 4 |
| initial | tcr\_pmhc/long | mask75 | 3.6956 | 0.032175 | 0 | 1562 | 1554 | 8 |
| initial | tcr\_pmhc/long | mask100 | 3.82046 | 0.0181373 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_pmhc/long | mask\_mean | 3.56349 | 0.0313187 | 0 | — | — | — |
| initial | tcr\_pmhc/medium | mask25 | 4.03386 | 0.0950413 | 0 | 244 | 242 | 2 |
| initial | tcr\_pmhc/medium | mask50 | 4.13424 | 0.0793319 | 0 | 484 | 479 | 5 |
| initial | tcr\_pmhc/medium | mask75 | 4.21764 | 0.0520156 | 0 | 777 | 769 | 8 |
| initial | tcr\_pmhc/medium | mask100 | 4.2521 | 0.011811 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_pmhc/medium | mask\_mean | 4.15946 | 0.05955 | 0 | — | — | — |
| initial | tcr\_pmhc/short | mask25 | 4.12421 | 0 | 0 | 51 | 48 | 3 |
| initial | tcr\_pmhc/short | mask50 | 4.20187 | 0 | 0 | 86 | 83 | 3 |
| initial | tcr\_pmhc/short | mask75 | 4.10686 | 0 | 0 | 152 | 147 | 5 |
| initial | tcr\_pmhc/short | mask100 | 4.11286 | 0 | 0 | 200 | 192 | 8 |
| initial | tcr\_pmhc/short | mask\_mean | 4.13645 | 0 | 0 | — | — | — |
| initial | unconditional\_ab/long | mask25 | 3.91889 | 0.0317757 | 0 | 538 | 535 | 3 |
| initial | unconditional\_ab/long | mask50 | 4.14326 | 0.0229446 | 0 | 1050 | 1046 | 4 |
| initial | unconditional\_ab/long | mask75 | 4.2586 | 0.0630689 | 0 | 1545 | 1538 | 7 |
| initial | unconditional\_ab/long | mask100 | 4.17438 | 0.0588235 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_ab/long | mask\_mean | 4.12378 | 0.0441532 | 0 | — | — | — |
| initial | unconditional\_ab/medium | mask25 | 4.23935 | 0.146245 | 0 | 256 | 253 | 3 |
| initial | unconditional\_ab/medium | mask50 | 4.34042 | 0.137795 | 0 | 512 | 508 | 4 |
| initial | unconditional\_ab/medium | mask75 | 4.31506 | 0.143045 | 0 | 768 | 762 | 6 |
| initial | unconditional\_ab/medium | mask100 | 4.25781 | 0.140748 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_ab/medium | mask\_mean | 4.28816 | 0.141958 | 0 | — | — | — |
| initial | unconditional\_ab/short | mask25 | 3.98838 | 0.0961538 | 0 | 55 | 52 | 3 |
| initial | unconditional\_ab/short | mask50 | 3.93455 | 0.0612245 | 0 | 103 | 98 | 5 |
| initial | unconditional\_ab/short | mask75 | 3.90859 | 0.0567376 | 0 | 148 | 141 | 7 |
| initial | unconditional\_ab/short | mask100 | 3.87788 | 0.03125 | 0 | 200 | 192 | 8 |
| initial | unconditional\_ab/short | mask\_mean | 3.92735 | 0.0613415 | 0 | — | — | — |
| initial | unconditional\_single/long | mask25 | 3.42862 | 0.0485075 | 0 | 269 | 268 | 1 |
| initial | unconditional\_single/long | mask50 | 3.37246 | 0.08 | 0 | 527 | 525 | 2 |
| initial | unconditional\_single/long | mask75 | 3.3423 | 0.192506 | 0 | 778 | 774 | 4 |
| initial | unconditional\_single/long | mask100 | 3.3784 | 0 | 0 | 1024 | 1020 | 4 |
| initial | unconditional\_single/long | mask\_mean | 3.38045 | 0.0802535 | 0 | — | — | — |
| initial | unconditional\_single/medium | mask25 | 3.71448 | 0.0366972 | 0 | 110 | 109 | 1 |
| initial | unconditional\_single/medium | mask50 | 3.49804 | 0.0818966 | 0 | 234 | 232 | 2 |
| initial | unconditional\_single/medium | mask75 | 3.50775 | 0.163488 | 0 | 371 | 367 | 4 |
| initial | unconditional\_single/medium | mask100 | 3.501 | 0 | 0 | 512 | 508 | 4 |
| initial | unconditional\_single/medium | mask\_mean | 3.55532 | 0.0705204 | 0 | — | — | — |
| initial | unconditional\_single/short | mask25 | 3.11568 | 0 | 0 | 28 | 26 | 2 |
| initial | unconditional\_single/short | mask50 | 3.29004 | 0 | 0 | 52 | 50 | 2 |
| initial | unconditional\_single/short | mask75 | 3.10684 | 0 | 0 | 80 | 77 | 3 |
| initial | unconditional\_single/short | mask100 | 3.10388 | 0 | 0 | 100 | 96 | 4 |
| initial | unconditional\_single/short | mask\_mean | 3.15411 | 0 | 0 | — | — | — |
| initial | unconditional\_tcr/long | mask25 | 3.72215 | 0.0368932 | 0 | 516 | 515 | 1 |
| initial | unconditional\_tcr/long | mask50 | 3.86969 | 0.0303951 | 0 | 989 | 987 | 2 |
| initial | unconditional\_tcr/long | mask75 | 3.91864 | 0 | 0 | 1527 | 1522 | 5 |
| initial | unconditional\_tcr/long | mask100 | 3.85547 | 0 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_tcr/long | mask\_mean | 3.84149 | 0.0168221 | 0 | — | — | — |
| initial | unconditional\_tcr/medium | mask25 | 3.84478 | 0 | — | 247 | 247 | 0 |
| initial | unconditional\_tcr/medium | mask50 | 3.96335 | 0 | 0 | 500 | 497 | 3 |
| initial | unconditional\_tcr/medium | mask75 | 3.94344 | 0 | 0 | 744 | 738 | 6 |
| initial | unconditional\_tcr/medium | mask100 | 3.92848 | 0 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_tcr/medium | mask\_mean | 3.92001 | 0 | 0 | — | — | — |
| initial | unconditional\_tcr/short | mask25 | 4.04066 | 0 | — | 40 | 40 | 0 |
| initial | unconditional\_tcr/short | mask50 | 3.91056 | 0 | 0 | 93 | 90 | 3 |
| initial | unconditional\_tcr/short | mask75 | 3.96515 | 0 | 0 | 142 | 136 | 6 |
| initial | unconditional\_tcr/short | mask100 | 3.91533 | 0 | 0 | 200 | 192 | 8 |
| initial | unconditional\_tcr/short | mask\_mean | 3.95792 | 0 | 0 | — | — | — |
| latest | ab\_antigen/long | mask25 | 1.81207 | 0.355513 | 0 | 530 | 526 | 4 |
| latest | ab\_antigen/long | mask50 | 1.78914 | 0.371175 | 0 | 1019 | 1013 | 6 |
| latest | ab\_antigen/long | mask75 | 1.76735 | 0.368799 | 0 | 1538 | 1532 | 6 |
| latest | ab\_antigen/long | mask100 | 1.81485 | 0.241176 | 0 | 2048 | 2040 | 8 |
| latest | ab\_antigen/long | mask\_mean | 1.79585 | 0.334166 | 0 | — | — | — |
| latest | ab\_antigen/medium | mask25 | 1.90586 | 0.312757 | 0 | 246 | 243 | 3 |
| latest | ab\_antigen/medium | mask50 | 1.94491 | 0.284861 | 0 | 507 | 502 | 5 |
| latest | ab\_antigen/medium | mask75 | 1.90018 | 0.290237 | 0 | 765 | 758 | 7 |
| latest | ab\_antigen/medium | mask100 | 1.88474 | 0.294291 | 0 | 1024 | 1016 | 8 |
| latest | ab\_antigen/medium | mask\_mean | 1.90892 | 0.295537 | 0 | — | — | — |
| latest | ab\_antigen/short | mask25 | 2.36092 | 0.264706 | 0 | 38 | 34 | 4 |
| latest | ab\_antigen/short | mask50 | 2.26105 | 0.261364 | 0 | 93 | 88 | 5 |
| latest | ab\_antigen/short | mask75 | 2.17604 | 0.221429 | 0 | 145 | 140 | 5 |
| latest | ab\_antigen/short | mask100 | 2.22386 | 0.244792 | 0 | 200 | 192 | 8 |
| latest | ab\_antigen/short | mask\_mean | 2.25547 | 0.248072 | 0 | — | — | — |
| latest | tcr\_peptide/long | mask25 | 1.90772 | 0.266667 | 0 | 466 | 465 | 1 |
| latest | tcr\_peptide/long | mask50 | 1.9032 | 0.277268 | 0 | 985 | 981 | 4 |
| latest | tcr\_peptide/long | mask75 | 1.89932 | 0.253721 | 0 | 1484 | 1478 | 6 |
| latest | tcr\_peptide/long | mask100 | 1.93077 | 0.219118 | 0 | 2048 | 2040 | 8 |
| latest | tcr\_peptide/long | mask\_mean | 1.91025 | 0.254193 | 0 | — | — | — |
| latest | tcr\_peptide/medium | mask25 | 2.02957 | 0.232365 | 0 | 242 | 241 | 1 |
| latest | tcr\_peptide/medium | mask50 | 2.01297 | 0.262948 | 0 | 506 | 502 | 4 |
| latest | tcr\_peptide/medium | mask75 | 2.02391 | 0.267105 | 0 | 767 | 760 | 7 |
| latest | tcr\_peptide/medium | mask100 | 2.07008 | 0.253937 | 0 | 1024 | 1016 | 8 |
| latest | tcr\_peptide/medium | mask\_mean | 2.03413 | 0.254089 | 0 | — | — | — |
| latest | tcr\_peptide/short | mask25 | 2.46253 | 0.229167 | 0 | 51 | 48 | 3 |
| latest | tcr\_peptide/short | mask50 | 2.44452 | 0.221154 | 0 | 110 | 104 | 6 |
| latest | tcr\_peptide/short | mask75 | 2.33045 | 0.233577 | 0 | 144 | 137 | 7 |
| latest | tcr\_peptide/short | mask100 | 2.34526 | 0.213542 | 0 | 200 | 192 | 8 |
| latest | tcr\_peptide/short | mask\_mean | 2.39569 | 0.22436 | 0 | — | — | — |
| latest | tcr\_pmhc/long | mask25 | 1.81048 | 0.279851 | 0 | 540 | 536 | 4 |
| latest | tcr\_pmhc/long | mask50 | 1.71621 | 0.304642 | 0 | 1038 | 1034 | 4 |
| latest | tcr\_pmhc/long | mask75 | 1.69145 | 0.293436 | 0 | 1562 | 1554 | 8 |
| latest | tcr\_pmhc/long | mask100 | 1.69913 | 0.296078 | 0 | 2048 | 2040 | 8 |
| latest | tcr\_pmhc/long | mask\_mean | 1.72932 | 0.293502 | 0 | — | — | — |
| latest | tcr\_pmhc/medium | mask25 | 1.56237 | 0.446281 | 0 | 244 | 242 | 2 |
| latest | tcr\_pmhc/medium | mask50 | 1.60941 | 0.417537 | 0 | 484 | 479 | 5 |
| latest | tcr\_pmhc/medium | mask75 | 1.63333 | 0.410923 | 0 | 777 | 769 | 8 |
| latest | tcr\_pmhc/medium | mask100 | 1.68249 | 0.399606 | 0 | 1024 | 1016 | 8 |
| latest | tcr\_pmhc/medium | mask\_mean | 1.6219 | 0.418587 | 0 | — | — | — |
| latest | tcr\_pmhc/short | mask25 | 2.05654 | 0.354167 | 0 | 51 | 48 | 3 |
| latest | tcr\_pmhc/short | mask50 | 1.97759 | 0.337349 | 0 | 86 | 83 | 3 |
| latest | tcr\_pmhc/short | mask75 | 2.01569 | 0.285714 | 0 | 152 | 147 | 5 |
| latest | tcr\_pmhc/short | mask100 | 2.12023 | 0.296875 | 0 | 200 | 192 | 8 |
| latest | tcr\_pmhc/short | mask\_mean | 2.04251 | 0.318526 | 0 | — | — | — |
| latest | unconditional\_ab/long | mask25 | 1.90896 | 0.314019 | 0 | 538 | 535 | 3 |
| latest | unconditional\_ab/long | mask50 | 1.92513 | 0.319312 | 0 | 1050 | 1046 | 4 |
| latest | unconditional\_ab/long | mask75 | 1.90763 | 0.341352 | 0 | 1545 | 1538 | 7 |
| latest | unconditional\_ab/long | mask100 | 2.12002 | 0.336765 | 0 | 2048 | 2040 | 8 |
| latest | unconditional\_ab/long | mask\_mean | 1.96544 | 0.327862 | 0 | — | — | — |
| latest | unconditional\_ab/medium | mask25 | 1.96053 | 0.3083 | 0 | 256 | 253 | 3 |
| latest | unconditional\_ab/medium | mask50 | 2.01056 | 0.295276 | 0 | 512 | 508 | 4 |
| latest | unconditional\_ab/medium | mask75 | 1.98245 | 0.301837 | 0 | 768 | 762 | 6 |
| latest | unconditional\_ab/medium | mask100 | 2.18233 | 0.298228 | 0 | 1024 | 1016 | 8 |
| latest | unconditional\_ab/medium | mask\_mean | 2.03397 | 0.30091 | 0 | — | — | — |
| latest | unconditional\_ab/short | mask25 | 2.20269 | 0.25 | 0 | 55 | 52 | 3 |
| latest | unconditional\_ab/short | mask50 | 2.37665 | 0.234694 | 0 | 103 | 98 | 5 |
| latest | unconditional\_ab/short | mask75 | 2.50024 | 0.22695 | 0 | 148 | 141 | 7 |
| latest | unconditional\_ab/short | mask100 | 2.72009 | 0.229167 | 0 | 200 | 192 | 8 |
| latest | unconditional\_ab/short | mask\_mean | 2.44992 | 0.235203 | 0 | — | — | — |
| latest | unconditional\_single/long | mask25 | 1.74359 | 0.320896 | 0 | 269 | 268 | 1 |
| latest | unconditional\_single/long | mask50 | 1.75626 | 0.310476 | 0 | 527 | 525 | 2 |
| latest | unconditional\_single/long | mask75 | 1.78521 | 0.303618 | 0 | 778 | 774 | 4 |
| latest | unconditional\_single/long | mask100 | 2.18481 | 0.188235 | 0 | 1024 | 1020 | 4 |
| latest | unconditional\_single/long | mask\_mean | 1.86747 | 0.280806 | 0 | — | — | — |
| latest | unconditional\_single/medium | mask25 | 1.79416 | 0.321101 | 0 | 110 | 109 | 1 |
| latest | unconditional\_single/medium | mask50 | 1.79364 | 0.344828 | 0 | 234 | 232 | 2 |
| latest | unconditional\_single/medium | mask75 | 1.86905 | 0.258856 | 0 | 371 | 367 | 4 |
| latest | unconditional\_single/medium | mask100 | 2.21275 | 0.15748 | 0 | 512 | 508 | 4 |
| latest | unconditional\_single/medium | mask\_mean | 1.9174 | 0.270566 | 0 | — | — | — |
| latest | unconditional\_single/short | mask25 | 2.84172 | 0.346154 | 0 | 28 | 26 | 2 |
| latest | unconditional\_single/short | mask50 | 2.62139 | 0.2 | 0 | 52 | 50 | 2 |
| latest | unconditional\_single/short | mask75 | 2.51592 | 0.194805 | 0 | 80 | 77 | 3 |
| latest | unconditional\_single/short | mask100 | 2.49391 | 0.0625 | 0 | 100 | 96 | 4 |
| latest | unconditional\_single/short | mask\_mean | 2.61823 | 0.200865 | 0 | — | — | — |
| latest | unconditional\_tcr/long | mask25 | 1.90185 | 0.31068 | 0 | 516 | 515 | 1 |
| latest | unconditional\_tcr/long | mask50 | 1.90189 | 0.300912 | 0 | 989 | 987 | 2 |
| latest | unconditional\_tcr/long | mask75 | 1.91228 | 0.309461 | 0 | 1527 | 1522 | 5 |
| latest | unconditional\_tcr/long | mask100 | 2.31408 | 0.214706 | 0 | 2048 | 2040 | 8 |
| latest | unconditional\_tcr/long | mask\_mean | 2.00753 | 0.28394 | 0 | — | — | — |
| latest | unconditional\_tcr/medium | mask25 | 1.66852 | 0.352227 | — | 247 | 247 | 0 |
| latest | unconditional\_tcr/medium | mask50 | 1.67174 | 0.368209 | 0 | 500 | 497 | 3 |
| latest | unconditional\_tcr/medium | mask75 | 1.72023 | 0.355014 | 0 | 744 | 738 | 6 |
| latest | unconditional\_tcr/medium | mask100 | 2.07341 | 0.305118 | 0 | 1024 | 1016 | 8 |
| latest | unconditional\_tcr/medium | mask\_mean | 1.78347 | 0.345142 | 0 | — | — | — |
| latest | unconditional\_tcr/short | mask25 | 2.36947 | 0.25 | — | 40 | 40 | 0 |
| latest | unconditional\_tcr/short | mask50 | 2.37019 | 0.2 | 0 | 93 | 90 | 3 |
| latest | unconditional\_tcr/short | mask75 | 2.38267 | 0.198529 | 0 | 142 | 136 | 6 |
| latest | unconditional\_tcr/short | mask100 | 2.46754 | 0.177083 | 0 | 200 | 192 | 8 |
| latest | unconditional\_tcr/short | mask\_mean | 2.39747 | 0.206403 | 0 | — | — | — |
| train\_probe | ab\_antigen/long | mask25 | 1.60398 | 0.358491 | — | 530 | 530 | 0 |
| train\_probe | ab\_antigen/long | mask50 | 1.62215 | 0.373424 | 0 | 1034 | 1031 | 3 |
| train\_probe | ab\_antigen/long | mask75 | 1.61509 | 0.357739 | 0 | 1563 | 1557 | 6 |
| train\_probe | ab\_antigen/long | mask100 | 1.70238 | 0.330392 | 0 | 2048 | 2040 | 8 |
| train\_probe | ab\_antigen/long | mask\_mean | 1.6359 | 0.355011 | 0 | — | — | — |
| train\_probe | ab\_antigen/medium | mask25 | 2.1208 | 0.289377 | 0 | 275 | 273 | 2 |
| train\_probe | ab\_antigen/medium | mask50 | 2.04291 | 0.301115 | 0 | 542 | 538 | 4 |
| train\_probe | ab\_antigen/medium | mask75 | 1.99499 | 0.313144 | 0 | 781 | 776 | 5 |
| train\_probe | ab\_antigen/medium | mask100 | 1.97326 | 0.324803 | 0 | 1024 | 1016 | 8 |
| train\_probe | ab\_antigen/medium | mask\_mean | 2.03299 | 0.30711 | 0 | — | — | — |
| train\_probe | ab\_antigen/short | mask25 | 2.25151 | 0.361702 | 0 | 52 | 47 | 5 |
| train\_probe | ab\_antigen/short | mask50 | 2.21879 | 0.294118 | 0 | 109 | 102 | 7 |
| train\_probe | ab\_antigen/short | mask75 | 2.16484 | 0.297297 | 0 | 155 | 148 | 7 |
| train\_probe | ab\_antigen/short | mask100 | 2.23587 | 0.291667 | 0 | 200 | 192 | 8 |
| train\_probe | ab\_antigen/short | mask\_mean | 2.21775 | 0.311196 | 0 | — | — | — |
| train\_probe | tcr\_peptide/long | mask25 | 1.73328 | 0.384615 | 0 | 495 | 494 | 1 |
| train\_probe | tcr\_peptide/long | mask50 | 1.72661 | 0.373541 | 0 | 1033 | 1028 | 5 |
| train\_probe | tcr\_peptide/long | mask75 | 1.75632 | 0.344941 | 0 | 1530 | 1522 | 8 |
| train\_probe | tcr\_peptide/long | mask100 | 1.81279 | 0.321078 | 0 | 2048 | 2040 | 8 |
| train\_probe | tcr\_peptide/long | mask\_mean | 1.75725 | 0.356044 | 0 | — | — | — |
| train\_probe | tcr\_peptide/medium | mask25 | 1.86735 | 0.293651 | — | 252 | 252 | 0 |
| train\_probe | tcr\_peptide/medium | mask50 | 1.87341 | 0.339181 | 0 | 517 | 513 | 4 |
| train\_probe | tcr\_peptide/medium | mask75 | 1.86679 | 0.323607 | 0 | 760 | 754 | 6 |
| train\_probe | tcr\_peptide/medium | mask100 | 1.96583 | 0.292323 | 0 | 1024 | 1016 | 8 |
| train\_probe | tcr\_peptide/medium | mask\_mean | 1.89335 | 0.312191 | 0 | — | — | — |
| train\_probe | tcr\_peptide/short | mask25 | 2.19307 | 0.347826 | 0 | 51 | 46 | 5 |
| train\_probe | tcr\_peptide/short | mask50 | 2.04968 | 0.311111 | 0 | 95 | 90 | 5 |
| train\_probe | tcr\_peptide/short | mask75 | 2.10322 | 0.345588 | 0 | 142 | 136 | 6 |
| train\_probe | tcr\_peptide/short | mask100 | 2.15467 | 0.302083 | 0 | 200 | 192 | 8 |
| train\_probe | tcr\_peptide/short | mask\_mean | 2.12516 | 0.326652 | 0 | — | — | — |
| train\_probe | tcr\_pmhc/long | mask25 | 1.69406 | 0.344828 | 0 | 526 | 522 | 4 |
| train\_probe | tcr\_pmhc/long | mask50 | 1.73884 | 0.326867 | 0 | 1036 | 1031 | 5 |
| train\_probe | tcr\_pmhc/long | mask75 | 1.75181 | 0.324952 | 0 | 1557 | 1551 | 6 |
| train\_probe | tcr\_pmhc/long | mask100 | 1.78172 | 0.291176 | 0 | 2048 | 2040 | 8 |
| train\_probe | tcr\_pmhc/long | mask\_mean | 1.74161 | 0.321956 | 0 | — | — | — |
| train\_probe | tcr\_pmhc/medium | mask25 | 1.67741 | 0.373626 | 0 | 276 | 273 | 3 |
| train\_probe | tcr\_pmhc/medium | mask50 | 1.72195 | 0.336414 | 0 | 549 | 541 | 8 |
| train\_probe | tcr\_pmhc/medium | mask75 | 1.69933 | 0.332484 | 0 | 793 | 785 | 8 |
| train\_probe | tcr\_pmhc/medium | mask100 | 1.75316 | 0.320866 | 0 | 1024 | 1016 | 8 |
| train\_probe | tcr\_pmhc/medium | mask\_mean | 1.71296 | 0.340848 | 0 | — | — | — |
| train\_probe | tcr\_pmhc/short | mask25 | 2.43993 | 0.229167 | 0 | 50 | 48 | 2 |
| train\_probe | tcr\_pmhc/short | mask50 | 2.45888 | 0.17 | 0 | 103 | 100 | 3 |
| train\_probe | tcr\_pmhc/short | mask75 | 2.51013 | 0.137931 | 0 | 152 | 145 | 7 |
| train\_probe | tcr\_pmhc/short | mask100 | 2.39688 | 0.135417 | 0 | 200 | 192 | 8 |
| train\_probe | tcr\_pmhc/short | mask\_mean | 2.45145 | 0.168129 | 0 | — | — | — |
| train\_probe | unconditional\_ab/long | mask25 | 1.79413 | 0.298419 | 0 | 509 | 506 | 3 |
| train\_probe | unconditional\_ab/long | mask50 | 1.78235 | 0.308911 | 0 | 1016 | 1010 | 6 |
| train\_probe | unconditional\_ab/long | mask75 | 1.77464 | 0.319026 | 0 | 1569 | 1561 | 8 |
| train\_probe | unconditional\_ab/long | mask100 | 2.1486 | 0.321078 | 0 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_ab/long | mask\_mean | 1.87493 | 0.311859 | 0 | — | — | — |
| train\_probe | unconditional\_ab/medium | mask25 | 1.96557 | 0.33195 | 0 | 243 | 241 | 2 |
| train\_probe | unconditional\_ab/medium | mask50 | 2.00952 | 0.315891 | 0 | 521 | 516 | 5 |
| train\_probe | unconditional\_ab/medium | mask75 | 1.98062 | 0.327201 | 0 | 766 | 761 | 5 |
| train\_probe | unconditional\_ab/medium | mask100 | 2.16718 | 0.329724 | 0 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_ab/medium | mask\_mean | 2.03072 | 0.326192 | 0 | — | — | — |
| train\_probe | unconditional\_ab/short | mask25 | 2.54934 | 0.153846 | 0 | 53 | 52 | 1 |
| train\_probe | unconditional\_ab/short | mask50 | 2.41957 | 0.192308 | 0 | 106 | 104 | 2 |
| train\_probe | unconditional\_ab/short | mask75 | 2.40619 | 0.21519 | 0 | 163 | 158 | 5 |
| train\_probe | unconditional\_ab/short | mask100 | 2.68994 | 0.197917 | 0 | 200 | 192 | 8 |
| train\_probe | unconditional\_ab/short | mask\_mean | 2.51626 | 0.189815 | 0 | — | — | — |
| train\_probe | unconditional\_single/long | mask25 | 1.94747 | 0.286885 | 0 | 245 | 244 | 1 |
| train\_probe | unconditional\_single/long | mask50 | 1.94856 | 0.262729 | 0 | 492 | 491 | 1 |
| train\_probe | unconditional\_single/long | mask75 | 1.9732 | 0.252604 | 0 | 771 | 768 | 3 |
| train\_probe | unconditional\_single/long | mask100 | 2.30853 | 0 | 0 | 1024 | 1020 | 4 |
| train\_probe | unconditional\_single/long | mask\_mean | 2.04444 | 0.200555 | 0 | — | — | — |
| train\_probe | unconditional\_single/medium | mask25 | 1.9718 | 0.130769 | 0 | 132 | 130 | 2 |
| train\_probe | unconditional\_single/medium | mask50 | 2.04517 | 0.130952 | 0 | 255 | 252 | 3 |
| train\_probe | unconditional\_single/medium | mask75 | 1.97922 | 0.234829 | 0 | 382 | 379 | 3 |
| train\_probe | unconditional\_single/medium | mask100 | 2.19407 | 0.0944882 | 0 | 512 | 508 | 4 |
| train\_probe | unconditional\_single/medium | mask\_mean | 2.04756 | 0.14776 | 0 | — | — | — |
| train\_probe | unconditional\_single/short | mask25 | 2.54231 | 0.192308 | 0 | 28 | 26 | 2 |
| train\_probe | unconditional\_single/short | mask50 | 2.44944 | 0.204082 | 0 | 52 | 49 | 3 |
| train\_probe | unconditional\_single/short | mask75 | 2.43991 | 0.171429 | 0 | 73 | 70 | 3 |
| train\_probe | unconditional\_single/short | mask100 | 2.49813 | 0 | 0 | 100 | 96 | 4 |
| train\_probe | unconditional\_single/short | mask\_mean | 2.48245 | 0.141954 | 0 | — | — | — |
| train\_probe | unconditional\_tcr/long | mask25 | 1.85062 | 0.329524 | — | 525 | 525 | 0 |
| train\_probe | unconditional\_tcr/long | mask50 | 1.86423 | 0.3311 | 0 | 1048 | 1045 | 3 |
| train\_probe | unconditional\_tcr/long | mask75 | 1.87507 | 0.307595 | 0 | 1585 | 1580 | 5 |
| train\_probe | unconditional\_tcr/long | mask100 | 2.32771 | 0.151961 | 0 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_tcr/long | mask\_mean | 1.97941 | 0.280045 | 0 | — | — | — |
| train\_probe | unconditional\_tcr/medium | mask25 | 1.71062 | 0.333333 | — | 261 | 261 | 0 |
| train\_probe | unconditional\_tcr/medium | mask50 | 1.70769 | 0.372233 | 0 | 498 | 497 | 1 |
| train\_probe | unconditional\_tcr/medium | mask75 | 1.70474 | 0.391478 | 0 | 755 | 751 | 4 |
| train\_probe | unconditional\_tcr/medium | mask100 | 1.93818 | 0.36811 | 0 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_tcr/medium | mask\_mean | 1.76531 | 0.366289 | 0 | — | — | — |
| train\_probe | unconditional\_tcr/short | mask25 | 2.15643 | 0.173913 | — | 46 | 46 | 0 |
| train\_probe | unconditional\_tcr/short | mask50 | 2.51043 | 0.141304 | 0 | 97 | 92 | 5 |
| train\_probe | unconditional\_tcr/short | mask75 | 2.49024 | 0.130137 | 0 | 152 | 146 | 6 |
| train\_probe | unconditional\_tcr/short | mask100 | 2.7334 | 0.114583 | 0 | 200 | 192 | 8 |
| train\_probe | unconditional\_tcr/short | mask\_mean | 2.47262 | 0.139984 | 0 | — | — | — |

### 1790802160343981887\_r21\_refined\_head\_gate\_80d10a6d：分层 mask 指标

源码目录：/vepfs-mlp2/c20250601/251105016/project/dllm\_test/output/immune\_research\_releases/r20\_r27\_20261001；完成/目标 step：25/25。

kind=synthetic\_generalization\_diagnostic；dataset_kind=deterministic\_artificial\_sequences；biological_quality_claim=False。

| 时点 | mask level | 指标 | 值 |
| --- | --- | --- | --- |
| initial | mask25 | mask\_ratio | 0.25 |
| initial | mask25 | selected\_count | 4467 |
| initial | mask25 | aa\_count | 4428 |
| initial | mask25 | terminal\_count | 39 |
| initial | mask25 | loss | 3.91472 |
| initial | mask25 | aa\_loss | 3.91377 |
| initial | mask25 | aa\_accuracy | 0.0682023 |
| initial | mask25 | terminal\_loss | 4.02257 |
| initial | mask25 | terminal\_accuracy | 0 |
| initial | mask50 | mask\_ratio | 0.5 |
| initial | mask50 | selected\_count | 8888 |
| initial | mask50 | aa\_count | 8819 |
| initial | mask50 | terminal\_count | 69 |
| initial | mask50 | loss | 4.07252 |
| initial | mask50 | aa\_loss | 4.0726 |
| initial | mask50 | aa\_accuracy | 0.0734777 |
| initial | mask50 | terminal\_loss | 4.06312 |
| initial | mask50 | terminal\_accuracy | 0 |
| initial | mask75 | mask\_ratio | 0.75 |
| initial | mask75 | selected\_count | 13437 |
| initial | mask75 | aa\_count | 13330 |
| initial | mask75 | terminal\_count | 107 |
| initial | mask75 | loss | 4.1958 |
| initial | mask75 | aa\_loss | 4.1954 |
| initial | mask75 | aa\_accuracy | 0.0578395 |
| initial | mask75 | terminal\_loss | 4.24564 |
| initial | mask75 | terminal\_accuracy | 0 |
| initial | mask100 | mask\_ratio | 1 |
| initial | mask100 | selected\_count | 17996 |
| initial | mask100 | aa\_count | 17864 |
| initial | mask100 | terminal\_count | 132 |
| initial | mask100 | loss | 4.1696 |
| initial | mask100 | aa\_loss | 4.1687 |
| initial | mask100 | aa\_accuracy | 0.0558665 |
| initial | mask100 | terminal\_loss | 4.29045 |
| initial | mask100 | terminal\_accuracy | 0 |
| initial | mask\_mean | loss | 4.08816 |
| initial | mask\_mean | aa\_loss | 4.08762 |
| initial | mask\_mean | terminal\_loss | 4.15545 |
| initial | mask\_mean | aa\_accuracy | 0.0638465 |
| initial | mask\_mean | terminal\_accuracy | 0 |
| latest | mask25 | mask\_ratio | 0.25 |
| latest | mask25 | selected\_count | 4467 |
| latest | mask25 | aa\_count | 4428 |
| latest | mask25 | terminal\_count | 39 |
| latest | mask25 | loss | 1.79264 |
| latest | mask25 | aa\_loss | 1.76509 |
| latest | mask25 | aa\_accuracy | 0.327236 |
| latest | mask25 | terminal\_loss | 4.92022 |
| latest | mask25 | terminal\_accuracy | 0 |
| latest | mask50 | mask\_ratio | 0.5 |
| latest | mask50 | selected\_count | 8888 |
| latest | mask50 | aa\_count | 8819 |
| latest | mask50 | terminal\_count | 69 |
| latest | mask50 | loss | 1.80289 |
| latest | mask50 | aa\_loss | 1.77939 |
| latest | mask50 | aa\_accuracy | 0.327361 |
| latest | mask50 | terminal\_loss | 4.80722 |
| latest | mask50 | terminal\_accuracy | 0 |
| latest | mask75 | mask\_ratio | 0.75 |
| latest | mask75 | selected\_count | 13437 |
| latest | mask75 | aa\_count | 13330 |
| latest | mask75 | terminal\_count | 107 |
| latest | mask75 | loss | 1.81571 |
| latest | mask75 | aa\_loss | 1.79244 |
| latest | mask75 | aa\_accuracy | 0.327457 |
| latest | mask75 | terminal\_loss | 4.7137 |
| latest | mask75 | terminal\_accuracy | 0 |
| latest | mask100 | mask\_ratio | 1 |
| latest | mask100 | selected\_count | 17996 |
| latest | mask100 | aa\_count | 17864 |
| latest | mask100 | terminal\_count | 132 |
| latest | mask100 | loss | 1.95769 |
| latest | mask100 | aa\_loss | 1.94188 |
| latest | mask100 | aa\_accuracy | 0.317174 |
| latest | mask100 | terminal\_loss | 4.09698 |
| latest | mask100 | terminal\_accuracy | 0 |
| latest | mask\_mean | loss | 1.84223 |
| latest | mask\_mean | aa\_loss | 1.8197 |
| latest | mask\_mean | terminal\_loss | 4.63453 |
| latest | mask\_mean | aa\_accuracy | 0.324807 |
| latest | mask\_mean | terminal\_accuracy | 0 |

评测曲线（原始 step，四档等权均值）：

| step | mean loss | mean AA acc | mean end acc | gradient norm |
| --- | --- | --- | --- | --- |
| 25 | 1.84223 | 0.324807 | 0 | 3.44503 |

有条件/无条件曲线（每档按token计数合并，再平均四档；缺失分组不推断数值）：

| step | 子集 | mean loss | mean AA | mean end | Full-mask AA | Full-mask end | Full-mask AA count | Full-mask end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | conditional | 4.18144 | 0.0639687 | 0 | 0.0632184 | 0 | 9744 | 72 |
| initial | unconditional | 3.97672 | 0.0636479 | 0 | 0.0470443 | 0 | 8120 | 60 |
| 25 | conditional | 1.78296 | 0.330896 | 0 | 0.319068 | 0 | 9744 | 72 |
| 25 | unconditional | 1.91318 | 0.317589 | 0 | 0.314901 | 0 | 8120 | 60 |

人工数据 manifest：d40bbbbc1c4294eed843d88bcdd0cf193886de6c2d7e9db9069bbf86e931c022；精确 target sequence 重叠=0；精确 record 重叠=0。

跨 radius32 覆盖：{&quot;possible&quot;: true, &quot;train&quot;: true, &quot;valid&quot;: true}。生成规则共享是诊断设计的一部分，不能当作真实下游去重证书。


按任务/链长分组（全局 level 按 token 计数汇总，下面保留各组计数；不同组不得直接比较未经加权的总 loss）：

| 时点 | group/length | mask level | loss | AA acc | end acc | selected count | AA count | end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | ab\_antigen/long | mask25 | 4.10407 | 0.0285171 | 0 | 530 | 526 | 4 |
| initial | ab\_antigen/long | mask50 | 4.40782 | 0.0306022 | 0 | 1019 | 1013 | 6 |
| initial | ab\_antigen/long | mask75 | 4.7473 | 0.0274151 | 0 | 1538 | 1532 | 6 |
| initial | ab\_antigen/long | mask100 | 4.66581 | 0.0303922 | 0 | 2048 | 2040 | 8 |
| initial | ab\_antigen/long | mask\_mean | 4.48125 | 0.0292316 | 0 | — | — | — |
| initial | ab\_antigen/medium | mask25 | 4.17571 | 0.144033 | 0 | 246 | 243 | 3 |
| initial | ab\_antigen/medium | mask50 | 4.27198 | 0.105578 | 0 | 507 | 502 | 5 |
| initial | ab\_antigen/medium | mask75 | 4.37624 | 0.0620053 | 0 | 765 | 758 | 7 |
| initial | ab\_antigen/medium | mask100 | 4.37027 | 0.0610236 | 0 | 1024 | 1016 | 8 |
| initial | ab\_antigen/medium | mask\_mean | 4.29855 | 0.0931599 | 0 | — | — | — |
| initial | ab\_antigen/short | mask25 | 4.83735 | 0.0588235 | 0 | 38 | 34 | 4 |
| initial | ab\_antigen/short | mask50 | 4.64596 | 0.0795455 | 0 | 93 | 88 | 5 |
| initial | ab\_antigen/short | mask75 | 4.54142 | 0.0785714 | 0 | 145 | 140 | 5 |
| initial | ab\_antigen/short | mask100 | 4.46303 | 0.09375 | 0 | 200 | 192 | 8 |
| initial | ab\_antigen/short | mask\_mean | 4.62194 | 0.0776726 | 0 | — | — | — |
| initial | tcr\_peptide/long | mask25 | 3.79134 | 0.083871 | 0 | 466 | 465 | 1 |
| initial | tcr\_peptide/long | mask50 | 3.99306 | 0.114169 | 0 | 985 | 981 | 4 |
| initial | tcr\_peptide/long | mask75 | 4.12436 | 0.118403 | 0 | 1484 | 1478 | 6 |
| initial | tcr\_peptide/long | mask100 | 4.06361 | 0.12451 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_peptide/long | mask\_mean | 3.99309 | 0.110238 | 0 | — | — | — |
| initial | tcr\_peptide/medium | mask25 | 4.1572 | 0.0580913 | 0 | 242 | 241 | 1 |
| initial | tcr\_peptide/medium | mask50 | 4.14345 | 0.0537849 | 0 | 506 | 502 | 4 |
| initial | tcr\_peptide/medium | mask75 | 4.06713 | 0.0578947 | 0 | 767 | 760 | 7 |
| initial | tcr\_peptide/medium | mask100 | 3.95053 | 0.0610236 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_peptide/medium | mask\_mean | 4.07958 | 0.0576986 | 0 | — | — | — |
| initial | tcr\_peptide/short | mask25 | 3.90693 | 0.104167 | 0 | 51 | 48 | 3 |
| initial | tcr\_peptide/short | mask50 | 3.85339 | 0.0865385 | 0 | 110 | 104 | 6 |
| initial | tcr\_peptide/short | mask75 | 3.88939 | 0.0875912 | 0 | 144 | 137 | 7 |
| initial | tcr\_peptide/short | mask100 | 3.88457 | 0.09375 | 0 | 200 | 192 | 8 |
| initial | tcr\_peptide/short | mask\_mean | 3.88357 | 0.0930116 | 0 | — | — | — |
| initial | tcr\_pmhc/long | mask25 | 3.54222 | 0.0391791 | 0 | 540 | 536 | 4 |
| initial | tcr\_pmhc/long | mask50 | 3.83329 | 0.0357834 | 0 | 1038 | 1034 | 4 |
| initial | tcr\_pmhc/long | mask75 | 4.11564 | 0.032175 | 0 | 1562 | 1554 | 8 |
| initial | tcr\_pmhc/long | mask100 | 4.29233 | 0.0313726 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_pmhc/long | mask\_mean | 3.94587 | 0.0346275 | 0 | — | — | — |
| initial | tcr\_pmhc/medium | mask25 | 4.27294 | 0.0909091 | 0 | 244 | 242 | 2 |
| initial | tcr\_pmhc/medium | mask50 | 4.34512 | 0.0793319 | 0 | 484 | 479 | 5 |
| initial | tcr\_pmhc/medium | mask75 | 4.43825 | 0.0702211 | 0 | 777 | 769 | 8 |
| initial | tcr\_pmhc/medium | mask100 | 4.45656 | 0.0629921 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_pmhc/medium | mask\_mean | 4.37822 | 0.0758636 | 0 | — | — | — |
| initial | tcr\_pmhc/short | mask25 | 4.41788 | 0.0416667 | 0 | 51 | 48 | 3 |
| initial | tcr\_pmhc/short | mask50 | 4.43505 | 0.0481928 | 0 | 86 | 83 | 3 |
| initial | tcr\_pmhc/short | mask75 | 4.25766 | 0.0680272 | 0 | 152 | 147 | 5 |
| initial | tcr\_pmhc/short | mask100 | 4.2994 | 0.0625 | 0 | 200 | 192 | 8 |
| initial | tcr\_pmhc/short | mask\_mean | 4.3525 | 0.0550967 | 0 | — | — | — |
| initial | unconditional\_ab/long | mask25 | 3.45445 | 0.157009 | 0 | 538 | 535 | 3 |
| initial | unconditional\_ab/long | mask50 | 3.68616 | 0.179732 | 0 | 1050 | 1046 | 4 |
| initial | unconditional\_ab/long | mask75 | 3.89091 | 0.0923277 | 0 | 1545 | 1538 | 7 |
| initial | unconditional\_ab/long | mask100 | 3.94998 | 0.0627451 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_ab/long | mask\_mean | 3.74538 | 0.122954 | 0 | — | — | — |
| initial | unconditional\_ab/medium | mask25 | 4.11098 | 0.055336 | 0 | 256 | 253 | 3 |
| initial | unconditional\_ab/medium | mask50 | 4.24527 | 0.139764 | 0 | 512 | 508 | 4 |
| initial | unconditional\_ab/medium | mask75 | 4.21312 | 0.149606 | 0 | 768 | 762 | 6 |
| initial | unconditional\_ab/medium | mask100 | 4.17067 | 0.15748 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_ab/medium | mask\_mean | 4.18501 | 0.125547 | 0 | — | — | — |
| initial | unconditional\_ab/short | mask25 | 4.10634 | 0.0961538 | 0 | 55 | 52 | 3 |
| initial | unconditional\_ab/short | mask50 | 4.061 | 0.0612245 | 0 | 103 | 98 | 5 |
| initial | unconditional\_ab/short | mask75 | 4.02084 | 0.0567376 | 0 | 148 | 141 | 7 |
| initial | unconditional\_ab/short | mask100 | 3.95762 | 0.0625 | 0 | 200 | 192 | 8 |
| initial | unconditional\_ab/short | mask\_mean | 4.03645 | 0.069154 | 0 | — | — | — |
| initial | unconditional\_single/long | mask25 | 3.41733 | 0.0820896 | 0 | 269 | 268 | 1 |
| initial | unconditional\_single/long | mask50 | 3.40878 | 0.0533333 | 0 | 527 | 525 | 2 |
| initial | unconditional\_single/long | mask75 | 3.53938 | 0.00516796 | 0 | 778 | 774 | 4 |
| initial | unconditional\_single/long | mask100 | 3.60678 | 0 | 0 | 1024 | 1020 | 4 |
| initial | unconditional\_single/long | mask\_mean | 3.49307 | 0.0351477 | 0 | — | — | — |
| initial | unconditional\_single/medium | mask25 | 3.85475 | 0 | 0 | 110 | 109 | 1 |
| initial | unconditional\_single/medium | mask50 | 3.72801 | 0 | 0 | 234 | 232 | 2 |
| initial | unconditional\_single/medium | mask75 | 3.72224 | 0 | 0 | 371 | 367 | 4 |
| initial | unconditional\_single/medium | mask100 | 3.73258 | 0 | 0 | 512 | 508 | 4 |
| initial | unconditional\_single/medium | mask\_mean | 3.75939 | 0 | 0 | — | — | — |
| initial | unconditional\_single/short | mask25 | 3.47104 | 0 | 0 | 28 | 26 | 2 |
| initial | unconditional\_single/short | mask50 | 3.6937 | 0 | 0 | 52 | 50 | 2 |
| initial | unconditional\_single/short | mask75 | 3.49417 | 0 | 0 | 80 | 77 | 3 |
| initial | unconditional\_single/short | mask100 | 3.44943 | 0 | 0 | 100 | 96 | 4 |
| initial | unconditional\_single/short | mask\_mean | 3.52708 | 0 | 0 | — | — | — |
| initial | unconditional\_tcr/long | mask25 | 4.12367 | 0.0368932 | 0 | 516 | 515 | 1 |
| initial | unconditional\_tcr/long | mask50 | 4.32329 | 0.0303951 | 0 | 989 | 987 | 2 |
| initial | unconditional\_tcr/long | mask75 | 4.35912 | 0.0302234 | 0 | 1527 | 1522 | 5 |
| initial | unconditional\_tcr/long | mask100 | 4.13527 | 0.0313726 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_tcr/long | mask\_mean | 4.23534 | 0.0322211 | 0 | — | — | — |
| initial | unconditional\_tcr/medium | mask25 | 4.32901 | 0 | — | 247 | 247 | 0 |
| initial | unconditional\_tcr/medium | mask50 | 4.40907 | 0 | 0 | 500 | 497 | 3 |
| initial | unconditional\_tcr/medium | mask75 | 4.36242 | 0 | 0 | 744 | 738 | 6 |
| initial | unconditional\_tcr/medium | mask100 | 4.26923 | 0 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_tcr/medium | mask\_mean | 4.34243 | 0 | 0 | — | — | — |
| initial | unconditional\_tcr/short | mask25 | 4.34405 | 0.075 | — | 40 | 40 | 0 |
| initial | unconditional\_tcr/short | mask50 | 4.15658 | 0.0777778 | 0 | 93 | 90 | 3 |
| initial | unconditional\_tcr/short | mask75 | 4.14406 | 0.0882353 | 0 | 142 | 136 | 6 |
| initial | unconditional\_tcr/short | mask100 | 4.06085 | 0.09375 | 0 | 200 | 192 | 8 |
| initial | unconditional\_tcr/short | mask\_mean | 4.17639 | 0.0836908 | 0 | — | — | — |
| latest | ab\_antigen/long | mask25 | 1.80503 | 0.336502 | 0 | 530 | 526 | 4 |
| latest | ab\_antigen/long | mask50 | 1.78696 | 0.351431 | 0 | 1019 | 1013 | 6 |
| latest | ab\_antigen/long | mask75 | 1.78903 | 0.3453 | 0 | 1538 | 1532 | 6 |
| latest | ab\_antigen/long | mask100 | 1.84438 | 0.313726 | 0 | 2048 | 2040 | 8 |
| latest | ab\_antigen/long | mask\_mean | 1.80635 | 0.33674 | 0 | — | — | — |
| latest | ab\_antigen/medium | mask25 | 1.83467 | 0.337449 | 0 | 246 | 243 | 3 |
| latest | ab\_antigen/medium | mask50 | 1.88438 | 0.302789 | 0 | 507 | 502 | 5 |
| latest | ab\_antigen/medium | mask75 | 1.86263 | 0.28628 | 0 | 765 | 758 | 7 |
| latest | ab\_antigen/medium | mask100 | 1.87681 | 0.255906 | 0 | 1024 | 1016 | 8 |
| latest | ab\_antigen/medium | mask\_mean | 1.86462 | 0.295606 | 0 | — | — | — |
| latest | ab\_antigen/short | mask25 | 2.26947 | 0.294118 | 0 | 38 | 34 | 4 |
| latest | ab\_antigen/short | mask50 | 2.20542 | 0.25 | 0 | 93 | 88 | 5 |
| latest | ab\_antigen/short | mask75 | 2.15364 | 0.214286 | 0 | 145 | 140 | 5 |
| latest | ab\_antigen/short | mask100 | 2.15694 | 0.25 | 0 | 200 | 192 | 8 |
| latest | ab\_antigen/short | mask\_mean | 2.19637 | 0.252101 | 0 | — | — | — |
| latest | tcr\_peptide/long | mask25 | 1.89328 | 0.249462 | 0 | 466 | 465 | 1 |
| latest | tcr\_peptide/long | mask50 | 1.89359 | 0.278287 | 0 | 985 | 981 | 4 |
| latest | tcr\_peptide/long | mask75 | 1.8832 | 0.275372 | 0 | 1484 | 1478 | 6 |
| latest | tcr\_peptide/long | mask100 | 1.88627 | 0.263726 | 0 | 2048 | 2040 | 8 |
| latest | tcr\_peptide/long | mask\_mean | 1.88908 | 0.266712 | 0 | — | — | — |
| latest | tcr\_peptide/medium | mask25 | 1.91021 | 0.290456 | 0 | 242 | 241 | 1 |
| latest | tcr\_peptide/medium | mask50 | 1.92697 | 0.300797 | 0 | 506 | 502 | 4 |
| latest | tcr\_peptide/medium | mask75 | 1.94752 | 0.285526 | 0 | 767 | 760 | 7 |
| latest | tcr\_peptide/medium | mask100 | 1.93972 | 0.277559 | 0 | 1024 | 1016 | 8 |
| latest | tcr\_peptide/medium | mask\_mean | 1.9311 | 0.288585 | 0 | — | — | — |
| latest | tcr\_peptide/short | mask25 | 2.29162 | 0.25 | 0 | 51 | 48 | 3 |
| latest | tcr\_peptide/short | mask50 | 2.29256 | 0.230769 | 0 | 110 | 104 | 6 |
| latest | tcr\_peptide/short | mask75 | 2.19988 | 0.255474 | 0 | 144 | 137 | 7 |
| latest | tcr\_peptide/short | mask100 | 2.17706 | 0.260417 | 0 | 200 | 192 | 8 |
| latest | tcr\_peptide/short | mask\_mean | 2.24028 | 0.249165 | 0 | — | — | — |
| latest | tcr\_pmhc/long | mask25 | 1.58602 | 0.38806 | 0 | 540 | 536 | 4 |
| latest | tcr\_pmhc/long | mask50 | 1.54685 | 0.389749 | 0 | 1038 | 1034 | 4 |
| latest | tcr\_pmhc/long | mask75 | 1.54518 | 0.407979 | 0 | 1562 | 1554 | 8 |
| latest | tcr\_pmhc/long | mask100 | 1.56539 | 0.404412 | 0 | 2048 | 2040 | 8 |
| latest | tcr\_pmhc/long | mask\_mean | 1.56086 | 0.39755 | 0 | — | — | — |
| latest | tcr\_pmhc/medium | mask25 | 1.50062 | 0.442149 | 0 | 244 | 242 | 2 |
| latest | tcr\_pmhc/medium | mask50 | 1.5357 | 0.407098 | 0 | 484 | 479 | 5 |
| latest | tcr\_pmhc/medium | mask75 | 1.56955 | 0.414824 | 0 | 777 | 769 | 8 |
| latest | tcr\_pmhc/medium | mask100 | 1.62957 | 0.398622 | 0 | 1024 | 1016 | 8 |
| latest | tcr\_pmhc/medium | mask\_mean | 1.55886 | 0.415673 | 0 | — | — | — |
| latest | tcr\_pmhc/short | mask25 | 1.87502 | 0.333333 | 0 | 51 | 48 | 3 |
| latest | tcr\_pmhc/short | mask50 | 1.91819 | 0.325301 | 0 | 86 | 83 | 3 |
| latest | tcr\_pmhc/short | mask75 | 1.92458 | 0.319728 | 0 | 152 | 147 | 5 |
| latest | tcr\_pmhc/short | mask100 | 2.04221 | 0.317708 | 0 | 200 | 192 | 8 |
| latest | tcr\_pmhc/short | mask\_mean | 1.94 | 0.324018 | 0 | — | — | — |
| latest | unconditional\_ab/long | mask25 | 1.883 | 0.306542 | 0 | 538 | 535 | 3 |
| latest | unconditional\_ab/long | mask50 | 1.87723 | 0.315488 | 0 | 1050 | 1046 | 4 |
| latest | unconditional\_ab/long | mask75 | 1.88626 | 0.323797 | 0 | 1545 | 1538 | 7 |
| latest | unconditional\_ab/long | mask100 | 2.1076 | 0.336765 | 0 | 2048 | 2040 | 8 |
| latest | unconditional\_ab/long | mask\_mean | 1.93853 | 0.320648 | 0 | — | — | — |
| latest | unconditional\_ab/medium | mask25 | 1.88533 | 0.312253 | 0 | 256 | 253 | 3 |
| latest | unconditional\_ab/medium | mask50 | 1.94576 | 0.318898 | 0 | 512 | 508 | 4 |
| latest | unconditional\_ab/medium | mask75 | 1.94925 | 0.305774 | 0 | 768 | 762 | 6 |
| latest | unconditional\_ab/medium | mask100 | 2.19174 | 0.298228 | 0 | 1024 | 1016 | 8 |
| latest | unconditional\_ab/medium | mask\_mean | 1.99302 | 0.308788 | 0 | — | — | — |
| latest | unconditional\_ab/short | mask25 | 2.01462 | 0.269231 | 0 | 55 | 52 | 3 |
| latest | unconditional\_ab/short | mask50 | 2.2136 | 0.234694 | 0 | 103 | 98 | 5 |
| latest | unconditional\_ab/short | mask75 | 2.32282 | 0.248227 | 0 | 148 | 141 | 7 |
| latest | unconditional\_ab/short | mask100 | 2.48465 | 0.229167 | 0 | 200 | 192 | 8 |
| latest | unconditional\_ab/short | mask\_mean | 2.25893 | 0.24533 | 0 | — | — | — |
| latest | unconditional\_single/long | mask25 | 1.70182 | 0.291045 | 0 | 269 | 268 | 1 |
| latest | unconditional\_single/long | mask50 | 1.69461 | 0.264762 | 0 | 527 | 525 | 2 |
| latest | unconditional\_single/long | mask75 | 1.73009 | 0.286822 | 0 | 778 | 774 | 4 |
| latest | unconditional\_single/long | mask100 | 2.08233 | 0.305882 | 0 | 1024 | 1020 | 4 |
| latest | unconditional\_single/long | mask\_mean | 1.80221 | 0.287128 | 0 | — | — | — |
| latest | unconditional\_single/medium | mask25 | 1.77246 | 0.33945 | 0 | 110 | 109 | 1 |
| latest | unconditional\_single/medium | mask50 | 1.72533 | 0.37069 | 0 | 234 | 232 | 2 |
| latest | unconditional\_single/medium | mask75 | 1.81041 | 0.359673 | 0 | 371 | 367 | 4 |
| latest | unconditional\_single/medium | mask100 | 2.14827 | 0.330709 | 0 | 512 | 508 | 4 |
| latest | unconditional\_single/medium | mask\_mean | 1.86412 | 0.35013 | 0 | — | — | — |
| latest | unconditional\_single/short | mask25 | 2.68254 | 0.269231 | 0 | 28 | 26 | 2 |
| latest | unconditional\_single/short | mask50 | 2.64146 | 0.2 | 0 | 52 | 50 | 2 |
| latest | unconditional\_single/short | mask75 | 2.37879 | 0.246753 | 0 | 80 | 77 | 3 |
| latest | unconditional\_single/short | mask100 | 2.32001 | 0.291667 | 0 | 100 | 96 | 4 |
| latest | unconditional\_single/short | mask\_mean | 2.5057 | 0.251913 | 0 | — | — | — |
| latest | unconditional\_tcr/long | mask25 | 1.77651 | 0.332039 | 0 | 516 | 515 | 1 |
| latest | unconditional\_tcr/long | mask50 | 1.80461 | 0.33232 | 0 | 989 | 987 | 2 |
| latest | unconditional\_tcr/long | mask75 | 1.85951 | 0.325887 | 0 | 1527 | 1522 | 5 |
| latest | unconditional\_tcr/long | mask100 | 2.1811 | 0.305392 | 0 | 2048 | 2040 | 8 |
| latest | unconditional\_tcr/long | mask\_mean | 1.90543 | 0.32391 | 0 | — | — | — |
| latest | unconditional\_tcr/medium | mask25 | 1.59124 | 0.356275 | — | 247 | 247 | 0 |
| latest | unconditional\_tcr/medium | mask50 | 1.63943 | 0.360161 | 0 | 500 | 497 | 3 |
| latest | unconditional\_tcr/medium | mask75 | 1.68285 | 0.355014 | 0 | 744 | 738 | 6 |
| latest | unconditional\_tcr/medium | mask100 | 2.03111 | 0.345472 | 0 | 1024 | 1016 | 8 |
| latest | unconditional\_tcr/medium | mask\_mean | 1.73615 | 0.354231 | 0 | — | — | — |
| latest | unconditional\_tcr/short | mask25 | 2.2441 | 0.325 | — | 40 | 40 | 0 |
| latest | unconditional\_tcr/short | mask50 | 2.1771 | 0.3 | 0 | 93 | 90 | 3 |
| latest | unconditional\_tcr/short | mask75 | 2.27837 | 0.242647 | 0 | 142 | 136 | 6 |
| latest | unconditional\_tcr/short | mask100 | 2.31783 | 0.213542 | 0 | 200 | 192 | 8 |
| latest | unconditional\_tcr/short | mask\_mean | 2.25435 | 0.270297 | 0 | — | — | — |
| train\_probe | ab\_antigen/long | mask25 | 1.61558 | 0.403774 | — | 530 | 530 | 0 |
| train\_probe | ab\_antigen/long | mask50 | 1.61443 | 0.390883 | 0 | 1034 | 1031 | 3 |
| train\_probe | ab\_antigen/long | mask75 | 1.62556 | 0.376365 | 0 | 1563 | 1557 | 6 |
| train\_probe | ab\_antigen/long | mask100 | 1.72425 | 0.34902 | 0 | 2048 | 2040 | 8 |
| train\_probe | ab\_antigen/long | mask\_mean | 1.64496 | 0.38001 | 0 | — | — | — |
| train\_probe | ab\_antigen/medium | mask25 | 1.95172 | 0.296703 | 0 | 275 | 273 | 2 |
| train\_probe | ab\_antigen/medium | mask50 | 1.93768 | 0.263941 | 0 | 542 | 538 | 4 |
| train\_probe | ab\_antigen/medium | mask75 | 1.91263 | 0.282216 | 0 | 781 | 776 | 5 |
| train\_probe | ab\_antigen/medium | mask100 | 1.95695 | 0.307087 | 0 | 1024 | 1016 | 8 |
| train\_probe | ab\_antigen/medium | mask\_mean | 1.93974 | 0.287487 | 0 | — | — | — |
| train\_probe | ab\_antigen/short | mask25 | 2.0583 | 0.319149 | 0 | 52 | 47 | 5 |
| train\_probe | ab\_antigen/short | mask50 | 2.09067 | 0.313726 | 0 | 109 | 102 | 7 |
| train\_probe | ab\_antigen/short | mask75 | 2.09361 | 0.337838 | 0 | 155 | 148 | 7 |
| train\_probe | ab\_antigen/short | mask100 | 2.15529 | 0.317708 | 0 | 200 | 192 | 8 |
| train\_probe | ab\_antigen/short | mask\_mean | 2.09947 | 0.322105 | 0 | — | — | — |
| train\_probe | tcr\_peptide/long | mask25 | 1.63872 | 0.336032 | 0 | 495 | 494 | 1 |
| train\_probe | tcr\_peptide/long | mask50 | 1.6333 | 0.332685 | 0 | 1033 | 1028 | 5 |
| train\_probe | tcr\_peptide/long | mask75 | 1.68368 | 0.324573 | 0 | 1530 | 1522 | 8 |
| train\_probe | tcr\_peptide/long | mask100 | 1.64938 | 0.338726 | 0 | 2048 | 2040 | 8 |
| train\_probe | tcr\_peptide/long | mask\_mean | 1.65127 | 0.333004 | 0 | — | — | — |
| train\_probe | tcr\_peptide/medium | mask25 | 1.67998 | 0.321429 | — | 252 | 252 | 0 |
| train\_probe | tcr\_peptide/medium | mask50 | 1.71125 | 0.346979 | 0 | 517 | 513 | 4 |
| train\_probe | tcr\_peptide/medium | mask75 | 1.71183 | 0.33687 | 0 | 760 | 754 | 6 |
| train\_probe | tcr\_peptide/medium | mask100 | 1.76747 | 0.332677 | 0 | 1024 | 1016 | 8 |
| train\_probe | tcr\_peptide/medium | mask\_mean | 1.71763 | 0.334489 | 0 | — | — | — |
| train\_probe | tcr\_peptide/short | mask25 | 2.09263 | 0.26087 | 0 | 51 | 46 | 5 |
| train\_probe | tcr\_peptide/short | mask50 | 2.00938 | 0.266667 | 0 | 95 | 90 | 5 |
| train\_probe | tcr\_peptide/short | mask75 | 2.05165 | 0.294118 | 0 | 142 | 136 | 6 |
| train\_probe | tcr\_peptide/short | mask100 | 2.04103 | 0.354167 | 0 | 200 | 192 | 8 |
| train\_probe | tcr\_peptide/short | mask\_mean | 2.04868 | 0.293955 | 0 | — | — | — |
| train\_probe | tcr\_pmhc/long | mask25 | 1.63844 | 0.335249 | 0 | 526 | 522 | 4 |
| train\_probe | tcr\_pmhc/long | mask50 | 1.66877 | 0.313288 | 0 | 1036 | 1031 | 5 |
| train\_probe | tcr\_pmhc/long | mask75 | 1.6721 | 0.319794 | 0 | 1557 | 1551 | 6 |
| train\_probe | tcr\_pmhc/long | mask100 | 1.69926 | 0.325 | 0 | 2048 | 2040 | 8 |
| train\_probe | tcr\_pmhc/long | mask\_mean | 1.66964 | 0.323333 | 0 | — | — | — |
| train\_probe | tcr\_pmhc/medium | mask25 | 1.5787 | 0.347985 | 0 | 276 | 273 | 3 |
| train\_probe | tcr\_pmhc/medium | mask50 | 1.63234 | 0.332717 | 0 | 549 | 541 | 8 |
| train\_probe | tcr\_pmhc/medium | mask75 | 1.62525 | 0.332484 | 0 | 793 | 785 | 8 |
| train\_probe | tcr\_pmhc/medium | mask100 | 1.63535 | 0.34252 | 0 | 1024 | 1016 | 8 |
| train\_probe | tcr\_pmhc/medium | mask\_mean | 1.61791 | 0.338927 | 0 | — | — | — |
| train\_probe | tcr\_pmhc/short | mask25 | 2.34885 | 0.3125 | 0 | 50 | 48 | 2 |
| train\_probe | tcr\_pmhc/short | mask50 | 2.27749 | 0.23 | 0 | 103 | 100 | 3 |
| train\_probe | tcr\_pmhc/short | mask75 | 2.30187 | 0.193103 | 0 | 152 | 145 | 7 |
| train\_probe | tcr\_pmhc/short | mask100 | 2.22362 | 0.223958 | 0 | 200 | 192 | 8 |
| train\_probe | tcr\_pmhc/short | mask\_mean | 2.28796 | 0.23989 | 0 | — | — | — |
| train\_probe | unconditional\_ab/long | mask25 | 1.75849 | 0.237154 | 0 | 509 | 506 | 3 |
| train\_probe | unconditional\_ab/long | mask50 | 1.73623 | 0.241584 | 0 | 1016 | 1010 | 6 |
| train\_probe | unconditional\_ab/long | mask75 | 1.71146 | 0.276105 | 0 | 1569 | 1561 | 8 |
| train\_probe | unconditional\_ab/long | mask100 | 2.0038 | 0.321078 | 0 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_ab/long | mask\_mean | 1.80249 | 0.26898 | 0 | — | — | — |
| train\_probe | unconditional\_ab/medium | mask25 | 1.85958 | 0.273859 | 0 | 243 | 241 | 2 |
| train\_probe | unconditional\_ab/medium | mask50 | 1.88127 | 0.25969 | 0 | 521 | 516 | 5 |
| train\_probe | unconditional\_ab/medium | mask75 | 1.87772 | 0.278581 | 0 | 766 | 761 | 5 |
| train\_probe | unconditional\_ab/medium | mask100 | 2.07098 | 0.329724 | 0 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_ab/medium | mask\_mean | 1.92239 | 0.285464 | 0 | — | — | — |
| train\_probe | unconditional\_ab/short | mask25 | 2.23573 | 0.230769 | 0 | 53 | 52 | 1 |
| train\_probe | unconditional\_ab/short | mask50 | 2.15022 | 0.230769 | 0 | 106 | 104 | 2 |
| train\_probe | unconditional\_ab/short | mask75 | 2.21099 | 0.253165 | 0 | 163 | 158 | 5 |
| train\_probe | unconditional\_ab/short | mask100 | 2.43012 | 0.197917 | 0 | 200 | 192 | 8 |
| train\_probe | unconditional\_ab/short | mask\_mean | 2.25677 | 0.228155 | 0 | — | — | — |
| train\_probe | unconditional\_single/long | mask25 | 1.78851 | 0.278689 | 0 | 245 | 244 | 1 |
| train\_probe | unconditional\_single/long | mask50 | 1.75862 | 0.283096 | 0 | 492 | 491 | 1 |
| train\_probe | unconditional\_single/long | mask75 | 1.78258 | 0.279948 | 0 | 771 | 768 | 3 |
| train\_probe | unconditional\_single/long | mask100 | 2.13983 | 0.243137 | 0 | 1024 | 1020 | 4 |
| train\_probe | unconditional\_single/long | mask\_mean | 1.86738 | 0.271217 | 0 | — | — | — |
| train\_probe | unconditional\_single/medium | mask25 | 1.85395 | 0.338462 | 0 | 132 | 130 | 2 |
| train\_probe | unconditional\_single/medium | mask50 | 1.90829 | 0.321429 | 0 | 255 | 252 | 3 |
| train\_probe | unconditional\_single/medium | mask75 | 1.84946 | 0.332454 | 0 | 382 | 379 | 3 |
| train\_probe | unconditional\_single/medium | mask100 | 2.00337 | 0.362205 | 0 | 512 | 508 | 4 |
| train\_probe | unconditional\_single/medium | mask\_mean | 1.90377 | 0.338637 | 0 | — | — | — |
| train\_probe | unconditional\_single/short | mask25 | 2.641 | 0.0769231 | 0 | 28 | 26 | 2 |
| train\_probe | unconditional\_single/short | mask50 | 2.50846 | 0.163265 | 0 | 52 | 49 | 3 |
| train\_probe | unconditional\_single/short | mask75 | 2.34098 | 0.214286 | 0 | 73 | 70 | 3 |
| train\_probe | unconditional\_single/short | mask100 | 2.36782 | 0.229167 | 0 | 100 | 96 | 4 |
| train\_probe | unconditional\_single/short | mask\_mean | 2.46456 | 0.17091 | 0 | — | — | — |
| train\_probe | unconditional\_tcr/long | mask25 | 1.75974 | 0.32381 | — | 525 | 525 | 0 |
| train\_probe | unconditional\_tcr/long | mask50 | 1.75606 | 0.326316 | 0 | 1048 | 1045 | 3 |
| train\_probe | unconditional\_tcr/long | mask75 | 1.76949 | 0.344304 | 0 | 1585 | 1580 | 5 |
| train\_probe | unconditional\_tcr/long | mask100 | 2.11358 | 0.27402 | 0 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_tcr/long | mask\_mean | 1.84972 | 0.317112 | 0 | — | — | — |
| train\_probe | unconditional\_tcr/medium | mask25 | 1.70953 | 0.318008 | — | 261 | 261 | 0 |
| train\_probe | unconditional\_tcr/medium | mask50 | 1.70316 | 0.331992 | 0 | 498 | 497 | 1 |
| train\_probe | unconditional\_tcr/medium | mask75 | 1.76289 | 0.335553 | 0 | 755 | 751 | 4 |
| train\_probe | unconditional\_tcr/medium | mask100 | 1.93595 | 0.376969 | 0 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_tcr/medium | mask\_mean | 1.77788 | 0.34063 | 0 | — | — | — |
| train\_probe | unconditional\_tcr/short | mask25 | 1.93939 | 0.26087 | — | 46 | 46 | 0 |
| train\_probe | unconditional\_tcr/short | mask50 | 2.20461 | 0.271739 | 0 | 97 | 92 | 5 |
| train\_probe | unconditional\_tcr/short | mask75 | 2.19763 | 0.267123 | 0 | 152 | 146 | 6 |
| train\_probe | unconditional\_tcr/short | mask100 | 2.4022 | 0.182292 | 0 | 200 | 192 | 8 |
| train\_probe | unconditional\_tcr/short | mask\_mean | 2.18596 | 0.245506 | 0 | — | — | — |

### 1790802160346881056\_r22\_refined\_fixed\_gate\_263516a3：分层 mask 指标

源码目录：/vepfs-mlp2/c20250601/251105016/project/dllm\_test/output/immune\_research\_releases/r20\_r27\_20261001；完成/目标 step：25/25。

kind=synthetic\_generalization\_diagnostic；dataset_kind=deterministic\_artificial\_sequences；biological_quality_claim=False。

| 时点 | mask level | 指标 | 值 |
| --- | --- | --- | --- |
| initial | mask25 | mask\_ratio | 0.25 |
| initial | mask25 | selected\_count | 4467 |
| initial | mask25 | aa\_count | 4428 |
| initial | mask25 | terminal\_count | 39 |
| initial | mask25 | loss | 3.94671 |
| initial | mask25 | aa\_loss | 3.9454 |
| initial | mask25 | aa\_accuracy | 0.0546522 |
| initial | mask25 | terminal\_loss | 4.09557 |
| initial | mask25 | terminal\_accuracy | 0 |
| initial | mask50 | mask\_ratio | 0.5 |
| initial | mask50 | selected\_count | 8888 |
| initial | mask50 | aa\_count | 8819 |
| initial | mask50 | terminal\_count | 69 |
| initial | mask50 | loss | 4.10037 |
| initial | mask50 | aa\_loss | 4.10079 |
| initial | mask50 | aa\_accuracy | 0.0520467 |
| initial | mask50 | terminal\_loss | 4.04707 |
| initial | mask50 | terminal\_accuracy | 0 |
| initial | mask75 | mask\_ratio | 0.75 |
| initial | mask75 | selected\_count | 13437 |
| initial | mask75 | aa\_count | 13330 |
| initial | mask75 | terminal\_count | 107 |
| initial | mask75 | loss | 4.20503 |
| initial | mask75 | aa\_loss | 4.20545 |
| initial | mask75 | aa\_accuracy | 0.0533383 |
| initial | mask75 | terminal\_loss | 4.15238 |
| initial | mask75 | terminal\_accuracy | 0 |
| initial | mask100 | mask\_ratio | 1 |
| initial | mask100 | selected\_count | 17996 |
| initial | mask100 | aa\_count | 17864 |
| initial | mask100 | terminal\_count | 132 |
| initial | mask100 | loss | 4.1714 |
| initial | mask100 | aa\_loss | 4.17087 |
| initial | mask100 | aa\_accuracy | 0.0558665 |
| initial | mask100 | terminal\_loss | 4.24331 |
| initial | mask100 | terminal\_accuracy | 0 |
| initial | mask\_mean | loss | 4.10588 |
| initial | mask\_mean | aa\_loss | 4.10563 |
| initial | mask\_mean | terminal\_loss | 4.13458 |
| initial | mask\_mean | aa\_accuracy | 0.053976 |
| initial | mask\_mean | terminal\_accuracy | 0 |
| latest | mask25 | mask\_ratio | 0.25 |
| latest | mask25 | selected\_count | 4467 |
| latest | mask25 | aa\_count | 4428 |
| latest | mask25 | terminal\_count | 39 |
| latest | mask25 | loss | 1.79917 |
| latest | mask25 | aa\_loss | 1.77074 |
| latest | mask25 | aa\_accuracy | 0.328365 |
| latest | mask25 | terminal\_loss | 5.02705 |
| latest | mask25 | terminal\_accuracy | 0 |
| latest | mask50 | mask\_ratio | 0.5 |
| latest | mask50 | selected\_count | 8888 |
| latest | mask50 | aa\_count | 8819 |
| latest | mask50 | terminal\_count | 69 |
| latest | mask50 | loss | 1.80596 |
| latest | mask50 | aa\_loss | 1.78172 |
| latest | mask50 | aa\_accuracy | 0.327928 |
| latest | mask50 | terminal\_loss | 4.90437 |
| latest | mask50 | terminal\_accuracy | 0 |
| latest | mask75 | mask\_ratio | 0.75 |
| latest | mask75 | selected\_count | 13437 |
| latest | mask75 | aa\_count | 13330 |
| latest | mask75 | terminal\_count | 107 |
| latest | mask75 | loss | 1.81648 |
| latest | mask75 | aa\_loss | 1.79281 |
| latest | mask75 | aa\_accuracy | 0.330608 |
| latest | mask75 | terminal\_loss | 4.76544 |
| latest | mask75 | terminal\_accuracy | 0 |
| latest | mask100 | mask\_ratio | 1 |
| latest | mask100 | selected\_count | 17996 |
| latest | mask100 | aa\_count | 17864 |
| latest | mask100 | terminal\_count | 132 |
| latest | mask100 | loss | 1.98071 |
| latest | mask100 | aa\_loss | 1.96411 |
| latest | mask100 | aa\_accuracy | 0.305587 |
| latest | mask100 | terminal\_loss | 4.22622 |
| latest | mask100 | terminal\_accuracy | 0 |
| latest | mask\_mean | loss | 1.85058 |
| latest | mask\_mean | aa\_loss | 1.82735 |
| latest | mask\_mean | terminal\_loss | 4.73077 |
| latest | mask\_mean | aa\_accuracy | 0.323122 |
| latest | mask\_mean | terminal\_accuracy | 0 |

评测曲线（原始 step，四档等权均值）：

| step | mean loss | mean AA acc | mean end acc | gradient norm |
| --- | --- | --- | --- | --- |
| 25 | 1.85058 | 0.323122 | 0 | 4.19729 |

有条件/无条件曲线（每档按token计数合并，再平均四档；缺失分组不推断数值）：

| step | 子集 | mean loss | mean AA | mean end | Full-mask AA | Full-mask end | Full-mask AA count | Full-mask end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | conditional | 4.17776 | 0.0591387 | 0 | 0.0632184 | 0 | 9744 | 72 |
| initial | unconditional | 4.02 | 0.0478017 | 0 | 0.0470443 | 0 | 8120 | 60 |
| 25 | conditional | 1.79185 | 0.332756 | 0 | 0.321736 | 0 | 9744 | 72 |
| 25 | unconditional | 1.92091 | 0.311614 | 0 | 0.286207 | 0 | 8120 | 60 |

人工数据 manifest：d40bbbbc1c4294eed843d88bcdd0cf193886de6c2d7e9db9069bbf86e931c022；精确 target sequence 重叠=0；精确 record 重叠=0。

跨 radius32 覆盖：{&quot;possible&quot;: true, &quot;train&quot;: true, &quot;valid&quot;: true}。生成规则共享是诊断设计的一部分，不能当作真实下游去重证书。


按任务/链长分组（全局 level 按 token 计数汇总，下面保留各组计数；不同组不得直接比较未经加权的总 loss）：

| 时点 | group/length | mask level | loss | AA acc | end acc | selected count | AA count | end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | ab\_antigen/long | mask25 | 4.02088 | 0.0095057 | 0 | 530 | 526 | 4 |
| initial | ab\_antigen/long | mask50 | 4.33442 | 0.011846 | 0 | 1019 | 1013 | 6 |
| initial | ab\_antigen/long | mask75 | 4.57156 | 0.0221932 | 0 | 1538 | 1532 | 6 |
| initial | ab\_antigen/long | mask100 | 4.47569 | 0.0303922 | 0 | 2048 | 2040 | 8 |
| initial | ab\_antigen/long | mask\_mean | 4.35064 | 0.0184843 | 0 | — | — | — |
| initial | ab\_antigen/medium | mask25 | 4.39362 | 0.0452675 | 0 | 246 | 243 | 3 |
| initial | ab\_antigen/medium | mask50 | 4.39276 | 0.0657371 | 0 | 507 | 502 | 5 |
| initial | ab\_antigen/medium | mask75 | 4.38496 | 0.0620053 | 0 | 765 | 758 | 7 |
| initial | ab\_antigen/medium | mask100 | 4.3142 | 0.0610236 | 0 | 1024 | 1016 | 8 |
| initial | ab\_antigen/medium | mask\_mean | 4.37138 | 0.0585084 | 0 | — | — | — |
| initial | ab\_antigen/short | mask25 | 4.68396 | 0.0588235 | 0 | 38 | 34 | 4 |
| initial | ab\_antigen/short | mask50 | 4.45858 | 0.0795455 | 0 | 93 | 88 | 5 |
| initial | ab\_antigen/short | mask75 | 4.38017 | 0.0785714 | 0 | 145 | 140 | 5 |
| initial | ab\_antigen/short | mask100 | 4.30045 | 0.09375 | 0 | 200 | 192 | 8 |
| initial | ab\_antigen/short | mask\_mean | 4.45579 | 0.0776726 | 0 | — | — | — |
| initial | tcr\_peptide/long | mask25 | 3.7745 | 0.105376 | 0 | 466 | 465 | 1 |
| initial | tcr\_peptide/long | mask50 | 3.9799 | 0.114169 | 0 | 985 | 981 | 4 |
| initial | tcr\_peptide/long | mask75 | 4.09007 | 0.118403 | 0 | 1484 | 1478 | 6 |
| initial | tcr\_peptide/long | mask100 | 4.07857 | 0.12451 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_peptide/long | mask\_mean | 3.98076 | 0.115615 | 0 | — | — | — |
| initial | tcr\_peptide/medium | mask25 | 4.26187 | 0.0580913 | 0 | 242 | 241 | 1 |
| initial | tcr\_peptide/medium | mask50 | 4.24434 | 0.0537849 | 0 | 506 | 502 | 4 |
| initial | tcr\_peptide/medium | mask75 | 4.19083 | 0.0578947 | 0 | 767 | 760 | 7 |
| initial | tcr\_peptide/medium | mask100 | 4.09311 | 0.0610236 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_peptide/medium | mask\_mean | 4.19754 | 0.0576986 | 0 | — | — | — |
| initial | tcr\_peptide/short | mask25 | 3.96389 | 0.104167 | 0 | 51 | 48 | 3 |
| initial | tcr\_peptide/short | mask50 | 3.88309 | 0.0865385 | 0 | 110 | 104 | 6 |
| initial | tcr\_peptide/short | mask75 | 3.91848 | 0.0875912 | 0 | 144 | 137 | 7 |
| initial | tcr\_peptide/short | mask100 | 3.92809 | 0.09375 | 0 | 200 | 192 | 8 |
| initial | tcr\_peptide/short | mask\_mean | 3.92339 | 0.0930116 | 0 | — | — | — |
| initial | tcr\_pmhc/long | mask25 | 3.54116 | 0.0391791 | 0 | 540 | 536 | 4 |
| initial | tcr\_pmhc/long | mask50 | 3.78469 | 0.0357834 | 0 | 1038 | 1034 | 4 |
| initial | tcr\_pmhc/long | mask75 | 4.09161 | 0.032175 | 0 | 1562 | 1554 | 8 |
| initial | tcr\_pmhc/long | mask100 | 4.2881 | 0.0313726 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_pmhc/long | mask\_mean | 3.92639 | 0.0346275 | 0 | — | — | — |
| initial | tcr\_pmhc/medium | mask25 | 4.3718 | 0.0909091 | 0 | 244 | 242 | 2 |
| initial | tcr\_pmhc/medium | mask50 | 4.4696 | 0.0793319 | 0 | 484 | 479 | 5 |
| initial | tcr\_pmhc/medium | mask75 | 4.54578 | 0.0702211 | 0 | 777 | 769 | 8 |
| initial | tcr\_pmhc/medium | mask100 | 4.55209 | 0.0629921 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_pmhc/medium | mask\_mean | 4.48482 | 0.0758636 | 0 | — | — | — |
| initial | tcr\_pmhc/short | mask25 | 4.52547 | 0.0416667 | 0 | 51 | 48 | 3 |
| initial | tcr\_pmhc/short | mask50 | 4.53172 | 0.0481928 | 0 | 86 | 83 | 3 |
| initial | tcr\_pmhc/short | mask75 | 4.37626 | 0.0680272 | 0 | 152 | 147 | 5 |
| initial | tcr\_pmhc/short | mask100 | 4.39867 | 0.0625 | 0 | 200 | 192 | 8 |
| initial | tcr\_pmhc/short | mask\_mean | 4.45803 | 0.0550967 | 0 | — | — | — |
| initial | unconditional\_ab/long | mask25 | 3.74204 | 0.0672897 | 0 | 538 | 535 | 3 |
| initial | unconditional\_ab/long | mask50 | 3.94065 | 0.0554493 | 0 | 1050 | 1046 | 4 |
| initial | unconditional\_ab/long | mask75 | 4.06046 | 0.0630689 | 0 | 1545 | 1538 | 7 |
| initial | unconditional\_ab/long | mask100 | 3.98622 | 0.0627451 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_ab/long | mask\_mean | 3.93234 | 0.0621383 | 0 | — | — | — |
| initial | unconditional\_ab/medium | mask25 | 4.16708 | 0.13834 | 0 | 256 | 253 | 3 |
| initial | unconditional\_ab/medium | mask50 | 4.26042 | 0.147638 | 0 | 512 | 508 | 4 |
| initial | unconditional\_ab/medium | mask75 | 4.22652 | 0.149606 | 0 | 768 | 762 | 6 |
| initial | unconditional\_ab/medium | mask100 | 4.1788 | 0.15748 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_ab/medium | mask\_mean | 4.20821 | 0.148266 | 0 | — | — | — |
| initial | unconditional\_ab/short | mask25 | 4.0033 | 0.0961538 | 0 | 55 | 52 | 3 |
| initial | unconditional\_ab/short | mask50 | 3.92814 | 0.0612245 | 0 | 103 | 98 | 5 |
| initial | unconditional\_ab/short | mask75 | 3.90161 | 0.0567376 | 0 | 148 | 141 | 7 |
| initial | unconditional\_ab/short | mask100 | 3.86558 | 0.0625 | 0 | 200 | 192 | 8 |
| initial | unconditional\_ab/short | mask\_mean | 3.92466 | 0.069154 | 0 | — | — | — |
| initial | unconditional\_single/long | mask25 | 3.36228 | 0.0485075 | 0 | 269 | 268 | 1 |
| initial | unconditional\_single/long | mask50 | 3.37615 | 0.00761905 | 0 | 527 | 525 | 2 |
| initial | unconditional\_single/long | mask75 | 3.4834 | 0 | 0 | 778 | 774 | 4 |
| initial | unconditional\_single/long | mask100 | 3.57495 | 0 | 0 | 1024 | 1020 | 4 |
| initial | unconditional\_single/long | mask\_mean | 3.4492 | 0.0140316 | 0 | — | — | — |
| initial | unconditional\_single/medium | mask25 | 3.80027 | 0 | 0 | 110 | 109 | 1 |
| initial | unconditional\_single/medium | mask50 | 3.61845 | 0 | 0 | 234 | 232 | 2 |
| initial | unconditional\_single/medium | mask75 | 3.63511 | 0 | 0 | 371 | 367 | 4 |
| initial | unconditional\_single/medium | mask100 | 3.62178 | 0 | 0 | 512 | 508 | 4 |
| initial | unconditional\_single/medium | mask\_mean | 3.6689 | 0 | 0 | — | — | — |
| initial | unconditional\_single/short | mask25 | 3.31903 | 0 | 0 | 28 | 26 | 2 |
| initial | unconditional\_single/short | mask50 | 3.46652 | 0 | 0 | 52 | 50 | 2 |
| initial | unconditional\_single/short | mask75 | 3.30701 | 0 | 0 | 80 | 77 | 3 |
| initial | unconditional\_single/short | mask100 | 3.31712 | 0 | 0 | 100 | 96 | 4 |
| initial | unconditional\_single/short | mask\_mean | 3.35242 | 0 | 0 | — | — | — |
| initial | unconditional\_tcr/long | mask25 | 4.04654 | 0.0368932 | 0 | 516 | 515 | 1 |
| initial | unconditional\_tcr/long | mask50 | 4.34307 | 0.0303951 | 0 | 989 | 987 | 2 |
| initial | unconditional\_tcr/long | mask75 | 4.44465 | 0.0282523 | 0 | 1527 | 1522 | 5 |
| initial | unconditional\_tcr/long | mask100 | 4.25195 | 0.0313726 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_tcr/long | mask\_mean | 4.27155 | 0.0317283 | 0 | — | — | — |
| initial | unconditional\_tcr/medium | mask25 | 4.29023 | 0 | — | 247 | 247 | 0 |
| initial | unconditional\_tcr/medium | mask50 | 4.39133 | 0 | 0 | 500 | 497 | 3 |
| initial | unconditional\_tcr/medium | mask75 | 4.36145 | 0 | 0 | 744 | 738 | 6 |
| initial | unconditional\_tcr/medium | mask100 | 4.27926 | 0 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_tcr/medium | mask\_mean | 4.33057 | 0 | 0 | — | — | — |
| initial | unconditional\_tcr/short | mask25 | 4.36457 | 0.075 | — | 40 | 40 | 0 |
| initial | unconditional\_tcr/short | mask50 | 4.17515 | 0.0777778 | 0 | 93 | 90 | 3 |
| initial | unconditional\_tcr/short | mask75 | 4.1806 | 0.0882353 | 0 | 142 | 136 | 6 |
| initial | unconditional\_tcr/short | mask100 | 4.0935 | 0.09375 | 0 | 200 | 192 | 8 |
| initial | unconditional\_tcr/short | mask\_mean | 4.20345 | 0.0836908 | 0 | — | — | — |
| latest | ab\_antigen/long | mask25 | 1.81625 | 0.361217 | 0 | 530 | 526 | 4 |
| latest | ab\_antigen/long | mask50 | 1.78063 | 0.372162 | 0 | 1019 | 1013 | 6 |
| latest | ab\_antigen/long | mask75 | 1.767 | 0.369452 | 0 | 1538 | 1532 | 6 |
| latest | ab\_antigen/long | mask100 | 1.81599 | 0.331373 | 0 | 2048 | 2040 | 8 |
| latest | ab\_antigen/long | mask\_mean | 1.79497 | 0.358551 | 0 | — | — | — |
| latest | ab\_antigen/medium | mask25 | 1.86308 | 0.333333 | 0 | 246 | 243 | 3 |
| latest | ab\_antigen/medium | mask50 | 1.91309 | 0.306773 | 0 | 507 | 502 | 5 |
| latest | ab\_antigen/medium | mask75 | 1.87212 | 0.306069 | 0 | 765 | 758 | 7 |
| latest | ab\_antigen/medium | mask100 | 1.88609 | 0.29626 | 0 | 1024 | 1016 | 8 |
| latest | ab\_antigen/medium | mask\_mean | 1.8836 | 0.310609 | 0 | — | — | — |
| latest | ab\_antigen/short | mask25 | 2.34664 | 0.147059 | 0 | 38 | 34 | 4 |
| latest | ab\_antigen/short | mask50 | 2.19755 | 0.204545 | 0 | 93 | 88 | 5 |
| latest | ab\_antigen/short | mask75 | 2.14094 | 0.2 | 0 | 145 | 140 | 5 |
| latest | ab\_antigen/short | mask100 | 2.18034 | 0.213542 | 0 | 200 | 192 | 8 |
| latest | ab\_antigen/short | mask\_mean | 2.21637 | 0.191286 | 0 | — | — | — |
| latest | tcr\_peptide/long | mask25 | 1.89918 | 0.249462 | 0 | 466 | 465 | 1 |
| latest | tcr\_peptide/long | mask50 | 1.89398 | 0.271152 | 0 | 985 | 981 | 4 |
| latest | tcr\_peptide/long | mask75 | 1.87609 | 0.284168 | 0 | 1484 | 1478 | 6 |
| latest | tcr\_peptide/long | mask100 | 1.91341 | 0.251961 | 0 | 2048 | 2040 | 8 |
| latest | tcr\_peptide/long | mask\_mean | 1.89566 | 0.264186 | 0 | — | — | — |
| latest | tcr\_peptide/medium | mask25 | 1.98983 | 0.290456 | 0 | 242 | 241 | 1 |
| latest | tcr\_peptide/medium | mask50 | 1.97968 | 0.300797 | 0 | 506 | 502 | 4 |
| latest | tcr\_peptide/medium | mask75 | 1.98106 | 0.285526 | 0 | 767 | 760 | 7 |
| latest | tcr\_peptide/medium | mask100 | 2.02866 | 0.288386 | 0 | 1024 | 1016 | 8 |
| latest | tcr\_peptide/medium | mask\_mean | 1.99481 | 0.291291 | 0 | — | — | — |
| latest | tcr\_peptide/short | mask25 | 2.34054 | 0.229167 | 0 | 51 | 48 | 3 |
| latest | tcr\_peptide/short | mask50 | 2.32336 | 0.221154 | 0 | 110 | 104 | 6 |
| latest | tcr\_peptide/short | mask75 | 2.20971 | 0.248175 | 0 | 144 | 137 | 7 |
| latest | tcr\_peptide/short | mask100 | 2.24667 | 0.244792 | 0 | 200 | 192 | 8 |
| latest | tcr\_peptide/short | mask\_mean | 2.28007 | 0.235822 | 0 | — | — | — |
| latest | tcr\_pmhc/long | mask25 | 1.53883 | 0.378731 | 0 | 540 | 536 | 4 |
| latest | tcr\_pmhc/long | mask50 | 1.51217 | 0.389749 | 0 | 1038 | 1034 | 4 |
| latest | tcr\_pmhc/long | mask75 | 1.51816 | 0.411197 | 0 | 1562 | 1554 | 8 |
| latest | tcr\_pmhc/long | mask100 | 1.55364 | 0.403922 | 0 | 2048 | 2040 | 8 |
| latest | tcr\_pmhc/long | mask\_mean | 1.5307 | 0.3959 | 0 | — | — | — |
| latest | tcr\_pmhc/medium | mask25 | 1.53492 | 0.404959 | 0 | 244 | 242 | 2 |
| latest | tcr\_pmhc/medium | mask50 | 1.57537 | 0.390397 | 0 | 484 | 479 | 5 |
| latest | tcr\_pmhc/medium | mask75 | 1.62276 | 0.396619 | 0 | 777 | 769 | 8 |
| latest | tcr\_pmhc/medium | mask100 | 1.72255 | 0.393701 | 0 | 1024 | 1016 | 8 |
| latest | tcr\_pmhc/medium | mask\_mean | 1.6139 | 0.396419 | 0 | — | — | — |
| latest | tcr\_pmhc/short | mask25 | 1.93707 | 0.354167 | 0 | 51 | 48 | 3 |
| latest | tcr\_pmhc/short | mask50 | 1.9297 | 0.337349 | 0 | 86 | 83 | 3 |
| latest | tcr\_pmhc/short | mask75 | 1.98441 | 0.29932 | 0 | 152 | 147 | 5 |
| latest | tcr\_pmhc/short | mask100 | 2.11086 | 0.203125 | 0 | 200 | 192 | 8 |
| latest | tcr\_pmhc/short | mask\_mean | 1.99051 | 0.29849 | 0 | — | — | — |
| latest | unconditional\_ab/long | mask25 | 1.84568 | 0.338318 | 0 | 538 | 535 | 3 |
| latest | unconditional\_ab/long | mask50 | 1.84843 | 0.325048 | 0 | 1050 | 1046 | 4 |
| latest | unconditional\_ab/long | mask75 | 1.865 | 0.33225 | 0 | 1545 | 1538 | 7 |
| latest | unconditional\_ab/long | mask100 | 2.13282 | 0.336765 | 0 | 2048 | 2040 | 8 |
| latest | unconditional\_ab/long | mask\_mean | 1.92298 | 0.333095 | 0 | — | — | — |
| latest | unconditional\_ab/medium | mask25 | 1.90692 | 0.3083 | 0 | 256 | 253 | 3 |
| latest | unconditional\_ab/medium | mask50 | 1.97036 | 0.307087 | 0 | 512 | 508 | 4 |
| latest | unconditional\_ab/medium | mask75 | 1.95092 | 0.295276 | 0 | 768 | 762 | 6 |
| latest | unconditional\_ab/medium | mask100 | 2.17722 | 0.298228 | 0 | 1024 | 1016 | 8 |
| latest | unconditional\_ab/medium | mask\_mean | 2.00136 | 0.302223 | 0 | — | — | — |
| latest | unconditional\_ab/short | mask25 | 2.16796 | 0.25 | 0 | 55 | 52 | 3 |
| latest | unconditional\_ab/short | mask50 | 2.28936 | 0.234694 | 0 | 103 | 98 | 5 |
| latest | unconditional\_ab/short | mask75 | 2.38299 | 0.22695 | 0 | 148 | 141 | 7 |
| latest | unconditional\_ab/short | mask100 | 2.53548 | 0.229167 | 0 | 200 | 192 | 8 |
| latest | unconditional\_ab/short | mask\_mean | 2.34395 | 0.235203 | 0 | — | — | — |
| latest | unconditional\_single/long | mask25 | 1.69934 | 0.328358 | 0 | 269 | 268 | 1 |
| latest | unconditional\_single/long | mask50 | 1.70341 | 0.314286 | 0 | 527 | 525 | 2 |
| latest | unconditional\_single/long | mask75 | 1.79075 | 0.328165 | 0 | 778 | 774 | 4 |
| latest | unconditional\_single/long | mask100 | 2.12939 | 0.305882 | 0 | 1024 | 1020 | 4 |
| latest | unconditional\_single/long | mask\_mean | 1.83072 | 0.319173 | 0 | — | — | — |
| latest | unconditional\_single/medium | mask25 | 1.78002 | 0.330275 | 0 | 110 | 109 | 1 |
| latest | unconditional\_single/medium | mask50 | 1.80913 | 0.353448 | 0 | 234 | 232 | 2 |
| latest | unconditional\_single/medium | mask75 | 1.86657 | 0.321526 | 0 | 371 | 367 | 4 |
| latest | unconditional\_single/medium | mask100 | 2.1643 | 0.330709 | 0 | 512 | 508 | 4 |
| latest | unconditional\_single/medium | mask\_mean | 1.90501 | 0.33399 | 0 | — | — | — |
| latest | unconditional\_single/short | mask25 | 2.76583 | 0.269231 | 0 | 28 | 26 | 2 |
| latest | unconditional\_single/short | mask50 | 2.70742 | 0.2 | 0 | 52 | 50 | 2 |
| latest | unconditional\_single/short | mask75 | 2.45272 | 0.272727 | 0 | 80 | 77 | 3 |
| latest | unconditional\_single/short | mask100 | 2.38563 | 0.291667 | 0 | 100 | 96 | 4 |
| latest | unconditional\_single/short | mask\_mean | 2.5779 | 0.258406 | 0 | — | — | — |
| latest | unconditional\_tcr/long | mask25 | 1.78563 | 0.312621 | 0 | 516 | 515 | 1 |
| latest | unconditional\_tcr/long | mask50 | 1.79136 | 0.31307 | 0 | 989 | 987 | 2 |
| latest | unconditional\_tcr/long | mask75 | 1.84173 | 0.305519 | 0 | 1527 | 1522 | 5 |
| latest | unconditional\_tcr/long | mask100 | 2.21544 | 0.214706 | 0 | 2048 | 2040 | 8 |
| latest | unconditional\_tcr/long | mask\_mean | 1.90854 | 0.286479 | 0 | — | — | — |
| latest | unconditional\_tcr/medium | mask25 | 1.59358 | 0.360324 | — | 247 | 247 | 0 |
| latest | unconditional\_tcr/medium | mask50 | 1.62249 | 0.366197 | 0 | 500 | 497 | 3 |
| latest | unconditional\_tcr/medium | mask75 | 1.66318 | 0.364499 | 0 | 744 | 738 | 6 |
| latest | unconditional\_tcr/medium | mask100 | 2.04619 | 0.305118 | 0 | 1024 | 1016 | 8 |
| latest | unconditional\_tcr/medium | mask\_mean | 1.73136 | 0.349034 | 0 | — | — | — |
| latest | unconditional\_tcr/short | mask25 | 2.27119 | 0.25 | — | 40 | 40 | 0 |
| latest | unconditional\_tcr/short | mask50 | 2.27441 | 0.2 | 0 | 93 | 90 | 3 |
| latest | unconditional\_tcr/short | mask75 | 2.31013 | 0.198529 | 0 | 142 | 136 | 6 |
| latest | unconditional\_tcr/short | mask100 | 2.40347 | 0.177083 | 0 | 200 | 192 | 8 |
| latest | unconditional\_tcr/short | mask\_mean | 2.3148 | 0.206403 | 0 | — | — | — |
| train\_probe | ab\_antigen/long | mask25 | 1.583 | 0.386792 | — | 530 | 530 | 0 |
| train\_probe | ab\_antigen/long | mask50 | 1.58102 | 0.394762 | 0 | 1034 | 1031 | 3 |
| train\_probe | ab\_antigen/long | mask75 | 1.58644 | 0.387926 | 0 | 1563 | 1557 | 6 |
| train\_probe | ab\_antigen/long | mask100 | 1.71042 | 0.363235 | 0 | 2048 | 2040 | 8 |
| train\_probe | ab\_antigen/long | mask\_mean | 1.61522 | 0.383179 | 0 | — | — | — |
| train\_probe | ab\_antigen/medium | mask25 | 1.9607 | 0.285714 | 0 | 275 | 273 | 2 |
| train\_probe | ab\_antigen/medium | mask50 | 1.94381 | 0.315985 | 0 | 542 | 538 | 4 |
| train\_probe | ab\_antigen/medium | mask75 | 1.90747 | 0.331186 | 0 | 781 | 776 | 5 |
| train\_probe | ab\_antigen/medium | mask100 | 1.95682 | 0.317913 | 0 | 1024 | 1016 | 8 |
| train\_probe | ab\_antigen/medium | mask\_mean | 1.9422 | 0.3127 | 0 | — | — | — |
| train\_probe | ab\_antigen/short | mask25 | 2.17564 | 0.361702 | 0 | 52 | 47 | 5 |
| train\_probe | ab\_antigen/short | mask50 | 2.15772 | 0.284314 | 0 | 109 | 102 | 7 |
| train\_probe | ab\_antigen/short | mask75 | 2.15017 | 0.310811 | 0 | 155 | 148 | 7 |
| train\_probe | ab\_antigen/short | mask100 | 2.21324 | 0.291667 | 0 | 200 | 192 | 8 |
| train\_probe | ab\_antigen/short | mask\_mean | 2.17419 | 0.312123 | 0 | — | — | — |
| train\_probe | tcr\_peptide/long | mask25 | 1.604 | 0.376518 | 0 | 495 | 494 | 1 |
| train\_probe | tcr\_peptide/long | mask50 | 1.61461 | 0.367704 | 0 | 1033 | 1028 | 5 |
| train\_probe | tcr\_peptide/long | mask75 | 1.67075 | 0.354139 | 0 | 1530 | 1522 | 8 |
| train\_probe | tcr\_peptide/long | mask100 | 1.70957 | 0.348529 | 0 | 2048 | 2040 | 8 |
| train\_probe | tcr\_peptide/long | mask\_mean | 1.64973 | 0.361723 | 0 | — | — | — |
| train\_probe | tcr\_peptide/medium | mask25 | 1.75014 | 0.337302 | — | 252 | 252 | 0 |
| train\_probe | tcr\_peptide/medium | mask50 | 1.76082 | 0.339181 | 0 | 517 | 513 | 4 |
| train\_probe | tcr\_peptide/medium | mask75 | 1.74963 | 0.330239 | 0 | 760 | 754 | 6 |
| train\_probe | tcr\_peptide/medium | mask100 | 1.87358 | 0.312008 | 0 | 1024 | 1016 | 8 |
| train\_probe | tcr\_peptide/medium | mask\_mean | 1.78354 | 0.329682 | 0 | — | — | — |
| train\_probe | tcr\_peptide/short | mask25 | 2.06545 | 0.347826 | 0 | 51 | 46 | 5 |
| train\_probe | tcr\_peptide/short | mask50 | 1.95139 | 0.366667 | 0 | 95 | 90 | 5 |
| train\_probe | tcr\_peptide/short | mask75 | 1.9874 | 0.352941 | 0 | 142 | 136 | 6 |
| train\_probe | tcr\_peptide/short | mask100 | 2.05012 | 0.333333 | 0 | 200 | 192 | 8 |
| train\_probe | tcr\_peptide/short | mask\_mean | 2.01359 | 0.350192 | 0 | — | — | — |
| train\_probe | tcr\_pmhc/long | mask25 | 1.63279 | 0.335249 | 0 | 526 | 522 | 4 |
| train\_probe | tcr\_pmhc/long | mask50 | 1.67229 | 0.324927 | 0 | 1036 | 1031 | 5 |
| train\_probe | tcr\_pmhc/long | mask75 | 1.68119 | 0.327531 | 0 | 1557 | 1551 | 6 |
| train\_probe | tcr\_pmhc/long | mask100 | 1.69575 | 0.32402 | 0 | 2048 | 2040 | 8 |
| train\_probe | tcr\_pmhc/long | mask\_mean | 1.6705 | 0.327932 | 0 | — | — | — |
| train\_probe | tcr\_pmhc/medium | mask25 | 1.60198 | 0.355311 | 0 | 276 | 273 | 3 |
| train\_probe | tcr\_pmhc/medium | mask50 | 1.65019 | 0.340111 | 0 | 549 | 541 | 8 |
| train\_probe | tcr\_pmhc/medium | mask75 | 1.65478 | 0.329936 | 0 | 793 | 785 | 8 |
| train\_probe | tcr\_pmhc/medium | mask100 | 1.75109 | 0.329724 | 0 | 1024 | 1016 | 8 |
| train\_probe | tcr\_pmhc/medium | mask\_mean | 1.66451 | 0.338771 | 0 | — | — | — |
| train\_probe | tcr\_pmhc/short | mask25 | 2.37592 | 0.25 | 0 | 50 | 48 | 2 |
| train\_probe | tcr\_pmhc/short | mask50 | 2.40077 | 0.18 | 0 | 103 | 100 | 3 |
| train\_probe | tcr\_pmhc/short | mask75 | 2.43885 | 0.17931 | 0 | 152 | 145 | 7 |
| train\_probe | tcr\_pmhc/short | mask100 | 2.32783 | 0.21875 | 0 | 200 | 192 | 8 |
| train\_probe | tcr\_pmhc/short | mask\_mean | 2.38584 | 0.207015 | 0 | — | — | — |
| train\_probe | unconditional\_ab/long | mask25 | 1.66267 | 0.310277 | 0 | 509 | 506 | 3 |
| train\_probe | unconditional\_ab/long | mask50 | 1.66483 | 0.327723 | 0 | 1016 | 1010 | 6 |
| train\_probe | unconditional\_ab/long | mask75 | 1.68087 | 0.310698 | 0 | 1569 | 1561 | 8 |
| train\_probe | unconditional\_ab/long | mask100 | 2.05469 | 0.321078 | 0 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_ab/long | mask\_mean | 1.76576 | 0.317444 | 0 | — | — | — |
| train\_probe | unconditional\_ab/medium | mask25 | 1.88225 | 0.33195 | 0 | 243 | 241 | 2 |
| train\_probe | unconditional\_ab/medium | mask50 | 1.93179 | 0.315891 | 0 | 521 | 516 | 5 |
| train\_probe | unconditional\_ab/medium | mask75 | 1.91239 | 0.327201 | 0 | 766 | 761 | 5 |
| train\_probe | unconditional\_ab/medium | mask100 | 2.12073 | 0.329724 | 0 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_ab/medium | mask\_mean | 1.96179 | 0.326192 | 0 | — | — | — |
| train\_probe | unconditional\_ab/short | mask25 | 2.37205 | 0.173077 | 0 | 53 | 52 | 1 |
| train\_probe | unconditional\_ab/short | mask50 | 2.2793 | 0.211538 | 0 | 106 | 104 | 2 |
| train\_probe | unconditional\_ab/short | mask75 | 2.34203 | 0.208861 | 0 | 163 | 158 | 5 |
| train\_probe | unconditional\_ab/short | mask100 | 2.54478 | 0.197917 | 0 | 200 | 192 | 8 |
| train\_probe | unconditional\_ab/short | mask\_mean | 2.38454 | 0.197848 | 0 | — | — | — |
| train\_probe | unconditional\_single/long | mask25 | 1.83614 | 0.286885 | 0 | 245 | 244 | 1 |
| train\_probe | unconditional\_single/long | mask50 | 1.82373 | 0.303462 | 0 | 492 | 491 | 1 |
| train\_probe | unconditional\_single/long | mask75 | 1.85218 | 0.292969 | 0 | 771 | 768 | 3 |
| train\_probe | unconditional\_single/long | mask100 | 2.26748 | 0.243137 | 0 | 1024 | 1020 | 4 |
| train\_probe | unconditional\_single/long | mask\_mean | 1.94488 | 0.281613 | 0 | — | — | — |
| train\_probe | unconditional\_single/medium | mask25 | 1.77816 | 0.392308 | 0 | 132 | 130 | 2 |
| train\_probe | unconditional\_single/medium | mask50 | 1.83451 | 0.313492 | 0 | 255 | 252 | 3 |
| train\_probe | unconditional\_single/medium | mask75 | 1.79942 | 0.345646 | 0 | 382 | 379 | 3 |
| train\_probe | unconditional\_single/medium | mask100 | 2.03564 | 0.362205 | 0 | 512 | 508 | 4 |
| train\_probe | unconditional\_single/medium | mask\_mean | 1.86193 | 0.353413 | 0 | — | — | — |
| train\_probe | unconditional\_single/short | mask25 | 2.53608 | 0.0769231 | 0 | 28 | 26 | 2 |
| train\_probe | unconditional\_single/short | mask50 | 2.38438 | 0.22449 | 0 | 52 | 49 | 3 |
| train\_probe | unconditional\_single/short | mask75 | 2.36919 | 0.228571 | 0 | 73 | 70 | 3 |
| train\_probe | unconditional\_single/short | mask100 | 2.4011 | 0.229167 | 0 | 100 | 96 | 4 |
| train\_probe | unconditional\_single/short | mask\_mean | 2.42269 | 0.189788 | 0 | — | — | — |
| train\_probe | unconditional\_tcr/long | mask25 | 1.73995 | 0.380952 | — | 525 | 525 | 0 |
| train\_probe | unconditional\_tcr/long | mask50 | 1.76169 | 0.346411 | 0 | 1048 | 1045 | 3 |
| train\_probe | unconditional\_tcr/long | mask75 | 1.79538 | 0.327848 | 0 | 1585 | 1580 | 5 |
| train\_probe | unconditional\_tcr/long | mask100 | 2.21942 | 0.151961 | 0 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_tcr/long | mask\_mean | 1.87911 | 0.301793 | 0 | — | — | — |
| train\_probe | unconditional\_tcr/medium | mask25 | 1.70944 | 0.298851 | — | 261 | 261 | 0 |
| train\_probe | unconditional\_tcr/medium | mask50 | 1.70169 | 0.348089 | 0 | 498 | 497 | 1 |
| train\_probe | unconditional\_tcr/medium | mask75 | 1.71662 | 0.375499 | 0 | 755 | 751 | 4 |
| train\_probe | unconditional\_tcr/medium | mask100 | 1.92357 | 0.36811 | 0 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_tcr/medium | mask\_mean | 1.76283 | 0.347637 | 0 | — | — | — |
| train\_probe | unconditional\_tcr/short | mask25 | 2.08643 | 0.173913 | — | 46 | 46 | 0 |
| train\_probe | unconditional\_tcr/short | mask50 | 2.35772 | 0.141304 | 0 | 97 | 92 | 5 |
| train\_probe | unconditional\_tcr/short | mask75 | 2.34201 | 0.130137 | 0 | 152 | 146 | 6 |
| train\_probe | unconditional\_tcr/short | mask100 | 2.5241 | 0.114583 | 0 | 200 | 192 | 8 |
| train\_probe | unconditional\_tcr/short | mask\_mean | 2.32757 | 0.139984 | 0 | — | — | — |

### 1790802160350276097\_r23\_refined\_layerscale\_eb29f0d4：分层 mask 指标

源码目录：/vepfs-mlp2/c20250601/251105016/project/dllm\_test/output/immune\_research\_releases/r20\_r27\_20261001；完成/目标 step：25/25。

kind=synthetic\_generalization\_diagnostic；dataset_kind=deterministic\_artificial\_sequences；biological_quality_claim=False。

| 时点 | mask level | 指标 | 值 |
| --- | --- | --- | --- |
| initial | mask25 | mask\_ratio | 0.25 |
| initial | mask25 | selected\_count | 4467 |
| initial | mask25 | aa\_count | 4428 |
| initial | mask25 | terminal\_count | 39 |
| initial | mask25 | loss | 3.81518 |
| initial | mask25 | aa\_loss | 3.8107 |
| initial | mask25 | aa\_accuracy | 0.0589431 |
| initial | mask25 | terminal\_loss | 4.32412 |
| initial | mask25 | terminal\_accuracy | 0 |
| initial | mask50 | mask\_ratio | 0.5 |
| initial | mask50 | selected\_count | 8888 |
| initial | mask50 | aa\_count | 8819 |
| initial | mask50 | terminal\_count | 69 |
| initial | mask50 | loss | 3.85226 |
| initial | mask50 | aa\_loss | 3.8493 |
| initial | mask50 | aa\_accuracy | 0.0426352 |
| initial | mask50 | terminal\_loss | 4.23099 |
| initial | mask50 | terminal\_accuracy | 0 |
| initial | mask75 | mask\_ratio | 0.75 |
| initial | mask75 | selected\_count | 13437 |
| initial | mask75 | aa\_count | 13330 |
| initial | mask75 | terminal\_count | 107 |
| initial | mask75 | loss | 3.83517 |
| initial | mask75 | aa\_loss | 3.83171 |
| initial | mask75 | aa\_accuracy | 0.0306827 |
| initial | mask75 | terminal\_loss | 4.26675 |
| initial | mask75 | terminal\_accuracy | 0 |
| initial | mask100 | mask\_ratio | 1 |
| initial | mask100 | selected\_count | 17996 |
| initial | mask100 | aa\_count | 17864 |
| initial | mask100 | terminal\_count | 132 |
| initial | mask100 | loss | 3.80145 |
| initial | mask100 | aa\_loss | 3.79761 |
| initial | mask100 | aa\_accuracy | 0.0301164 |
| initial | mask100 | terminal\_loss | 4.32137 |
| initial | mask100 | terminal\_accuracy | 0 |
| initial | mask\_mean | loss | 3.82602 |
| initial | mask\_mean | aa\_loss | 3.82233 |
| initial | mask\_mean | terminal\_loss | 4.28581 |
| initial | mask\_mean | aa\_accuracy | 0.0405944 |
| initial | mask\_mean | terminal\_accuracy | 0 |
| latest | mask25 | mask\_ratio | 0.25 |
| latest | mask25 | selected\_count | 4467 |
| latest | mask25 | aa\_count | 4428 |
| latest | mask25 | terminal\_count | 39 |
| latest | mask25 | loss | 1.79676 |
| latest | mask25 | aa\_loss | 1.7662 |
| latest | mask25 | aa\_accuracy | 0.320009 |
| latest | mask25 | terminal\_loss | 5.26583 |
| latest | mask25 | terminal\_accuracy | 0 |
| latest | mask50 | mask\_ratio | 0.5 |
| latest | mask50 | selected\_count | 8888 |
| latest | mask50 | aa\_count | 8819 |
| latest | mask50 | terminal\_count | 69 |
| latest | mask50 | loss | 1.80018 |
| latest | mask50 | aa\_loss | 1.77433 |
| latest | mask50 | aa\_accuracy | 0.322826 |
| latest | mask50 | terminal\_loss | 5.1045 |
| latest | mask50 | terminal\_accuracy | 0 |
| latest | mask75 | mask\_ratio | 0.75 |
| latest | mask75 | selected\_count | 13437 |
| latest | mask75 | aa\_count | 13330 |
| latest | mask75 | terminal\_count | 107 |
| latest | mask75 | loss | 1.81045 |
| latest | mask75 | aa\_loss | 1.7852 |
| latest | mask75 | aa\_accuracy | 0.326932 |
| latest | mask75 | terminal\_loss | 4.9563 |
| latest | mask75 | terminal\_accuracy | 0 |
| latest | mask100 | mask\_ratio | 1 |
| latest | mask100 | selected\_count | 17996 |
| latest | mask100 | aa\_count | 17864 |
| latest | mask100 | terminal\_count | 132 |
| latest | mask100 | loss | 1.97572 |
| latest | mask100 | aa\_loss | 1.95825 |
| latest | mask100 | aa\_accuracy | 0.311688 |
| latest | mask100 | terminal\_loss | 4.34007 |
| latest | mask100 | terminal\_accuracy | 0 |
| latest | mask\_mean | loss | 1.84578 |
| latest | mask\_mean | aa\_loss | 1.82099 |
| latest | mask\_mean | terminal\_loss | 4.91668 |
| latest | mask\_mean | aa\_accuracy | 0.320364 |
| latest | mask\_mean | terminal\_accuracy | 0 |

评测曲线（原始 step，四档等权均值）：

| step | mean loss | mean AA acc | mean end acc | gradient norm |
| --- | --- | --- | --- | --- |
| 25 | 1.84578 | 0.320364 | 0 | 1.56676 |

有条件/无条件曲线（每档按token计数合并，再平均四档；缺失分组不推断数值）：

| step | 子集 | mean loss | mean AA | mean end | Full-mask AA | Full-mask end | Full-mask AA count | Full-mask end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | conditional | 3.79117 | 0.0444692 | 0 | 0.0298645 | 0 | 9744 | 72 |
| initial | unconditional | 3.86732 | 0.0360518 | 0 | 0.0304187 | 0 | 8120 | 60 |
| 25 | conditional | 1.79854 | 0.317915 | 0 | 0.309011 | 0 | 9744 | 72 |
| 25 | unconditional | 1.90253 | 0.323239 | 0 | 0.314901 | 0 | 8120 | 60 |

人工数据 manifest：d40bbbbc1c4294eed843d88bcdd0cf193886de6c2d7e9db9069bbf86e931c022；精确 target sequence 重叠=0；精确 record 重叠=0。

跨 radius32 覆盖：{&quot;possible&quot;: true, &quot;train&quot;: true, &quot;valid&quot;: true}。生成规则共享是诊断设计的一部分，不能当作真实下游去重证书。


按任务/链长分组（全局 level 按 token 计数汇总，下面保留各组计数；不同组不得直接比较未经加权的总 loss）：

| 时点 | group/length | mask level | loss | AA acc | end acc | selected count | AA count | end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | ab\_antigen/long | mask25 | 4.10241 | 0 | 0 | 530 | 526 | 4 |
| initial | ab\_antigen/long | mask50 | 4.14621 | 0 | 0 | 1019 | 1013 | 6 |
| initial | ab\_antigen/long | mask75 | 4.10335 | 0 | 0 | 1538 | 1532 | 6 |
| initial | ab\_antigen/long | mask100 | 4.03516 | 0 | 0 | 2048 | 2040 | 8 |
| initial | ab\_antigen/long | mask\_mean | 4.09678 | 0 | 0 | — | — | — |
| initial | ab\_antigen/medium | mask25 | 4.1109 | 0 | 0 | 246 | 243 | 3 |
| initial | ab\_antigen/medium | mask50 | 4.0947 | 0 | 0 | 507 | 502 | 5 |
| initial | ab\_antigen/medium | mask75 | 4.07023 | 0 | 0 | 765 | 758 | 7 |
| initial | ab\_antigen/medium | mask100 | 4.03235 | 0 | 0 | 1024 | 1016 | 8 |
| initial | ab\_antigen/medium | mask\_mean | 4.07704 | 0 | 0 | — | — | — |
| initial | ab\_antigen/short | mask25 | 3.94668 | 0.0588235 | 0 | 38 | 34 | 4 |
| initial | ab\_antigen/short | mask50 | 3.8665 | 0.0454545 | 0 | 93 | 88 | 5 |
| initial | ab\_antigen/short | mask75 | 3.78397 | 0.05 | 0 | 145 | 140 | 5 |
| initial | ab\_antigen/short | mask100 | 3.7871 | 0.046875 | 0 | 200 | 192 | 8 |
| initial | ab\_antigen/short | mask\_mean | 3.84606 | 0.0502883 | 0 | — | — | — |
| initial | tcr\_peptide/long | mask25 | 3.70198 | 0.0258065 | 0 | 466 | 465 | 1 |
| initial | tcr\_peptide/long | mask50 | 3.77676 | 0.011213 | 0 | 985 | 981 | 4 |
| initial | tcr\_peptide/long | mask75 | 3.77699 | 0.0121786 | 0 | 1484 | 1478 | 6 |
| initial | tcr\_peptide/long | mask100 | 3.76535 | 0.0151961 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_peptide/long | mask\_mean | 3.75527 | 0.0160985 | 0 | — | — | — |
| initial | tcr\_peptide/medium | mask25 | 3.97216 | 0.00829876 | 0 | 242 | 241 | 1 |
| initial | tcr\_peptide/medium | mask50 | 3.90908 | 0.00996016 | 0 | 506 | 502 | 4 |
| initial | tcr\_peptide/medium | mask75 | 3.85215 | 0.0118421 | 0 | 767 | 760 | 7 |
| initial | tcr\_peptide/medium | mask100 | 3.78247 | 0.0147638 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_peptide/medium | mask\_mean | 3.87897 | 0.0112162 | 0 | — | — | — |
| initial | tcr\_peptide/short | mask25 | 3.64878 | 0.0625 | 0 | 51 | 48 | 3 |
| initial | tcr\_peptide/short | mask50 | 3.61911 | 0.0865385 | 0 | 110 | 104 | 6 |
| initial | tcr\_peptide/short | mask75 | 3.66048 | 0.0656934 | 0 | 144 | 137 | 7 |
| initial | tcr\_peptide/short | mask100 | 3.65702 | 0.046875 | 0 | 200 | 192 | 8 |
| initial | tcr\_peptide/short | mask\_mean | 3.64635 | 0.0654017 | 0 | — | — | — |
| initial | tcr\_pmhc/long | mask25 | 3.19962 | 0.266791 | 0 | 540 | 536 | 4 |
| initial | tcr\_pmhc/long | mask50 | 3.30848 | 0.185687 | 0 | 1038 | 1034 | 4 |
| initial | tcr\_pmhc/long | mask75 | 3.37315 | 0.117761 | 0 | 1562 | 1554 | 8 |
| initial | tcr\_pmhc/long | mask100 | 3.41892 | 0.109804 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_pmhc/long | mask\_mean | 3.32504 | 0.170011 | 0 | — | — | — |
| initial | tcr\_pmhc/medium | mask25 | 3.82185 | 0.0165289 | 0 | 244 | 242 | 2 |
| initial | tcr\_pmhc/medium | mask50 | 3.84317 | 0.00626305 | 0 | 484 | 479 | 5 |
| initial | tcr\_pmhc/medium | mask75 | 3.85431 | 0 | 0 | 777 | 769 | 8 |
| initial | tcr\_pmhc/medium | mask100 | 3.8816 | 0 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_pmhc/medium | mask\_mean | 3.85023 | 0.00569799 | 0 | — | — | — |
| initial | tcr\_pmhc/short | mask25 | 3.76194 | 0 | 0 | 51 | 48 | 3 |
| initial | tcr\_pmhc/short | mask50 | 3.81972 | 0.0120482 | 0 | 86 | 83 | 3 |
| initial | tcr\_pmhc/short | mask75 | 3.80688 | 0.0136054 | 0 | 152 | 147 | 5 |
| initial | tcr\_pmhc/short | mask100 | 3.83313 | 0.015625 | 0 | 200 | 192 | 8 |
| initial | tcr\_pmhc/short | mask\_mean | 3.80542 | 0.0103197 | 0 | — | — | — |
| initial | unconditional\_ab/long | mask25 | 3.9701 | 0.0616822 | 0 | 538 | 535 | 3 |
| initial | unconditional\_ab/long | mask50 | 3.92095 | 0.0602294 | 0 | 1050 | 1046 | 4 |
| initial | unconditional\_ab/long | mask75 | 3.88596 | 0.0318596 | 0 | 1545 | 1538 | 7 |
| initial | unconditional\_ab/long | mask100 | 3.79481 | 0.0308824 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_ab/long | mask\_mean | 3.89296 | 0.0461634 | 0 | — | — | — |
| initial | unconditional\_ab/medium | mask25 | 3.7778 | 0.0592885 | 0 | 256 | 253 | 3 |
| initial | unconditional\_ab/medium | mask50 | 3.8424 | 0.0393701 | 0 | 512 | 508 | 4 |
| initial | unconditional\_ab/medium | mask75 | 3.8188 | 0.0354331 | 0 | 768 | 762 | 6 |
| initial | unconditional\_ab/medium | mask100 | 3.7911 | 0.0305118 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_ab/medium | mask\_mean | 3.80753 | 0.0411509 | 0 | — | — | — |
| initial | unconditional\_ab/short | mask25 | 3.85957 | 0.0576923 | 0 | 55 | 52 | 3 |
| initial | unconditional\_ab/short | mask50 | 3.84337 | 0.0306122 | 0 | 103 | 98 | 5 |
| initial | unconditional\_ab/short | mask75 | 3.80081 | 0.035461 | 0 | 148 | 141 | 7 |
| initial | unconditional\_ab/short | mask100 | 3.75868 | 0.03125 | 0 | 200 | 192 | 8 |
| initial | unconditional\_ab/short | mask\_mean | 3.8156 | 0.0387539 | 0 | — | — | — |
| initial | unconditional\_single/long | mask25 | 3.94339 | 0.0485075 | 0 | 269 | 268 | 1 |
| initial | unconditional\_single/long | mask50 | 4.00614 | 0.0514286 | 0 | 527 | 525 | 2 |
| initial | unconditional\_single/long | mask75 | 3.96455 | 0.0516796 | 0 | 778 | 774 | 4 |
| initial | unconditional\_single/long | mask100 | 3.88226 | 0.0617647 | 0 | 1024 | 1020 | 4 |
| initial | unconditional\_single/long | mask\_mean | 3.94909 | 0.0533451 | 0 | — | — | — |
| initial | unconditional\_single/medium | mask25 | 4.15354 | 0.0366972 | 0 | 110 | 109 | 1 |
| initial | unconditional\_single/medium | mask50 | 4.00297 | 0.0258621 | 0 | 234 | 232 | 2 |
| initial | unconditional\_single/medium | mask75 | 3.95779 | 0.0245232 | 0 | 371 | 367 | 4 |
| initial | unconditional\_single/medium | mask100 | 3.90136 | 0.0295276 | 0 | 512 | 508 | 4 |
| initial | unconditional\_single/medium | mask\_mean | 4.00392 | 0.0291525 | 0 | — | — | — |
| initial | unconditional\_single/short | mask25 | 3.41472 | 0.269231 | 0 | 28 | 26 | 2 |
| initial | unconditional\_single/short | mask50 | 3.5961 | 0.18 | 0 | 52 | 50 | 2 |
| initial | unconditional\_single/short | mask75 | 3.42945 | 0.207792 | 0 | 80 | 77 | 3 |
| initial | unconditional\_single/short | mask100 | 3.41298 | 0.21875 | 0 | 100 | 96 | 4 |
| initial | unconditional\_single/short | mask\_mean | 3.46331 | 0.218943 | 0 | — | — | — |
| initial | unconditional\_tcr/long | mask25 | 3.81511 | 0.0291262 | 0 | 516 | 515 | 1 |
| initial | unconditional\_tcr/long | mask50 | 3.90053 | 0.0151976 | 0 | 989 | 987 | 2 |
| initial | unconditional\_tcr/long | mask75 | 3.85577 | 0.0157687 | 0 | 1527 | 1522 | 5 |
| initial | unconditional\_tcr/long | mask100 | 3.77771 | 0.0156863 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_tcr/long | mask\_mean | 3.83728 | 0.0189447 | 0 | — | — | — |
| initial | unconditional\_tcr/medium | mask25 | 3.74311 | 0.0202429 | — | 247 | 247 | 0 |
| initial | unconditional\_tcr/medium | mask50 | 3.86175 | 0.0160966 | 0 | 500 | 497 | 3 |
| initial | unconditional\_tcr/medium | mask75 | 3.85697 | 0.0149051 | 0 | 744 | 738 | 6 |
| initial | unconditional\_tcr/medium | mask100 | 3.8777 | 0.015748 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_tcr/medium | mask\_mean | 3.83488 | 0.0167482 | 0 | — | — | — |
| initial | unconditional\_tcr/short | mask25 | 4.01228 | 0 | — | 40 | 40 | 0 |
| initial | unconditional\_tcr/short | mask50 | 3.82482 | 0 | 0 | 93 | 90 | 3 |
| initial | unconditional\_tcr/short | mask75 | 3.85293 | 0 | 0 | 142 | 136 | 6 |
| initial | unconditional\_tcr/short | mask100 | 3.86799 | 0 | 0 | 200 | 192 | 8 |
| initial | unconditional\_tcr/short | mask\_mean | 3.8895 | 0 | 0 | — | — | — |
| latest | ab\_antigen/long | mask25 | 1.85321 | 0.326996 | 0 | 530 | 526 | 4 |
| latest | ab\_antigen/long | mask50 | 1.81461 | 0.342547 | 0 | 1019 | 1013 | 6 |
| latest | ab\_antigen/long | mask75 | 1.81355 | 0.33094 | 0 | 1538 | 1532 | 6 |
| latest | ab\_antigen/long | mask100 | 1.81114 | 0.333824 | 0 | 2048 | 2040 | 8 |
| latest | ab\_antigen/long | mask\_mean | 1.82313 | 0.333577 | 0 | — | — | — |
| latest | ab\_antigen/medium | mask25 | 1.83084 | 0.316872 | 0 | 246 | 243 | 3 |
| latest | ab\_antigen/medium | mask50 | 1.88523 | 0.282869 | 0 | 507 | 502 | 5 |
| latest | ab\_antigen/medium | mask75 | 1.86675 | 0.294195 | 0 | 765 | 758 | 7 |
| latest | ab\_antigen/medium | mask100 | 1.88047 | 0.288386 | 0 | 1024 | 1016 | 8 |
| latest | ab\_antigen/medium | mask\_mean | 1.86582 | 0.295581 | 0 | — | — | — |
| latest | ab\_antigen/short | mask25 | 2.25887 | 0.264706 | 0 | 38 | 34 | 4 |
| latest | ab\_antigen/short | mask50 | 2.17915 | 0.272727 | 0 | 93 | 88 | 5 |
| latest | ab\_antigen/short | mask75 | 2.09627 | 0.242857 | 0 | 145 | 140 | 5 |
| latest | ab\_antigen/short | mask100 | 2.1452 | 0.260417 | 0 | 200 | 192 | 8 |
| latest | ab\_antigen/short | mask\_mean | 2.16987 | 0.260177 | 0 | — | — | — |
| latest | tcr\_peptide/long | mask25 | 1.87826 | 0.236559 | 0 | 466 | 465 | 1 |
| latest | tcr\_peptide/long | mask50 | 1.87828 | 0.271152 | 0 | 985 | 981 | 4 |
| latest | tcr\_peptide/long | mask75 | 1.88667 | 0.280785 | 0 | 1484 | 1478 | 6 |
| latest | tcr\_peptide/long | mask100 | 1.89225 | 0.221078 | 0 | 2048 | 2040 | 8 |
| latest | tcr\_peptide/long | mask\_mean | 1.88386 | 0.252394 | 0 | — | — | — |
| latest | tcr\_peptide/medium | mask25 | 1.99047 | 0.244813 | 0 | 242 | 241 | 1 |
| latest | tcr\_peptide/medium | mask50 | 1.96612 | 0.26494 | 0 | 506 | 502 | 4 |
| latest | tcr\_peptide/medium | mask75 | 1.97836 | 0.271053 | 0 | 767 | 760 | 7 |
| latest | tcr\_peptide/medium | mask100 | 2.00863 | 0.219488 | 0 | 1024 | 1016 | 8 |
| latest | tcr\_peptide/medium | mask\_mean | 1.98589 | 0.250074 | 0 | — | — | — |
| latest | tcr\_peptide/short | mask25 | 2.38988 | 0.25 | 0 | 51 | 48 | 3 |
| latest | tcr\_peptide/short | mask50 | 2.38422 | 0.211538 | 0 | 110 | 104 | 6 |
| latest | tcr\_peptide/short | mask75 | 2.26369 | 0.240876 | 0 | 144 | 137 | 7 |
| latest | tcr\_peptide/short | mask100 | 2.27846 | 0.244792 | 0 | 200 | 192 | 8 |
| latest | tcr\_peptide/short | mask\_mean | 2.32906 | 0.236802 | 0 | — | — | — |
| latest | tcr\_pmhc/long | mask25 | 1.65673 | 0.343284 | 0 | 540 | 536 | 4 |
| latest | tcr\_pmhc/long | mask50 | 1.5733 | 0.360735 | 0 | 1038 | 1034 | 4 |
| latest | tcr\_pmhc/long | mask75 | 1.55468 | 0.393179 | 0 | 1562 | 1554 | 8 |
| latest | tcr\_pmhc/long | mask100 | 1.53989 | 0.412255 | 0 | 2048 | 2040 | 8 |
| latest | tcr\_pmhc/long | mask\_mean | 1.58115 | 0.377363 | 0 | — | — | — |
| latest | tcr\_pmhc/medium | mask25 | 1.48052 | 0.450413 | 0 | 244 | 242 | 2 |
| latest | tcr\_pmhc/medium | mask50 | 1.53415 | 0.413361 | 0 | 484 | 479 | 5 |
| latest | tcr\_pmhc/medium | mask75 | 1.57649 | 0.410923 | 0 | 777 | 769 | 8 |
| latest | tcr\_pmhc/medium | mask100 | 1.65913 | 0.384843 | 0 | 1024 | 1016 | 8 |
| latest | tcr\_pmhc/medium | mask\_mean | 1.56257 | 0.414885 | 0 | — | — | — |
| latest | tcr\_pmhc/short | mask25 | 1.92443 | 0.333333 | 0 | 51 | 48 | 3 |
| latest | tcr\_pmhc/short | mask50 | 1.89302 | 0.325301 | 0 | 86 | 83 | 3 |
| latest | tcr\_pmhc/short | mask75 | 1.96692 | 0.312925 | 0 | 152 | 147 | 5 |
| latest | tcr\_pmhc/short | mask100 | 2.1338 | 0.177083 | 0 | 200 | 192 | 8 |
| latest | tcr\_pmhc/short | mask\_mean | 1.97954 | 0.287161 | 0 | — | — | — |
| latest | unconditional\_ab/long | mask25 | 1.84021 | 0.325234 | 0 | 538 | 535 | 3 |
| latest | unconditional\_ab/long | mask50 | 1.84075 | 0.338432 | 0 | 1050 | 1046 | 4 |
| latest | unconditional\_ab/long | mask75 | 1.85282 | 0.338752 | 0 | 1545 | 1538 | 7 |
| latest | unconditional\_ab/long | mask100 | 2.19912 | 0.336765 | 0 | 2048 | 2040 | 8 |
| latest | unconditional\_ab/long | mask\_mean | 1.93322 | 0.334796 | 0 | — | — | — |
| latest | unconditional\_ab/medium | mask25 | 1.83023 | 0.343874 | 0 | 256 | 253 | 3 |
| latest | unconditional\_ab/medium | mask50 | 1.89713 | 0.34252 | 0 | 512 | 508 | 4 |
| latest | unconditional\_ab/medium | mask75 | 1.87401 | 0.321522 | 0 | 768 | 762 | 6 |
| latest | unconditional\_ab/medium | mask100 | 2.20014 | 0.298228 | 0 | 1024 | 1016 | 8 |
| latest | unconditional\_ab/medium | mask\_mean | 1.95038 | 0.326536 | 0 | — | — | — |
| latest | unconditional\_ab/short | mask25 | 2.1411 | 0.25 | 0 | 55 | 52 | 3 |
| latest | unconditional\_ab/short | mask50 | 2.35395 | 0.234694 | 0 | 103 | 98 | 5 |
| latest | unconditional\_ab/short | mask75 | 2.44411 | 0.22695 | 0 | 148 | 141 | 7 |
| latest | unconditional\_ab/short | mask100 | 2.56348 | 0.229167 | 0 | 200 | 192 | 8 |
| latest | unconditional\_ab/short | mask\_mean | 2.37566 | 0.235203 | 0 | — | — | — |
| latest | unconditional\_single/long | mask25 | 1.65765 | 0.358209 | 0 | 269 | 268 | 1 |
| latest | unconditional\_single/long | mask50 | 1.67761 | 0.327619 | 0 | 527 | 525 | 2 |
| latest | unconditional\_single/long | mask75 | 1.7615 | 0.337209 | 0 | 778 | 774 | 4 |
| latest | unconditional\_single/long | mask100 | 2.13647 | 0.305882 | 0 | 1024 | 1020 | 4 |
| latest | unconditional\_single/long | mask\_mean | 1.80831 | 0.33223 | 0 | — | — | — |
| latest | unconditional\_single/medium | mask25 | 1.69337 | 0.376147 | 0 | 110 | 109 | 1 |
| latest | unconditional\_single/medium | mask50 | 1.69924 | 0.375 | 0 | 234 | 232 | 2 |
| latest | unconditional\_single/medium | mask75 | 1.73357 | 0.362398 | 0 | 371 | 367 | 4 |
| latest | unconditional\_single/medium | mask100 | 2.16843 | 0.330709 | 0 | 512 | 508 | 4 |
| latest | unconditional\_single/medium | mask\_mean | 1.82366 | 0.361063 | 0 | — | — | — |
| latest | unconditional\_single/short | mask25 | 2.65494 | 0.230769 | 0 | 28 | 26 | 2 |
| latest | unconditional\_single/short | mask50 | 2.73855 | 0.2 | 0 | 52 | 50 | 2 |
| latest | unconditional\_single/short | mask75 | 2.45847 | 0.168831 | 0 | 80 | 77 | 3 |
| latest | unconditional\_single/short | mask100 | 2.31766 | 0.291667 | 0 | 100 | 96 | 4 |
| latest | unconditional\_single/short | mask\_mean | 2.54241 | 0.222817 | 0 | — | — | — |
| latest | unconditional\_tcr/long | mask25 | 1.78103 | 0.300971 | 0 | 516 | 515 | 1 |
| latest | unconditional\_tcr/long | mask50 | 1.79659 | 0.305978 | 0 | 989 | 987 | 2 |
| latest | unconditional\_tcr/long | mask75 | 1.83516 | 0.310775 | 0 | 1527 | 1522 | 5 |
| latest | unconditional\_tcr/long | mask100 | 2.18033 | 0.305392 | 0 | 2048 | 2040 | 8 |
| latest | unconditional\_tcr/long | mask\_mean | 1.89828 | 0.305779 | 0 | — | — | — |
| latest | unconditional\_tcr/medium | mask25 | 1.54719 | 0.356275 | — | 247 | 247 | 0 |
| latest | unconditional\_tcr/medium | mask50 | 1.58776 | 0.350101 | 0 | 500 | 497 | 3 |
| latest | unconditional\_tcr/medium | mask75 | 1.63101 | 0.350949 | 0 | 744 | 738 | 6 |
| latest | unconditional\_tcr/medium | mask100 | 2.02614 | 0.345472 | 0 | 1024 | 1016 | 8 |
| latest | unconditional\_tcr/medium | mask\_mean | 1.69803 | 0.350699 | 0 | — | — | — |
| latest | unconditional\_tcr/short | mask25 | 2.26995 | 0.225 | — | 40 | 40 | 0 |
| latest | unconditional\_tcr/short | mask50 | 2.20878 | 0.211111 | 0 | 93 | 90 | 3 |
| latest | unconditional\_tcr/short | mask75 | 2.25791 | 0.220588 | 0 | 142 | 136 | 6 |
| latest | unconditional\_tcr/short | mask100 | 2.42328 | 0.213542 | 0 | 200 | 192 | 8 |
| latest | unconditional\_tcr/short | mask\_mean | 2.28998 | 0.21756 | 0 | — | — | — |
| train\_probe | ab\_antigen/long | mask25 | 1.53978 | 0.4 | — | 530 | 530 | 0 |
| train\_probe | ab\_antigen/long | mask50 | 1.55197 | 0.399612 | 0 | 1034 | 1031 | 3 |
| train\_probe | ab\_antigen/long | mask75 | 1.57111 | 0.384714 | 0 | 1563 | 1557 | 6 |
| train\_probe | ab\_antigen/long | mask100 | 1.65688 | 0.359804 | 0 | 2048 | 2040 | 8 |
| train\_probe | ab\_antigen/long | mask\_mean | 1.57994 | 0.386033 | 0 | — | — | — |
| train\_probe | ab\_antigen/medium | mask25 | 1.94226 | 0.296703 | 0 | 275 | 273 | 2 |
| train\_probe | ab\_antigen/medium | mask50 | 1.92656 | 0.32342 | 0 | 542 | 538 | 4 |
| train\_probe | ab\_antigen/medium | mask75 | 1.90509 | 0.331186 | 0 | 781 | 776 | 5 |
| train\_probe | ab\_antigen/medium | mask100 | 1.91624 | 0.327756 | 0 | 1024 | 1016 | 8 |
| train\_probe | ab\_antigen/medium | mask\_mean | 1.92254 | 0.319766 | 0 | — | — | — |
| train\_probe | ab\_antigen/short | mask25 | 2.10165 | 0.404255 | 0 | 52 | 47 | 5 |
| train\_probe | ab\_antigen/short | mask50 | 2.11373 | 0.284314 | 0 | 109 | 102 | 7 |
| train\_probe | ab\_antigen/short | mask75 | 2.10478 | 0.304054 | 0 | 155 | 148 | 7 |
| train\_probe | ab\_antigen/short | mask100 | 2.1438 | 0.291667 | 0 | 200 | 192 | 8 |
| train\_probe | ab\_antigen/short | mask\_mean | 2.11599 | 0.321072 | 0 | — | — | — |
| train\_probe | tcr\_peptide/long | mask25 | 1.61067 | 0.376518 | 0 | 495 | 494 | 1 |
| train\_probe | tcr\_peptide/long | mask50 | 1.60776 | 0.361868 | 0 | 1033 | 1028 | 5 |
| train\_probe | tcr\_peptide/long | mask75 | 1.65272 | 0.343627 | 0 | 1530 | 1522 | 8 |
| train\_probe | tcr\_peptide/long | mask100 | 1.66589 | 0.353431 | 0 | 2048 | 2040 | 8 |
| train\_probe | tcr\_peptide/long | mask\_mean | 1.63426 | 0.358861 | 0 | — | — | — |
| train\_probe | tcr\_peptide/medium | mask25 | 1.67752 | 0.31746 | — | 252 | 252 | 0 |
| train\_probe | tcr\_peptide/medium | mask50 | 1.73175 | 0.345029 | 0 | 517 | 513 | 4 |
| train\_probe | tcr\_peptide/medium | mask75 | 1.73983 | 0.338196 | 0 | 760 | 754 | 6 |
| train\_probe | tcr\_peptide/medium | mask100 | 1.82658 | 0.305118 | 0 | 1024 | 1016 | 8 |
| train\_probe | tcr\_peptide/medium | mask\_mean | 1.74392 | 0.326451 | 0 | — | — | — |
| train\_probe | tcr\_peptide/short | mask25 | 2.10909 | 0.391304 | 0 | 51 | 46 | 5 |
| train\_probe | tcr\_peptide/short | mask50 | 2.00552 | 0.366667 | 0 | 95 | 90 | 5 |
| train\_probe | tcr\_peptide/short | mask75 | 2.06612 | 0.360294 | 0 | 142 | 136 | 6 |
| train\_probe | tcr\_peptide/short | mask100 | 2.10563 | 0.322917 | 0 | 200 | 192 | 8 |
| train\_probe | tcr\_peptide/short | mask\_mean | 2.07159 | 0.360295 | 0 | — | — | — |
| train\_probe | tcr\_pmhc/long | mask25 | 1.67834 | 0.340996 | 0 | 526 | 522 | 4 |
| train\_probe | tcr\_pmhc/long | mask50 | 1.70989 | 0.317168 | 0 | 1036 | 1031 | 5 |
| train\_probe | tcr\_pmhc/long | mask75 | 1.7191 | 0.285622 | 0 | 1557 | 1551 | 6 |
| train\_probe | tcr\_pmhc/long | mask100 | 1.70841 | 0.308333 | 0 | 2048 | 2040 | 8 |
| train\_probe | tcr\_pmhc/long | mask\_mean | 1.70394 | 0.31303 | 0 | — | — | — |
| train\_probe | tcr\_pmhc/medium | mask25 | 1.5996 | 0.32967 | 0 | 276 | 273 | 3 |
| train\_probe | tcr\_pmhc/medium | mask50 | 1.62417 | 0.341959 | 0 | 549 | 541 | 8 |
| train\_probe | tcr\_pmhc/medium | mask75 | 1.61517 | 0.338854 | 0 | 793 | 785 | 8 |
| train\_probe | tcr\_pmhc/medium | mask100 | 1.6729 | 0.347441 | 0 | 1024 | 1016 | 8 |
| train\_probe | tcr\_pmhc/medium | mask\_mean | 1.62796 | 0.339481 | 0 | — | — | — |
| train\_probe | tcr\_pmhc/short | mask25 | 2.40585 | 0.229167 | 0 | 50 | 48 | 2 |
| train\_probe | tcr\_pmhc/short | mask50 | 2.42332 | 0.2 | 0 | 103 | 100 | 3 |
| train\_probe | tcr\_pmhc/short | mask75 | 2.49462 | 0.144828 | 0 | 152 | 145 | 7 |
| train\_probe | tcr\_pmhc/short | mask100 | 2.37991 | 0.161458 | 0 | 200 | 192 | 8 |
| train\_probe | tcr\_pmhc/short | mask\_mean | 2.42593 | 0.183863 | 0 | — | — | — |
| train\_probe | unconditional\_ab/long | mask25 | 1.63768 | 0.347826 | 0 | 509 | 506 | 3 |
| train\_probe | unconditional\_ab/long | mask50 | 1.64499 | 0.337624 | 0 | 1016 | 1010 | 6 |
| train\_probe | unconditional\_ab/long | mask75 | 1.65392 | 0.320948 | 0 | 1569 | 1561 | 8 |
| train\_probe | unconditional\_ab/long | mask100 | 2.0283 | 0.321078 | 0 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_ab/long | mask\_mean | 1.74122 | 0.331869 | 0 | — | — | — |
| train\_probe | unconditional\_ab/medium | mask25 | 1.83984 | 0.302905 | 0 | 243 | 241 | 2 |
| train\_probe | unconditional\_ab/medium | mask50 | 1.88078 | 0.29845 | 0 | 521 | 516 | 5 |
| train\_probe | unconditional\_ab/medium | mask75 | 1.85987 | 0.319317 | 0 | 766 | 761 | 5 |
| train\_probe | unconditional\_ab/medium | mask100 | 2.11793 | 0.329724 | 0 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_ab/medium | mask\_mean | 1.9246 | 0.312599 | 0 | — | — | — |
| train\_probe | unconditional\_ab/short | mask25 | 2.36907 | 0.153846 | 0 | 53 | 52 | 1 |
| train\_probe | unconditional\_ab/short | mask50 | 2.24456 | 0.211538 | 0 | 106 | 104 | 2 |
| train\_probe | unconditional\_ab/short | mask75 | 2.34498 | 0.202532 | 0 | 163 | 158 | 5 |
| train\_probe | unconditional\_ab/short | mask100 | 2.6777 | 0.197917 | 0 | 200 | 192 | 8 |
| train\_probe | unconditional\_ab/short | mask\_mean | 2.40908 | 0.191458 | 0 | — | — | — |
| train\_probe | unconditional\_single/long | mask25 | 1.79511 | 0.254098 | 0 | 245 | 244 | 1 |
| train\_probe | unconditional\_single/long | mask50 | 1.7751 | 0.287169 | 0 | 492 | 491 | 1 |
| train\_probe | unconditional\_single/long | mask75 | 1.84096 | 0.264323 | 0 | 771 | 768 | 3 |
| train\_probe | unconditional\_single/long | mask100 | 2.26719 | 0.243137 | 0 | 1024 | 1020 | 4 |
| train\_probe | unconditional\_single/long | mask\_mean | 1.91959 | 0.262182 | 0 | — | — | — |
| train\_probe | unconditional\_single/medium | mask25 | 1.76012 | 0.330769 | 0 | 132 | 130 | 2 |
| train\_probe | unconditional\_single/medium | mask50 | 1.79012 | 0.388889 | 0 | 255 | 252 | 3 |
| train\_probe | unconditional\_single/medium | mask75 | 1.76607 | 0.385224 | 0 | 382 | 379 | 3 |
| train\_probe | unconditional\_single/medium | mask100 | 2.06474 | 0.362205 | 0 | 512 | 508 | 4 |
| train\_probe | unconditional\_single/medium | mask\_mean | 1.84526 | 0.366772 | 0 | — | — | — |
| train\_probe | unconditional\_single/short | mask25 | 2.64068 | 0.0769231 | 0 | 28 | 26 | 2 |
| train\_probe | unconditional\_single/short | mask50 | 2.45868 | 0.204082 | 0 | 52 | 49 | 3 |
| train\_probe | unconditional\_single/short | mask75 | 2.44048 | 0.228571 | 0 | 73 | 70 | 3 |
| train\_probe | unconditional\_single/short | mask100 | 2.37219 | 0.229167 | 0 | 100 | 96 | 4 |
| train\_probe | unconditional\_single/short | mask\_mean | 2.47801 | 0.184686 | 0 | — | — | — |
| train\_probe | unconditional\_tcr/long | mask25 | 1.72074 | 0.36 | — | 525 | 525 | 0 |
| train\_probe | unconditional\_tcr/long | mask50 | 1.76069 | 0.34067 | 0 | 1048 | 1045 | 3 |
| train\_probe | unconditional\_tcr/long | mask75 | 1.79425 | 0.322785 | 0 | 1585 | 1580 | 5 |
| train\_probe | unconditional\_tcr/long | mask100 | 2.27279 | 0.27402 | 0 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_tcr/long | mask\_mean | 1.88712 | 0.324369 | 0 | — | — | — |
| train\_probe | unconditional\_tcr/medium | mask25 | 1.64639 | 0.32567 | — | 261 | 261 | 0 |
| train\_probe | unconditional\_tcr/medium | mask50 | 1.65987 | 0.307847 | 0 | 498 | 497 | 1 |
| train\_probe | unconditional\_tcr/medium | mask75 | 1.66931 | 0.332889 | 0 | 755 | 751 | 4 |
| train\_probe | unconditional\_tcr/medium | mask100 | 1.97814 | 0.376969 | 0 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_tcr/medium | mask\_mean | 1.73843 | 0.335844 | 0 | — | — | — |
| train\_probe | unconditional\_tcr/short | mask25 | 2.01692 | 0.217391 | — | 46 | 46 | 0 |
| train\_probe | unconditional\_tcr/short | mask50 | 2.34098 | 0.195652 | 0 | 97 | 92 | 5 |
| train\_probe | unconditional\_tcr/short | mask75 | 2.28485 | 0.219178 | 0 | 152 | 146 | 6 |
| train\_probe | unconditional\_tcr/short | mask100 | 2.61351 | 0.182292 | 0 | 200 | 192 | 8 |
| train\_probe | unconditional\_tcr/short | mask\_mean | 2.31407 | 0.203628 | 0 | — | — | — |

### 1790802160353131370\_r24\_refined\_fixed\_scale\_c6e0089d：分层 mask 指标

源码目录：/vepfs-mlp2/c20250601/251105016/project/dllm\_test/output/immune\_research\_releases/r20\_r27\_20261001；完成/目标 step：25/25。

kind=synthetic\_generalization\_diagnostic；dataset_kind=deterministic\_artificial\_sequences；biological_quality_claim=False。

| 时点 | mask level | 指标 | 值 |
| --- | --- | --- | --- |
| initial | mask25 | mask\_ratio | 0.25 |
| initial | mask25 | selected\_count | 4467 |
| initial | mask25 | aa\_count | 4428 |
| initial | mask25 | terminal\_count | 39 |
| initial | mask25 | loss | 3.81518 |
| initial | mask25 | aa\_loss | 3.8107 |
| initial | mask25 | aa\_accuracy | 0.0589431 |
| initial | mask25 | terminal\_loss | 4.32412 |
| initial | mask25 | terminal\_accuracy | 0 |
| initial | mask50 | mask\_ratio | 0.5 |
| initial | mask50 | selected\_count | 8888 |
| initial | mask50 | aa\_count | 8819 |
| initial | mask50 | terminal\_count | 69 |
| initial | mask50 | loss | 3.85226 |
| initial | mask50 | aa\_loss | 3.8493 |
| initial | mask50 | aa\_accuracy | 0.0426352 |
| initial | mask50 | terminal\_loss | 4.23099 |
| initial | mask50 | terminal\_accuracy | 0 |
| initial | mask75 | mask\_ratio | 0.75 |
| initial | mask75 | selected\_count | 13437 |
| initial | mask75 | aa\_count | 13330 |
| initial | mask75 | terminal\_count | 107 |
| initial | mask75 | loss | 3.83517 |
| initial | mask75 | aa\_loss | 3.83171 |
| initial | mask75 | aa\_accuracy | 0.0306827 |
| initial | mask75 | terminal\_loss | 4.26675 |
| initial | mask75 | terminal\_accuracy | 0 |
| initial | mask100 | mask\_ratio | 1 |
| initial | mask100 | selected\_count | 17996 |
| initial | mask100 | aa\_count | 17864 |
| initial | mask100 | terminal\_count | 132 |
| initial | mask100 | loss | 3.80145 |
| initial | mask100 | aa\_loss | 3.79761 |
| initial | mask100 | aa\_accuracy | 0.0301164 |
| initial | mask100 | terminal\_loss | 4.32137 |
| initial | mask100 | terminal\_accuracy | 0 |
| initial | mask\_mean | loss | 3.82602 |
| initial | mask\_mean | aa\_loss | 3.82233 |
| initial | mask\_mean | terminal\_loss | 4.28581 |
| initial | mask\_mean | aa\_accuracy | 0.0405944 |
| initial | mask\_mean | terminal\_accuracy | 0 |
| latest | mask25 | mask\_ratio | 0.25 |
| latest | mask25 | selected\_count | 4467 |
| latest | mask25 | aa\_count | 4428 |
| latest | mask25 | terminal\_count | 39 |
| latest | mask25 | loss | 1.79634 |
| latest | mask25 | aa\_loss | 1.76574 |
| latest | mask25 | aa\_accuracy | 0.320687 |
| latest | mask25 | terminal\_loss | 5.27028 |
| latest | mask25 | terminal\_accuracy | 0 |
| latest | mask50 | mask\_ratio | 0.5 |
| latest | mask50 | selected\_count | 8888 |
| latest | mask50 | aa\_count | 8819 |
| latest | mask50 | terminal\_count | 69 |
| latest | mask50 | loss | 1.79961 |
| latest | mask50 | aa\_loss | 1.77373 |
| latest | mask50 | aa\_accuracy | 0.32464 |
| latest | mask50 | terminal\_loss | 5.10825 |
| latest | mask50 | terminal\_accuracy | 0 |
| latest | mask75 | mask\_ratio | 0.75 |
| latest | mask75 | selected\_count | 13437 |
| latest | mask75 | aa\_count | 13330 |
| latest | mask75 | terminal\_count | 107 |
| latest | mask75 | loss | 1.80997 |
| latest | mask75 | aa\_loss | 1.7847 |
| latest | mask75 | aa\_accuracy | 0.326482 |
| latest | mask75 | terminal\_loss | 4.95807 |
| latest | mask75 | terminal\_accuracy | 0 |
| latest | mask100 | mask\_ratio | 1 |
| latest | mask100 | selected\_count | 17996 |
| latest | mask100 | aa\_count | 17864 |
| latest | mask100 | terminal\_count | 132 |
| latest | mask100 | loss | 1.97552 |
| latest | mask100 | aa\_loss | 1.95804 |
| latest | mask100 | aa\_accuracy | 0.31236 |
| latest | mask100 | terminal\_loss | 4.34235 |
| latest | mask100 | terminal\_accuracy | 0 |
| latest | mask\_mean | loss | 1.84536 |
| latest | mask\_mean | aa\_loss | 1.82055 |
| latest | mask\_mean | terminal\_loss | 4.91974 |
| latest | mask\_mean | aa\_accuracy | 0.321042 |
| latest | mask\_mean | terminal\_accuracy | 0 |

评测曲线（原始 step，四档等权均值）：

| step | mean loss | mean AA acc | mean end acc | gradient norm |
| --- | --- | --- | --- | --- |
| 25 | 1.84536 | 0.321042 | 0 | 1.5845 |

有条件/无条件曲线（每档按token计数合并，再平均四档；缺失分组不推断数值）：

| step | 子集 | mean loss | mean AA | mean end | Full-mask AA | Full-mask end | Full-mask AA count | Full-mask end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | conditional | 3.79117 | 0.0444692 | 0 | 0.0298645 | 0 | 9744 | 72 |
| initial | unconditional | 3.86732 | 0.0360518 | 0 | 0.0304187 | 0 | 8120 | 60 |
| 25 | conditional | 1.79789 | 0.31892 | 0 | 0.310242 | 0 | 9744 | 72 |
| 25 | unconditional | 1.90239 | 0.323527 | 0 | 0.314901 | 0 | 8120 | 60 |

人工数据 manifest：d40bbbbc1c4294eed843d88bcdd0cf193886de6c2d7e9db9069bbf86e931c022；精确 target sequence 重叠=0；精确 record 重叠=0。

跨 radius32 覆盖：{&quot;possible&quot;: true, &quot;train&quot;: true, &quot;valid&quot;: true}。生成规则共享是诊断设计的一部分，不能当作真实下游去重证书。


按任务/链长分组（全局 level 按 token 计数汇总，下面保留各组计数；不同组不得直接比较未经加权的总 loss）：

| 时点 | group/length | mask level | loss | AA acc | end acc | selected count | AA count | end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | ab\_antigen/long | mask25 | 4.10241 | 0 | 0 | 530 | 526 | 4 |
| initial | ab\_antigen/long | mask50 | 4.14621 | 0 | 0 | 1019 | 1013 | 6 |
| initial | ab\_antigen/long | mask75 | 4.10335 | 0 | 0 | 1538 | 1532 | 6 |
| initial | ab\_antigen/long | mask100 | 4.03516 | 0 | 0 | 2048 | 2040 | 8 |
| initial | ab\_antigen/long | mask\_mean | 4.09678 | 0 | 0 | — | — | — |
| initial | ab\_antigen/medium | mask25 | 4.1109 | 0 | 0 | 246 | 243 | 3 |
| initial | ab\_antigen/medium | mask50 | 4.0947 | 0 | 0 | 507 | 502 | 5 |
| initial | ab\_antigen/medium | mask75 | 4.07023 | 0 | 0 | 765 | 758 | 7 |
| initial | ab\_antigen/medium | mask100 | 4.03235 | 0 | 0 | 1024 | 1016 | 8 |
| initial | ab\_antigen/medium | mask\_mean | 4.07704 | 0 | 0 | — | — | — |
| initial | ab\_antigen/short | mask25 | 3.94668 | 0.0588235 | 0 | 38 | 34 | 4 |
| initial | ab\_antigen/short | mask50 | 3.8665 | 0.0454545 | 0 | 93 | 88 | 5 |
| initial | ab\_antigen/short | mask75 | 3.78397 | 0.05 | 0 | 145 | 140 | 5 |
| initial | ab\_antigen/short | mask100 | 3.7871 | 0.046875 | 0 | 200 | 192 | 8 |
| initial | ab\_antigen/short | mask\_mean | 3.84606 | 0.0502883 | 0 | — | — | — |
| initial | tcr\_peptide/long | mask25 | 3.70198 | 0.0258065 | 0 | 466 | 465 | 1 |
| initial | tcr\_peptide/long | mask50 | 3.77676 | 0.011213 | 0 | 985 | 981 | 4 |
| initial | tcr\_peptide/long | mask75 | 3.77699 | 0.0121786 | 0 | 1484 | 1478 | 6 |
| initial | tcr\_peptide/long | mask100 | 3.76535 | 0.0151961 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_peptide/long | mask\_mean | 3.75527 | 0.0160985 | 0 | — | — | — |
| initial | tcr\_peptide/medium | mask25 | 3.97216 | 0.00829876 | 0 | 242 | 241 | 1 |
| initial | tcr\_peptide/medium | mask50 | 3.90908 | 0.00996016 | 0 | 506 | 502 | 4 |
| initial | tcr\_peptide/medium | mask75 | 3.85215 | 0.0118421 | 0 | 767 | 760 | 7 |
| initial | tcr\_peptide/medium | mask100 | 3.78247 | 0.0147638 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_peptide/medium | mask\_mean | 3.87897 | 0.0112162 | 0 | — | — | — |
| initial | tcr\_peptide/short | mask25 | 3.64878 | 0.0625 | 0 | 51 | 48 | 3 |
| initial | tcr\_peptide/short | mask50 | 3.61911 | 0.0865385 | 0 | 110 | 104 | 6 |
| initial | tcr\_peptide/short | mask75 | 3.66048 | 0.0656934 | 0 | 144 | 137 | 7 |
| initial | tcr\_peptide/short | mask100 | 3.65702 | 0.046875 | 0 | 200 | 192 | 8 |
| initial | tcr\_peptide/short | mask\_mean | 3.64635 | 0.0654017 | 0 | — | — | — |
| initial | tcr\_pmhc/long | mask25 | 3.19962 | 0.266791 | 0 | 540 | 536 | 4 |
| initial | tcr\_pmhc/long | mask50 | 3.30848 | 0.185687 | 0 | 1038 | 1034 | 4 |
| initial | tcr\_pmhc/long | mask75 | 3.37315 | 0.117761 | 0 | 1562 | 1554 | 8 |
| initial | tcr\_pmhc/long | mask100 | 3.41892 | 0.109804 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_pmhc/long | mask\_mean | 3.32504 | 0.170011 | 0 | — | — | — |
| initial | tcr\_pmhc/medium | mask25 | 3.82185 | 0.0165289 | 0 | 244 | 242 | 2 |
| initial | tcr\_pmhc/medium | mask50 | 3.84317 | 0.00626305 | 0 | 484 | 479 | 5 |
| initial | tcr\_pmhc/medium | mask75 | 3.85431 | 0 | 0 | 777 | 769 | 8 |
| initial | tcr\_pmhc/medium | mask100 | 3.8816 | 0 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_pmhc/medium | mask\_mean | 3.85023 | 0.00569799 | 0 | — | — | — |
| initial | tcr\_pmhc/short | mask25 | 3.76194 | 0 | 0 | 51 | 48 | 3 |
| initial | tcr\_pmhc/short | mask50 | 3.81972 | 0.0120482 | 0 | 86 | 83 | 3 |
| initial | tcr\_pmhc/short | mask75 | 3.80688 | 0.0136054 | 0 | 152 | 147 | 5 |
| initial | tcr\_pmhc/short | mask100 | 3.83313 | 0.015625 | 0 | 200 | 192 | 8 |
| initial | tcr\_pmhc/short | mask\_mean | 3.80542 | 0.0103197 | 0 | — | — | — |
| initial | unconditional\_ab/long | mask25 | 3.9701 | 0.0616822 | 0 | 538 | 535 | 3 |
| initial | unconditional\_ab/long | mask50 | 3.92095 | 0.0602294 | 0 | 1050 | 1046 | 4 |
| initial | unconditional\_ab/long | mask75 | 3.88596 | 0.0318596 | 0 | 1545 | 1538 | 7 |
| initial | unconditional\_ab/long | mask100 | 3.79481 | 0.0308824 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_ab/long | mask\_mean | 3.89296 | 0.0461634 | 0 | — | — | — |
| initial | unconditional\_ab/medium | mask25 | 3.7778 | 0.0592885 | 0 | 256 | 253 | 3 |
| initial | unconditional\_ab/medium | mask50 | 3.8424 | 0.0393701 | 0 | 512 | 508 | 4 |
| initial | unconditional\_ab/medium | mask75 | 3.8188 | 0.0354331 | 0 | 768 | 762 | 6 |
| initial | unconditional\_ab/medium | mask100 | 3.7911 | 0.0305118 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_ab/medium | mask\_mean | 3.80753 | 0.0411509 | 0 | — | — | — |
| initial | unconditional\_ab/short | mask25 | 3.85957 | 0.0576923 | 0 | 55 | 52 | 3 |
| initial | unconditional\_ab/short | mask50 | 3.84337 | 0.0306122 | 0 | 103 | 98 | 5 |
| initial | unconditional\_ab/short | mask75 | 3.80081 | 0.035461 | 0 | 148 | 141 | 7 |
| initial | unconditional\_ab/short | mask100 | 3.75868 | 0.03125 | 0 | 200 | 192 | 8 |
| initial | unconditional\_ab/short | mask\_mean | 3.8156 | 0.0387539 | 0 | — | — | — |
| initial | unconditional\_single/long | mask25 | 3.94339 | 0.0485075 | 0 | 269 | 268 | 1 |
| initial | unconditional\_single/long | mask50 | 4.00614 | 0.0514286 | 0 | 527 | 525 | 2 |
| initial | unconditional\_single/long | mask75 | 3.96455 | 0.0516796 | 0 | 778 | 774 | 4 |
| initial | unconditional\_single/long | mask100 | 3.88226 | 0.0617647 | 0 | 1024 | 1020 | 4 |
| initial | unconditional\_single/long | mask\_mean | 3.94909 | 0.0533451 | 0 | — | — | — |
| initial | unconditional\_single/medium | mask25 | 4.15354 | 0.0366972 | 0 | 110 | 109 | 1 |
| initial | unconditional\_single/medium | mask50 | 4.00297 | 0.0258621 | 0 | 234 | 232 | 2 |
| initial | unconditional\_single/medium | mask75 | 3.95779 | 0.0245232 | 0 | 371 | 367 | 4 |
| initial | unconditional\_single/medium | mask100 | 3.90136 | 0.0295276 | 0 | 512 | 508 | 4 |
| initial | unconditional\_single/medium | mask\_mean | 4.00392 | 0.0291525 | 0 | — | — | — |
| initial | unconditional\_single/short | mask25 | 3.41472 | 0.269231 | 0 | 28 | 26 | 2 |
| initial | unconditional\_single/short | mask50 | 3.5961 | 0.18 | 0 | 52 | 50 | 2 |
| initial | unconditional\_single/short | mask75 | 3.42945 | 0.207792 | 0 | 80 | 77 | 3 |
| initial | unconditional\_single/short | mask100 | 3.41298 | 0.21875 | 0 | 100 | 96 | 4 |
| initial | unconditional\_single/short | mask\_mean | 3.46331 | 0.218943 | 0 | — | — | — |
| initial | unconditional\_tcr/long | mask25 | 3.81511 | 0.0291262 | 0 | 516 | 515 | 1 |
| initial | unconditional\_tcr/long | mask50 | 3.90053 | 0.0151976 | 0 | 989 | 987 | 2 |
| initial | unconditional\_tcr/long | mask75 | 3.85577 | 0.0157687 | 0 | 1527 | 1522 | 5 |
| initial | unconditional\_tcr/long | mask100 | 3.77771 | 0.0156863 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_tcr/long | mask\_mean | 3.83728 | 0.0189447 | 0 | — | — | — |
| initial | unconditional\_tcr/medium | mask25 | 3.74311 | 0.0202429 | — | 247 | 247 | 0 |
| initial | unconditional\_tcr/medium | mask50 | 3.86175 | 0.0160966 | 0 | 500 | 497 | 3 |
| initial | unconditional\_tcr/medium | mask75 | 3.85697 | 0.0149051 | 0 | 744 | 738 | 6 |
| initial | unconditional\_tcr/medium | mask100 | 3.8777 | 0.015748 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_tcr/medium | mask\_mean | 3.83488 | 0.0167482 | 0 | — | — | — |
| initial | unconditional\_tcr/short | mask25 | 4.01228 | 0 | — | 40 | 40 | 0 |
| initial | unconditional\_tcr/short | mask50 | 3.82482 | 0 | 0 | 93 | 90 | 3 |
| initial | unconditional\_tcr/short | mask75 | 3.85293 | 0 | 0 | 142 | 136 | 6 |
| initial | unconditional\_tcr/short | mask100 | 3.86799 | 0 | 0 | 200 | 192 | 8 |
| initial | unconditional\_tcr/short | mask\_mean | 3.8895 | 0 | 0 | — | — | — |
| latest | ab\_antigen/long | mask25 | 1.85469 | 0.326996 | 0 | 530 | 526 | 4 |
| latest | ab\_antigen/long | mask50 | 1.81471 | 0.344521 | 0 | 1019 | 1013 | 6 |
| latest | ab\_antigen/long | mask75 | 1.81332 | 0.33094 | 0 | 1538 | 1532 | 6 |
| latest | ab\_antigen/long | mask100 | 1.81031 | 0.333824 | 0 | 2048 | 2040 | 8 |
| latest | ab\_antigen/long | mask\_mean | 1.82326 | 0.33407 | 0 | — | — | — |
| latest | ab\_antigen/medium | mask25 | 1.83256 | 0.320988 | 0 | 246 | 243 | 3 |
| latest | ab\_antigen/medium | mask50 | 1.88448 | 0.296813 | 0 | 507 | 502 | 5 |
| latest | ab\_antigen/medium | mask75 | 1.86698 | 0.291557 | 0 | 765 | 758 | 7 |
| latest | ab\_antigen/medium | mask100 | 1.88072 | 0.291339 | 0 | 1024 | 1016 | 8 |
| latest | ab\_antigen/medium | mask\_mean | 1.86619 | 0.300174 | 0 | — | — | — |
| latest | ab\_antigen/short | mask25 | 2.2597 | 0.264706 | 0 | 38 | 34 | 4 |
| latest | ab\_antigen/short | mask50 | 2.18047 | 0.261364 | 0 | 93 | 88 | 5 |
| latest | ab\_antigen/short | mask75 | 2.09577 | 0.235714 | 0 | 145 | 140 | 5 |
| latest | ab\_antigen/short | mask100 | 2.14515 | 0.255208 | 0 | 200 | 192 | 8 |
| latest | ab\_antigen/short | mask\_mean | 2.17027 | 0.254248 | 0 | — | — | — |
| latest | tcr\_peptide/long | mask25 | 1.87558 | 0.234409 | 0 | 466 | 465 | 1 |
| latest | tcr\_peptide/long | mask50 | 1.87644 | 0.269113 | 0 | 985 | 981 | 4 |
| latest | tcr\_peptide/long | mask75 | 1.88513 | 0.282138 | 0 | 1484 | 1478 | 6 |
| latest | tcr\_peptide/long | mask100 | 1.89082 | 0.220588 | 0 | 2048 | 2040 | 8 |
| latest | tcr\_peptide/long | mask\_mean | 1.88199 | 0.251562 | 0 | — | — | — |
| latest | tcr\_peptide/medium | mask25 | 1.98935 | 0.244813 | 0 | 242 | 241 | 1 |
| latest | tcr\_peptide/medium | mask50 | 1.96571 | 0.26494 | 0 | 506 | 502 | 4 |
| latest | tcr\_peptide/medium | mask75 | 1.97957 | 0.272368 | 0 | 767 | 760 | 7 |
| latest | tcr\_peptide/medium | mask100 | 2.0093 | 0.220472 | 0 | 1024 | 1016 | 8 |
| latest | tcr\_peptide/medium | mask\_mean | 1.98598 | 0.250649 | 0 | — | — | — |
| latest | tcr\_peptide/short | mask25 | 2.38956 | 0.25 | 0 | 51 | 48 | 3 |
| latest | tcr\_peptide/short | mask50 | 2.38439 | 0.211538 | 0 | 110 | 104 | 6 |
| latest | tcr\_peptide/short | mask75 | 2.26393 | 0.255474 | 0 | 144 | 137 | 7 |
| latest | tcr\_peptide/short | mask100 | 2.27829 | 0.244792 | 0 | 200 | 192 | 8 |
| latest | tcr\_peptide/short | mask\_mean | 2.32904 | 0.240451 | 0 | — | — | — |
| latest | tcr\_pmhc/long | mask25 | 1.65498 | 0.347015 | 0 | 540 | 536 | 4 |
| latest | tcr\_pmhc/long | mask50 | 1.57199 | 0.365571 | 0 | 1038 | 1034 | 4 |
| latest | tcr\_pmhc/long | mask75 | 1.5532 | 0.389961 | 0 | 1562 | 1554 | 8 |
| latest | tcr\_pmhc/long | mask100 | 1.53762 | 0.415196 | 0 | 2048 | 2040 | 8 |
| latest | tcr\_pmhc/long | mask\_mean | 1.57945 | 0.379436 | 0 | — | — | — |
| latest | tcr\_pmhc/medium | mask25 | 1.47937 | 0.450413 | 0 | 244 | 242 | 2 |
| latest | tcr\_pmhc/medium | mask50 | 1.53436 | 0.407098 | 0 | 484 | 479 | 5 |
| latest | tcr\_pmhc/medium | mask75 | 1.57715 | 0.420026 | 0 | 777 | 769 | 8 |
| latest | tcr\_pmhc/medium | mask100 | 1.65983 | 0.387795 | 0 | 1024 | 1016 | 8 |
| latest | tcr\_pmhc/medium | mask\_mean | 1.56268 | 0.416333 | 0 | — | — | — |
| latest | tcr\_pmhc/short | mask25 | 1.92634 | 0.333333 | 0 | 51 | 48 | 3 |
| latest | tcr\_pmhc/short | mask50 | 1.89303 | 0.325301 | 0 | 86 | 83 | 3 |
| latest | tcr\_pmhc/short | mask75 | 1.96623 | 0.29932 | 0 | 152 | 147 | 5 |
| latest | tcr\_pmhc/short | mask100 | 2.13199 | 0.182292 | 0 | 200 | 192 | 8 |
| latest | tcr\_pmhc/short | mask\_mean | 1.9794 | 0.285061 | 0 | — | — | — |
| latest | unconditional\_ab/long | mask25 | 1.84011 | 0.327103 | 0 | 538 | 535 | 3 |
| latest | unconditional\_ab/long | mask50 | 1.84086 | 0.337476 | 0 | 1050 | 1046 | 4 |
| latest | unconditional\_ab/long | mask75 | 1.85197 | 0.338752 | 0 | 1545 | 1538 | 7 |
| latest | unconditional\_ab/long | mask100 | 2.19956 | 0.336765 | 0 | 2048 | 2040 | 8 |
| latest | unconditional\_ab/long | mask\_mean | 1.93313 | 0.335024 | 0 | — | — | — |
| latest | unconditional\_ab/medium | mask25 | 1.83025 | 0.343874 | 0 | 256 | 253 | 3 |
| latest | unconditional\_ab/medium | mask50 | 1.89585 | 0.34252 | 0 | 512 | 508 | 4 |
| latest | unconditional\_ab/medium | mask75 | 1.87263 | 0.321522 | 0 | 768 | 762 | 6 |
| latest | unconditional\_ab/medium | mask100 | 2.20289 | 0.298228 | 0 | 1024 | 1016 | 8 |
| latest | unconditional\_ab/medium | mask\_mean | 1.9504 | 0.326536 | 0 | — | — | — |
| latest | unconditional\_ab/short | mask25 | 2.14351 | 0.25 | 0 | 55 | 52 | 3 |
| latest | unconditional\_ab/short | mask50 | 2.35324 | 0.234694 | 0 | 103 | 98 | 5 |
| latest | unconditional\_ab/short | mask75 | 2.44297 | 0.22695 | 0 | 148 | 141 | 7 |
| latest | unconditional\_ab/short | mask100 | 2.56365 | 0.229167 | 0 | 200 | 192 | 8 |
| latest | unconditional\_ab/short | mask\_mean | 2.37584 | 0.235203 | 0 | — | — | — |
| latest | unconditional\_single/long | mask25 | 1.65599 | 0.358209 | 0 | 269 | 268 | 1 |
| latest | unconditional\_single/long | mask50 | 1.67764 | 0.331429 | 0 | 527 | 525 | 2 |
| latest | unconditional\_single/long | mask75 | 1.76148 | 0.334625 | 0 | 778 | 774 | 4 |
| latest | unconditional\_single/long | mask100 | 2.13788 | 0.305882 | 0 | 1024 | 1020 | 4 |
| latest | unconditional\_single/long | mask\_mean | 1.80825 | 0.332536 | 0 | — | — | — |
| latest | unconditional\_single/medium | mask25 | 1.68792 | 0.376147 | 0 | 110 | 109 | 1 |
| latest | unconditional\_single/medium | mask50 | 1.69762 | 0.375 | 0 | 234 | 232 | 2 |
| latest | unconditional\_single/medium | mask75 | 1.73296 | 0.356948 | 0 | 371 | 367 | 4 |
| latest | unconditional\_single/medium | mask100 | 2.16916 | 0.330709 | 0 | 512 | 508 | 4 |
| latest | unconditional\_single/medium | mask\_mean | 1.82191 | 0.359701 | 0 | — | — | — |
| latest | unconditional\_single/short | mask25 | 2.64973 | 0.230769 | 0 | 28 | 26 | 2 |
| latest | unconditional\_single/short | mask50 | 2.73402 | 0.2 | 0 | 52 | 50 | 2 |
| latest | unconditional\_single/short | mask75 | 2.45579 | 0.168831 | 0 | 80 | 77 | 3 |
| latest | unconditional\_single/short | mask100 | 2.3176 | 0.291667 | 0 | 100 | 96 | 4 |
| latest | unconditional\_single/short | mask\_mean | 2.53929 | 0.222817 | 0 | — | — | — |
| latest | unconditional\_tcr/long | mask25 | 1.7823 | 0.300971 | 0 | 516 | 515 | 1 |
| latest | unconditional\_tcr/long | mask50 | 1.79706 | 0.31307 | 0 | 989 | 987 | 2 |
| latest | unconditional\_tcr/long | mask75 | 1.83582 | 0.30749 | 0 | 1527 | 1522 | 5 |
| latest | unconditional\_tcr/long | mask100 | 2.17896 | 0.305392 | 0 | 2048 | 2040 | 8 |
| latest | unconditional\_tcr/long | mask\_mean | 1.89854 | 0.306731 | 0 | — | — | — |
| latest | unconditional\_tcr/medium | mask25 | 1.54755 | 0.356275 | — | 247 | 247 | 0 |
| latest | unconditional\_tcr/medium | mask50 | 1.58677 | 0.350101 | 0 | 500 | 497 | 3 |
| latest | unconditional\_tcr/medium | mask75 | 1.63027 | 0.350949 | 0 | 744 | 738 | 6 |
| latest | unconditional\_tcr/medium | mask100 | 2.02796 | 0.345472 | 0 | 1024 | 1016 | 8 |
| latest | unconditional\_tcr/medium | mask\_mean | 1.69814 | 0.350699 | 0 | — | — | — |
| latest | unconditional\_tcr/short | mask25 | 2.26751 | 0.225 | — | 40 | 40 | 0 |
| latest | unconditional\_tcr/short | mask50 | 2.20471 | 0.211111 | 0 | 93 | 90 | 3 |
| latest | unconditional\_tcr/short | mask75 | 2.25451 | 0.227941 | 0 | 142 | 136 | 6 |
| latest | unconditional\_tcr/short | mask100 | 2.42294 | 0.213542 | 0 | 200 | 192 | 8 |
| latest | unconditional\_tcr/short | mask\_mean | 2.28742 | 0.219398 | 0 | — | — | — |
| train\_probe | ab\_antigen/long | mask25 | 1.53998 | 0.4 | — | 530 | 530 | 0 |
| train\_probe | ab\_antigen/long | mask50 | 1.55125 | 0.398642 | 0 | 1034 | 1031 | 3 |
| train\_probe | ab\_antigen/long | mask75 | 1.57039 | 0.382787 | 0 | 1563 | 1557 | 6 |
| train\_probe | ab\_antigen/long | mask100 | 1.6566 | 0.361275 | 0 | 2048 | 2040 | 8 |
| train\_probe | ab\_antigen/long | mask\_mean | 1.57956 | 0.385676 | 0 | — | — | — |
| train\_probe | ab\_antigen/medium | mask25 | 1.94161 | 0.296703 | 0 | 275 | 273 | 2 |
| train\_probe | ab\_antigen/medium | mask50 | 1.9254 | 0.321561 | 0 | 542 | 538 | 4 |
| train\_probe | ab\_antigen/medium | mask75 | 1.9032 | 0.335052 | 0 | 781 | 776 | 5 |
| train\_probe | ab\_antigen/medium | mask100 | 1.91571 | 0.329724 | 0 | 1024 | 1016 | 8 |
| train\_probe | ab\_antigen/medium | mask\_mean | 1.92148 | 0.32076 | 0 | — | — | — |
| train\_probe | ab\_antigen/short | mask25 | 2.09968 | 0.382979 | 0 | 52 | 47 | 5 |
| train\_probe | ab\_antigen/short | mask50 | 2.1154 | 0.284314 | 0 | 109 | 102 | 7 |
| train\_probe | ab\_antigen/short | mask75 | 2.10511 | 0.310811 | 0 | 155 | 148 | 7 |
| train\_probe | ab\_antigen/short | mask100 | 2.14411 | 0.291667 | 0 | 200 | 192 | 8 |
| train\_probe | ab\_antigen/short | mask\_mean | 2.11607 | 0.317442 | 0 | — | — | — |
| train\_probe | tcr\_peptide/long | mask25 | 1.60954 | 0.376518 | 0 | 495 | 494 | 1 |
| train\_probe | tcr\_peptide/long | mask50 | 1.60873 | 0.363813 | 0 | 1033 | 1028 | 5 |
| train\_probe | tcr\_peptide/long | mask75 | 1.65365 | 0.343627 | 0 | 1530 | 1522 | 8 |
| train\_probe | tcr\_peptide/long | mask100 | 1.66558 | 0.355392 | 0 | 2048 | 2040 | 8 |
| train\_probe | tcr\_peptide/long | mask\_mean | 1.63437 | 0.359838 | 0 | — | — | — |
| train\_probe | tcr\_peptide/medium | mask25 | 1.678 | 0.31746 | — | 252 | 252 | 0 |
| train\_probe | tcr\_peptide/medium | mask50 | 1.73051 | 0.341131 | 0 | 517 | 513 | 4 |
| train\_probe | tcr\_peptide/medium | mask75 | 1.73932 | 0.344828 | 0 | 760 | 754 | 6 |
| train\_probe | tcr\_peptide/medium | mask100 | 1.82673 | 0.311024 | 0 | 1024 | 1016 | 8 |
| train\_probe | tcr\_peptide/medium | mask\_mean | 1.74364 | 0.328611 | 0 | — | — | — |
| train\_probe | tcr\_peptide/short | mask25 | 2.10999 | 0.391304 | 0 | 51 | 46 | 5 |
| train\_probe | tcr\_peptide/short | mask50 | 2.00442 | 0.366667 | 0 | 95 | 90 | 5 |
| train\_probe | tcr\_peptide/short | mask75 | 2.06431 | 0.360294 | 0 | 142 | 136 | 6 |
| train\_probe | tcr\_peptide/short | mask100 | 2.10317 | 0.322917 | 0 | 200 | 192 | 8 |
| train\_probe | tcr\_peptide/short | mask\_mean | 2.07047 | 0.360295 | 0 | — | — | — |
| train\_probe | tcr\_pmhc/long | mask25 | 1.67836 | 0.346743 | 0 | 526 | 522 | 4 |
| train\_probe | tcr\_pmhc/long | mask50 | 1.70926 | 0.321048 | 0 | 1036 | 1031 | 5 |
| train\_probe | tcr\_pmhc/long | mask75 | 1.71735 | 0.286912 | 0 | 1557 | 1551 | 6 |
| train\_probe | tcr\_pmhc/long | mask100 | 1.70584 | 0.313726 | 0 | 2048 | 2040 | 8 |
| train\_probe | tcr\_pmhc/long | mask\_mean | 1.7027 | 0.317107 | 0 | — | — | — |
| train\_probe | tcr\_pmhc/medium | mask25 | 1.59979 | 0.326007 | 0 | 276 | 273 | 3 |
| train\_probe | tcr\_pmhc/medium | mask50 | 1.62487 | 0.338262 | 0 | 549 | 541 | 8 |
| train\_probe | tcr\_pmhc/medium | mask75 | 1.61411 | 0.338854 | 0 | 793 | 785 | 8 |
| train\_probe | tcr\_pmhc/medium | mask100 | 1.67147 | 0.351378 | 0 | 1024 | 1016 | 8 |
| train\_probe | tcr\_pmhc/medium | mask\_mean | 1.62756 | 0.338625 | 0 | — | — | — |
| train\_probe | tcr\_pmhc/short | mask25 | 2.4019 | 0.229167 | 0 | 50 | 48 | 2 |
| train\_probe | tcr\_pmhc/short | mask50 | 2.42004 | 0.19 | 0 | 103 | 100 | 3 |
| train\_probe | tcr\_pmhc/short | mask75 | 2.49158 | 0.151724 | 0 | 152 | 145 | 7 |
| train\_probe | tcr\_pmhc/short | mask100 | 2.37818 | 0.161458 | 0 | 200 | 192 | 8 |
| train\_probe | tcr\_pmhc/short | mask\_mean | 2.42292 | 0.183087 | 0 | — | — | — |
| train\_probe | unconditional\_ab/long | mask25 | 1.63649 | 0.347826 | 0 | 509 | 506 | 3 |
| train\_probe | unconditional\_ab/long | mask50 | 1.64406 | 0.334653 | 0 | 1016 | 1010 | 6 |
| train\_probe | unconditional\_ab/long | mask75 | 1.65193 | 0.323511 | 0 | 1569 | 1561 | 8 |
| train\_probe | unconditional\_ab/long | mask100 | 2.02703 | 0.321078 | 0 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_ab/long | mask\_mean | 1.73988 | 0.331767 | 0 | — | — | — |
| train\_probe | unconditional\_ab/medium | mask25 | 1.83891 | 0.307054 | 0 | 243 | 241 | 2 |
| train\_probe | unconditional\_ab/medium | mask50 | 1.87926 | 0.296512 | 0 | 521 | 516 | 5 |
| train\_probe | unconditional\_ab/medium | mask75 | 1.85871 | 0.324573 | 0 | 766 | 761 | 5 |
| train\_probe | unconditional\_ab/medium | mask100 | 2.11983 | 0.329724 | 0 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_ab/medium | mask\_mean | 1.92418 | 0.314466 | 0 | — | — | — |
| train\_probe | unconditional\_ab/short | mask25 | 2.36855 | 0.153846 | 0 | 53 | 52 | 1 |
| train\_probe | unconditional\_ab/short | mask50 | 2.24506 | 0.211538 | 0 | 106 | 104 | 2 |
| train\_probe | unconditional\_ab/short | mask75 | 2.34345 | 0.202532 | 0 | 163 | 158 | 5 |
| train\_probe | unconditional\_ab/short | mask100 | 2.67662 | 0.197917 | 0 | 200 | 192 | 8 |
| train\_probe | unconditional\_ab/short | mask\_mean | 2.40842 | 0.191458 | 0 | — | — | — |
| train\_probe | unconditional\_single/long | mask25 | 1.79578 | 0.254098 | 0 | 245 | 244 | 1 |
| train\_probe | unconditional\_single/long | mask50 | 1.77423 | 0.287169 | 0 | 492 | 491 | 1 |
| train\_probe | unconditional\_single/long | mask75 | 1.83981 | 0.265625 | 0 | 771 | 768 | 3 |
| train\_probe | unconditional\_single/long | mask100 | 2.26655 | 0.243137 | 0 | 1024 | 1020 | 4 |
| train\_probe | unconditional\_single/long | mask\_mean | 1.91909 | 0.262507 | 0 | — | — | — |
| train\_probe | unconditional\_single/medium | mask25 | 1.75986 | 0.330769 | 0 | 132 | 130 | 2 |
| train\_probe | unconditional\_single/medium | mask50 | 1.79086 | 0.392857 | 0 | 255 | 252 | 3 |
| train\_probe | unconditional\_single/medium | mask75 | 1.76642 | 0.385224 | 0 | 382 | 379 | 3 |
| train\_probe | unconditional\_single/medium | mask100 | 2.06759 | 0.362205 | 0 | 512 | 508 | 4 |
| train\_probe | unconditional\_single/medium | mask\_mean | 1.84618 | 0.367764 | 0 | — | — | — |
| train\_probe | unconditional\_single/short | mask25 | 2.6369 | 0.0769231 | 0 | 28 | 26 | 2 |
| train\_probe | unconditional\_single/short | mask50 | 2.4541 | 0.204082 | 0 | 52 | 49 | 3 |
| train\_probe | unconditional\_single/short | mask75 | 2.43753 | 0.228571 | 0 | 73 | 70 | 3 |
| train\_probe | unconditional\_single/short | mask100 | 2.37042 | 0.229167 | 0 | 100 | 96 | 4 |
| train\_probe | unconditional\_single/short | mask\_mean | 2.47473 | 0.184686 | 0 | — | — | — |
| train\_probe | unconditional\_tcr/long | mask25 | 1.72076 | 0.36 | — | 525 | 525 | 0 |
| train\_probe | unconditional\_tcr/long | mask50 | 1.75872 | 0.34067 | 0 | 1048 | 1045 | 3 |
| train\_probe | unconditional\_tcr/long | mask75 | 1.79175 | 0.322785 | 0 | 1585 | 1580 | 5 |
| train\_probe | unconditional\_tcr/long | mask100 | 2.27164 | 0.27402 | 0 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_tcr/long | mask\_mean | 1.88572 | 0.324369 | 0 | — | — | — |
| train\_probe | unconditional\_tcr/medium | mask25 | 1.64685 | 0.340996 | — | 261 | 261 | 0 |
| train\_probe | unconditional\_tcr/medium | mask50 | 1.65907 | 0.305835 | 0 | 498 | 497 | 1 |
| train\_probe | unconditional\_tcr/medium | mask75 | 1.66943 | 0.328895 | 0 | 755 | 751 | 4 |
| train\_probe | unconditional\_tcr/medium | mask100 | 1.97712 | 0.376969 | 0 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_tcr/medium | mask\_mean | 1.73812 | 0.338174 | 0 | — | — | — |
| train\_probe | unconditional\_tcr/short | mask25 | 2.01804 | 0.217391 | — | 46 | 46 | 0 |
| train\_probe | unconditional\_tcr/short | mask50 | 2.34077 | 0.195652 | 0 | 97 | 92 | 5 |
| train\_probe | unconditional\_tcr/short | mask75 | 2.28662 | 0.219178 | 0 | 152 | 146 | 6 |
| train\_probe | unconditional\_tcr/short | mask100 | 2.61419 | 0.182292 | 0 | 200 | 192 | 8 |
| train\_probe | unconditional\_tcr/short | mask\_mean | 2.3149 | 0.203628 | 0 | — | — | — |

### 1790802160356028347\_r25\_refined\_late\_fusion\_4640596c：分层 mask 指标

源码目录：/vepfs-mlp2/c20250601/251105016/project/dllm\_test/output/immune\_research\_releases/r20\_r27\_20261001；完成/目标 step：25/25。

kind=synthetic\_generalization\_diagnostic；dataset_kind=deterministic\_artificial\_sequences；biological_quality_claim=False。

| 时点 | mask level | 指标 | 值 |
| --- | --- | --- | --- |
| initial | mask25 | mask\_ratio | 0.25 |
| initial | mask25 | selected\_count | 4467 |
| initial | mask25 | aa\_count | 4428 |
| initial | mask25 | terminal\_count | 39 |
| initial | mask25 | loss | 3.94605 |
| initial | mask25 | aa\_loss | 3.94325 |
| initial | mask25 | aa\_accuracy | 0.0372629 |
| initial | mask25 | terminal\_loss | 4.26441 |
| initial | mask25 | terminal\_accuracy | 0 |
| initial | mask50 | mask\_ratio | 0.5 |
| initial | mask50 | selected\_count | 8888 |
| initial | mask50 | aa\_count | 8819 |
| initial | mask50 | terminal\_count | 69 |
| initial | mask50 | loss | 4.03979 |
| initial | mask50 | aa\_loss | 4.0364 |
| initial | mask50 | aa\_accuracy | 0.0413879 |
| initial | mask50 | terminal\_loss | 4.47271 |
| initial | mask50 | terminal\_accuracy | 0 |
| initial | mask75 | mask\_ratio | 0.75 |
| initial | mask75 | selected\_count | 13437 |
| initial | mask75 | aa\_count | 13330 |
| initial | mask75 | terminal\_count | 107 |
| initial | mask75 | loss | 4.09853 |
| initial | mask75 | aa\_loss | 4.09435 |
| initial | mask75 | aa\_accuracy | 0.0528132 |
| initial | mask75 | terminal\_loss | 4.61916 |
| initial | mask75 | terminal\_accuracy | 0 |
| initial | mask100 | mask\_ratio | 1 |
| initial | mask100 | selected\_count | 17996 |
| initial | mask100 | aa\_count | 17864 |
| initial | mask100 | terminal\_count | 132 |
| initial | mask100 | loss | 4.08227 |
| initial | mask100 | aa\_loss | 4.07792 |
| initial | mask100 | aa\_accuracy | 0.0315719 |
| initial | mask100 | terminal\_loss | 4.67092 |
| initial | mask100 | terminal\_accuracy | 0 |
| initial | mask\_mean | loss | 4.04166 |
| initial | mask\_mean | aa\_loss | 4.03798 |
| initial | mask\_mean | terminal\_loss | 4.5068 |
| initial | mask\_mean | aa\_accuracy | 0.040759 |
| initial | mask\_mean | terminal\_accuracy | 0 |
| latest | mask25 | mask\_ratio | 0.25 |
| latest | mask25 | selected\_count | 4467 |
| latest | mask25 | aa\_count | 4428 |
| latest | mask25 | terminal\_count | 39 |
| latest | mask25 | loss | 1.80555 |
| latest | mask25 | aa\_loss | 1.77595 |
| latest | mask25 | aa\_accuracy | 0.315041 |
| latest | mask25 | terminal\_loss | 5.1666 |
| latest | mask25 | terminal\_accuracy | 0 |
| latest | mask50 | mask\_ratio | 0.5 |
| latest | mask50 | selected\_count | 8888 |
| latest | mask50 | aa\_count | 8819 |
| latest | mask50 | terminal\_count | 69 |
| latest | mask50 | loss | 1.81376 |
| latest | mask50 | aa\_loss | 1.78848 |
| latest | mask50 | aa\_accuracy | 0.317723 |
| latest | mask50 | terminal\_loss | 5.04559 |
| latest | mask50 | terminal\_accuracy | 0 |
| latest | mask75 | mask\_ratio | 0.75 |
| latest | mask75 | selected\_count | 13437 |
| latest | mask75 | aa\_count | 13330 |
| latest | mask75 | terminal\_count | 107 |
| latest | mask75 | loss | 1.82704 |
| latest | mask75 | aa\_loss | 1.80213 |
| latest | mask75 | aa\_accuracy | 0.317479 |
| latest | mask75 | terminal\_loss | 4.93026 |
| latest | mask75 | terminal\_accuracy | 0 |
| latest | mask100 | mask\_ratio | 1 |
| latest | mask100 | selected\_count | 17996 |
| latest | mask100 | aa\_count | 17864 |
| latest | mask100 | terminal\_count | 132 |
| latest | mask100 | loss | 2.14857 |
| latest | mask100 | aa\_loss | 2.13304 |
| latest | mask100 | aa\_accuracy | 0.295231 |
| latest | mask100 | terminal\_loss | 4.25111 |
| latest | mask100 | terminal\_accuracy | 0 |
| latest | mask\_mean | loss | 1.89873 |
| latest | mask\_mean | aa\_loss | 1.8749 |
| latest | mask\_mean | terminal\_loss | 4.84839 |
| latest | mask\_mean | aa\_accuracy | 0.311368 |
| latest | mask\_mean | terminal\_accuracy | 0 |

评测曲线（原始 step，四档等权均值）：

| step | mean loss | mean AA acc | mean end acc | gradient norm |
| --- | --- | --- | --- | --- |
| 25 | 1.89873 | 0.311368 | 0 | 2.06952 |

有条件/无条件曲线（每档按token计数合并，再平均四档；缺失分组不推断数值）：

| step | 子集 | mean loss | mean AA | mean end | Full-mask AA | Full-mask end | Full-mask AA count | Full-mask end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | conditional | 4.10076 | 0.0341571 | 0 | 0.0332512 | 0 | 9744 | 72 |
| initial | unconditional | 3.97121 | 0.0486268 | 0 | 0.0295567 | 0 | 8120 | 60 |
| 25 | conditional | 1.87798 | 0.307129 | 0 | 0.278838 | 0 | 9744 | 72 |
| 25 | unconditional | 1.92347 | 0.316475 | 0 | 0.314901 | 0 | 8120 | 60 |

人工数据 manifest：d40bbbbc1c4294eed843d88bcdd0cf193886de6c2d7e9db9069bbf86e931c022；精确 target sequence 重叠=0；精确 record 重叠=0。

跨 radius32 覆盖：{&quot;possible&quot;: true, &quot;train&quot;: true, &quot;valid&quot;: true}。生成规则共享是诊断设计的一部分，不能当作真实下游去重证书。


按任务/链长分组（全局 level 按 token 计数汇总，下面保留各组计数；不同组不得直接比较未经加权的总 loss）：

| 时点 | group/length | mask level | loss | AA acc | end acc | selected count | AA count | end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | ab\_antigen/long | mask25 | 4.10376 | 0.026616 | 0 | 530 | 526 | 4 |
| initial | ab\_antigen/long | mask50 | 4.30155 | 0.0266535 | 0 | 1019 | 1013 | 6 |
| initial | ab\_antigen/long | mask75 | 4.40906 | 0.0261097 | 0 | 1538 | 1532 | 6 |
| initial | ab\_antigen/long | mask100 | 4.34778 | 0.0156863 | 0 | 2048 | 2040 | 8 |
| initial | ab\_antigen/long | mask\_mean | 4.29054 | 0.0237664 | 0 | — | — | — |
| initial | ab\_antigen/medium | mask25 | 4.45436 | 0.0452675 | 0 | 246 | 243 | 3 |
| initial | ab\_antigen/medium | mask50 | 4.46488 | 0.0278884 | 0 | 507 | 502 | 5 |
| initial | ab\_antigen/medium | mask75 | 4.49795 | 0.0277045 | 0 | 765 | 758 | 7 |
| initial | ab\_antigen/medium | mask100 | 4.45486 | 0.0295276 | 0 | 1024 | 1016 | 8 |
| initial | ab\_antigen/medium | mask\_mean | 4.46801 | 0.032597 | 0 | — | — | — |
| initial | ab\_antigen/short | mask25 | 4.69282 | 0.0294118 | 0 | 38 | 34 | 4 |
| initial | ab\_antigen/short | mask50 | 4.39749 | 0.0340909 | 0 | 93 | 88 | 5 |
| initial | ab\_antigen/short | mask75 | 4.33879 | 0.0357143 | 0 | 145 | 140 | 5 |
| initial | ab\_antigen/short | mask100 | 4.28528 | 0.015625 | 0 | 200 | 192 | 8 |
| initial | ab\_antigen/short | mask\_mean | 4.4286 | 0.0287105 | 0 | — | — | — |
| initial | tcr\_peptide/long | mask25 | 3.81168 | 0.0451613 | 0 | 466 | 465 | 1 |
| initial | tcr\_peptide/long | mask50 | 3.89662 | 0.0458716 | 0 | 985 | 981 | 4 |
| initial | tcr\_peptide/long | mask75 | 3.96553 | 0.0500677 | 0 | 1484 | 1478 | 6 |
| initial | tcr\_peptide/long | mask100 | 3.98102 | 0.0470588 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_peptide/long | mask\_mean | 3.91372 | 0.0470398 | 0 | — | — | — |
| initial | tcr\_peptide/medium | mask25 | 3.96461 | 0.0373444 | 0 | 242 | 241 | 1 |
| initial | tcr\_peptide/medium | mask50 | 3.94159 | 0.0517928 | 0 | 506 | 502 | 4 |
| initial | tcr\_peptide/medium | mask75 | 3.88725 | 0.0513158 | 0 | 767 | 760 | 7 |
| initial | tcr\_peptide/medium | mask100 | 3.83488 | 0.0472441 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_peptide/medium | mask\_mean | 3.90708 | 0.0469243 | 0 | — | — | — |
| initial | tcr\_peptide/short | mask25 | 3.86838 | 0.0416667 | 0 | 51 | 48 | 3 |
| initial | tcr\_peptide/short | mask50 | 3.6784 | 0.0961538 | 0 | 110 | 104 | 6 |
| initial | tcr\_peptide/short | mask75 | 3.75761 | 0.0875912 | 0 | 144 | 137 | 7 |
| initial | tcr\_peptide/short | mask100 | 3.77943 | 0.09375 | 0 | 200 | 192 | 8 |
| initial | tcr\_peptide/short | mask\_mean | 3.77096 | 0.0797904 | 0 | — | — | — |
| initial | tcr\_pmhc/long | mask25 | 3.6662 | 0.0261194 | 0 | 540 | 536 | 4 |
| initial | tcr\_pmhc/long | mask50 | 3.87821 | 0.0367505 | 0 | 1038 | 1034 | 4 |
| initial | tcr\_pmhc/long | mask75 | 4.0134 | 0.0212355 | 0 | 1562 | 1554 | 8 |
| initial | tcr\_pmhc/long | mask100 | 4.05544 | 0.0308824 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_pmhc/long | mask\_mean | 3.90331 | 0.0287469 | 0 | — | — | — |
| initial | tcr\_pmhc/medium | mask25 | 4.19948 | 0.0123967 | 0 | 244 | 242 | 2 |
| initial | tcr\_pmhc/medium | mask50 | 4.2887 | 0.0250522 | 0 | 484 | 479 | 5 |
| initial | tcr\_pmhc/medium | mask75 | 4.35228 | 0.0312094 | 0 | 777 | 769 | 8 |
| initial | tcr\_pmhc/medium | mask100 | 4.34503 | 0.0305118 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_pmhc/medium | mask\_mean | 4.29637 | 0.0247925 | 0 | — | — | — |
| initial | tcr\_pmhc/short | mask25 | 4.38912 | 0.0208333 | 0 | 51 | 48 | 3 |
| initial | tcr\_pmhc/short | mask50 | 4.37921 | 0.0240964 | 0 | 86 | 83 | 3 |
| initial | tcr\_pmhc/short | mask75 | 4.18772 | 0.0204082 | 0 | 152 | 147 | 5 |
| initial | tcr\_pmhc/short | mask100 | 4.20912 | 0.015625 | 0 | 200 | 192 | 8 |
| initial | tcr\_pmhc/short | mask\_mean | 4.29129 | 0.0202407 | 0 | — | — | — |
| initial | unconditional\_ab/long | mask25 | 3.95361 | 0.0373832 | 0 | 538 | 535 | 3 |
| initial | unconditional\_ab/long | mask50 | 4.10352 | 0.0334608 | 0 | 1050 | 1046 | 4 |
| initial | unconditional\_ab/long | mask75 | 4.21672 | 0.0318596 | 0 | 1545 | 1538 | 7 |
| initial | unconditional\_ab/long | mask100 | 4.16261 | 0.0308824 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_ab/long | mask\_mean | 4.10912 | 0.0333965 | 0 | — | — | — |
| initial | unconditional\_ab/medium | mask25 | 4.21185 | 0.0948617 | 0 | 256 | 253 | 3 |
| initial | unconditional\_ab/medium | mask50 | 4.29512 | 0.0807087 | 0 | 512 | 508 | 4 |
| initial | unconditional\_ab/medium | mask75 | 4.2954 | 0.0748032 | 0 | 768 | 762 | 6 |
| initial | unconditional\_ab/medium | mask100 | 4.24567 | 0.0147638 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_ab/medium | mask\_mean | 4.26201 | 0.0662843 | 0 | — | — | — |
| initial | unconditional\_ab/short | mask25 | 3.86841 | 0.0576923 | 0 | 55 | 52 | 3 |
| initial | unconditional\_ab/short | mask50 | 3.89384 | 0.0714286 | 0 | 103 | 98 | 5 |
| initial | unconditional\_ab/short | mask75 | 3.85869 | 0.106383 | 0 | 148 | 141 | 7 |
| initial | unconditional\_ab/short | mask100 | 3.84764 | 0.125 | 0 | 200 | 192 | 8 |
| initial | unconditional\_ab/short | mask\_mean | 3.86715 | 0.090126 | 0 | — | — | — |
| initial | unconditional\_single/long | mask25 | 3.42299 | 0.0485075 | 0 | 269 | 268 | 1 |
| initial | unconditional\_single/long | mask50 | 3.36982 | 0.0838095 | 0 | 527 | 525 | 2 |
| initial | unconditional\_single/long | mask75 | 3.34001 | 0.22739 | 0 | 778 | 774 | 4 |
| initial | unconditional\_single/long | mask100 | 3.37679 | 0 | 0 | 1024 | 1020 | 4 |
| initial | unconditional\_single/long | mask\_mean | 3.3774 | 0.0899268 | 0 | — | — | — |
| initial | unconditional\_single/medium | mask25 | 3.71585 | 0.0366972 | 0 | 110 | 109 | 1 |
| initial | unconditional\_single/medium | mask50 | 3.50228 | 0.0818966 | 0 | 234 | 232 | 2 |
| initial | unconditional\_single/medium | mask75 | 3.51101 | 0.19346 | 0 | 371 | 367 | 4 |
| initial | unconditional\_single/medium | mask100 | 3.50481 | 0 | 0 | 512 | 508 | 4 |
| initial | unconditional\_single/medium | mask\_mean | 3.55849 | 0.0780136 | 0 | — | — | — |
| initial | unconditional\_single/short | mask25 | 3.10833 | 0 | 0 | 28 | 26 | 2 |
| initial | unconditional\_single/short | mask50 | 3.28518 | 0 | 0 | 52 | 50 | 2 |
| initial | unconditional\_single/short | mask75 | 3.10215 | 0 | 0 | 80 | 77 | 3 |
| initial | unconditional\_single/short | mask100 | 3.10264 | 0 | 0 | 100 | 96 | 4 |
| initial | unconditional\_single/short | mask\_mean | 3.14957 | 0 | 0 | — | — | — |
| initial | unconditional\_tcr/long | mask25 | 3.86257 | 0.0194175 | 0 | 516 | 515 | 1 |
| initial | unconditional\_tcr/long | mask50 | 3.99619 | 0.0182371 | 0 | 989 | 987 | 2 |
| initial | unconditional\_tcr/long | mask75 | 4.09982 | 0.0157687 | 0 | 1527 | 1522 | 5 |
| initial | unconditional\_tcr/long | mask100 | 4.10163 | 0.0151961 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_tcr/long | mask\_mean | 4.01505 | 0.0171548 | 0 | — | — | — |
| initial | unconditional\_tcr/medium | mask25 | 4.14186 | 0.0566802 | — | 247 | 247 | 0 |
| initial | unconditional\_tcr/medium | mask50 | 4.26738 | 0.0382294 | 0 | 500 | 497 | 3 |
| initial | unconditional\_tcr/medium | mask75 | 4.23126 | 0.0745257 | 0 | 744 | 738 | 6 |
| initial | unconditional\_tcr/medium | mask100 | 4.18619 | 0.0935039 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_tcr/medium | mask\_mean | 4.20667 | 0.0657348 | 0 | — | — | — |
| initial | unconditional\_tcr/short | mask25 | 4.14871 | 0.025 | — | 40 | 40 | 0 |
| initial | unconditional\_tcr/short | mask50 | 4.01632 | 0.0555556 | 0 | 93 | 90 | 3 |
| initial | unconditional\_tcr/short | mask75 | 4.10977 | 0.0441176 | 0 | 142 | 136 | 6 |
| initial | unconditional\_tcr/short | mask100 | 4.08673 | 0.0625 | 0 | 200 | 192 | 8 |
| initial | unconditional\_tcr/short | mask\_mean | 4.09038 | 0.0467933 | 0 | — | — | — |
| latest | ab\_antigen/long | mask25 | 1.76776 | 0.3327 | 0 | 530 | 526 | 4 |
| latest | ab\_antigen/long | mask50 | 1.75389 | 0.324778 | 0 | 1019 | 1013 | 6 |
| latest | ab\_antigen/long | mask75 | 1.75906 | 0.331593 | 0 | 1538 | 1532 | 6 |
| latest | ab\_antigen/long | mask100 | 2.22273 | 0.183824 | 0 | 2048 | 2040 | 8 |
| latest | ab\_antigen/long | mask\_mean | 1.87586 | 0.293223 | 0 | — | — | — |
| latest | ab\_antigen/medium | mask25 | 1.87738 | 0.325103 | 0 | 246 | 243 | 3 |
| latest | ab\_antigen/medium | mask50 | 1.89952 | 0.292829 | 0 | 507 | 502 | 5 |
| latest | ab\_antigen/medium | mask75 | 1.87463 | 0.291557 | 0 | 765 | 758 | 7 |
| latest | ab\_antigen/medium | mask100 | 2.14938 | 0.227362 | 0 | 1024 | 1016 | 8 |
| latest | ab\_antigen/medium | mask\_mean | 1.95023 | 0.284213 | 0 | — | — | — |
| latest | ab\_antigen/short | mask25 | 2.2386 | 0.235294 | 0 | 38 | 34 | 4 |
| latest | ab\_antigen/short | mask50 | 2.18572 | 0.193182 | 0 | 93 | 88 | 5 |
| latest | ab\_antigen/short | mask75 | 2.22317 | 0.221429 | 0 | 145 | 140 | 5 |
| latest | ab\_antigen/short | mask100 | 2.32516 | 0.208333 | 0 | 200 | 192 | 8 |
| latest | ab\_antigen/short | mask\_mean | 2.24316 | 0.214559 | 0 | — | — | — |
| latest | tcr\_peptide/long | mask25 | 1.91439 | 0.258065 | 0 | 466 | 465 | 1 |
| latest | tcr\_peptide/long | mask50 | 1.90455 | 0.276249 | 0 | 985 | 981 | 4 |
| latest | tcr\_peptide/long | mask75 | 1.91393 | 0.271313 | 0 | 1484 | 1478 | 6 |
| latest | tcr\_peptide/long | mask100 | 2.18582 | 0.27402 | 0 | 2048 | 2040 | 8 |
| latest | tcr\_peptide/long | mask\_mean | 1.97967 | 0.269911 | 0 | — | — | — |
| latest | tcr\_peptide/medium | mask25 | 2.00595 | 0.244813 | 0 | 242 | 241 | 1 |
| latest | tcr\_peptide/medium | mask50 | 1.9951 | 0.26494 | 0 | 506 | 502 | 4 |
| latest | tcr\_peptide/medium | mask75 | 2.03934 | 0.267105 | 0 | 767 | 760 | 7 |
| latest | tcr\_peptide/medium | mask100 | 2.31254 | 0.266732 | 0 | 1024 | 1016 | 8 |
| latest | tcr\_peptide/medium | mask\_mean | 2.08823 | 0.260898 | 0 | — | — | — |
| latest | tcr\_peptide/short | mask25 | 2.55162 | 0.229167 | 0 | 51 | 48 | 3 |
| latest | tcr\_peptide/short | mask50 | 2.51436 | 0.211538 | 0 | 110 | 104 | 6 |
| latest | tcr\_peptide/short | mask75 | 2.39987 | 0.248175 | 0 | 144 | 137 | 7 |
| latest | tcr\_peptide/short | mask100 | 2.38354 | 0.244792 | 0 | 200 | 192 | 8 |
| latest | tcr\_peptide/short | mask\_mean | 2.46235 | 0.233418 | 0 | — | — | — |
| latest | tcr\_pmhc/long | mask25 | 1.56955 | 0.348881 | 0 | 540 | 536 | 4 |
| latest | tcr\_pmhc/long | mask50 | 1.51647 | 0.368472 | 0 | 1038 | 1034 | 4 |
| latest | tcr\_pmhc/long | mask75 | 1.50887 | 0.37323 | 0 | 1562 | 1554 | 8 |
| latest | tcr\_pmhc/long | mask100 | 1.8776 | 0.415196 | 0 | 2048 | 2040 | 8 |
| latest | tcr\_pmhc/long | mask\_mean | 1.61813 | 0.376445 | 0 | — | — | — |
| latest | tcr\_pmhc/medium | mask25 | 1.52432 | 0.413223 | 0 | 244 | 242 | 2 |
| latest | tcr\_pmhc/medium | mask50 | 1.57535 | 0.394572 | 0 | 484 | 479 | 5 |
| latest | tcr\_pmhc/medium | mask75 | 1.62356 | 0.364109 | 0 | 777 | 769 | 8 |
| latest | tcr\_pmhc/medium | mask100 | 2.07513 | 0.298228 | 0 | 1024 | 1016 | 8 |
| latest | tcr\_pmhc/medium | mask\_mean | 1.69959 | 0.367533 | 0 | — | — | — |
| latest | tcr\_pmhc/short | mask25 | 2.05279 | 0.375 | 0 | 51 | 48 | 3 |
| latest | tcr\_pmhc/short | mask50 | 1.99268 | 0.313253 | 0 | 86 | 83 | 3 |
| latest | tcr\_pmhc/short | mask75 | 2.15604 | 0.251701 | 0 | 152 | 147 | 5 |
| latest | tcr\_pmhc/short | mask100 | 2.45448 | 0.229167 | 0 | 200 | 192 | 8 |
| latest | tcr\_pmhc/short | mask\_mean | 2.164 | 0.29228 | 0 | — | — | — |
| latest | unconditional\_ab/long | mask25 | 1.85578 | 0.336449 | 0 | 538 | 535 | 3 |
| latest | unconditional\_ab/long | mask50 | 1.87041 | 0.333652 | 0 | 1050 | 1046 | 4 |
| latest | unconditional\_ab/long | mask75 | 1.86989 | 0.340052 | 0 | 1545 | 1538 | 7 |
| latest | unconditional\_ab/long | mask100 | 2.19925 | 0.336765 | 0 | 2048 | 2040 | 8 |
| latest | unconditional\_ab/long | mask\_mean | 1.94883 | 0.336729 | 0 | — | — | — |
| latest | unconditional\_ab/medium | mask25 | 1.86128 | 0.3083 | 0 | 256 | 253 | 3 |
| latest | unconditional\_ab/medium | mask50 | 1.93965 | 0.320866 | 0 | 512 | 508 | 4 |
| latest | unconditional\_ab/medium | mask75 | 1.90681 | 0.304462 | 0 | 768 | 762 | 6 |
| latest | unconditional\_ab/medium | mask100 | 2.08508 | 0.298228 | 0 | 1024 | 1016 | 8 |
| latest | unconditional\_ab/medium | mask\_mean | 1.9482 | 0.307964 | 0 | — | — | — |
| latest | unconditional\_ab/short | mask25 | 2.20381 | 0.211538 | 0 | 55 | 52 | 3 |
| latest | unconditional\_ab/short | mask50 | 2.41612 | 0.234694 | 0 | 103 | 98 | 5 |
| latest | unconditional\_ab/short | mask75 | 2.5605 | 0.241135 | 0 | 148 | 141 | 7 |
| latest | unconditional\_ab/short | mask100 | 2.64478 | 0.229167 | 0 | 200 | 192 | 8 |
| latest | unconditional\_ab/short | mask\_mean | 2.4563 | 0.229133 | 0 | — | — | — |
| latest | unconditional\_single/long | mask25 | 1.71983 | 0.298507 | 0 | 269 | 268 | 1 |
| latest | unconditional\_single/long | mask50 | 1.7243 | 0.32381 | 0 | 527 | 525 | 2 |
| latest | unconditional\_single/long | mask75 | 1.79484 | 0.320413 | 0 | 778 | 774 | 4 |
| latest | unconditional\_single/long | mask100 | 2.17189 | 0.305882 | 0 | 1024 | 1020 | 4 |
| latest | unconditional\_single/long | mask\_mean | 1.85271 | 0.312153 | 0 | — | — | — |
| latest | unconditional\_single/medium | mask25 | 1.76377 | 0.284404 | 0 | 110 | 109 | 1 |
| latest | unconditional\_single/medium | mask50 | 1.75281 | 0.323276 | 0 | 234 | 232 | 2 |
| latest | unconditional\_single/medium | mask75 | 1.774 | 0.337875 | 0 | 371 | 367 | 4 |
| latest | unconditional\_single/medium | mask100 | 2.14129 | 0.330709 | 0 | 512 | 508 | 4 |
| latest | unconditional\_single/medium | mask\_mean | 1.85797 | 0.319066 | 0 | — | — | — |
| latest | unconditional\_single/short | mask25 | 2.71468 | 0.230769 | 0 | 28 | 26 | 2 |
| latest | unconditional\_single/short | mask50 | 2.77759 | 0.2 | 0 | 52 | 50 | 2 |
| latest | unconditional\_single/short | mask75 | 2.4832 | 0.25974 | 0 | 80 | 77 | 3 |
| latest | unconditional\_single/short | mask100 | 2.29851 | 0.291667 | 0 | 100 | 96 | 4 |
| latest | unconditional\_single/short | mask\_mean | 2.56849 | 0.245544 | 0 | — | — | — |
| latest | unconditional\_tcr/long | mask25 | 1.81128 | 0.299029 | 0 | 516 | 515 | 1 |
| latest | unconditional\_tcr/long | mask50 | 1.81966 | 0.302938 | 0 | 989 | 987 | 2 |
| latest | unconditional\_tcr/long | mask75 | 1.84783 | 0.30092 | 0 | 1527 | 1522 | 5 |
| latest | unconditional\_tcr/long | mask100 | 2.19574 | 0.305392 | 0 | 2048 | 2040 | 8 |
| latest | unconditional\_tcr/long | mask\_mean | 1.91863 | 0.30207 | 0 | — | — | — |
| latest | unconditional\_tcr/medium | mask25 | 1.60332 | 0.356275 | — | 247 | 247 | 0 |
| latest | unconditional\_tcr/medium | mask50 | 1.63482 | 0.358149 | 0 | 500 | 497 | 3 |
| latest | unconditional\_tcr/medium | mask75 | 1.68184 | 0.361789 | 0 | 744 | 738 | 6 |
| latest | unconditional\_tcr/medium | mask100 | 1.92217 | 0.345472 | 0 | 1024 | 1016 | 8 |
| latest | unconditional\_tcr/medium | mask\_mean | 1.71054 | 0.355421 | 0 | — | — | — |
| latest | unconditional\_tcr/short | mask25 | 2.25791 | 0.25 | — | 40 | 40 | 0 |
| latest | unconditional\_tcr/short | mask50 | 2.2678 | 0.222222 | 0 | 93 | 90 | 3 |
| latest | unconditional\_tcr/short | mask75 | 2.34853 | 0.227941 | 0 | 142 | 136 | 6 |
| latest | unconditional\_tcr/short | mask100 | 2.40801 | 0.213542 | 0 | 200 | 192 | 8 |
| latest | unconditional\_tcr/short | mask\_mean | 2.32056 | 0.228426 | 0 | — | — | — |
| train\_probe | ab\_antigen/long | mask25 | 1.58513 | 0.343396 | — | 530 | 530 | 0 |
| train\_probe | ab\_antigen/long | mask50 | 1.60265 | 0.369544 | 0 | 1034 | 1031 | 3 |
| train\_probe | ab\_antigen/long | mask75 | 1.62497 | 0.344252 | 0 | 1563 | 1557 | 6 |
| train\_probe | ab\_antigen/long | mask100 | 2.34089 | 0.183824 | 0 | 2048 | 2040 | 8 |
| train\_probe | ab\_antigen/long | mask\_mean | 1.78841 | 0.310254 | 0 | — | — | — |
| train\_probe | ab\_antigen/medium | mask25 | 1.98897 | 0.29304 | 0 | 275 | 273 | 2 |
| train\_probe | ab\_antigen/medium | mask50 | 1.95143 | 0.317844 | 0 | 542 | 538 | 4 |
| train\_probe | ab\_antigen/medium | mask75 | 1.94659 | 0.326031 | 0 | 781 | 776 | 5 |
| train\_probe | ab\_antigen/medium | mask100 | 2.10697 | 0.299213 | 0 | 1024 | 1016 | 8 |
| train\_probe | ab\_antigen/medium | mask\_mean | 1.99849 | 0.309032 | 0 | — | — | — |
| train\_probe | ab\_antigen/short | mask25 | 2.21896 | 0.361702 | 0 | 52 | 47 | 5 |
| train\_probe | ab\_antigen/short | mask50 | 2.16482 | 0.294118 | 0 | 109 | 102 | 7 |
| train\_probe | ab\_antigen/short | mask75 | 2.2072 | 0.297297 | 0 | 155 | 148 | 7 |
| train\_probe | ab\_antigen/short | mask100 | 2.43027 | 0.270833 | 0 | 200 | 192 | 8 |
| train\_probe | ab\_antigen/short | mask\_mean | 2.25531 | 0.305988 | 0 | — | — | — |
| train\_probe | tcr\_peptide/long | mask25 | 1.57561 | 0.388664 | 0 | 495 | 494 | 1 |
| train\_probe | tcr\_peptide/long | mask50 | 1.58391 | 0.361868 | 0 | 1033 | 1028 | 5 |
| train\_probe | tcr\_peptide/long | mask75 | 1.63622 | 0.347569 | 0 | 1530 | 1522 | 8 |
| train\_probe | tcr\_peptide/long | mask100 | 2.33483 | 0.305392 | 0 | 2048 | 2040 | 8 |
| train\_probe | tcr\_peptide/long | mask\_mean | 1.78264 | 0.350873 | 0 | — | — | — |
| train\_probe | tcr\_peptide/medium | mask25 | 1.71487 | 0.293651 | — | 252 | 252 | 0 |
| train\_probe | tcr\_peptide/medium | mask50 | 1.74767 | 0.327485 | 0 | 517 | 513 | 4 |
| train\_probe | tcr\_peptide/medium | mask75 | 1.75325 | 0.334217 | 0 | 760 | 754 | 6 |
| train\_probe | tcr\_peptide/medium | mask100 | 2.25184 | 0.298228 | 0 | 1024 | 1016 | 8 |
| train\_probe | tcr\_peptide/medium | mask\_mean | 1.86691 | 0.313396 | 0 | — | — | — |
| train\_probe | tcr\_peptide/short | mask25 | 2.12941 | 0.369565 | 0 | 51 | 46 | 5 |
| train\_probe | tcr\_peptide/short | mask50 | 1.96728 | 0.355556 | 0 | 95 | 90 | 5 |
| train\_probe | tcr\_peptide/short | mask75 | 2.05529 | 0.338235 | 0 | 142 | 136 | 6 |
| train\_probe | tcr\_peptide/short | mask100 | 2.31739 | 0.322917 | 0 | 200 | 192 | 8 |
| train\_probe | tcr\_peptide/short | mask\_mean | 2.11734 | 0.346568 | 0 | — | — | — |
| train\_probe | tcr\_pmhc/long | mask25 | 1.62198 | 0.362069 | 0 | 526 | 522 | 4 |
| train\_probe | tcr\_pmhc/long | mask50 | 1.67113 | 0.332687 | 0 | 1036 | 1031 | 5 |
| train\_probe | tcr\_pmhc/long | mask75 | 1.71047 | 0.323662 | 0 | 1557 | 1551 | 6 |
| train\_probe | tcr\_pmhc/long | mask100 | 2.10911 | 0.321078 | 0 | 2048 | 2040 | 8 |
| train\_probe | tcr\_pmhc/long | mask\_mean | 1.77817 | 0.334874 | 0 | — | — | — |
| train\_probe | tcr\_pmhc/medium | mask25 | 1.6385 | 0.351648 | 0 | 276 | 273 | 3 |
| train\_probe | tcr\_pmhc/medium | mask50 | 1.68858 | 0.330869 | 0 | 549 | 541 | 8 |
| train\_probe | tcr\_pmhc/medium | mask75 | 1.67467 | 0.346497 | 0 | 793 | 785 | 8 |
| train\_probe | tcr\_pmhc/medium | mask100 | 2.22327 | 0.266732 | 0 | 1024 | 1016 | 8 |
| train\_probe | tcr\_pmhc/medium | mask\_mean | 1.80626 | 0.323937 | 0 | — | — | — |
| train\_probe | tcr\_pmhc/short | mask25 | 2.42342 | 0.229167 | 0 | 50 | 48 | 2 |
| train\_probe | tcr\_pmhc/short | mask50 | 2.45246 | 0.18 | 0 | 103 | 100 | 3 |
| train\_probe | tcr\_pmhc/short | mask75 | 2.56933 | 0.17931 | 0 | 152 | 145 | 7 |
| train\_probe | tcr\_pmhc/short | mask100 | 2.48394 | 0.213542 | 0 | 200 | 192 | 8 |
| train\_probe | tcr\_pmhc/short | mask\_mean | 2.48229 | 0.200505 | 0 | — | — | — |
| train\_probe | unconditional\_ab/long | mask25 | 1.66278 | 0.337945 | 0 | 509 | 506 | 3 |
| train\_probe | unconditional\_ab/long | mask50 | 1.67044 | 0.30396 | 0 | 1016 | 1010 | 6 |
| train\_probe | unconditional\_ab/long | mask75 | 1.68121 | 0.316464 | 0 | 1569 | 1561 | 8 |
| train\_probe | unconditional\_ab/long | mask100 | 2.16732 | 0.321078 | 0 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_ab/long | mask\_mean | 1.79544 | 0.319862 | 0 | — | — | — |
| train\_probe | unconditional\_ab/medium | mask25 | 1.8604 | 0.33195 | 0 | 243 | 241 | 2 |
| train\_probe | unconditional\_ab/medium | mask50 | 1.89031 | 0.315891 | 0 | 521 | 516 | 5 |
| train\_probe | unconditional\_ab/medium | mask75 | 1.89955 | 0.327201 | 0 | 766 | 761 | 5 |
| train\_probe | unconditional\_ab/medium | mask100 | 2.18882 | 0.329724 | 0 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_ab/medium | mask\_mean | 1.95977 | 0.326192 | 0 | — | — | — |
| train\_probe | unconditional\_ab/short | mask25 | 2.48151 | 0.173077 | 0 | 53 | 52 | 1 |
| train\_probe | unconditional\_ab/short | mask50 | 2.36865 | 0.192308 | 0 | 106 | 104 | 2 |
| train\_probe | unconditional\_ab/short | mask75 | 2.42662 | 0.189873 | 0 | 163 | 158 | 5 |
| train\_probe | unconditional\_ab/short | mask100 | 2.71267 | 0.197917 | 0 | 200 | 192 | 8 |
| train\_probe | unconditional\_ab/short | mask\_mean | 2.49736 | 0.188294 | 0 | — | — | — |
| train\_probe | unconditional\_single/long | mask25 | 1.7049 | 0.258197 | 0 | 245 | 244 | 1 |
| train\_probe | unconditional\_single/long | mask50 | 1.6916 | 0.283096 | 0 | 492 | 491 | 1 |
| train\_probe | unconditional\_single/long | mask75 | 1.75786 | 0.279948 | 0 | 771 | 768 | 3 |
| train\_probe | unconditional\_single/long | mask100 | 2.30177 | 0.243137 | 0 | 1024 | 1020 | 4 |
| train\_probe | unconditional\_single/long | mask\_mean | 1.86403 | 0.266094 | 0 | — | — | — |
| train\_probe | unconditional\_single/medium | mask25 | 1.80678 | 0.330769 | 0 | 132 | 130 | 2 |
| train\_probe | unconditional\_single/medium | mask50 | 1.79579 | 0.380952 | 0 | 255 | 252 | 3 |
| train\_probe | unconditional\_single/medium | mask75 | 1.77873 | 0.366755 | 0 | 382 | 379 | 3 |
| train\_probe | unconditional\_single/medium | mask100 | 2.10985 | 0.362205 | 0 | 512 | 508 | 4 |
| train\_probe | unconditional\_single/medium | mask\_mean | 1.87279 | 0.36017 | 0 | — | — | — |
| train\_probe | unconditional\_single/short | mask25 | 2.62882 | 0.0769231 | 0 | 28 | 26 | 2 |
| train\_probe | unconditional\_single/short | mask50 | 2.44509 | 0.183673 | 0 | 52 | 49 | 3 |
| train\_probe | unconditional\_single/short | mask75 | 2.45784 | 0.228571 | 0 | 73 | 70 | 3 |
| train\_probe | unconditional\_single/short | mask100 | 2.41773 | 0.229167 | 0 | 100 | 96 | 4 |
| train\_probe | unconditional\_single/short | mask\_mean | 2.48737 | 0.179584 | 0 | — | — | — |
| train\_probe | unconditional\_tcr/long | mask25 | 1.76303 | 0.297143 | — | 525 | 525 | 0 |
| train\_probe | unconditional\_tcr/long | mask50 | 1.80192 | 0.282297 | 0 | 1048 | 1045 | 3 |
| train\_probe | unconditional\_tcr/long | mask75 | 1.83886 | 0.277848 | 0 | 1585 | 1580 | 5 |
| train\_probe | unconditional\_tcr/long | mask100 | 2.31775 | 0.27402 | 0 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_tcr/long | mask\_mean | 1.93039 | 0.282827 | 0 | — | — | — |
| train\_probe | unconditional\_tcr/medium | mask25 | 1.70889 | 0.356322 | — | 261 | 261 | 0 |
| train\_probe | unconditional\_tcr/medium | mask50 | 1.72552 | 0.394366 | 0 | 498 | 497 | 1 |
| train\_probe | unconditional\_tcr/medium | mask75 | 1.74913 | 0.375499 | 0 | 755 | 751 | 4 |
| train\_probe | unconditional\_tcr/medium | mask100 | 1.97784 | 0.376969 | 0 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_tcr/medium | mask\_mean | 1.79034 | 0.375789 | 0 | — | — | — |
| train\_probe | unconditional\_tcr/short | mask25 | 2.11804 | 0.152174 | — | 46 | 46 | 0 |
| train\_probe | unconditional\_tcr/short | mask50 | 2.45187 | 0.152174 | 0 | 97 | 92 | 5 |
| train\_probe | unconditional\_tcr/short | mask75 | 2.40705 | 0.184932 | 0 | 152 | 146 | 6 |
| train\_probe | unconditional\_tcr/short | mask100 | 2.65528 | 0.182292 | 0 | 200 | 192 | 8 |
| train\_probe | unconditional\_tcr/short | mask\_mean | 2.40806 | 0.167893 | 0 | — | — | — |

### 1790802160358902166\_r26\_refined\_local\_global\_905c9acd：分层 mask 指标

源码目录：/vepfs-mlp2/c20250601/251105016/project/dllm\_test/output/immune\_research\_releases/r20\_r27\_20261001；完成/目标 step：25/25。

kind=synthetic\_generalization\_diagnostic；dataset_kind=deterministic\_artificial\_sequences；biological_quality_claim=False。

| 时点 | mask level | 指标 | 值 |
| --- | --- | --- | --- |
| initial | mask25 | mask\_ratio | 0.25 |
| initial | mask25 | selected\_count | 4467 |
| initial | mask25 | aa\_count | 4428 |
| initial | mask25 | terminal\_count | 39 |
| initial | mask25 | loss | 3.89278 |
| initial | mask25 | aa\_loss | 3.88838 |
| initial | mask25 | aa\_accuracy | 0.0350045 |
| initial | mask25 | terminal\_loss | 4.39224 |
| initial | mask25 | terminal\_accuracy | 0 |
| initial | mask50 | mask\_ratio | 0.5 |
| initial | mask50 | selected\_count | 8888 |
| initial | mask50 | aa\_count | 8819 |
| initial | mask50 | terminal\_count | 69 |
| initial | mask50 | loss | 4.00888 |
| initial | mask50 | aa\_loss | 4.00457 |
| initial | mask50 | aa\_accuracy | 0.035038 |
| initial | mask50 | terminal\_loss | 4.55927 |
| initial | mask50 | terminal\_accuracy | 0 |
| initial | mask75 | mask\_ratio | 0.75 |
| initial | mask75 | selected\_count | 13437 |
| initial | mask75 | aa\_count | 13330 |
| initial | mask75 | terminal\_count | 107 |
| initial | mask75 | loss | 4.07546 |
| initial | mask75 | aa\_loss | 4.07043 |
| initial | mask75 | aa\_accuracy | 0.0447862 |
| initial | mask75 | terminal\_loss | 4.70148 |
| initial | mask75 | terminal\_accuracy | 0 |
| initial | mask100 | mask\_ratio | 1 |
| initial | mask100 | selected\_count | 17996 |
| initial | mask100 | aa\_count | 17864 |
| initial | mask100 | terminal\_count | 132 |
| initial | mask100 | loss | 4.07558 |
| initial | mask100 | aa\_loss | 4.07093 |
| initial | mask100 | aa\_accuracy | 0.0292208 |
| initial | mask100 | terminal\_loss | 4.7046 |
| initial | mask100 | terminal\_accuracy | 0 |
| initial | mask\_mean | loss | 4.01317 |
| initial | mask\_mean | aa\_loss | 4.00858 |
| initial | mask\_mean | terminal\_loss | 4.58939 |
| initial | mask\_mean | aa\_accuracy | 0.0360124 |
| initial | mask\_mean | terminal\_accuracy | 0 |
| latest | mask25 | mask\_ratio | 0.25 |
| latest | mask25 | selected\_count | 4467 |
| latest | mask25 | aa\_count | 4428 |
| latest | mask25 | terminal\_count | 39 |
| latest | mask25 | loss | 1.60561 |
| latest | mask25 | aa\_loss | 1.5739 |
| latest | mask25 | aa\_accuracy | 0.397922 |
| latest | mask25 | terminal\_loss | 5.20576 |
| latest | mask25 | terminal\_accuracy | 0 |
| latest | mask50 | mask\_ratio | 0.5 |
| latest | mask50 | selected\_count | 8888 |
| latest | mask50 | aa\_count | 8819 |
| latest | mask50 | terminal\_count | 69 |
| latest | mask50 | loss | 1.65163 |
| latest | mask50 | aa\_loss | 1.62493 |
| latest | mask50 | aa\_accuracy | 0.385531 |
| latest | mask50 | terminal\_loss | 5.06349 |
| latest | mask50 | terminal\_accuracy | 0 |
| latest | mask75 | mask\_ratio | 0.75 |
| latest | mask75 | selected\_count | 13437 |
| latest | mask75 | aa\_count | 13330 |
| latest | mask75 | terminal\_count | 107 |
| latest | mask75 | loss | 1.7315 |
| latest | mask75 | aa\_loss | 1.70578 |
| latest | mask75 | aa\_accuracy | 0.374419 |
| latest | mask75 | terminal\_loss | 4.93561 |
| latest | mask75 | terminal\_accuracy | 0 |
| latest | mask100 | mask\_ratio | 1 |
| latest | mask100 | selected\_count | 17996 |
| latest | mask100 | aa\_count | 17864 |
| latest | mask100 | terminal\_count | 132 |
| latest | mask100 | loss | 2.08457 |
| latest | mask100 | aa\_loss | 2.06975 |
| latest | mask100 | aa\_accuracy | 0.297302 |
| latest | mask100 | terminal\_loss | 4.08998 |
| latest | mask100 | terminal\_accuracy | 0 |
| latest | mask\_mean | loss | 1.76833 |
| latest | mask\_mean | aa\_loss | 1.74359 |
| latest | mask\_mean | terminal\_loss | 4.82371 |
| latest | mask\_mean | aa\_accuracy | 0.363794 |
| latest | mask\_mean | terminal\_accuracy | 0 |

评测曲线（原始 step，四档等权均值）：

| step | mean loss | mean AA acc | mean end acc | gradient norm |
| --- | --- | --- | --- | --- |
| 25 | 1.76833 | 0.363794 | 0 | 2.39835 |

有条件/无条件曲线（每档按token计数合并，再平均四档；缺失分组不推断数值）：

| step | 子集 | mean loss | mean AA | mean end | Full-mask AA | Full-mask end | Full-mask AA count | Full-mask end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | conditional | 4.08659 | 0.0313433 | 0 | 0.0339696 | 0 | 9744 | 72 |
| initial | unconditional | 3.92552 | 0.0415568 | 0 | 0.0235222 | 0 | 8120 | 60 |
| 25 | conditional | 1.74062 | 0.369605 | 0 | 0.282635 | 0 | 9744 | 72 |
| 25 | unconditional | 1.80139 | 0.356951 | 0 | 0.314901 | 0 | 8120 | 60 |

人工数据 manifest：d40bbbbc1c4294eed843d88bcdd0cf193886de6c2d7e9db9069bbf86e931c022；精确 target sequence 重叠=0；精确 record 重叠=0。

跨 radius32 覆盖：{&quot;possible&quot;: true, &quot;train&quot;: true, &quot;valid&quot;: true}。生成规则共享是诊断设计的一部分，不能当作真实下游去重证书。


按任务/链长分组（全局 level 按 token 计数汇总，下面保留各组计数；不同组不得直接比较未经加权的总 loss）：

| 时点 | group/length | mask level | loss | AA acc | end acc | selected count | AA count | end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | ab\_antigen/long | mask25 | 4.24973 | 0.0190114 | 0 | 530 | 526 | 4 |
| initial | ab\_antigen/long | mask50 | 4.38742 | 0.0227048 | 0 | 1019 | 1013 | 6 |
| initial | ab\_antigen/long | mask75 | 4.39272 | 0.0267624 | 0 | 1538 | 1532 | 6 |
| initial | ab\_antigen/long | mask100 | 4.27875 | 0.0264706 | 0 | 2048 | 2040 | 8 |
| initial | ab\_antigen/long | mask\_mean | 4.32715 | 0.0237373 | 0 | — | — | — |
| initial | ab\_antigen/medium | mask25 | 4.25892 | 0.00823045 | 0 | 246 | 243 | 3 |
| initial | ab\_antigen/medium | mask50 | 4.34305 | 0.0159363 | 0 | 507 | 502 | 5 |
| initial | ab\_antigen/medium | mask75 | 4.4131 | 0.0422164 | 0 | 765 | 758 | 7 |
| initial | ab\_antigen/medium | mask100 | 4.39726 | 0.0305118 | 0 | 1024 | 1016 | 8 |
| initial | ab\_antigen/medium | mask\_mean | 4.35308 | 0.0242237 | 0 | — | — | — |
| initial | ab\_antigen/short | mask25 | 4.42935 | 0.0588235 | 0 | 38 | 34 | 4 |
| initial | ab\_antigen/short | mask50 | 4.26535 | 0.0340909 | 0 | 93 | 88 | 5 |
| initial | ab\_antigen/short | mask75 | 4.24109 | 0.0428571 | 0 | 145 | 140 | 5 |
| initial | ab\_antigen/short | mask100 | 4.20677 | 0.046875 | 0 | 200 | 192 | 8 |
| initial | ab\_antigen/short | mask\_mean | 4.28564 | 0.0456616 | 0 | — | — | — |
| initial | tcr\_peptide/long | mask25 | 3.75574 | 0.0473118 | 0 | 466 | 465 | 1 |
| initial | tcr\_peptide/long | mask50 | 3.86912 | 0.0417941 | 0 | 985 | 981 | 4 |
| initial | tcr\_peptide/long | mask75 | 3.97539 | 0.0575101 | 0 | 1484 | 1478 | 6 |
| initial | tcr\_peptide/long | mask100 | 4.05611 | 0.0622549 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_peptide/long | mask\_mean | 3.91409 | 0.0522177 | 0 | — | — | — |
| initial | tcr\_peptide/medium | mask25 | 3.83075 | 0.0290456 | 0 | 242 | 241 | 1 |
| initial | tcr\_peptide/medium | mask50 | 3.90211 | 0.0239044 | 0 | 506 | 502 | 4 |
| initial | tcr\_peptide/medium | mask75 | 3.94099 | 0.0276316 | 0 | 767 | 760 | 7 |
| initial | tcr\_peptide/medium | mask100 | 3.89363 | 0.0305118 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_peptide/medium | mask\_mean | 3.89187 | 0.0277734 | 0 | — | — | — |
| initial | tcr\_peptide/short | mask25 | 3.7955 | 0.0625 | 0 | 51 | 48 | 3 |
| initial | tcr\_peptide/short | mask50 | 3.6999 | 0.0480769 | 0 | 110 | 104 | 6 |
| initial | tcr\_peptide/short | mask75 | 3.76546 | 0.0510949 | 0 | 144 | 137 | 7 |
| initial | tcr\_peptide/short | mask100 | 3.7947 | 0.046875 | 0 | 200 | 192 | 8 |
| initial | tcr\_peptide/short | mask\_mean | 3.76389 | 0.0521367 | 0 | — | — | — |
| initial | tcr\_pmhc/long | mask25 | 3.64688 | 0.0317164 | 0 | 540 | 536 | 4 |
| initial | tcr\_pmhc/long | mask50 | 3.87128 | 0.0135397 | 0 | 1038 | 1034 | 4 |
| initial | tcr\_pmhc/long | mask75 | 4.05111 | 0.0135135 | 0 | 1562 | 1554 | 8 |
| initial | tcr\_pmhc/long | mask100 | 4.16231 | 0.0156863 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_pmhc/long | mask\_mean | 3.9329 | 0.018614 | 0 | — | — | — |
| initial | tcr\_pmhc/medium | mask25 | 3.91128 | 0.0371901 | 0 | 244 | 242 | 2 |
| initial | tcr\_pmhc/medium | mask50 | 4.13464 | 0.0375783 | 0 | 484 | 479 | 5 |
| initial | tcr\_pmhc/medium | mask75 | 4.33153 | 0.0351105 | 0 | 777 | 769 | 8 |
| initial | tcr\_pmhc/medium | mask100 | 4.36849 | 0.0314961 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_pmhc/medium | mask\_mean | 4.18648 | 0.0353437 | 0 | — | — | — |
| initial | tcr\_pmhc/short | mask25 | 4.16854 | 0.0416667 | 0 | 51 | 48 | 3 |
| initial | tcr\_pmhc/short | mask50 | 4.28146 | 0.0361446 | 0 | 86 | 83 | 3 |
| initial | tcr\_pmhc/short | mask75 | 4.21468 | 0.0408163 | 0 | 152 | 147 | 5 |
| initial | tcr\_pmhc/short | mask100 | 4.19548 | 0.03125 | 0 | 200 | 192 | 8 |
| initial | tcr\_pmhc/short | mask\_mean | 4.21504 | 0.0374694 | 0 | — | — | — |
| initial | unconditional\_ab/long | mask25 | 4.01602 | 0.0429907 | 0 | 538 | 535 | 3 |
| initial | unconditional\_ab/long | mask50 | 4.10145 | 0.0650096 | 0 | 1050 | 1046 | 4 |
| initial | unconditional\_ab/long | mask75 | 4.15189 | 0.0591678 | 0 | 1545 | 1538 | 7 |
| initial | unconditional\_ab/long | mask100 | 4.10116 | 0.0313726 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_ab/long | mask\_mean | 4.09263 | 0.0496351 | 0 | — | — | — |
| initial | unconditional\_ab/medium | mask25 | 3.9054 | 0.083004 | 0 | 256 | 253 | 3 |
| initial | unconditional\_ab/medium | mask50 | 4.09812 | 0.0826772 | 0 | 512 | 508 | 4 |
| initial | unconditional\_ab/medium | mask75 | 4.16791 | 0.0813648 | 0 | 768 | 762 | 6 |
| initial | unconditional\_ab/medium | mask100 | 4.17491 | 0.0787402 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_ab/medium | mask\_mean | 4.08659 | 0.0814465 | 0 | — | — | — |
| initial | unconditional\_ab/short | mask25 | 3.8656 | 0.0384615 | 0 | 55 | 52 | 3 |
| initial | unconditional\_ab/short | mask50 | 3.90127 | 0.0408163 | 0 | 103 | 98 | 5 |
| initial | unconditional\_ab/short | mask75 | 3.87111 | 0.0212766 | 0 | 148 | 141 | 7 |
| initial | unconditional\_ab/short | mask100 | 3.84389 | 0.03125 | 0 | 200 | 192 | 8 |
| initial | unconditional\_ab/short | mask\_mean | 3.87047 | 0.0329511 | 0 | — | — | — |
| initial | unconditional\_single/long | mask25 | 3.41248 | 0.0485075 | 0 | 269 | 268 | 1 |
| initial | unconditional\_single/long | mask50 | 3.39562 | 0.0761905 | 0 | 527 | 525 | 2 |
| initial | unconditional\_single/long | mask75 | 3.36794 | 0.193798 | 0 | 778 | 774 | 4 |
| initial | unconditional\_single/long | mask100 | 3.38543 | 0 | 0 | 1024 | 1020 | 4 |
| initial | unconditional\_single/long | mask\_mean | 3.39037 | 0.0796241 | 0 | — | — | — |
| initial | unconditional\_single/medium | mask25 | 3.7779 | 0.0366972 | 0 | 110 | 109 | 1 |
| initial | unconditional\_single/medium | mask50 | 3.60855 | 0.0258621 | 0 | 234 | 232 | 2 |
| initial | unconditional\_single/medium | mask75 | 3.57466 | 0.0435967 | 0 | 371 | 367 | 4 |
| initial | unconditional\_single/medium | mask100 | 3.51609 | 0 | 0 | 512 | 508 | 4 |
| initial | unconditional\_single/medium | mask\_mean | 3.6193 | 0.026539 | 0 | — | — | — |
| initial | unconditional\_single/short | mask25 | 3.17399 | 0.230769 | 0 | 28 | 26 | 2 |
| initial | unconditional\_single/short | mask50 | 3.39984 | 0 | 0 | 52 | 50 | 2 |
| initial | unconditional\_single/short | mask75 | 3.17481 | 0 | 0 | 80 | 77 | 3 |
| initial | unconditional\_single/short | mask100 | 3.12259 | 0 | 0 | 100 | 96 | 4 |
| initial | unconditional\_single/short | mask\_mean | 3.21781 | 0.0576923 | 0 | — | — | — |
| initial | unconditional\_tcr/long | mask25 | 3.89262 | 0.0213592 | 0 | 516 | 515 | 1 |
| initial | unconditional\_tcr/long | mask50 | 3.99228 | 0.0192503 | 0 | 989 | 987 | 2 |
| initial | unconditional\_tcr/long | mask75 | 4.04526 | 0.0151117 | 0 | 1527 | 1522 | 5 |
| initial | unconditional\_tcr/long | mask100 | 4.0541 | 0.0156863 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_tcr/long | mask\_mean | 3.99607 | 0.0178519 | 0 | — | — | — |
| initial | unconditional\_tcr/medium | mask25 | 3.83404 | 0 | — | 247 | 247 | 0 |
| initial | unconditional\_tcr/medium | mask50 | 4.08607 | 0 | 0 | 500 | 497 | 3 |
| initial | unconditional\_tcr/medium | mask75 | 4.12407 | 0 | 0 | 744 | 738 | 6 |
| initial | unconditional\_tcr/medium | mask100 | 4.11728 | 0 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_tcr/medium | mask\_mean | 4.04036 | 0 | 0 | — | — | — |
| initial | unconditional\_tcr/short | mask25 | 4.06521 | 0.025 | — | 40 | 40 | 0 |
| initial | unconditional\_tcr/short | mask50 | 4.00589 | 0.0333333 | 0 | 93 | 90 | 3 |
| initial | unconditional\_tcr/short | mask75 | 4.09862 | 0.0441176 | 0 | 142 | 136 | 6 |
| initial | unconditional\_tcr/short | mask100 | 4.03079 | 0.046875 | 0 | 200 | 192 | 8 |
| initial | unconditional\_tcr/short | mask\_mean | 4.05013 | 0.0373315 | 0 | — | — | — |
| latest | ab\_antigen/long | mask25 | 1.5634 | 0.425856 | 0 | 530 | 526 | 4 |
| latest | ab\_antigen/long | mask50 | 1.56137 | 0.427443 | 0 | 1019 | 1013 | 6 |
| latest | ab\_antigen/long | mask75 | 1.63022 | 0.412533 | 0 | 1538 | 1532 | 6 |
| latest | ab\_antigen/long | mask100 | 2.03742 | 0.270588 | 0 | 2048 | 2040 | 8 |
| latest | ab\_antigen/long | mask\_mean | 1.6981 | 0.384105 | 0 | — | — | — |
| latest | ab\_antigen/medium | mask25 | 1.63034 | 0.390946 | 0 | 246 | 243 | 3 |
| latest | ab\_antigen/medium | mask50 | 1.73246 | 0.348606 | 0 | 507 | 502 | 5 |
| latest | ab\_antigen/medium | mask75 | 1.76936 | 0.336412 | 0 | 765 | 758 | 7 |
| latest | ab\_antigen/medium | mask100 | 2.0336 | 0.298228 | 0 | 1024 | 1016 | 8 |
| latest | ab\_antigen/medium | mask\_mean | 1.79144 | 0.343548 | 0 | — | — | — |
| latest | ab\_antigen/short | mask25 | 2.28628 | 0.235294 | 0 | 38 | 34 | 4 |
| latest | ab\_antigen/short | mask50 | 2.19973 | 0.215909 | 0 | 93 | 88 | 5 |
| latest | ab\_antigen/short | mask75 | 2.25779 | 0.214286 | 0 | 145 | 140 | 5 |
| latest | ab\_antigen/short | mask100 | 2.30089 | 0.197917 | 0 | 200 | 192 | 8 |
| latest | ab\_antigen/short | mask\_mean | 2.26117 | 0.215851 | 0 | — | — | — |
| latest | tcr\_peptide/long | mask25 | 1.67532 | 0.335484 | 0 | 466 | 465 | 1 |
| latest | tcr\_peptide/long | mask50 | 1.73075 | 0.374108 | 0 | 985 | 981 | 4 |
| latest | tcr\_peptide/long | mask75 | 1.78592 | 0.364682 | 0 | 1484 | 1478 | 6 |
| latest | tcr\_peptide/long | mask100 | 2.09166 | 0.275 | 0 | 2048 | 2040 | 8 |
| latest | tcr\_peptide/long | mask\_mean | 1.82091 | 0.337318 | 0 | — | — | — |
| latest | tcr\_peptide/medium | mask25 | 1.75039 | 0.311203 | 0 | 242 | 241 | 1 |
| latest | tcr\_peptide/medium | mask50 | 1.78666 | 0.318725 | 0 | 506 | 502 | 4 |
| latest | tcr\_peptide/medium | mask75 | 1.88855 | 0.336842 | 0 | 767 | 760 | 7 |
| latest | tcr\_peptide/medium | mask100 | 2.18065 | 0.265748 | 0 | 1024 | 1016 | 8 |
| latest | tcr\_peptide/medium | mask\_mean | 1.90156 | 0.30813 | 0 | — | — | — |
| latest | tcr\_peptide/short | mask25 | 2.54271 | 0.25 | 0 | 51 | 48 | 3 |
| latest | tcr\_peptide/short | mask50 | 2.54252 | 0.230769 | 0 | 110 | 104 | 6 |
| latest | tcr\_peptide/short | mask75 | 2.4099 | 0.240876 | 0 | 144 | 137 | 7 |
| latest | tcr\_peptide/short | mask100 | 2.33067 | 0.244792 | 0 | 200 | 192 | 8 |
| latest | tcr\_peptide/short | mask\_mean | 2.45645 | 0.241609 | 0 | — | — | — |
| latest | tcr\_pmhc/long | mask25 | 1.39834 | 0.490672 | 0 | 540 | 536 | 4 |
| latest | tcr\_pmhc/long | mask50 | 1.39775 | 0.459381 | 0 | 1038 | 1034 | 4 |
| latest | tcr\_pmhc/long | mask75 | 1.4649 | 0.438867 | 0 | 1562 | 1554 | 8 |
| latest | tcr\_pmhc/long | mask100 | 1.91536 | 0.314706 | 0 | 2048 | 2040 | 8 |
| latest | tcr\_pmhc/long | mask\_mean | 1.54409 | 0.425907 | 0 | — | — | — |
| latest | tcr\_pmhc/medium | mask25 | 1.29155 | 0.504132 | 0 | 244 | 242 | 2 |
| latest | tcr\_pmhc/medium | mask50 | 1.37281 | 0.48643 | 0 | 484 | 479 | 5 |
| latest | tcr\_pmhc/medium | mask75 | 1.49431 | 0.456437 | 0 | 777 | 769 | 8 |
| latest | tcr\_pmhc/medium | mask100 | 2.04427 | 0.294291 | 0 | 1024 | 1016 | 8 |
| latest | tcr\_pmhc/medium | mask\_mean | 1.55074 | 0.435323 | 0 | — | — | — |
| latest | tcr\_pmhc/short | mask25 | 2.07877 | 0.354167 | 0 | 51 | 48 | 3 |
| latest | tcr\_pmhc/short | mask50 | 1.96207 | 0.373494 | 0 | 86 | 83 | 3 |
| latest | tcr\_pmhc/short | mask75 | 2.1062 | 0.272109 | 0 | 152 | 147 | 5 |
| latest | tcr\_pmhc/short | mask100 | 2.37999 | 0.21875 | 0 | 200 | 192 | 8 |
| latest | tcr\_pmhc/short | mask\_mean | 2.13176 | 0.30463 | 0 | — | — | — |
| latest | unconditional\_ab/long | mask25 | 1.61213 | 0.414953 | 0 | 538 | 535 | 3 |
| latest | unconditional\_ab/long | mask50 | 1.68033 | 0.361377 | 0 | 1050 | 1046 | 4 |
| latest | unconditional\_ab/long | mask75 | 1.75856 | 0.378414 | 0 | 1545 | 1538 | 7 |
| latest | unconditional\_ab/long | mask100 | 2.07177 | 0.336765 | 0 | 2048 | 2040 | 8 |
| latest | unconditional\_ab/long | mask\_mean | 1.7807 | 0.372877 | 0 | — | — | — |
| latest | unconditional\_ab/medium | mask25 | 1.7003 | 0.351779 | 0 | 256 | 253 | 3 |
| latest | unconditional\_ab/medium | mask50 | 1.7656 | 0.314961 | 0 | 512 | 508 | 4 |
| latest | unconditional\_ab/medium | mask75 | 1.8318 | 0.332021 | 0 | 768 | 762 | 6 |
| latest | unconditional\_ab/medium | mask100 | 2.10758 | 0.298228 | 0 | 1024 | 1016 | 8 |
| latest | unconditional\_ab/medium | mask\_mean | 1.85132 | 0.324247 | 0 | — | — | — |
| latest | unconditional\_ab/short | mask25 | 2.28647 | 0.115385 | 0 | 55 | 52 | 3 |
| latest | unconditional\_ab/short | mask50 | 2.44009 | 0.193878 | 0 | 103 | 98 | 5 |
| latest | unconditional\_ab/short | mask75 | 2.55354 | 0.219858 | 0 | 148 | 141 | 7 |
| latest | unconditional\_ab/short | mask100 | 2.52694 | 0.229167 | 0 | 200 | 192 | 8 |
| latest | unconditional\_ab/short | mask\_mean | 2.45176 | 0.189572 | 0 | — | — | — |
| latest | unconditional\_single/long | mask25 | 1.46337 | 0.432836 | 0 | 269 | 268 | 1 |
| latest | unconditional\_single/long | mask50 | 1.52864 | 0.441905 | 0 | 527 | 525 | 2 |
| latest | unconditional\_single/long | mask75 | 1.65973 | 0.388889 | 0 | 778 | 774 | 4 |
| latest | unconditional\_single/long | mask100 | 2.12668 | 0.305882 | 0 | 1024 | 1020 | 4 |
| latest | unconditional\_single/long | mask\_mean | 1.69461 | 0.392378 | 0 | — | — | — |
| latest | unconditional\_single/medium | mask25 | 1.62919 | 0.40367 | 0 | 110 | 109 | 1 |
| latest | unconditional\_single/medium | mask50 | 1.61894 | 0.426724 | 0 | 234 | 232 | 2 |
| latest | unconditional\_single/medium | mask75 | 1.76261 | 0.384196 | 0 | 371 | 367 | 4 |
| latest | unconditional\_single/medium | mask100 | 2.17772 | 0.330709 | 0 | 512 | 508 | 4 |
| latest | unconditional\_single/medium | mask\_mean | 1.79712 | 0.386325 | 0 | — | — | — |
| latest | unconditional\_single/short | mask25 | 2.62139 | 0.307692 | 0 | 28 | 26 | 2 |
| latest | unconditional\_single/short | mask50 | 2.69099 | 0.18 | 0 | 52 | 50 | 2 |
| latest | unconditional\_single/short | mask75 | 2.50666 | 0.181818 | 0 | 80 | 77 | 3 |
| latest | unconditional\_single/short | mask100 | 2.3531 | 0.291667 | 0 | 100 | 96 | 4 |
| latest | unconditional\_single/short | mask\_mean | 2.54303 | 0.240294 | 0 | — | — | — |
| latest | unconditional\_tcr/long | mask25 | 1.59857 | 0.365049 | 0 | 516 | 515 | 1 |
| latest | unconditional\_tcr/long | mask50 | 1.64773 | 0.363728 | 0 | 989 | 987 | 2 |
| latest | unconditional\_tcr/long | mask75 | 1.76992 | 0.351511 | 0 | 1527 | 1522 | 5 |
| latest | unconditional\_tcr/long | mask100 | 2.13347 | 0.305392 | 0 | 2048 | 2040 | 8 |
| latest | unconditional\_tcr/long | mask\_mean | 1.78742 | 0.34642 | 0 | — | — | — |
| latest | unconditional\_tcr/medium | mask25 | 1.43886 | 0.433198 | — | 247 | 247 | 0 |
| latest | unconditional\_tcr/medium | mask50 | 1.49223 | 0.416499 | 0 | 500 | 497 | 3 |
| latest | unconditional\_tcr/medium | mask75 | 1.59024 | 0.384824 | 0 | 744 | 738 | 6 |
| latest | unconditional\_tcr/medium | mask100 | 1.99864 | 0.345472 | 0 | 1024 | 1016 | 8 |
| latest | unconditional\_tcr/medium | mask\_mean | 1.62999 | 0.394998 | 0 | — | — | — |
| latest | unconditional\_tcr/short | mask25 | 2.26578 | 0.25 | — | 40 | 40 | 0 |
| latest | unconditional\_tcr/short | mask50 | 2.29461 | 0.222222 | 0 | 93 | 90 | 3 |
| latest | unconditional\_tcr/short | mask75 | 2.31985 | 0.235294 | 0 | 142 | 136 | 6 |
| latest | unconditional\_tcr/short | mask100 | 2.36653 | 0.213542 | 0 | 200 | 192 | 8 |
| latest | unconditional\_tcr/short | mask\_mean | 2.31169 | 0.230265 | 0 | — | — | — |
| train\_probe | ab\_antigen/long | mask25 | 1.35206 | 0.462264 | — | 530 | 530 | 0 |
| train\_probe | ab\_antigen/long | mask50 | 1.38354 | 0.455868 | 0 | 1034 | 1031 | 3 |
| train\_probe | ab\_antigen/long | mask75 | 1.48476 | 0.41747 | 0 | 1563 | 1557 | 6 |
| train\_probe | ab\_antigen/long | mask100 | 2.04655 | 0.304412 | 0 | 2048 | 2040 | 8 |
| train\_probe | ab\_antigen/long | mask\_mean | 1.56673 | 0.410003 | 0 | — | — | — |
| train\_probe | ab\_antigen/medium | mask25 | 1.74606 | 0.377289 | 0 | 275 | 273 | 2 |
| train\_probe | ab\_antigen/medium | mask50 | 1.77831 | 0.366171 | 0 | 542 | 538 | 4 |
| train\_probe | ab\_antigen/medium | mask75 | 1.85242 | 0.359536 | 0 | 781 | 776 | 5 |
| train\_probe | ab\_antigen/medium | mask100 | 2.04081 | 0.329724 | 0 | 1024 | 1016 | 8 |
| train\_probe | ab\_antigen/medium | mask\_mean | 1.8544 | 0.35818 | 0 | — | — | — |
| train\_probe | ab\_antigen/short | mask25 | 2.25066 | 0.340426 | 0 | 52 | 47 | 5 |
| train\_probe | ab\_antigen/short | mask50 | 2.19737 | 0.254902 | 0 | 109 | 102 | 7 |
| train\_probe | ab\_antigen/short | mask75 | 2.18221 | 0.344595 | 0 | 155 | 148 | 7 |
| train\_probe | ab\_antigen/short | mask100 | 2.32077 | 0.291667 | 0 | 200 | 192 | 8 |
| train\_probe | ab\_antigen/short | mask\_mean | 2.23775 | 0.307897 | 0 | — | — | — |
| train\_probe | tcr\_peptide/long | mask25 | 1.35558 | 0.506073 | 0 | 495 | 494 | 1 |
| train\_probe | tcr\_peptide/long | mask50 | 1.41382 | 0.461089 | 0 | 1033 | 1028 | 5 |
| train\_probe | tcr\_peptide/long | mask75 | 1.55184 | 0.397503 | 0 | 1530 | 1522 | 8 |
| train\_probe | tcr\_peptide/long | mask100 | 2.04995 | 0.296569 | 0 | 2048 | 2040 | 8 |
| train\_probe | tcr\_peptide/long | mask\_mean | 1.5928 | 0.415309 | 0 | — | — | — |
| train\_probe | tcr\_peptide/medium | mask25 | 1.50435 | 0.412698 | — | 252 | 252 | 0 |
| train\_probe | tcr\_peptide/medium | mask50 | 1.59352 | 0.405458 | 0 | 517 | 513 | 4 |
| train\_probe | tcr\_peptide/medium | mask75 | 1.65525 | 0.385942 | 0 | 760 | 754 | 6 |
| train\_probe | tcr\_peptide/medium | mask100 | 2.10529 | 0.298228 | 0 | 1024 | 1016 | 8 |
| train\_probe | tcr\_peptide/medium | mask\_mean | 1.7146 | 0.375582 | 0 | — | — | — |
| train\_probe | tcr\_peptide/short | mask25 | 2.1835 | 0.23913 | 0 | 51 | 46 | 5 |
| train\_probe | tcr\_peptide/short | mask50 | 1.95779 | 0.3 | 0 | 95 | 90 | 5 |
| train\_probe | tcr\_peptide/short | mask75 | 2.03987 | 0.294118 | 0 | 142 | 136 | 6 |
| train\_probe | tcr\_peptide/short | mask100 | 2.19769 | 0.333333 | 0 | 200 | 192 | 8 |
| train\_probe | tcr\_peptide/short | mask\_mean | 2.09471 | 0.291645 | 0 | — | — | — |
| train\_probe | tcr\_pmhc/long | mask25 | 1.38008 | 0.484674 | 0 | 526 | 522 | 4 |
| train\_probe | tcr\_pmhc/long | mask50 | 1.46206 | 0.447139 | 0 | 1036 | 1031 | 5 |
| train\_probe | tcr\_pmhc/long | mask75 | 1.587 | 0.416505 | 0 | 1557 | 1551 | 6 |
| train\_probe | tcr\_pmhc/long | mask100 | 1.97469 | 0.318627 | 0 | 2048 | 2040 | 8 |
| train\_probe | tcr\_pmhc/long | mask\_mean | 1.60096 | 0.416736 | 0 | — | — | — |
| train\_probe | tcr\_pmhc/medium | mask25 | 1.39401 | 0.487179 | 0 | 276 | 273 | 3 |
| train\_probe | tcr\_pmhc/medium | mask50 | 1.48009 | 0.404806 | 0 | 549 | 541 | 8 |
| train\_probe | tcr\_pmhc/medium | mask75 | 1.58208 | 0.37707 | 0 | 793 | 785 | 8 |
| train\_probe | tcr\_pmhc/medium | mask100 | 2.11479 | 0.258858 | 0 | 1024 | 1016 | 8 |
| train\_probe | tcr\_pmhc/medium | mask\_mean | 1.64274 | 0.381978 | 0 | — | — | — |
| train\_probe | tcr\_pmhc/short | mask25 | 2.41218 | 0.229167 | 0 | 50 | 48 | 2 |
| train\_probe | tcr\_pmhc/short | mask50 | 2.40835 | 0.2 | 0 | 103 | 100 | 3 |
| train\_probe | tcr\_pmhc/short | mask75 | 2.51565 | 0.17931 | 0 | 152 | 145 | 7 |
| train\_probe | tcr\_pmhc/short | mask100 | 2.36296 | 0.213542 | 0 | 200 | 192 | 8 |
| train\_probe | tcr\_pmhc/short | mask\_mean | 2.42479 | 0.205505 | 0 | — | — | — |
| train\_probe | unconditional\_ab/long | mask25 | 1.43959 | 0.442688 | 0 | 509 | 506 | 3 |
| train\_probe | unconditional\_ab/long | mask50 | 1.44097 | 0.470297 | 0 | 1016 | 1010 | 6 |
| train\_probe | unconditional\_ab/long | mask75 | 1.53582 | 0.414478 | 0 | 1569 | 1561 | 8 |
| train\_probe | unconditional\_ab/long | mask100 | 2.01496 | 0.321078 | 0 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_ab/long | mask\_mean | 1.60784 | 0.412135 | 0 | — | — | — |
| train\_probe | unconditional\_ab/medium | mask25 | 1.6447 | 0.419087 | 0 | 243 | 241 | 2 |
| train\_probe | unconditional\_ab/medium | mask50 | 1.71789 | 0.372093 | 0 | 521 | 516 | 5 |
| train\_probe | unconditional\_ab/medium | mask75 | 1.81534 | 0.362681 | 0 | 766 | 761 | 5 |
| train\_probe | unconditional\_ab/medium | mask100 | 2.06553 | 0.329724 | 0 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_ab/medium | mask\_mean | 1.81087 | 0.370896 | 0 | — | — | — |
| train\_probe | unconditional\_ab/short | mask25 | 2.3965 | 0.211538 | 0 | 53 | 52 | 1 |
| train\_probe | unconditional\_ab/short | mask50 | 2.27064 | 0.153846 | 0 | 106 | 104 | 2 |
| train\_probe | unconditional\_ab/short | mask75 | 2.33135 | 0.221519 | 0 | 163 | 158 | 5 |
| train\_probe | unconditional\_ab/short | mask100 | 2.52921 | 0.197917 | 0 | 200 | 192 | 8 |
| train\_probe | unconditional\_ab/short | mask\_mean | 2.38193 | 0.196205 | 0 | — | — | — |
| train\_probe | unconditional\_single/long | mask25 | 1.51432 | 0.397541 | 0 | 245 | 244 | 1 |
| train\_probe | unconditional\_single/long | mask50 | 1.54634 | 0.352342 | 0 | 492 | 491 | 1 |
| train\_probe | unconditional\_single/long | mask75 | 1.72793 | 0.346354 | 0 | 771 | 768 | 3 |
| train\_probe | unconditional\_single/long | mask100 | 2.17946 | 0.243137 | 0 | 1024 | 1020 | 4 |
| train\_probe | unconditional\_single/long | mask\_mean | 1.74201 | 0.334844 | 0 | — | — | — |
| train\_probe | unconditional\_single/medium | mask25 | 1.58191 | 0.415385 | 0 | 132 | 130 | 2 |
| train\_probe | unconditional\_single/medium | mask50 | 1.65443 | 0.464286 | 0 | 255 | 252 | 3 |
| train\_probe | unconditional\_single/medium | mask75 | 1.65579 | 0.432718 | 0 | 382 | 379 | 3 |
| train\_probe | unconditional\_single/medium | mask100 | 2.02269 | 0.362205 | 0 | 512 | 508 | 4 |
| train\_probe | unconditional\_single/medium | mask\_mean | 1.72871 | 0.418648 | 0 | — | — | — |
| train\_probe | unconditional\_single/short | mask25 | 2.45983 | 0.0769231 | 0 | 28 | 26 | 2 |
| train\_probe | unconditional\_single/short | mask50 | 2.44674 | 0.204082 | 0 | 52 | 49 | 3 |
| train\_probe | unconditional\_single/short | mask75 | 2.43865 | 0.157143 | 0 | 73 | 70 | 3 |
| train\_probe | unconditional\_single/short | mask100 | 2.35185 | 0.229167 | 0 | 100 | 96 | 4 |
| train\_probe | unconditional\_single/short | mask\_mean | 2.42427 | 0.166829 | 0 | — | — | — |
| train\_probe | unconditional\_tcr/long | mask25 | 1.50952 | 0.500952 | — | 525 | 525 | 0 |
| train\_probe | unconditional\_tcr/long | mask50 | 1.55445 | 0.466986 | 0 | 1048 | 1045 | 3 |
| train\_probe | unconditional\_tcr/long | mask75 | 1.71235 | 0.391139 | 0 | 1585 | 1580 | 5 |
| train\_probe | unconditional\_tcr/long | mask100 | 2.16017 | 0.27402 | 0 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_tcr/long | mask\_mean | 1.73412 | 0.408274 | 0 | — | — | — |
| train\_probe | unconditional\_tcr/medium | mask25 | 1.48772 | 0.429119 | — | 261 | 261 | 0 |
| train\_probe | unconditional\_tcr/medium | mask50 | 1.53843 | 0.45674 | 0 | 498 | 497 | 1 |
| train\_probe | unconditional\_tcr/medium | mask75 | 1.66422 | 0.420772 | 0 | 755 | 751 | 4 |
| train\_probe | unconditional\_tcr/medium | mask100 | 1.91035 | 0.376969 | 0 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_tcr/medium | mask\_mean | 1.65018 | 0.4209 | 0 | — | — | — |
| train\_probe | unconditional\_tcr/short | mask25 | 2.07986 | 0.217391 | — | 46 | 46 | 0 |
| train\_probe | unconditional\_tcr/short | mask50 | 2.40265 | 0.173913 | 0 | 97 | 92 | 5 |
| train\_probe | unconditional\_tcr/short | mask75 | 2.34721 | 0.191781 | 0 | 152 | 146 | 6 |
| train\_probe | unconditional\_tcr/short | mask100 | 2.49435 | 0.182292 | 0 | 200 | 192 | 8 |
| train\_probe | unconditional\_tcr/short | mask\_mean | 2.33102 | 0.191344 | 0 | — | — | — |

### 1790802160361769452\_r27\_refined\_value\_residual\_0b46ff76：分层 mask 指标

源码目录：/vepfs-mlp2/c20250601/251105016/project/dllm\_test/output/immune\_research\_releases/r20\_r27\_20261001；完成/目标 step：25/25。

kind=synthetic\_generalization\_diagnostic；dataset_kind=deterministic\_artificial\_sequences；biological_quality_claim=False。

| 时点 | mask level | 指标 | 值 |
| --- | --- | --- | --- |
| initial | mask25 | mask\_ratio | 0.25 |
| initial | mask25 | selected\_count | 4467 |
| initial | mask25 | aa\_count | 4428 |
| initial | mask25 | terminal\_count | 39 |
| initial | mask25 | loss | 3.90573 |
| initial | mask25 | aa\_loss | 3.89496 |
| initial | mask25 | aa\_accuracy | 0.0643631 |
| initial | mask25 | terminal\_loss | 5.12936 |
| initial | mask25 | terminal\_accuracy | 0 |
| initial | mask50 | mask\_ratio | 0.5 |
| initial | mask50 | selected\_count | 8888 |
| initial | mask50 | aa\_count | 8819 |
| initial | mask50 | terminal\_count | 69 |
| initial | mask50 | loss | 4.11095 |
| initial | mask50 | aa\_loss | 4.10142 |
| initial | mask50 | aa\_accuracy | 0.0666742 |
| initial | mask50 | terminal\_loss | 5.32885 |
| initial | mask50 | terminal\_accuracy | 0 |
| initial | mask75 | mask\_ratio | 0.75 |
| initial | mask75 | selected\_count | 13437 |
| initial | mask75 | aa\_count | 13330 |
| initial | mask75 | terminal\_count | 107 |
| initial | mask75 | loss | 4.25822 |
| initial | mask75 | aa\_loss | 4.24812 |
| initial | mask75 | aa\_accuracy | 0.0588897 |
| initial | mask75 | terminal\_loss | 5.51605 |
| initial | mask75 | terminal\_accuracy | 0 |
| initial | mask100 | mask\_ratio | 1 |
| initial | mask100 | selected\_count | 17996 |
| initial | mask100 | aa\_count | 17864 |
| initial | mask100 | terminal\_count | 132 |
| initial | mask100 | loss | 4.24321 |
| initial | mask100 | aa\_loss | 4.23335 |
| initial | mask100 | aa\_accuracy | 0.0667824 |
| initial | mask100 | terminal\_loss | 5.57776 |
| initial | mask100 | terminal\_accuracy | 0 |
| initial | mask\_mean | loss | 4.12953 |
| initial | mask\_mean | aa\_loss | 4.11946 |
| initial | mask\_mean | terminal\_loss | 5.388 |
| initial | mask\_mean | aa\_accuracy | 0.0641774 |
| initial | mask\_mean | terminal\_accuracy | 0 |
| latest | mask25 | mask\_ratio | 0.25 |
| latest | mask25 | selected\_count | 4467 |
| latest | mask25 | aa\_count | 4428 |
| latest | mask25 | terminal\_count | 39 |
| latest | mask25 | loss | 1.76488 |
| latest | mask25 | aa\_loss | 1.73568 |
| latest | mask25 | aa\_accuracy | 0.342593 |
| latest | mask25 | terminal\_loss | 5.08032 |
| latest | mask25 | terminal\_accuracy | 0 |
| latest | mask50 | mask\_ratio | 0.5 |
| latest | mask50 | selected\_count | 8888 |
| latest | mask50 | aa\_count | 8819 |
| latest | mask50 | terminal\_count | 69 |
| latest | mask50 | loss | 1.76493 |
| latest | mask50 | aa\_loss | 1.7405 |
| latest | mask50 | aa\_accuracy | 0.342783 |
| latest | mask50 | terminal\_loss | 4.88774 |
| latest | mask50 | terminal\_accuracy | 0 |
| latest | mask75 | mask\_ratio | 0.75 |
| latest | mask75 | selected\_count | 13437 |
| latest | mask75 | aa\_count | 13330 |
| latest | mask75 | terminal\_count | 107 |
| latest | mask75 | loss | 1.78116 |
| latest | mask75 | aa\_loss | 1.75757 |
| latest | mask75 | aa\_accuracy | 0.335259 |
| latest | mask75 | terminal\_loss | 4.71934 |
| latest | mask75 | terminal\_accuracy | 0 |
| latest | mask100 | mask\_ratio | 1 |
| latest | mask100 | selected\_count | 17996 |
| latest | mask100 | aa\_count | 17864 |
| latest | mask100 | terminal\_count | 132 |
| latest | mask100 | loss | 1.9652 |
| latest | mask100 | aa\_loss | 1.94926 |
| latest | mask100 | aa\_accuracy | 0.28689 |
| latest | mask100 | terminal\_loss | 4.12293 |
| latest | mask100 | terminal\_accuracy | 0 |
| latest | mask\_mean | loss | 1.81904 |
| latest | mask\_mean | aa\_loss | 1.79575 |
| latest | mask\_mean | terminal\_loss | 4.70258 |
| latest | mask\_mean | aa\_accuracy | 0.326881 |
| latest | mask\_mean | terminal\_accuracy | 0 |

评测曲线（原始 step，四档等权均值）：

| step | mean loss | mean AA acc | mean end acc | gradient norm |
| --- | --- | --- | --- | --- |
| 25 | 1.81904 | 0.326881 | 0 | 3.94664 |

有条件/无条件曲线（每档按token计数合并，再平均四档；缺失分组不推断数值）：

| step | 子集 | mean loss | mean AA | mean end | Full-mask AA | Full-mask end | Full-mask AA count | Full-mask end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | conditional | 4.19569 | 0.0633803 | 0 | 0.0707102 | 0 | 9744 | 72 |
| initial | unconditional | 4.05032 | 0.065109 | 0 | 0.062069 | 0 | 8120 | 60 |
| 25 | conditional | 1.77652 | 0.324365 | 0 | 0.290127 | 0 | 9744 | 72 |
| 25 | unconditional | 1.87007 | 0.32979 | 0 | 0.283005 | 0 | 8120 | 60 |

人工数据 manifest：d40bbbbc1c4294eed843d88bcdd0cf193886de6c2d7e9db9069bbf86e931c022；精确 target sequence 重叠=0；精确 record 重叠=0。

跨 radius32 覆盖：{&quot;possible&quot;: true, &quot;train&quot;: true, &quot;valid&quot;: true}。生成规则共享是诊断设计的一部分，不能当作真实下游去重证书。


按任务/链长分组（全局 level 按 token 计数汇总，下面保留各组计数；不同组不得直接比较未经加权的总 loss）：

| 时点 | group/length | mask level | loss | AA acc | end acc | selected count | AA count | end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | ab\_antigen/long | mask25 | 4.06387 | 0.0304183 | 0 | 530 | 526 | 4 |
| initial | ab\_antigen/long | mask50 | 4.40229 | 0.0306022 | 0 | 1019 | 1013 | 6 |
| initial | ab\_antigen/long | mask75 | 4.65354 | 0.0306789 | 0 | 1538 | 1532 | 6 |
| initial | ab\_antigen/long | mask100 | 4.64382 | 0.0313726 | 0 | 2048 | 2040 | 8 |
| initial | ab\_antigen/long | mask\_mean | 4.44088 | 0.030768 | 0 | — | — | — |
| initial | ab\_antigen/medium | mask25 | 4.4492 | 0.156379 | 0 | 246 | 243 | 3 |
| initial | ab\_antigen/medium | mask50 | 4.52118 | 0.161355 | 0 | 507 | 502 | 5 |
| initial | ab\_antigen/medium | mask75 | 4.59706 | 0.154354 | 0 | 765 | 758 | 7 |
| initial | ab\_antigen/medium | mask100 | 4.58314 | 0.15748 | 0 | 1024 | 1016 | 8 |
| initial | ab\_antigen/medium | mask\_mean | 4.53764 | 0.157392 | 0 | — | — | — |
| initial | ab\_antigen/short | mask25 | 5.25912 | 0 | 0 | 38 | 34 | 4 |
| initial | ab\_antigen/short | mask50 | 4.90384 | 0.0340909 | 0 | 93 | 88 | 5 |
| initial | ab\_antigen/short | mask75 | 4.77905 | 0.0428571 | 0 | 145 | 140 | 5 |
| initial | ab\_antigen/short | mask100 | 4.70735 | 0.0625 | 0 | 200 | 192 | 8 |
| initial | ab\_antigen/short | mask\_mean | 4.91234 | 0.034862 | 0 | — | — | — |
| initial | tcr\_peptide/long | mask25 | 3.57421 | 0.105376 | 0 | 466 | 465 | 1 |
| initial | tcr\_peptide/long | mask50 | 3.84408 | 0.114169 | 0 | 985 | 981 | 4 |
| initial | tcr\_peptide/long | mask75 | 4.06469 | 0.0784844 | 0 | 1484 | 1478 | 6 |
| initial | tcr\_peptide/long | mask100 | 4.09122 | 0.12451 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_peptide/long | mask\_mean | 3.89355 | 0.105635 | 0 | — | — | — |
| initial | tcr\_peptide/medium | mask25 | 4.0786 | 0.0414938 | 0 | 242 | 241 | 1 |
| initial | tcr\_peptide/medium | mask50 | 4.10604 | 0.0179283 | 0 | 506 | 502 | 4 |
| initial | tcr\_peptide/medium | mask75 | 4.04149 | 0.0355263 | 0 | 767 | 760 | 7 |
| initial | tcr\_peptide/medium | mask100 | 3.93722 | 0.0610236 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_peptide/medium | mask\_mean | 4.04084 | 0.038993 | 0 | — | — | — |
| initial | tcr\_peptide/short | mask25 | 4.078 | 0.0416667 | 0 | 51 | 48 | 3 |
| initial | tcr\_peptide/short | mask50 | 4.01756 | 0.0576923 | 0 | 110 | 104 | 6 |
| initial | tcr\_peptide/short | mask75 | 4.03916 | 0.0656934 | 0 | 144 | 137 | 7 |
| initial | tcr\_peptide/short | mask100 | 4.01962 | 0.09375 | 0 | 200 | 192 | 8 |
| initial | tcr\_peptide/short | mask\_mean | 4.03858 | 0.0647006 | 0 | — | — | — |
| initial | tcr\_pmhc/long | mask25 | 3.45892 | 0.0391791 | 0 | 540 | 536 | 4 |
| initial | tcr\_pmhc/long | mask50 | 3.84629 | 0.0357834 | 0 | 1038 | 1034 | 4 |
| initial | tcr\_pmhc/long | mask75 | 4.25287 | 0.028314 | 0 | 1562 | 1554 | 8 |
| initial | tcr\_pmhc/long | mask100 | 4.43587 | 0.0210784 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_pmhc/long | mask\_mean | 3.99849 | 0.0310887 | 0 | — | — | — |
| initial | tcr\_pmhc/medium | mask25 | 4.24292 | 0.053719 | 0 | 244 | 242 | 2 |
| initial | tcr\_pmhc/medium | mask50 | 4.39061 | 0.039666 | 0 | 484 | 479 | 5 |
| initial | tcr\_pmhc/medium | mask75 | 4.47512 | 0.053316 | 0 | 777 | 769 | 8 |
| initial | tcr\_pmhc/medium | mask100 | 4.46935 | 0.0629921 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_pmhc/medium | mask\_mean | 4.3945 | 0.0524233 | 0 | — | — | — |
| initial | tcr\_pmhc/short | mask25 | 4.46829 | 0.0208333 | 0 | 51 | 48 | 3 |
| initial | tcr\_pmhc/short | mask50 | 4.44788 | 0.0240964 | 0 | 86 | 83 | 3 |
| initial | tcr\_pmhc/short | mask75 | 4.30161 | 0.0612245 | 0 | 152 | 147 | 5 |
| initial | tcr\_pmhc/short | mask100 | 4.29595 | 0.0625 | 0 | 200 | 192 | 8 |
| initial | tcr\_pmhc/short | mask\_mean | 4.37843 | 0.0421636 | 0 | — | — | — |
| initial | unconditional\_ab/long | mask25 | 3.72304 | 0.166355 | 0 | 538 | 535 | 3 |
| initial | unconditional\_ab/long | mask50 | 4.01507 | 0.179732 | 0 | 1050 | 1046 | 4 |
| initial | unconditional\_ab/long | mask75 | 4.18285 | 0.182055 | 0 | 1545 | 1538 | 7 |
| initial | unconditional\_ab/long | mask100 | 4.13411 | 0.187255 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_ab/long | mask\_mean | 4.01377 | 0.178849 | 0 | — | — | — |
| initial | unconditional\_ab/medium | mask25 | 4.52467 | 0.0474308 | 0 | 256 | 253 | 3 |
| initial | unconditional\_ab/medium | mask50 | 4.62929 | 0.0570866 | 0 | 512 | 508 | 4 |
| initial | unconditional\_ab/medium | mask75 | 4.59985 | 0.0629921 | 0 | 768 | 762 | 6 |
| initial | unconditional\_ab/medium | mask100 | 4.55057 | 0.0610236 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_ab/medium | mask\_mean | 4.5761 | 0.0571333 | 0 | — | — | — |
| initial | unconditional\_ab/short | mask25 | 4.32037 | 0.0384615 | 0 | 55 | 52 | 3 |
| initial | unconditional\_ab/short | mask50 | 4.14468 | 0.0408163 | 0 | 103 | 98 | 5 |
| initial | unconditional\_ab/short | mask75 | 4.10821 | 0.0567376 | 0 | 148 | 141 | 7 |
| initial | unconditional\_ab/short | mask100 | 4.07402 | 0.0625 | 0 | 200 | 192 | 8 |
| initial | unconditional\_ab/short | mask\_mean | 4.16182 | 0.0496289 | 0 | — | — | — |
| initial | unconditional\_single/long | mask25 | 3.56576 | 0.0485075 | 0 | 269 | 268 | 1 |
| initial | unconditional\_single/long | mask50 | 3.54172 | 0.0685714 | 0 | 527 | 525 | 2 |
| initial | unconditional\_single/long | mask75 | 3.59429 | 0.0335917 | 0 | 778 | 774 | 4 |
| initial | unconditional\_single/long | mask100 | 3.64062 | 0 | 0 | 1024 | 1020 | 4 |
| initial | unconditional\_single/long | mask\_mean | 3.5856 | 0.0376677 | 0 | — | — | — |
| initial | unconditional\_single/medium | mask25 | 3.96295 | 0 | 0 | 110 | 109 | 1 |
| initial | unconditional\_single/medium | mask50 | 3.81305 | 0 | 0 | 234 | 232 | 2 |
| initial | unconditional\_single/medium | mask75 | 3.80054 | 0 | 0 | 371 | 367 | 4 |
| initial | unconditional\_single/medium | mask100 | 3.77909 | 0 | 0 | 512 | 508 | 4 |
| initial | unconditional\_single/medium | mask\_mean | 3.83891 | 0 | 0 | — | — | — |
| initial | unconditional\_single/short | mask25 | 3.34487 | 0 | 0 | 28 | 26 | 2 |
| initial | unconditional\_single/short | mask50 | 3.54422 | 0 | 0 | 52 | 50 | 2 |
| initial | unconditional\_single/short | mask75 | 3.40152 | 0 | 0 | 80 | 77 | 3 |
| initial | unconditional\_single/short | mask100 | 3.41363 | 0 | 0 | 100 | 96 | 4 |
| initial | unconditional\_single/short | mask\_mean | 3.42606 | 0 | 0 | — | — | — |
| initial | unconditional\_tcr/long | mask25 | 3.66417 | 0.0368932 | 0 | 516 | 515 | 1 |
| initial | unconditional\_tcr/long | mask50 | 3.99255 | 0.029382 | 0 | 989 | 987 | 2 |
| initial | unconditional\_tcr/long | mask75 | 4.20409 | 0 | 0 | 1527 | 1522 | 5 |
| initial | unconditional\_tcr/long | mask100 | 4.05612 | 0.0156863 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_tcr/long | mask\_mean | 3.97923 | 0.0204904 | 0 | — | — | — |
| initial | unconditional\_tcr/medium | mask25 | 4.30167 | 0 | — | 247 | 247 | 0 |
| initial | unconditional\_tcr/medium | mask50 | 4.41837 | 0 | 0 | 500 | 497 | 3 |
| initial | unconditional\_tcr/medium | mask75 | 4.40098 | 0 | 0 | 744 | 738 | 6 |
| initial | unconditional\_tcr/medium | mask100 | 4.30471 | 0 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_tcr/medium | mask\_mean | 4.35643 | 0 | 0 | — | — | — |
| initial | unconditional\_tcr/short | mask25 | 4.1561 | 0 | — | 40 | 40 | 0 |
| initial | unconditional\_tcr/short | mask50 | 4.12996 | 0.0222222 | 0 | 93 | 90 | 3 |
| initial | unconditional\_tcr/short | mask75 | 4.13885 | 0.0514706 | 0 | 142 | 136 | 6 |
| initial | unconditional\_tcr/short | mask100 | 4.10155 | 0.0833333 | 0 | 200 | 192 | 8 |
| initial | unconditional\_tcr/short | mask\_mean | 4.13161 | 0.0392565 | 0 | — | — | — |
| latest | ab\_antigen/long | mask25 | 1.73483 | 0.361217 | 0 | 530 | 526 | 4 |
| latest | ab\_antigen/long | mask50 | 1.70727 | 0.39388 | 0 | 1019 | 1013 | 6 |
| latest | ab\_antigen/long | mask75 | 1.69819 | 0.375326 | 0 | 1538 | 1532 | 6 |
| latest | ab\_antigen/long | mask100 | 1.84711 | 0.317647 | 0 | 2048 | 2040 | 8 |
| latest | ab\_antigen/long | mask\_mean | 1.74685 | 0.362017 | 0 | — | — | — |
| latest | ab\_antigen/medium | mask25 | 1.83147 | 0.300412 | 0 | 246 | 243 | 3 |
| latest | ab\_antigen/medium | mask50 | 1.87609 | 0.294821 | 0 | 507 | 502 | 5 |
| latest | ab\_antigen/medium | mask75 | 1.86748 | 0.279683 | 0 | 765 | 758 | 7 |
| latest | ab\_antigen/medium | mask100 | 1.94556 | 0.249016 | 0 | 1024 | 1016 | 8 |
| latest | ab\_antigen/medium | mask\_mean | 1.88015 | 0.280983 | 0 | — | — | — |
| latest | ab\_antigen/short | mask25 | 2.20726 | 0.352941 | 0 | 38 | 34 | 4 |
| latest | ab\_antigen/short | mask50 | 2.11766 | 0.295455 | 0 | 93 | 88 | 5 |
| latest | ab\_antigen/short | mask75 | 2.06799 | 0.285714 | 0 | 145 | 140 | 5 |
| latest | ab\_antigen/short | mask100 | 2.11974 | 0.28125 | 0 | 200 | 192 | 8 |
| latest | ab\_antigen/short | mask\_mean | 2.12816 | 0.30384 | 0 | — | — | — |
| latest | tcr\_peptide/long | mask25 | 1.90565 | 0.275269 | 0 | 466 | 465 | 1 |
| latest | tcr\_peptide/long | mask50 | 1.88459 | 0.284404 | 0 | 985 | 981 | 4 |
| latest | tcr\_peptide/long | mask75 | 1.87959 | 0.304465 | 0 | 1484 | 1478 | 6 |
| latest | tcr\_peptide/long | mask100 | 1.93973 | 0.235784 | 0 | 2048 | 2040 | 8 |
| latest | tcr\_peptide/long | mask\_mean | 1.90239 | 0.274981 | 0 | — | — | — |
| latest | tcr\_peptide/medium | mask25 | 1.86093 | 0.290456 | 0 | 242 | 241 | 1 |
| latest | tcr\_peptide/medium | mask50 | 1.87225 | 0.298805 | 0 | 506 | 502 | 4 |
| latest | tcr\_peptide/medium | mask75 | 1.89622 | 0.289474 | 0 | 767 | 760 | 7 |
| latest | tcr\_peptide/medium | mask100 | 1.92583 | 0.268701 | 0 | 1024 | 1016 | 8 |
| latest | tcr\_peptide/medium | mask\_mean | 1.88881 | 0.286859 | 0 | — | — | — |
| latest | tcr\_peptide/short | mask25 | 2.31182 | 0.25 | 0 | 51 | 48 | 3 |
| latest | tcr\_peptide/short | mask50 | 2.2694 | 0.240385 | 0 | 110 | 104 | 6 |
| latest | tcr\_peptide/short | mask75 | 2.19417 | 0.233577 | 0 | 144 | 137 | 7 |
| latest | tcr\_peptide/short | mask100 | 2.20974 | 0.25 | 0 | 200 | 192 | 8 |
| latest | tcr\_peptide/short | mask\_mean | 2.24628 | 0.24349 | 0 | — | — | — |
| latest | tcr\_pmhc/long | mask25 | 1.63128 | 0.324627 | 0 | 540 | 536 | 4 |
| latest | tcr\_pmhc/long | mask50 | 1.55751 | 0.347195 | 0 | 1038 | 1034 | 4 |
| latest | tcr\_pmhc/long | mask75 | 1.5519 | 0.346203 | 0 | 1562 | 1554 | 8 |
| latest | tcr\_pmhc/long | mask100 | 1.59324 | 0.307353 | 0 | 2048 | 2040 | 8 |
| latest | tcr\_pmhc/long | mask\_mean | 1.58348 | 0.331345 | 0 | — | — | — |
| latest | tcr\_pmhc/medium | mask25 | 1.5104 | 0.46281 | 0 | 244 | 242 | 2 |
| latest | tcr\_pmhc/medium | mask50 | 1.55391 | 0.440501 | 0 | 484 | 479 | 5 |
| latest | tcr\_pmhc/medium | mask75 | 1.59656 | 0.43303 | 0 | 777 | 769 | 8 |
| latest | tcr\_pmhc/medium | mask100 | 1.6692 | 0.374016 | 0 | 1024 | 1016 | 8 |
| latest | tcr\_pmhc/medium | mask\_mean | 1.58252 | 0.427589 | 0 | — | — | — |
| latest | tcr\_pmhc/short | mask25 | 1.88391 | 0.375 | 0 | 51 | 48 | 3 |
| latest | tcr\_pmhc/short | mask50 | 1.90464 | 0.349398 | 0 | 86 | 83 | 3 |
| latest | tcr\_pmhc/short | mask75 | 1.93711 | 0.326531 | 0 | 152 | 147 | 5 |
| latest | tcr\_pmhc/short | mask100 | 2.04291 | 0.328125 | 0 | 200 | 192 | 8 |
| latest | tcr\_pmhc/short | mask\_mean | 1.94214 | 0.344763 | 0 | — | — | — |
| latest | unconditional\_ab/long | mask25 | 1.78885 | 0.342056 | 0 | 538 | 535 | 3 |
| latest | unconditional\_ab/long | mask50 | 1.78658 | 0.335564 | 0 | 1050 | 1046 | 4 |
| latest | unconditional\_ab/long | mask75 | 1.82178 | 0.340052 | 0 | 1545 | 1538 | 7 |
| latest | unconditional\_ab/long | mask100 | 2.15006 | 0.329412 | 0 | 2048 | 2040 | 8 |
| latest | unconditional\_ab/long | mask\_mean | 1.88682 | 0.336771 | 0 | — | — | — |
| latest | unconditional\_ab/medium | mask25 | 1.79825 | 0.347826 | 0 | 256 | 253 | 3 |
| latest | unconditional\_ab/medium | mask50 | 1.8587 | 0.346457 | 0 | 512 | 508 | 4 |
| latest | unconditional\_ab/medium | mask75 | 1.87067 | 0.330709 | 0 | 768 | 762 | 6 |
| latest | unconditional\_ab/medium | mask100 | 2.18309 | 0.281496 | 0 | 1024 | 1016 | 8 |
| latest | unconditional\_ab/medium | mask\_mean | 1.92768 | 0.326622 | 0 | — | — | — |
| latest | unconditional\_ab/short | mask25 | 1.98824 | 0.25 | 0 | 55 | 52 | 3 |
| latest | unconditional\_ab/short | mask50 | 2.18465 | 0.234694 | 0 | 103 | 98 | 5 |
| latest | unconditional\_ab/short | mask75 | 2.2801 | 0.234043 | 0 | 148 | 141 | 7 |
| latest | unconditional\_ab/short | mask100 | 2.4333 | 0.21875 | 0 | 200 | 192 | 8 |
| latest | unconditional\_ab/short | mask\_mean | 2.22157 | 0.234372 | 0 | — | — | — |
| latest | unconditional\_single/long | mask25 | 1.62985 | 0.41791 | 0 | 269 | 268 | 1 |
| latest | unconditional\_single/long | mask50 | 1.65509 | 0.350476 | 0 | 527 | 525 | 2 |
| latest | unconditional\_single/long | mask75 | 1.6962 | 0.330749 | 0 | 778 | 774 | 4 |
| latest | unconditional\_single/long | mask100 | 2.09687 | 0.305882 | 0 | 1024 | 1020 | 4 |
| latest | unconditional\_single/long | mask\_mean | 1.7695 | 0.351255 | 0 | — | — | — |
| latest | unconditional\_single/medium | mask25 | 1.66533 | 0.357798 | 0 | 110 | 109 | 1 |
| latest | unconditional\_single/medium | mask50 | 1.65248 | 0.400862 | 0 | 234 | 232 | 2 |
| latest | unconditional\_single/medium | mask75 | 1.77219 | 0.326975 | 0 | 371 | 367 | 4 |
| latest | unconditional\_single/medium | mask100 | 2.14418 | 0.330709 | 0 | 512 | 508 | 4 |
| latest | unconditional\_single/medium | mask\_mean | 1.80854 | 0.354086 | 0 | — | — | — |
| latest | unconditional\_single/short | mask25 | 2.73198 | 0.269231 | 0 | 28 | 26 | 2 |
| latest | unconditional\_single/short | mask50 | 2.66173 | 0.2 | 0 | 52 | 50 | 2 |
| latest | unconditional\_single/short | mask75 | 2.39821 | 0.246753 | 0 | 80 | 77 | 3 |
| latest | unconditional\_single/short | mask100 | 2.30694 | 0.291667 | 0 | 100 | 96 | 4 |
| latest | unconditional\_single/short | mask\_mean | 2.52471 | 0.251913 | 0 | — | — | — |
| latest | unconditional\_tcr/long | mask25 | 1.78685 | 0.357282 | 0 | 516 | 515 | 1 |
| latest | unconditional\_tcr/long | mask50 | 1.78495 | 0.356636 | 0 | 989 | 987 | 2 |
| latest | unconditional\_tcr/long | mask75 | 1.81261 | 0.343627 | 0 | 1527 | 1522 | 5 |
| latest | unconditional\_tcr/long | mask100 | 2.10251 | 0.216667 | 0 | 2048 | 2040 | 8 |
| latest | unconditional\_tcr/long | mask\_mean | 1.87173 | 0.318553 | 0 | — | — | — |
| latest | unconditional\_tcr/medium | mask25 | 1.5583 | 0.364372 | — | 247 | 247 | 0 |
| latest | unconditional\_tcr/medium | mask50 | 1.58089 | 0.372233 | 0 | 500 | 497 | 3 |
| latest | unconditional\_tcr/medium | mask75 | 1.64482 | 0.355014 | 0 | 744 | 738 | 6 |
| latest | unconditional\_tcr/medium | mask100 | 1.98008 | 0.309055 | 0 | 1024 | 1016 | 8 |
| latest | unconditional\_tcr/medium | mask\_mean | 1.69102 | 0.350169 | 0 | — | — | — |
| latest | unconditional\_tcr/short | mask25 | 2.27327 | 0.3 | — | 40 | 40 | 0 |
| latest | unconditional\_tcr/short | mask50 | 2.22927 | 0.255556 | 0 | 93 | 90 | 3 |
| latest | unconditional\_tcr/short | mask75 | 2.28889 | 0.242647 | 0 | 142 | 136 | 6 |
| latest | unconditional\_tcr/short | mask100 | 2.32321 | 0.177083 | 0 | 200 | 192 | 8 |
| latest | unconditional\_tcr/short | mask\_mean | 2.27866 | 0.243821 | 0 | — | — | — |
| train\_probe | ab\_antigen/long | mask25 | 1.61994 | 0.326415 | — | 530 | 530 | 0 |
| train\_probe | ab\_antigen/long | mask50 | 1.64573 | 0.304559 | 0 | 1034 | 1031 | 3 |
| train\_probe | ab\_antigen/long | mask75 | 1.68954 | 0.265896 | 0 | 1563 | 1557 | 6 |
| train\_probe | ab\_antigen/long | mask100 | 1.88491 | 0.25049 | 0 | 2048 | 2040 | 8 |
| train\_probe | ab\_antigen/long | mask\_mean | 1.71003 | 0.28684 | 0 | — | — | — |
| train\_probe | ab\_antigen/medium | mask25 | 1.9059 | 0.300366 | 0 | 275 | 273 | 2 |
| train\_probe | ab\_antigen/medium | mask50 | 1.91224 | 0.325279 | 0 | 542 | 538 | 4 |
| train\_probe | ab\_antigen/medium | mask75 | 1.90895 | 0.33634 | 0 | 781 | 776 | 5 |
| train\_probe | ab\_antigen/medium | mask100 | 2.03957 | 0.259843 | 0 | 1024 | 1016 | 8 |
| train\_probe | ab\_antigen/medium | mask\_mean | 1.94167 | 0.305457 | 0 | — | — | — |
| train\_probe | ab\_antigen/short | mask25 | 2.15229 | 0.361702 | 0 | 52 | 47 | 5 |
| train\_probe | ab\_antigen/short | mask50 | 2.18685 | 0.27451 | 0 | 109 | 102 | 7 |
| train\_probe | ab\_antigen/short | mask75 | 2.21962 | 0.283784 | 0 | 155 | 148 | 7 |
| train\_probe | ab\_antigen/short | mask100 | 2.24867 | 0.286458 | 0 | 200 | 192 | 8 |
| train\_probe | ab\_antigen/short | mask\_mean | 2.20186 | 0.301614 | 0 | — | — | — |
| train\_probe | tcr\_peptide/long | mask25 | 1.55391 | 0.366397 | 0 | 495 | 494 | 1 |
| train\_probe | tcr\_peptide/long | mask50 | 1.56872 | 0.364786 | 0 | 1033 | 1028 | 5 |
| train\_probe | tcr\_peptide/long | mask75 | 1.6161 | 0.344284 | 0 | 1530 | 1522 | 8 |
| train\_probe | tcr\_peptide/long | mask100 | 1.7144 | 0.291667 | 0 | 2048 | 2040 | 8 |
| train\_probe | tcr\_peptide/long | mask\_mean | 1.61329 | 0.341783 | 0 | — | — | — |
| train\_probe | tcr\_peptide/medium | mask25 | 1.63662 | 0.369048 | — | 252 | 252 | 0 |
| train\_probe | tcr\_peptide/medium | mask50 | 1.66661 | 0.34308 | 0 | 517 | 513 | 4 |
| train\_probe | tcr\_peptide/medium | mask75 | 1.69685 | 0.318302 | 0 | 760 | 754 | 6 |
| train\_probe | tcr\_peptide/medium | mask100 | 1.81347 | 0.269685 | 0 | 1024 | 1016 | 8 |
| train\_probe | tcr\_peptide/medium | mask\_mean | 1.70339 | 0.325029 | 0 | — | — | — |
| train\_probe | tcr\_peptide/short | mask25 | 2.08707 | 0.369565 | 0 | 51 | 46 | 5 |
| train\_probe | tcr\_peptide/short | mask50 | 1.98208 | 0.377778 | 0 | 95 | 90 | 5 |
| train\_probe | tcr\_peptide/short | mask75 | 2.01682 | 0.338235 | 0 | 142 | 136 | 6 |
| train\_probe | tcr\_peptide/short | mask100 | 2.09067 | 0.333333 | 0 | 200 | 192 | 8 |
| train\_probe | tcr\_peptide/short | mask\_mean | 2.04416 | 0.354728 | 0 | — | — | — |
| train\_probe | tcr\_pmhc/long | mask25 | 1.60361 | 0.369732 | 0 | 526 | 522 | 4 |
| train\_probe | tcr\_pmhc/long | mask50 | 1.66492 | 0.353055 | 0 | 1036 | 1031 | 5 |
| train\_probe | tcr\_pmhc/long | mask75 | 1.72533 | 0.334623 | 0 | 1557 | 1551 | 6 |
| train\_probe | tcr\_pmhc/long | mask100 | 1.77428 | 0.317157 | 0 | 2048 | 2040 | 8 |
| train\_probe | tcr\_pmhc/long | mask\_mean | 1.69204 | 0.343642 | 0 | — | — | — |
| train\_probe | tcr\_pmhc/medium | mask25 | 1.69473 | 0.380952 | 0 | 276 | 273 | 3 |
| train\_probe | tcr\_pmhc/medium | mask50 | 1.73605 | 0.332717 | 0 | 549 | 541 | 8 |
| train\_probe | tcr\_pmhc/medium | mask75 | 1.75391 | 0.342675 | 0 | 793 | 785 | 8 |
| train\_probe | tcr\_pmhc/medium | mask100 | 1.82248 | 0.300197 | 0 | 1024 | 1016 | 8 |
| train\_probe | tcr\_pmhc/medium | mask\_mean | 1.75179 | 0.339135 | 0 | — | — | — |
| train\_probe | tcr\_pmhc/short | mask25 | 2.37837 | 0.25 | 0 | 50 | 48 | 2 |
| train\_probe | tcr\_pmhc/short | mask50 | 2.38312 | 0.2 | 0 | 103 | 100 | 3 |
| train\_probe | tcr\_pmhc/short | mask75 | 2.4128 | 0.2 | 0 | 152 | 145 | 7 |
| train\_probe | tcr\_pmhc/short | mask100 | 2.30274 | 0.213542 | 0 | 200 | 192 | 8 |
| train\_probe | tcr\_pmhc/short | mask\_mean | 2.36926 | 0.215885 | 0 | — | — | — |
| train\_probe | unconditional\_ab/long | mask25 | 1.6212 | 0.312253 | 0 | 509 | 506 | 3 |
| train\_probe | unconditional\_ab/long | mask50 | 1.63611 | 0.276238 | 0 | 1016 | 1010 | 6 |
| train\_probe | unconditional\_ab/long | mask75 | 1.67974 | 0.294683 | 0 | 1569 | 1561 | 8 |
| train\_probe | unconditional\_ab/long | mask100 | 2.05616 | 0.314216 | 0 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_ab/long | mask\_mean | 1.7483 | 0.299347 | 0 | — | — | — |
| train\_probe | unconditional\_ab/medium | mask25 | 1.80613 | 0.294606 | 0 | 243 | 241 | 2 |
| train\_probe | unconditional\_ab/medium | mask50 | 1.8578 | 0.30814 | 0 | 521 | 516 | 5 |
| train\_probe | unconditional\_ab/medium | mask75 | 1.86253 | 0.323259 | 0 | 766 | 761 | 5 |
| train\_probe | unconditional\_ab/medium | mask100 | 2.14328 | 0.311024 | 0 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_ab/medium | mask\_mean | 1.91743 | 0.309257 | 0 | — | — | — |
| train\_probe | unconditional\_ab/short | mask25 | 2.27711 | 0.25 | 0 | 53 | 52 | 1 |
| train\_probe | unconditional\_ab/short | mask50 | 2.24233 | 0.25 | 0 | 106 | 104 | 2 |
| train\_probe | unconditional\_ab/short | mask75 | 2.29669 | 0.183544 | 0 | 163 | 158 | 5 |
| train\_probe | unconditional\_ab/short | mask100 | 2.47651 | 0.197917 | 0 | 200 | 192 | 8 |
| train\_probe | unconditional\_ab/short | mask\_mean | 2.32316 | 0.220365 | 0 | — | — | — |
| train\_probe | unconditional\_single/long | mask25 | 1.69465 | 0.307377 | 0 | 245 | 244 | 1 |
| train\_probe | unconditional\_single/long | mask50 | 1.68604 | 0.295316 | 0 | 492 | 491 | 1 |
| train\_probe | unconditional\_single/long | mask75 | 1.78007 | 0.236979 | 0 | 771 | 768 | 3 |
| train\_probe | unconditional\_single/long | mask100 | 2.10357 | 0.243137 | 0 | 1024 | 1020 | 4 |
| train\_probe | unconditional\_single/long | mask\_mean | 1.81608 | 0.270702 | 0 | — | — | — |
| train\_probe | unconditional\_single/medium | mask25 | 1.72048 | 0.415385 | 0 | 132 | 130 | 2 |
| train\_probe | unconditional\_single/medium | mask50 | 1.76865 | 0.376984 | 0 | 255 | 252 | 3 |
| train\_probe | unconditional\_single/medium | mask75 | 1.78367 | 0.366755 | 0 | 382 | 379 | 3 |
| train\_probe | unconditional\_single/medium | mask100 | 2.02104 | 0.362205 | 0 | 512 | 508 | 4 |
| train\_probe | unconditional\_single/medium | mask\_mean | 1.82346 | 0.380332 | 0 | — | — | — |
| train\_probe | unconditional\_single/short | mask25 | 2.55273 | 0.115385 | 0 | 28 | 26 | 2 |
| train\_probe | unconditional\_single/short | mask50 | 2.43305 | 0.163265 | 0 | 52 | 49 | 3 |
| train\_probe | unconditional\_single/short | mask75 | 2.32605 | 0.257143 | 0 | 73 | 70 | 3 |
| train\_probe | unconditional\_single/short | mask100 | 2.34756 | 0.229167 | 0 | 100 | 96 | 4 |
| train\_probe | unconditional\_single/short | mask\_mean | 2.41485 | 0.19124 | 0 | — | — | — |
| train\_probe | unconditional\_tcr/long | mask25 | 1.74558 | 0.339048 | — | 525 | 525 | 0 |
| train\_probe | unconditional\_tcr/long | mask50 | 1.79187 | 0.313876 | 0 | 1048 | 1045 | 3 |
| train\_probe | unconditional\_tcr/long | mask75 | 1.85561 | 0.289873 | 0 | 1585 | 1580 | 5 |
| train\_probe | unconditional\_tcr/long | mask100 | 2.2187 | 0.153922 | 0 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_tcr/long | mask\_mean | 1.90294 | 0.27418 | 0 | — | — | — |
| train\_probe | unconditional\_tcr/medium | mask25 | 1.70512 | 0.363985 | — | 261 | 261 | 0 |
| train\_probe | unconditional\_tcr/medium | mask50 | 1.71464 | 0.354125 | 0 | 498 | 497 | 1 |
| train\_probe | unconditional\_tcr/medium | mask75 | 1.75567 | 0.375499 | 0 | 755 | 751 | 4 |
| train\_probe | unconditional\_tcr/medium | mask100 | 1.95499 | 0.372047 | 0 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_tcr/medium | mask\_mean | 1.78261 | 0.366414 | 0 | — | — | — |
| train\_probe | unconditional\_tcr/short | mask25 | 2.0427 | 0.195652 | — | 46 | 46 | 0 |
| train\_probe | unconditional\_tcr/short | mask50 | 2.31668 | 0.184783 | 0 | 97 | 92 | 5 |
| train\_probe | unconditional\_tcr/short | mask75 | 2.29948 | 0.212329 | 0 | 152 | 146 | 6 |
| train\_probe | unconditional\_tcr/short | mask100 | 2.47267 | 0.114583 | 0 | 200 | 192 | 8 |
| train\_probe | unconditional\_tcr/short | mask\_mean | 2.28288 | 0.176837 | 0 | — | — | — |

### 1790803305045499169\_r20\_refined\_control\_a37dc5ab：分层 mask 指标

源码目录：/vepfs-mlp2/c20250601/251105016/project/dllm\_test/output/immune\_research\_releases/r20\_r27\_20261001；完成/目标 step：500/500。

kind=synthetic\_generalization\_diagnostic；dataset_kind=deterministic\_artificial\_sequences；biological_quality_claim=False。

| 时点 | mask level | 指标 | 值 |
| --- | --- | --- | --- |
| initial | mask25 | mask\_ratio | 0.25 |
| initial | mask25 | selected\_count | 4467 |
| initial | mask25 | aa\_count | 4428 |
| initial | mask25 | terminal\_count | 39 |
| initial | mask25 | loss | 3.83723 |
| initial | mask25 | aa\_loss | 3.83333 |
| initial | mask25 | aa\_accuracy | 0.0440379 |
| initial | mask25 | terminal\_loss | 4.27954 |
| initial | mask25 | terminal\_accuracy | 0 |
| initial | mask50 | mask\_ratio | 0.5 |
| initial | mask50 | selected\_count | 8888 |
| initial | mask50 | aa\_count | 8819 |
| initial | mask50 | terminal\_count | 69 |
| initial | mask50 | loss | 3.95698 |
| initial | mask50 | aa\_loss | 3.95337 |
| initial | mask50 | aa\_accuracy | 0.0481914 |
| initial | mask50 | terminal\_loss | 4.41734 |
| initial | mask50 | terminal\_accuracy | 0 |
| initial | mask75 | mask\_ratio | 0.75 |
| initial | mask75 | selected\_count | 13437 |
| initial | mask75 | aa\_count | 13330 |
| initial | mask75 | terminal\_count | 107 |
| initial | mask75 | loss | 4.0171 |
| initial | mask75 | aa\_loss | 4.01295 |
| initial | mask75 | aa\_accuracy | 0.0588897 |
| initial | mask75 | terminal\_loss | 4.53343 |
| initial | mask75 | terminal\_accuracy | 0 |
| initial | mask100 | mask\_ratio | 1 |
| initial | mask100 | selected\_count | 17996 |
| initial | mask100 | aa\_count | 17864 |
| initial | mask100 | terminal\_count | 132 |
| initial | mask100 | loss | 3.99692 |
| initial | mask100 | aa\_loss | 3.99259 |
| initial | mask100 | aa\_accuracy | 0.033979 |
| initial | mask100 | terminal\_loss | 4.58199 |
| initial | mask100 | terminal\_accuracy | 0 |
| initial | mask\_mean | loss | 3.95206 |
| initial | mask\_mean | aa\_loss | 3.94806 |
| initial | mask\_mean | terminal\_loss | 4.45307 |
| initial | mask\_mean | aa\_accuracy | 0.0462745 |
| initial | mask\_mean | terminal\_accuracy | 0 |
| latest | mask25 | mask\_ratio | 0.25 |
| latest | mask25 | selected\_count | 4467 |
| latest | mask25 | aa\_count | 4428 |
| latest | mask25 | terminal\_count | 39 |
| latest | mask25 | loss | 0.000122114 |
| latest | mask25 | aa\_loss | 5.94221e-05 |
| latest | mask25 | aa\_accuracy | 1 |
| latest | mask25 | terminal\_loss | 0.00724003 |
| latest | mask25 | terminal\_accuracy | 1 |
| latest | mask50 | mask\_ratio | 0.5 |
| latest | mask50 | selected\_count | 8888 |
| latest | mask50 | aa\_count | 8819 |
| latest | mask50 | terminal\_count | 69 |
| latest | mask50 | loss | 0.00341219 |
| latest | mask50 | aa\_loss | 0.00342677 |
| latest | mask50 | aa\_accuracy | 0.998639 |
| latest | mask50 | terminal\_loss | 0.00154863 |
| latest | mask50 | terminal\_accuracy | 1 |
| latest | mask75 | mask\_ratio | 0.75 |
| latest | mask75 | selected\_count | 13437 |
| latest | mask75 | aa\_count | 13330 |
| latest | mask75 | terminal\_count | 107 |
| latest | mask75 | loss | 0.0185435 |
| latest | mask75 | aa\_loss | 0.0186807 |
| latest | mask75 | aa\_accuracy | 0.992723 |
| latest | mask75 | terminal\_loss | 0.00144341 |
| latest | mask75 | terminal\_accuracy | 1 |
| latest | mask100 | mask\_ratio | 1 |
| latest | mask100 | selected\_count | 17996 |
| latest | mask100 | aa\_count | 17864 |
| latest | mask100 | terminal\_count | 132 |
| latest | mask100 | loss | 0.804176 |
| latest | mask100 | aa\_loss | 0.809785 |
| latest | mask100 | aa\_accuracy | 0.688368 |
| latest | mask100 | terminal\_loss | 0.0450202 |
| latest | mask100 | terminal\_accuracy | 1 |
| latest | mask\_mean | loss | 0.206563 |
| latest | mask\_mean | aa\_loss | 0.207988 |
| latest | mask\_mean | terminal\_loss | 0.0138131 |
| latest | mask\_mean | aa\_accuracy | 0.919933 |
| latest | mask\_mean | terminal\_accuracy | 1 |

评测曲线（原始 step，四档等权均值）：

| step | mean loss | mean AA acc | mean end acc | gradient norm |
| --- | --- | --- | --- | --- |
| 25 | 1.90782 | 0.300562 | 0 | 3.65246 |
| 50 | 1.46125 | 0.468203 | 0 | 3.62059 |
| 75 | 0.807219 | 0.733451 | 0 | 1.47287 |
| 100 | 0.581353 | 0.79743 | 0.298543 | 2.57774 |
| 125 | 0.497266 | 0.829814 | 0.926314 | 2.81366 |
| 150 | 0.424275 | 0.853103 | 0.847009 | 2.21197 |
| 175 | 0.387838 | 0.866344 | 0.989967 | 1.39138 |
| 200 | 0.359673 | 0.875424 | 0.959224 | 1.12381 |
| 225 | 0.279148 | 0.896209 | 0.974208 | 1.06439 |
| 250 | 0.247485 | 0.911683 | 0.970862 | 0.948095 |
| 275 | 0.237207 | 0.910596 | 0.984848 | 1.46468 |
| 300 | 0.267691 | 0.904493 | 0.992424 | 0.647583 |
| 325 | 0.21489 | 0.91924 | 0.963573 | 0.46808 |
| 350 | 0.220518 | 0.918916 | 1 | 0.574443 |
| 375 | 0.209315 | 0.920571 | 0.969697 | 1.06822 |
| 400 | 0.197626 | 0.924582 | 0.99053 | 0.323016 |
| 425 | 0.191129 | 0.919799 | 0.978724 | 0.809861 |
| 450 | 0.191286 | 0.92265 | 0.971591 | 0.430534 |
| 475 | 0.180679 | 0.929701 | 0.977273 | 0.308525 |
| 500 | 0.206563 | 0.919933 | 1 | 0.494194 |

有条件/无条件曲线（每档按token计数合并，再平均四档；缺失分组不推断数值）：

| step | 子集 | mean loss | mean AA | mean end | Full-mask AA | Full-mask end | Full-mask AA count | Full-mask end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | conditional | 3.99673 | 0.043826 | 0 | 0.034688 | 0 | 9744 | 72 |
| initial | unconditional | 3.89862 | 0.0492004 | 0 | 0.0331281 | 0 | 8120 | 60 |
| 25 | conditional | 1.84973 | 0.301396 | 0 | 0.272065 | 0 | 9744 | 72 |
| 25 | unconditional | 1.97741 | 0.299542 | 0 | 0.257882 | 0 | 8120 | 60 |
| 50 | conditional | 1.38361 | 0.49347 | 0 | 0.319479 | 0 | 9744 | 72 |
| 50 | unconditional | 1.55416 | 0.438024 | 0 | 0.18633 | 0 | 8120 | 60 |
| 75 | conditional | 0.735343 | 0.749662 | 0 | 0.311884 | 0 | 9744 | 72 |
| 75 | unconditional | 0.893138 | 0.714158 | 0 | 0.285591 | 0 | 8120 | 60 |
| 100 | conditional | 0.530096 | 0.813574 | 0.312032 | 0.342672 | 0.0138889 | 9744 | 72 |
| 100 | unconditional | 0.642765 | 0.778092 | 0.273859 | 0.232266 | 0.0666667 | 8120 | 60 |
| 125 | conditional | 0.439266 | 0.845607 | 0.965917 | 0.413896 | 0.972222 | 9744 | 72 |
| 125 | unconditional | 0.566826 | 0.810878 | 0.872073 | 0.296675 | 0.733333 | 8120 | 60 |
| 150 | conditional | 0.357778 | 0.876731 | 0.87281 | 0.524117 | 0.680556 | 9744 | 72 |
| 150 | unconditional | 0.504044 | 0.824757 | 0.81455 | 0.329803 | 0.533333 | 8120 | 60 |
| 175 | conditional | 0.296554 | 0.895512 | 1 | 0.595033 | 1 | 9744 | 72 |
| 175 | unconditional | 0.497361 | 0.831349 | 0.972884 | 0.352956 | 1 | 8120 | 60 |
| 200 | conditional | 0.27928 | 0.912458 | 0.958787 | 0.656096 | 0.916667 | 9744 | 72 |
| 200 | unconditional | 0.45614 | 0.830984 | 0.953869 | 0.344581 | 1 | 8120 | 60 |
| 225 | conditional | 0.152111 | 0.948344 | 0.99229 | 0.797516 | 0.986111 | 9744 | 72 |
| 225 | unconditional | 0.431581 | 0.833651 | 0.94881 | 0.356897 | 0.866667 | 8120 | 60 |
| 250 | conditional | 0.108094 | 0.964424 | 0.986111 | 0.861966 | 0.944444 | 9744 | 72 |
| 250 | unconditional | 0.414751 | 0.848398 | 0.94881 | 0.414655 | 0.866667 | 8120 | 60 |
| 275 | conditional | 0.0775455 | 0.974255 | 1 | 0.897989 | 1 | 9744 | 72 |
| 275 | unconditional | 0.428793 | 0.83421 | 0.966667 | 0.355911 | 0.866667 | 8120 | 60 |
| 300 | conditional | 0.115383 | 0.959775 | 1 | 0.846264 | 1 | 9744 | 72 |
| 300 | unconditional | 0.450453 | 0.838159 | 0.983333 | 0.371798 | 0.933333 | 8120 | 60 |
| 325 | conditional | 0.0379595 | 0.9883 | 0.974929 | 0.953407 | 0.916667 | 9744 | 72 |
| 325 | unconditional | 0.427193 | 0.836373 | 0.95 | 0.365148 | 0.8 | 8120 | 60 |
| 350 | conditional | 0.0380935 | 0.990088 | 1 | 0.960489 | 1 | 9744 | 72 |
| 350 | unconditional | 0.439423 | 0.833514 | 1 | 0.350985 | 1 | 8120 | 60 |
| 375 | conditional | 0.0394964 | 0.987972 | 1 | 0.955049 | 1 | 9744 | 72 |
| 375 | unconditional | 0.413095 | 0.839693 | 0.933333 | 0.38165 | 0.733333 | 8120 | 60 |
| 400 | conditional | 0.0196281 | 0.993329 | 0.982639 | 0.973317 | 0.930556 | 9744 | 72 |
| 400 | unconditional | 0.41122 | 0.842088 | 1 | 0.389901 | 1 | 8120 | 60 |
| 425 | conditional | 0.0151389 | 0.995664 | 0.974929 | 0.982656 | 0.916667 | 9744 | 72 |
| 425 | unconditional | 0.402315 | 0.828763 | 0.983333 | 0.331158 | 0.933333 | 8120 | 60 |
| 450 | conditional | 0.0189407 | 0.993618 | 0.989583 | 0.975985 | 0.958333 | 9744 | 72 |
| 450 | unconditional | 0.398087 | 0.837493 | 0.95 | 0.368227 | 0.8 | 8120 | 60 |
| 475 | conditional | 0.020099 | 0.991293 | 1 | 0.965723 | 1 | 9744 | 72 |
| 475 | unconditional | 0.373369 | 0.855795 | 0.95 | 0.439532 | 0.8 | 8120 | 60 |
| 500 | conditional | 0.035906 | 0.988103 | 1 | 0.952689 | 1 | 9744 | 72 |
| 500 | unconditional | 0.411351 | 0.838129 | 1 | 0.371182 | 1 | 8120 | 60 |

人工数据 manifest：d40bbbbc1c4294eed843d88bcdd0cf193886de6c2d7e9db9069bbf86e931c022；精确 target sequence 重叠=0；精确 record 重叠=0。

跨 radius32 覆盖：{&quot;possible&quot;: true, &quot;train&quot;: true, &quot;valid&quot;: true}。生成规则共享是诊断设计的一部分，不能当作真实下游去重证书。


按任务/链长分组（全局 level 按 token 计数汇总，下面保留各组计数；不同组不得直接比较未经加权的总 loss）：

| 时点 | group/length | mask level | loss | AA acc | end acc | selected count | AA count | end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | ab\_antigen/long | mask25 | 4.18791 | 0 | 0 | 530 | 526 | 4 |
| initial | ab\_antigen/long | mask50 | 4.46389 | 0 | 0 | 1019 | 1013 | 6 |
| initial | ab\_antigen/long | mask75 | 4.58208 | 0.0274151 | 0 | 1538 | 1532 | 6 |
| initial | ab\_antigen/long | mask100 | 4.47814 | 0.0303922 | 0 | 2048 | 2040 | 8 |
| initial | ab\_antigen/long | mask\_mean | 4.42801 | 0.0144518 | 0 | — | — | — |
| initial | ab\_antigen/medium | mask25 | 4.60088 | 0.0288066 | 0 | 246 | 243 | 3 |
| initial | ab\_antigen/medium | mask50 | 4.58408 | 0.0677291 | 0 | 507 | 502 | 5 |
| initial | ab\_antigen/medium | mask75 | 4.58341 | 0.0620053 | 0 | 765 | 758 | 7 |
| initial | ab\_antigen/medium | mask100 | 4.52137 | 0.0610236 | 0 | 1024 | 1016 | 8 |
| initial | ab\_antigen/medium | mask\_mean | 4.57243 | 0.0548911 | 0 | — | — | — |
| initial | ab\_antigen/short | mask25 | 4.71372 | 0.0588235 | 0 | 38 | 34 | 4 |
| initial | ab\_antigen/short | mask50 | 4.46987 | 0.0795455 | 0 | 93 | 88 | 5 |
| initial | ab\_antigen/short | mask75 | 4.39464 | 0.0785714 | 0 | 145 | 140 | 5 |
| initial | ab\_antigen/short | mask100 | 4.34276 | 0.09375 | 0 | 200 | 192 | 8 |
| initial | ab\_antigen/short | mask\_mean | 4.48025 | 0.0776726 | 0 | — | — | — |
| initial | tcr\_peptide/long | mask25 | 3.61712 | 0.088172 | 0 | 466 | 465 | 1 |
| initial | tcr\_peptide/long | mask50 | 3.7337 | 0.114169 | 0 | 985 | 981 | 4 |
| initial | tcr\_peptide/long | mask75 | 3.79636 | 0.116373 | 0 | 1484 | 1478 | 6 |
| initial | tcr\_peptide/long | mask100 | 3.82355 | 0.0720588 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_peptide/long | mask\_mean | 3.74268 | 0.0976934 | 0 | — | — | — |
| initial | tcr\_peptide/medium | mask25 | 3.81146 | 0.0248963 | 0 | 242 | 241 | 1 |
| initial | tcr\_peptide/medium | mask50 | 3.79279 | 0.0119522 | 0 | 506 | 502 | 4 |
| initial | tcr\_peptide/medium | mask75 | 3.72871 | 0 | 0 | 767 | 760 | 7 |
| initial | tcr\_peptide/medium | mask100 | 3.66509 | 0 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_peptide/medium | mask\_mean | 3.74952 | 0.00921211 | 0 | — | — | — |
| initial | tcr\_peptide/short | mask25 | 3.65752 | 0 | 0 | 51 | 48 | 3 |
| initial | tcr\_peptide/short | mask50 | 3.54852 | 0 | 0 | 110 | 104 | 6 |
| initial | tcr\_peptide/short | mask75 | 3.58623 | 0 | 0 | 144 | 137 | 7 |
| initial | tcr\_peptide/short | mask100 | 3.62608 | 0 | 0 | 200 | 192 | 8 |
| initial | tcr\_peptide/short | mask\_mean | 3.60459 | 0 | 0 | — | — | — |
| initial | tcr\_pmhc/long | mask25 | 3.25602 | 0.0391791 | 0 | 540 | 536 | 4 |
| initial | tcr\_pmhc/long | mask50 | 3.48187 | 0.0357834 | 0 | 1038 | 1034 | 4 |
| initial | tcr\_pmhc/long | mask75 | 3.6956 | 0.032175 | 0 | 1562 | 1554 | 8 |
| initial | tcr\_pmhc/long | mask100 | 3.82046 | 0.0181373 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_pmhc/long | mask\_mean | 3.56349 | 0.0313187 | 0 | — | — | — |
| initial | tcr\_pmhc/medium | mask25 | 4.03386 | 0.0950413 | 0 | 244 | 242 | 2 |
| initial | tcr\_pmhc/medium | mask50 | 4.13424 | 0.0793319 | 0 | 484 | 479 | 5 |
| initial | tcr\_pmhc/medium | mask75 | 4.21764 | 0.0520156 | 0 | 777 | 769 | 8 |
| initial | tcr\_pmhc/medium | mask100 | 4.2521 | 0.011811 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_pmhc/medium | mask\_mean | 4.15946 | 0.05955 | 0 | — | — | — |
| initial | tcr\_pmhc/short | mask25 | 4.12421 | 0 | 0 | 51 | 48 | 3 |
| initial | tcr\_pmhc/short | mask50 | 4.20187 | 0 | 0 | 86 | 83 | 3 |
| initial | tcr\_pmhc/short | mask75 | 4.10686 | 0 | 0 | 152 | 147 | 5 |
| initial | tcr\_pmhc/short | mask100 | 4.11286 | 0 | 0 | 200 | 192 | 8 |
| initial | tcr\_pmhc/short | mask\_mean | 4.13645 | 0 | 0 | — | — | — |
| initial | unconditional\_ab/long | mask25 | 3.91889 | 0.0317757 | 0 | 538 | 535 | 3 |
| initial | unconditional\_ab/long | mask50 | 4.14326 | 0.0229446 | 0 | 1050 | 1046 | 4 |
| initial | unconditional\_ab/long | mask75 | 4.2586 | 0.0630689 | 0 | 1545 | 1538 | 7 |
| initial | unconditional\_ab/long | mask100 | 4.17438 | 0.0588235 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_ab/long | mask\_mean | 4.12378 | 0.0441532 | 0 | — | — | — |
| initial | unconditional\_ab/medium | mask25 | 4.23935 | 0.146245 | 0 | 256 | 253 | 3 |
| initial | unconditional\_ab/medium | mask50 | 4.34042 | 0.137795 | 0 | 512 | 508 | 4 |
| initial | unconditional\_ab/medium | mask75 | 4.31506 | 0.143045 | 0 | 768 | 762 | 6 |
| initial | unconditional\_ab/medium | mask100 | 4.25781 | 0.140748 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_ab/medium | mask\_mean | 4.28816 | 0.141958 | 0 | — | — | — |
| initial | unconditional\_ab/short | mask25 | 3.98838 | 0.0961538 | 0 | 55 | 52 | 3 |
| initial | unconditional\_ab/short | mask50 | 3.93455 | 0.0612245 | 0 | 103 | 98 | 5 |
| initial | unconditional\_ab/short | mask75 | 3.90859 | 0.0567376 | 0 | 148 | 141 | 7 |
| initial | unconditional\_ab/short | mask100 | 3.87788 | 0.03125 | 0 | 200 | 192 | 8 |
| initial | unconditional\_ab/short | mask\_mean | 3.92735 | 0.0613415 | 0 | — | — | — |
| initial | unconditional\_single/long | mask25 | 3.42862 | 0.0485075 | 0 | 269 | 268 | 1 |
| initial | unconditional\_single/long | mask50 | 3.37246 | 0.08 | 0 | 527 | 525 | 2 |
| initial | unconditional\_single/long | mask75 | 3.3423 | 0.192506 | 0 | 778 | 774 | 4 |
| initial | unconditional\_single/long | mask100 | 3.3784 | 0 | 0 | 1024 | 1020 | 4 |
| initial | unconditional\_single/long | mask\_mean | 3.38045 | 0.0802535 | 0 | — | — | — |
| initial | unconditional\_single/medium | mask25 | 3.71448 | 0.0366972 | 0 | 110 | 109 | 1 |
| initial | unconditional\_single/medium | mask50 | 3.49804 | 0.0818966 | 0 | 234 | 232 | 2 |
| initial | unconditional\_single/medium | mask75 | 3.50775 | 0.163488 | 0 | 371 | 367 | 4 |
| initial | unconditional\_single/medium | mask100 | 3.501 | 0 | 0 | 512 | 508 | 4 |
| initial | unconditional\_single/medium | mask\_mean | 3.55532 | 0.0705204 | 0 | — | — | — |
| initial | unconditional\_single/short | mask25 | 3.11568 | 0 | 0 | 28 | 26 | 2 |
| initial | unconditional\_single/short | mask50 | 3.29004 | 0 | 0 | 52 | 50 | 2 |
| initial | unconditional\_single/short | mask75 | 3.10684 | 0 | 0 | 80 | 77 | 3 |
| initial | unconditional\_single/short | mask100 | 3.10388 | 0 | 0 | 100 | 96 | 4 |
| initial | unconditional\_single/short | mask\_mean | 3.15411 | 0 | 0 | — | — | — |
| initial | unconditional\_tcr/long | mask25 | 3.72215 | 0.0368932 | 0 | 516 | 515 | 1 |
| initial | unconditional\_tcr/long | mask50 | 3.86969 | 0.0303951 | 0 | 989 | 987 | 2 |
| initial | unconditional\_tcr/long | mask75 | 3.91864 | 0 | 0 | 1527 | 1522 | 5 |
| initial | unconditional\_tcr/long | mask100 | 3.85547 | 0 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_tcr/long | mask\_mean | 3.84149 | 0.0168221 | 0 | — | — | — |
| initial | unconditional\_tcr/medium | mask25 | 3.84478 | 0 | — | 247 | 247 | 0 |
| initial | unconditional\_tcr/medium | mask50 | 3.96335 | 0 | 0 | 500 | 497 | 3 |
| initial | unconditional\_tcr/medium | mask75 | 3.94344 | 0 | 0 | 744 | 738 | 6 |
| initial | unconditional\_tcr/medium | mask100 | 3.92848 | 0 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_tcr/medium | mask\_mean | 3.92001 | 0 | 0 | — | — | — |
| initial | unconditional\_tcr/short | mask25 | 4.04066 | 0 | — | 40 | 40 | 0 |
| initial | unconditional\_tcr/short | mask50 | 3.91056 | 0 | 0 | 93 | 90 | 3 |
| initial | unconditional\_tcr/short | mask75 | 3.96515 | 0 | 0 | 142 | 136 | 6 |
| initial | unconditional\_tcr/short | mask100 | 3.91533 | 0 | 0 | 200 | 192 | 8 |
| initial | unconditional\_tcr/short | mask\_mean | 3.95792 | 0 | 0 | — | — | — |
| latest | ab\_antigen/long | mask25 | 1.72164e-05 | 1 | 1 | 530 | 526 | 4 |
| latest | ab\_antigen/long | mask50 | 2.85444e-05 | 1 | 1 | 1019 | 1013 | 6 |
| latest | ab\_antigen/long | mask75 | 0.00312027 | 0.999347 | 1 | 1538 | 1532 | 6 |
| latest | ab\_antigen/long | mask100 | 0.168814 | 0.944118 | 1 | 2048 | 2040 | 8 |
| latest | ab\_antigen/long | mask\_mean | 0.0429949 | 0.985866 | 1 | — | — | — |
| latest | ab\_antigen/medium | mask25 | 4.54079e-05 | 1 | 1 | 246 | 243 | 3 |
| latest | ab\_antigen/medium | mask50 | 3.43782e-05 | 1 | 1 | 507 | 502 | 5 |
| latest | ab\_antigen/medium | mask75 | 0.000243781 | 1 | 1 | 765 | 758 | 7 |
| latest | ab\_antigen/medium | mask100 | 0.188683 | 0.925197 | 1 | 1024 | 1016 | 8 |
| latest | ab\_antigen/medium | mask\_mean | 0.0472516 | 0.981299 | 1 | — | — | — |
| latest | ab\_antigen/short | mask25 | 0.000234242 | 1 | 1 | 38 | 34 | 4 |
| latest | ab\_antigen/short | mask50 | 0.0011364 | 1 | 1 | 93 | 88 | 5 |
| latest | ab\_antigen/short | mask75 | 0.0309417 | 0.992857 | 1 | 145 | 140 | 5 |
| latest | ab\_antigen/short | mask100 | 0.0347628 | 0.989583 | 1 | 200 | 192 | 8 |
| latest | ab\_antigen/short | mask\_mean | 0.0167688 | 0.99561 | 1 | — | — | — |
| latest | tcr\_peptide/long | mask25 | 9.27873e-06 | 1 | 1 | 466 | 465 | 1 |
| latest | tcr\_peptide/long | mask50 | 1.41991e-05 | 1 | 1 | 985 | 981 | 4 |
| latest | tcr\_peptide/long | mask75 | 1.0608e-05 | 1 | 1 | 1484 | 1478 | 6 |
| latest | tcr\_peptide/long | mask100 | 0.0213011 | 0.995098 | 1 | 2048 | 2040 | 8 |
| latest | tcr\_peptide/long | mask\_mean | 0.0053338 | 0.998774 | 1 | — | — | — |
| latest | tcr\_peptide/medium | mask25 | 2.42864e-05 | 1 | 1 | 242 | 241 | 1 |
| latest | tcr\_peptide/medium | mask50 | 1.77001e-05 | 1 | 1 | 506 | 502 | 4 |
| latest | tcr\_peptide/medium | mask75 | 0.000276015 | 1 | 1 | 767 | 760 | 7 |
| latest | tcr\_peptide/medium | mask100 | 0.00849497 | 1 | 1 | 1024 | 1016 | 8 |
| latest | tcr\_peptide/medium | mask\_mean | 0.00220324 | 1 | 1 | — | — | — |
| latest | tcr\_peptide/short | mask25 | 1.40685e-05 | 1 | 1 | 51 | 48 | 3 |
| latest | tcr\_peptide/short | mask50 | 0.000154515 | 1 | 1 | 110 | 104 | 6 |
| latest | tcr\_peptide/short | mask75 | 0.000165229 | 1 | 1 | 144 | 137 | 7 |
| latest | tcr\_peptide/short | mask100 | 0.0117364 | 1 | 1 | 200 | 192 | 8 |
| latest | tcr\_peptide/short | mask\_mean | 0.00301754 | 1 | 1 | — | — | — |
| latest | tcr\_pmhc/long | mask25 | 1.14102e-05 | 1 | 1 | 540 | 536 | 4 |
| latest | tcr\_pmhc/long | mask50 | 1.07417e-05 | 1 | 1 | 1038 | 1034 | 4 |
| latest | tcr\_pmhc/long | mask75 | 3.79798e-05 | 1 | 1 | 1562 | 1554 | 8 |
| latest | tcr\_pmhc/long | mask100 | 0.373413 | 0.876471 | 1 | 2048 | 2040 | 8 |
| latest | tcr\_pmhc/long | mask\_mean | 0.0933683 | 0.969118 | 1 | — | — | — |
| latest | tcr\_pmhc/medium | mask25 | 0.00173504 | 1 | 1 | 244 | 242 | 2 |
| latest | tcr\_pmhc/medium | mask50 | 0.000277204 | 1 | 1 | 484 | 479 | 5 |
| latest | tcr\_pmhc/medium | mask75 | 0.000172537 | 1 | 1 | 777 | 769 | 8 |
| latest | tcr\_pmhc/medium | mask100 | 0.0224055 | 0.994094 | 1 | 1024 | 1016 | 8 |
| latest | tcr\_pmhc/medium | mask\_mean | 0.00614758 | 0.998524 | 1 | — | — | — |
| latest | tcr\_pmhc/short | mask25 | 2.34412e-05 | 1 | 1 | 51 | 48 | 3 |
| latest | tcr\_pmhc/short | mask50 | 3.2572e-05 | 1 | 1 | 86 | 83 | 3 |
| latest | tcr\_pmhc/short | mask75 | 0.00281872 | 1 | 1 | 152 | 147 | 5 |
| latest | tcr\_pmhc/short | mask100 | 0.0254957 | 0.994792 | 1 | 200 | 192 | 8 |
| latest | tcr\_pmhc/short | mask\_mean | 0.0070926 | 0.998698 | 1 | — | — | — |
| latest | unconditional\_ab/long | mask25 | 1.36599e-05 | 1 | 1 | 538 | 535 | 3 |
| latest | unconditional\_ab/long | mask50 | 1.9255e-05 | 1 | 1 | 1050 | 1046 | 4 |
| latest | unconditional\_ab/long | mask75 | 0.00174894 | 0.99935 | 1 | 1545 | 1538 | 7 |
| latest | unconditional\_ab/long | mask100 | 1.58018 | 0.342647 | 1 | 2048 | 2040 | 8 |
| latest | unconditional\_ab/long | mask\_mean | 0.39549 | 0.835499 | 1 | — | — | — |
| latest | unconditional\_ab/medium | mask25 | 1.39475e-05 | 1 | 1 | 256 | 253 | 3 |
| latest | unconditional\_ab/medium | mask50 | 3.51638e-05 | 1 | 1 | 512 | 508 | 4 |
| latest | unconditional\_ab/medium | mask75 | 0.00051606 | 1 | 1 | 768 | 762 | 6 |
| latest | unconditional\_ab/medium | mask100 | 1.57943 | 0.399606 | 1 | 1024 | 1016 | 8 |
| latest | unconditional\_ab/medium | mask\_mean | 0.394999 | 0.849902 | 1 | — | — | — |
| latest | unconditional\_ab/short | mask25 | 0.000294927 | 1 | 1 | 55 | 52 | 3 |
| latest | unconditional\_ab/short | mask50 | 0.15002 | 0.928571 | 1 | 103 | 98 | 5 |
| latest | unconditional\_ab/short | mask75 | 0.434167 | 0.836879 | 1 | 148 | 141 | 7 |
| latest | unconditional\_ab/short | mask100 | 1.15811 | 0.489583 | 1 | 200 | 192 | 8 |
| latest | unconditional\_ab/short | mask\_mean | 0.435649 | 0.813759 | 1 | — | — | — |
| latest | unconditional\_single/long | mask25 | 4.71613e-05 | 1 | 1 | 269 | 268 | 1 |
| latest | unconditional\_single/long | mask50 | 2.90553e-05 | 1 | 1 | 527 | 525 | 2 |
| latest | unconditional\_single/long | mask75 | 0.0476485 | 0.975452 | 1 | 778 | 774 | 4 |
| latest | unconditional\_single/long | mask100 | 1.68398 | 0.363726 | 1 | 1024 | 1020 | 4 |
| latest | unconditional\_single/long | mask\_mean | 0.432926 | 0.834794 | 1 | — | — | — |
| latest | unconditional\_single/medium | mask25 | 4.33011e-05 | 1 | 1 | 110 | 109 | 1 |
| latest | unconditional\_single/medium | mask50 | 8.83431e-05 | 1 | 1 | 234 | 232 | 2 |
| latest | unconditional\_single/medium | mask75 | 0.0252152 | 0.991826 | 1 | 371 | 367 | 4 |
| latest | unconditional\_single/medium | mask100 | 1.75638 | 0.348425 | 1 | 512 | 508 | 4 |
| latest | unconditional\_single/medium | mask\_mean | 0.445433 | 0.835063 | 1 | — | — | — |
| latest | unconditional\_single/short | mask25 | 0.00087708 | 1 | 1 | 28 | 26 | 2 |
| latest | unconditional\_single/short | mask50 | 0.266944 | 0.9 | 1 | 52 | 50 | 2 |
| latest | unconditional\_single/short | mask75 | 0.993905 | 0.558442 | 1 | 80 | 77 | 3 |
| latest | unconditional\_single/short | mask100 | 1.31979 | 0.385417 | 1 | 100 | 96 | 4 |
| latest | unconditional\_single/short | mask\_mean | 0.645379 | 0.710965 | 1 | — | — | — |
| latest | unconditional\_tcr/long | mask25 | 6.26517e-06 | 1 | 1 | 516 | 515 | 1 |
| latest | unconditional\_tcr/long | mask50 | 6.94866e-06 | 1 | 1 | 989 | 987 | 2 |
| latest | unconditional\_tcr/long | mask75 | 1.73249e-05 | 1 | 1 | 1527 | 1522 | 5 |
| latest | unconditional\_tcr/long | mask100 | 1.58632 | 0.371569 | 1 | 2048 | 2040 | 8 |
| latest | unconditional\_tcr/long | mask\_mean | 0.396587 | 0.842892 | 1 | — | — | — |
| latest | unconditional\_tcr/medium | mask25 | 8.08358e-06 | 1 | — | 247 | 247 | 0 |
| latest | unconditional\_tcr/medium | mask50 | 1.10744e-05 | 1 | 1 | 500 | 497 | 3 |
| latest | unconditional\_tcr/medium | mask75 | 1.31503e-05 | 1 | 1 | 744 | 738 | 6 |
| latest | unconditional\_tcr/medium | mask100 | 1.6776 | 0.379921 | 1 | 1024 | 1016 | 8 |
| latest | unconditional\_tcr/medium | mask\_mean | 0.419408 | 0.84498 | 1 | — | — | — |
| latest | unconditional\_tcr/short | mask25 | 7.06607e-06 | 1 | — | 40 | 40 | 0 |
| latest | unconditional\_tcr/short | mask50 | 0.00610194 | 1 | 1 | 93 | 90 | 3 |
| latest | unconditional\_tcr/short | mask75 | 0.320383 | 0.889706 | 1 | 142 | 136 | 6 |
| latest | unconditional\_tcr/short | mask100 | 1.35574 | 0.447917 | 1 | 200 | 192 | 8 |
| latest | unconditional\_tcr/short | mask\_mean | 0.420558 | 0.834406 | 1 | — | — | — |
| train\_probe | ab\_antigen/long | mask25 | 1.67385e-05 | 1 | — | 530 | 530 | 0 |
| train\_probe | ab\_antigen/long | mask50 | 2.66749e-05 | 1 | 1 | 1034 | 1031 | 3 |
| train\_probe | ab\_antigen/long | mask75 | 0.00644712 | 0.997431 | 1 | 1563 | 1557 | 6 |
| train\_probe | ab\_antigen/long | mask100 | 0.0440741 | 0.988726 | 1 | 2048 | 2040 | 8 |
| train\_probe | ab\_antigen/long | mask\_mean | 0.0126412 | 0.996539 | 1 | — | — | — |
| train\_probe | ab\_antigen/medium | mask25 | 8.62704e-06 | 1 | 1 | 275 | 273 | 2 |
| train\_probe | ab\_antigen/medium | mask50 | 0.00146927 | 0.998141 | 1 | 542 | 538 | 4 |
| train\_probe | ab\_antigen/medium | mask75 | 4.7625e-05 | 1 | 1 | 781 | 776 | 5 |
| train\_probe | ab\_antigen/medium | mask100 | 0.00759743 | 1 | 1 | 1024 | 1016 | 8 |
| train\_probe | ab\_antigen/medium | mask\_mean | 0.00228074 | 0.999535 | 1 | — | — | — |
| train\_probe | ab\_antigen/short | mask25 | 9.87789e-05 | 1 | 1 | 52 | 47 | 5 |
| train\_probe | ab\_antigen/short | mask50 | 0.000917129 | 1 | 1 | 109 | 102 | 7 |
| train\_probe | ab\_antigen/short | mask75 | 0.107977 | 0.966216 | 1 | 155 | 148 | 7 |
| train\_probe | ab\_antigen/short | mask100 | 0.0117256 | 1 | 1 | 200 | 192 | 8 |
| train\_probe | ab\_antigen/short | mask\_mean | 0.0301796 | 0.991554 | 1 | — | — | — |
| train\_probe | tcr\_peptide/long | mask25 | 3.01826e-05 | 1 | 1 | 495 | 494 | 1 |
| train\_probe | tcr\_peptide/long | mask50 | 1.68044e-05 | 1 | 1 | 1033 | 1028 | 5 |
| train\_probe | tcr\_peptide/long | mask75 | 1.9091e-05 | 1 | 1 | 1530 | 1522 | 8 |
| train\_probe | tcr\_peptide/long | mask100 | 0.0319509 | 0.992647 | 1 | 2048 | 2040 | 8 |
| train\_probe | tcr\_peptide/long | mask\_mean | 0.00800425 | 0.998162 | 1 | — | — | — |
| train\_probe | tcr\_peptide/medium | mask25 | 5.87293e-06 | 1 | — | 252 | 252 | 0 |
| train\_probe | tcr\_peptide/medium | mask50 | 8.54286e-06 | 1 | 1 | 517 | 513 | 4 |
| train\_probe | tcr\_peptide/medium | mask75 | 0.00364826 | 0.997347 | 1 | 760 | 754 | 6 |
| train\_probe | tcr\_peptide/medium | mask100 | 0.0168543 | 0.997047 | 1 | 1024 | 1016 | 8 |
| train\_probe | tcr\_peptide/medium | mask\_mean | 0.00512923 | 0.998599 | 1 | — | — | — |
| train\_probe | tcr\_peptide/short | mask25 | 0.00023574 | 1 | 1 | 51 | 46 | 5 |
| train\_probe | tcr\_peptide/short | mask50 | 6.27585e-05 | 1 | 1 | 95 | 90 | 5 |
| train\_probe | tcr\_peptide/short | mask75 | 0.00120742 | 1 | 1 | 142 | 136 | 6 |
| train\_probe | tcr\_peptide/short | mask100 | 0.0103739 | 0.994792 | 1 | 200 | 192 | 8 |
| train\_probe | tcr\_peptide/short | mask\_mean | 0.00296995 | 0.998698 | 1 | — | — | — |
| train\_probe | tcr\_pmhc/long | mask25 | 1.38869e-05 | 1 | 1 | 526 | 522 | 4 |
| train\_probe | tcr\_pmhc/long | mask50 | 1.06456e-05 | 1 | 1 | 1036 | 1031 | 5 |
| train\_probe | tcr\_pmhc/long | mask75 | 1.26964e-05 | 1 | 1 | 1557 | 1551 | 6 |
| train\_probe | tcr\_pmhc/long | mask100 | 0.237084 | 0.935784 | 1 | 2048 | 2040 | 8 |
| train\_probe | tcr\_pmhc/long | mask\_mean | 0.0592804 | 0.983946 | 1 | — | — | — |
| train\_probe | tcr\_pmhc/medium | mask25 | 8.97364e-05 | 1 | 1 | 276 | 273 | 3 |
| train\_probe | tcr\_pmhc/medium | mask50 | 5.76235e-05 | 1 | 1 | 549 | 541 | 8 |
| train\_probe | tcr\_pmhc/medium | mask75 | 2.26297e-05 | 1 | 1 | 793 | 785 | 8 |
| train\_probe | tcr\_pmhc/medium | mask100 | 0.0691424 | 0.969488 | 1 | 1024 | 1016 | 8 |
| train\_probe | tcr\_pmhc/medium | mask\_mean | 0.0173281 | 0.992372 | 1 | — | — | — |
| train\_probe | tcr\_pmhc/short | mask25 | 2.5964e-05 | 1 | 1 | 50 | 48 | 2 |
| train\_probe | tcr\_pmhc/short | mask50 | 2.24779e-05 | 1 | 1 | 103 | 100 | 3 |
| train\_probe | tcr\_pmhc/short | mask75 | 9.16497e-05 | 1 | 1 | 152 | 145 | 7 |
| train\_probe | tcr\_pmhc/short | mask100 | 0.00340673 | 1 | 1 | 200 | 192 | 8 |
| train\_probe | tcr\_pmhc/short | mask\_mean | 0.000886705 | 1 | 1 | — | — | — |
| train\_probe | unconditional\_ab/long | mask25 | 7.70791e-05 | 1 | 1 | 509 | 506 | 3 |
| train\_probe | unconditional\_ab/long | mask50 | 3.0897e-05 | 1 | 1 | 1016 | 1010 | 6 |
| train\_probe | unconditional\_ab/long | mask75 | 0.000339999 | 1 | 1 | 1569 | 1561 | 8 |
| train\_probe | unconditional\_ab/long | mask100 | 1.54136 | 0.293627 | 1 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_ab/long | mask\_mean | 0.385452 | 0.823407 | 1 | — | — | — |
| train\_probe | unconditional\_ab/medium | mask25 | 1.87715e-05 | 1 | 1 | 243 | 241 | 2 |
| train\_probe | unconditional\_ab/medium | mask50 | 0.0145558 | 0.996124 | 1 | 521 | 516 | 5 |
| train\_probe | unconditional\_ab/medium | mask75 | 0.0217638 | 0.99343 | 1 | 766 | 761 | 5 |
| train\_probe | unconditional\_ab/medium | mask100 | 1.72225 | 0.336614 | 1 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_ab/medium | mask\_mean | 0.439648 | 0.831542 | 1 | — | — | — |
| train\_probe | unconditional\_ab/short | mask25 | 0.00013346 | 1 | 1 | 53 | 52 | 1 |
| train\_probe | unconditional\_ab/short | mask50 | 0.14457 | 0.951923 | 1 | 106 | 104 | 2 |
| train\_probe | unconditional\_ab/short | mask75 | 0.561211 | 0.778481 | 1 | 163 | 158 | 5 |
| train\_probe | unconditional\_ab/short | mask100 | 1.50655 | 0.322917 | 1 | 200 | 192 | 8 |
| train\_probe | unconditional\_ab/short | mask\_mean | 0.553116 | 0.76333 | 1 | — | — | — |
| train\_probe | unconditional\_single/long | mask25 | 5.39311e-05 | 1 | 1 | 245 | 244 | 1 |
| train\_probe | unconditional\_single/long | mask50 | 2.44913e-05 | 1 | 1 | 492 | 491 | 1 |
| train\_probe | unconditional\_single/long | mask75 | 0.000553274 | 1 | 1 | 771 | 768 | 3 |
| train\_probe | unconditional\_single/long | mask100 | 1.75176 | 0.297059 | 1 | 1024 | 1020 | 4 |
| train\_probe | unconditional\_single/long | mask\_mean | 0.438097 | 0.824265 | 1 | — | — | — |
| train\_probe | unconditional\_single/medium | mask25 | 0.00878141 | 0.992308 | 1 | 132 | 130 | 2 |
| train\_probe | unconditional\_single/medium | mask50 | 0.000823442 | 1 | 1 | 255 | 252 | 3 |
| train\_probe | unconditional\_single/medium | mask75 | 0.000259179 | 1 | 1 | 382 | 379 | 3 |
| train\_probe | unconditional\_single/medium | mask100 | 1.59415 | 0.287402 | 1 | 512 | 508 | 4 |
| train\_probe | unconditional\_single/medium | mask\_mean | 0.401004 | 0.819927 | 1 | — | — | — |
| train\_probe | unconditional\_single/short | mask25 | 0.0080274 | 1 | 1 | 28 | 26 | 2 |
| train\_probe | unconditional\_single/short | mask50 | 0.308831 | 0.897959 | 1 | 52 | 49 | 3 |
| train\_probe | unconditional\_single/short | mask75 | 0.794583 | 0.671429 | 1 | 73 | 70 | 3 |
| train\_probe | unconditional\_single/short | mask100 | 1.2126 | 0.447917 | 1 | 100 | 96 | 4 |
| train\_probe | unconditional\_single/short | mask\_mean | 0.581011 | 0.754326 | 1 | — | — | — |
| train\_probe | unconditional\_tcr/long | mask25 | 4.05576e-05 | 1 | — | 525 | 525 | 0 |
| train\_probe | unconditional\_tcr/long | mask50 | 8.39568e-06 | 1 | 1 | 1048 | 1045 | 3 |
| train\_probe | unconditional\_tcr/long | mask75 | 1.30487e-05 | 1 | 1 | 1585 | 1580 | 5 |
| train\_probe | unconditional\_tcr/long | mask100 | 1.58085 | 0.364706 | 1 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_tcr/long | mask\_mean | 0.395229 | 0.841176 | 1 | — | — | — |
| train\_probe | unconditional\_tcr/medium | mask25 | 6.03273e-06 | 1 | — | 261 | 261 | 0 |
| train\_probe | unconditional\_tcr/medium | mask50 | 6.73013e-06 | 1 | 1 | 498 | 497 | 1 |
| train\_probe | unconditional\_tcr/medium | mask75 | 0.000124017 | 1 | 1 | 755 | 751 | 4 |
| train\_probe | unconditional\_tcr/medium | mask100 | 1.55846 | 0.387795 | 1 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_tcr/medium | mask\_mean | 0.389649 | 0.846949 | 1 | — | — | — |
| train\_probe | unconditional\_tcr/short | mask25 | 0.000131939 | 1 | — | 46 | 46 | 0 |
| train\_probe | unconditional\_tcr/short | mask50 | 0.43036 | 0.869565 | 1 | 97 | 92 | 5 |
| train\_probe | unconditional\_tcr/short | mask75 | 0.478098 | 0.80137 | 1 | 152 | 146 | 6 |
| train\_probe | unconditional\_tcr/short | mask100 | 1.43652 | 0.302083 | 1 | 200 | 192 | 8 |
| train\_probe | unconditional\_tcr/short | mask\_mean | 0.586278 | 0.743255 | 1 | — | — | — |

### 1790803305051081571\_r21\_refined\_head\_gate\_9cc7449e：分层 mask 指标

源码目录：/vepfs-mlp2/c20250601/251105016/project/dllm\_test/output/immune\_research\_releases/r20\_r27\_20261001；完成/目标 step：500/500。

kind=synthetic\_generalization\_diagnostic；dataset_kind=deterministic\_artificial\_sequences；biological_quality_claim=False。

| 时点 | mask level | 指标 | 值 |
| --- | --- | --- | --- |
| initial | mask25 | mask\_ratio | 0.25 |
| initial | mask25 | selected\_count | 4467 |
| initial | mask25 | aa\_count | 4428 |
| initial | mask25 | terminal\_count | 39 |
| initial | mask25 | loss | 3.91472 |
| initial | mask25 | aa\_loss | 3.91377 |
| initial | mask25 | aa\_accuracy | 0.0682023 |
| initial | mask25 | terminal\_loss | 4.02257 |
| initial | mask25 | terminal\_accuracy | 0 |
| initial | mask50 | mask\_ratio | 0.5 |
| initial | mask50 | selected\_count | 8888 |
| initial | mask50 | aa\_count | 8819 |
| initial | mask50 | terminal\_count | 69 |
| initial | mask50 | loss | 4.07252 |
| initial | mask50 | aa\_loss | 4.0726 |
| initial | mask50 | aa\_accuracy | 0.0734777 |
| initial | mask50 | terminal\_loss | 4.06312 |
| initial | mask50 | terminal\_accuracy | 0 |
| initial | mask75 | mask\_ratio | 0.75 |
| initial | mask75 | selected\_count | 13437 |
| initial | mask75 | aa\_count | 13330 |
| initial | mask75 | terminal\_count | 107 |
| initial | mask75 | loss | 4.1958 |
| initial | mask75 | aa\_loss | 4.1954 |
| initial | mask75 | aa\_accuracy | 0.0578395 |
| initial | mask75 | terminal\_loss | 4.24564 |
| initial | mask75 | terminal\_accuracy | 0 |
| initial | mask100 | mask\_ratio | 1 |
| initial | mask100 | selected\_count | 17996 |
| initial | mask100 | aa\_count | 17864 |
| initial | mask100 | terminal\_count | 132 |
| initial | mask100 | loss | 4.1696 |
| initial | mask100 | aa\_loss | 4.1687 |
| initial | mask100 | aa\_accuracy | 0.0558665 |
| initial | mask100 | terminal\_loss | 4.29045 |
| initial | mask100 | terminal\_accuracy | 0 |
| initial | mask\_mean | loss | 4.08816 |
| initial | mask\_mean | aa\_loss | 4.08762 |
| initial | mask\_mean | terminal\_loss | 4.15545 |
| initial | mask\_mean | aa\_accuracy | 0.0638465 |
| initial | mask\_mean | terminal\_accuracy | 0 |
| latest | mask25 | mask\_ratio | 0.25 |
| latest | mask25 | selected\_count | 4467 |
| latest | mask25 | aa\_count | 4428 |
| latest | mask25 | terminal\_count | 39 |
| latest | mask25 | loss | 3.81621e-05 |
| latest | mask25 | aa\_loss | 3.59545e-05 |
| latest | mask25 | aa\_accuracy | 1 |
| latest | mask25 | terminal\_loss | 0.000288804 |
| latest | mask25 | terminal\_accuracy | 1 |
| latest | mask50 | mask\_ratio | 0.5 |
| latest | mask50 | selected\_count | 8888 |
| latest | mask50 | aa\_count | 8819 |
| latest | mask50 | terminal\_count | 69 |
| latest | mask50 | loss | 0.00346393 |
| latest | mask50 | aa\_loss | 0.00348847 |
| latest | mask50 | aa\_accuracy | 0.998186 |
| latest | mask50 | terminal\_loss | 0.000327036 |
| latest | mask50 | terminal\_accuracy | 1 |
| latest | mask75 | mask\_ratio | 0.75 |
| latest | mask75 | selected\_count | 13437 |
| latest | mask75 | aa\_count | 13330 |
| latest | mask75 | terminal\_count | 107 |
| latest | mask75 | loss | 0.0264813 |
| latest | mask75 | aa\_loss | 0.0266861 |
| latest | mask75 | aa\_accuracy | 0.992723 |
| latest | mask75 | terminal\_loss | 0.00096699 |
| latest | mask75 | terminal\_accuracy | 1 |
| latest | mask100 | mask\_ratio | 1 |
| latest | mask100 | selected\_count | 17996 |
| latest | mask100 | aa\_count | 17864 |
| latest | mask100 | terminal\_count | 132 |
| latest | mask100 | loss | 0.75063 |
| latest | mask100 | aa\_loss | 0.755784 |
| latest | mask100 | aa\_accuracy | 0.708128 |
| latest | mask100 | terminal\_loss | 0.0531048 |
| latest | mask100 | terminal\_accuracy | 1 |
| latest | mask\_mean | loss | 0.195153 |
| latest | mask\_mean | aa\_loss | 0.196499 |
| latest | mask\_mean | terminal\_loss | 0.0136719 |
| latest | mask\_mean | aa\_accuracy | 0.924759 |
| latest | mask\_mean | terminal\_accuracy | 1 |

评测曲线（原始 step，四档等权均值）：

| step | mean loss | mean AA acc | mean end acc | gradient norm |
| --- | --- | --- | --- | --- |
| 25 | 1.84223 | 0.324807 | 0 | 3.44503 |
| 50 | 1.34797 | 0.521351 | 0 | 3.16969 |
| 75 | 0.815306 | 0.738949 | 0 | 2.53618 |
| 100 | 0.575922 | 0.806623 | 0.338618 | 2.23957 |
| 125 | 0.504221 | 0.822141 | 0.433715 | 2.37979 |
| 150 | 0.445816 | 0.847793 | 0.889747 | 1.59946 |
| 175 | 0.41263 | 0.852909 | 0.995327 | 2.38945 |
| 200 | 0.433864 | 0.855536 | 0.981071 | 2.08802 |
| 225 | 0.348254 | 0.878117 | 0.966354 | 1.52655 |
| 250 | 0.263497 | 0.904139 | 0.990088 | 0.922332 |
| 275 | 0.261013 | 0.901522 | 0.93327 | 1.17925 |
| 300 | 0.252108 | 0.908065 | 0.959064 | 0.31829 |
| 325 | 0.217236 | 0.916762 | 0.952652 | 0.742701 |
| 350 | 0.191756 | 0.922982 | 0.954545 | 0.490538 |
| 375 | 0.197641 | 0.921618 | 0.962121 | 0.495146 |
| 400 | 0.189672 | 0.927618 | 1 | 0.340932 |
| 425 | 0.187341 | 0.928723 | 1 | 0.431983 |
| 450 | 0.183921 | 0.928035 | 0.940845 | 0.32383 |
| 475 | 0.176513 | 0.930979 | 1 | 0.256539 |
| 500 | 0.195153 | 0.924759 | 1 | 0.362284 |

有条件/无条件曲线（每档按token计数合并，再平均四档；缺失分组不推断数值）：

| step | 子集 | mean loss | mean AA | mean end | Full-mask AA | Full-mask end | Full-mask AA count | Full-mask end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | conditional | 4.18144 | 0.0639687 | 0 | 0.0632184 | 0 | 9744 | 72 |
| initial | unconditional | 3.97672 | 0.0636479 | 0 | 0.0470443 | 0 | 8120 | 60 |
| 25 | conditional | 1.78296 | 0.330896 | 0 | 0.319068 | 0 | 9744 | 72 |
| 25 | unconditional | 1.91318 | 0.317589 | 0 | 0.314901 | 0 | 8120 | 60 |
| 50 | conditional | 1.26859 | 0.539727 | 0 | 0.310961 | 0 | 9744 | 72 |
| 50 | unconditional | 1.4429 | 0.499559 | 0 | 0.276232 | 0 | 8120 | 60 |
| 75 | conditional | 0.728661 | 0.75479 | 0 | 0.339696 | 0 | 9744 | 72 |
| 75 | unconditional | 0.918955 | 0.720099 | 0 | 0.313916 | 0 | 8120 | 60 |
| 100 | conditional | 0.520878 | 0.81753 | 0.408321 | 0.366071 | 0 | 9744 | 72 |
| 100 | unconditional | 0.641853 | 0.793562 | 0.221313 | 0.306527 | 0 | 8120 | 60 |
| 125 | conditional | 0.470037 | 0.834439 | 0.404166 | 0.381876 | 0.0138889 | 9744 | 72 |
| 125 | unconditional | 0.545177 | 0.807422 | 0.475298 | 0.3 | 0.133333 | 8120 | 60 |
| 150 | conditional | 0.379143 | 0.872016 | 0.932449 | 0.508005 | 0.930556 | 9744 | 72 |
| 150 | unconditional | 0.525796 | 0.818739 | 0.832192 | 0.316379 | 0.8 | 8120 | 60 |
| 175 | conditional | 0.338932 | 0.882675 | 0.995763 | 0.542282 | 1 | 9744 | 72 |
| 175 | unconditional | 0.501032 | 0.817209 | 0.994792 | 0.305049 | 1 | 8120 | 60 |
| 200 | conditional | 0.357689 | 0.881721 | 0.98981 | 0.538793 | 1 | 9744 | 72 |
| 200 | unconditional | 0.525252 | 0.824126 | 0.969907 | 0.321675 | 1 | 8120 | 60 |
| 225 | conditional | 0.253872 | 0.915898 | 0.993056 | 0.672003 | 0.972222 | 9744 | 72 |
| 225 | unconditional | 0.461492 | 0.832783 | 0.929134 | 0.348892 | 0.866667 | 8120 | 60 |
| 250 | conditional | 0.14472 | 0.95234 | 1 | 0.813218 | 1 | 9744 | 72 |
| 250 | unconditional | 0.406017 | 0.846305 | 0.978125 | 0.40702 | 0.933333 | 8120 | 60 |
| 275 | conditional | 0.107912 | 0.962415 | 0.993056 | 0.852833 | 0.972222 | 9744 | 72 |
| 275 | unconditional | 0.444722 | 0.828455 | 0.861458 | 0.3367 | 0.466667 | 8120 | 60 |
| 300 | conditional | 0.102284 | 0.967121 | 1 | 0.872332 | 1 | 9744 | 72 |
| 300 | unconditional | 0.431889 | 0.837204 | 0.908449 | 0.368227 | 0.733333 | 8120 | 60 |
| 325 | conditional | 0.0413521 | 0.986332 | 0.996528 | 0.946429 | 0.986111 | 9744 | 72 |
| 325 | unconditional | 0.42829 | 0.833281 | 0.9 | 0.35468 | 0.6 | 8120 | 60 |
| 350 | conditional | 0.0208948 | 0.992781 | 1 | 0.97147 | 1 | 9744 | 72 |
| 350 | unconditional | 0.396786 | 0.839226 | 0.9 | 0.376478 | 0.6 | 8120 | 60 |
| 375 | conditional | 0.0294486 | 0.992269 | 1 | 0.969212 | 1 | 9744 | 72 |
| 375 | unconditional | 0.399459 | 0.836846 | 0.916667 | 0.368596 | 0.666667 | 8120 | 60 |
| 400 | conditional | 0.0242211 | 0.990829 | 1 | 0.964696 | 1 | 9744 | 72 |
| 400 | unconditional | 0.388208 | 0.851767 | 1 | 0.42697 | 1 | 8120 | 60 |
| 425 | conditional | 0.0130575 | 0.995869 | 1 | 0.983888 | 1 | 9744 | 72 |
| 425 | unconditional | 0.396477 | 0.848152 | 1 | 0.413916 | 1 | 8120 | 60 |
| 450 | conditional | 0.00430156 | 0.998922 | 0.993056 | 0.9961 | 0.972222 | 9744 | 72 |
| 450 | unconditional | 0.399458 | 0.842974 | 0.878125 | 0.390148 | 0.533333 | 8120 | 60 |
| 475 | conditional | 0.0076792 | 0.997305 | 1 | 0.990456 | 1 | 9744 | 72 |
| 475 | unconditional | 0.379107 | 0.851393 | 1 | 0.421182 | 1 | 8120 | 60 |
| 500 | conditional | 0.0071366 | 0.997716 | 1 | 0.991277 | 1 | 9744 | 72 |
| 500 | unconditional | 0.420775 | 0.837215 | 1 | 0.36835 | 1 | 8120 | 60 |

人工数据 manifest：d40bbbbc1c4294eed843d88bcdd0cf193886de6c2d7e9db9069bbf86e931c022；精确 target sequence 重叠=0；精确 record 重叠=0。

跨 radius32 覆盖：{&quot;possible&quot;: true, &quot;train&quot;: true, &quot;valid&quot;: true}。生成规则共享是诊断设计的一部分，不能当作真实下游去重证书。


按任务/链长分组（全局 level 按 token 计数汇总，下面保留各组计数；不同组不得直接比较未经加权的总 loss）：

| 时点 | group/length | mask level | loss | AA acc | end acc | selected count | AA count | end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | ab\_antigen/long | mask25 | 4.10407 | 0.0285171 | 0 | 530 | 526 | 4 |
| initial | ab\_antigen/long | mask50 | 4.40782 | 0.0306022 | 0 | 1019 | 1013 | 6 |
| initial | ab\_antigen/long | mask75 | 4.7473 | 0.0274151 | 0 | 1538 | 1532 | 6 |
| initial | ab\_antigen/long | mask100 | 4.66581 | 0.0303922 | 0 | 2048 | 2040 | 8 |
| initial | ab\_antigen/long | mask\_mean | 4.48125 | 0.0292316 | 0 | — | — | — |
| initial | ab\_antigen/medium | mask25 | 4.17571 | 0.144033 | 0 | 246 | 243 | 3 |
| initial | ab\_antigen/medium | mask50 | 4.27198 | 0.105578 | 0 | 507 | 502 | 5 |
| initial | ab\_antigen/medium | mask75 | 4.37624 | 0.0620053 | 0 | 765 | 758 | 7 |
| initial | ab\_antigen/medium | mask100 | 4.37027 | 0.0610236 | 0 | 1024 | 1016 | 8 |
| initial | ab\_antigen/medium | mask\_mean | 4.29855 | 0.0931599 | 0 | — | — | — |
| initial | ab\_antigen/short | mask25 | 4.83735 | 0.0588235 | 0 | 38 | 34 | 4 |
| initial | ab\_antigen/short | mask50 | 4.64596 | 0.0795455 | 0 | 93 | 88 | 5 |
| initial | ab\_antigen/short | mask75 | 4.54142 | 0.0785714 | 0 | 145 | 140 | 5 |
| initial | ab\_antigen/short | mask100 | 4.46303 | 0.09375 | 0 | 200 | 192 | 8 |
| initial | ab\_antigen/short | mask\_mean | 4.62194 | 0.0776726 | 0 | — | — | — |
| initial | tcr\_peptide/long | mask25 | 3.79134 | 0.083871 | 0 | 466 | 465 | 1 |
| initial | tcr\_peptide/long | mask50 | 3.99306 | 0.114169 | 0 | 985 | 981 | 4 |
| initial | tcr\_peptide/long | mask75 | 4.12436 | 0.118403 | 0 | 1484 | 1478 | 6 |
| initial | tcr\_peptide/long | mask100 | 4.06361 | 0.12451 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_peptide/long | mask\_mean | 3.99309 | 0.110238 | 0 | — | — | — |
| initial | tcr\_peptide/medium | mask25 | 4.1572 | 0.0580913 | 0 | 242 | 241 | 1 |
| initial | tcr\_peptide/medium | mask50 | 4.14345 | 0.0537849 | 0 | 506 | 502 | 4 |
| initial | tcr\_peptide/medium | mask75 | 4.06713 | 0.0578947 | 0 | 767 | 760 | 7 |
| initial | tcr\_peptide/medium | mask100 | 3.95053 | 0.0610236 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_peptide/medium | mask\_mean | 4.07958 | 0.0576986 | 0 | — | — | — |
| initial | tcr\_peptide/short | mask25 | 3.90693 | 0.104167 | 0 | 51 | 48 | 3 |
| initial | tcr\_peptide/short | mask50 | 3.85339 | 0.0865385 | 0 | 110 | 104 | 6 |
| initial | tcr\_peptide/short | mask75 | 3.88939 | 0.0875912 | 0 | 144 | 137 | 7 |
| initial | tcr\_peptide/short | mask100 | 3.88457 | 0.09375 | 0 | 200 | 192 | 8 |
| initial | tcr\_peptide/short | mask\_mean | 3.88357 | 0.0930116 | 0 | — | — | — |
| initial | tcr\_pmhc/long | mask25 | 3.54222 | 0.0391791 | 0 | 540 | 536 | 4 |
| initial | tcr\_pmhc/long | mask50 | 3.83329 | 0.0357834 | 0 | 1038 | 1034 | 4 |
| initial | tcr\_pmhc/long | mask75 | 4.11564 | 0.032175 | 0 | 1562 | 1554 | 8 |
| initial | tcr\_pmhc/long | mask100 | 4.29233 | 0.0313726 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_pmhc/long | mask\_mean | 3.94587 | 0.0346275 | 0 | — | — | — |
| initial | tcr\_pmhc/medium | mask25 | 4.27294 | 0.0909091 | 0 | 244 | 242 | 2 |
| initial | tcr\_pmhc/medium | mask50 | 4.34512 | 0.0793319 | 0 | 484 | 479 | 5 |
| initial | tcr\_pmhc/medium | mask75 | 4.43825 | 0.0702211 | 0 | 777 | 769 | 8 |
| initial | tcr\_pmhc/medium | mask100 | 4.45656 | 0.0629921 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_pmhc/medium | mask\_mean | 4.37822 | 0.0758636 | 0 | — | — | — |
| initial | tcr\_pmhc/short | mask25 | 4.41788 | 0.0416667 | 0 | 51 | 48 | 3 |
| initial | tcr\_pmhc/short | mask50 | 4.43505 | 0.0481928 | 0 | 86 | 83 | 3 |
| initial | tcr\_pmhc/short | mask75 | 4.25766 | 0.0680272 | 0 | 152 | 147 | 5 |
| initial | tcr\_pmhc/short | mask100 | 4.2994 | 0.0625 | 0 | 200 | 192 | 8 |
| initial | tcr\_pmhc/short | mask\_mean | 4.3525 | 0.0550967 | 0 | — | — | — |
| initial | unconditional\_ab/long | mask25 | 3.45445 | 0.157009 | 0 | 538 | 535 | 3 |
| initial | unconditional\_ab/long | mask50 | 3.68616 | 0.179732 | 0 | 1050 | 1046 | 4 |
| initial | unconditional\_ab/long | mask75 | 3.89091 | 0.0923277 | 0 | 1545 | 1538 | 7 |
| initial | unconditional\_ab/long | mask100 | 3.94998 | 0.0627451 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_ab/long | mask\_mean | 3.74538 | 0.122954 | 0 | — | — | — |
| initial | unconditional\_ab/medium | mask25 | 4.11098 | 0.055336 | 0 | 256 | 253 | 3 |
| initial | unconditional\_ab/medium | mask50 | 4.24527 | 0.139764 | 0 | 512 | 508 | 4 |
| initial | unconditional\_ab/medium | mask75 | 4.21312 | 0.149606 | 0 | 768 | 762 | 6 |
| initial | unconditional\_ab/medium | mask100 | 4.17067 | 0.15748 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_ab/medium | mask\_mean | 4.18501 | 0.125547 | 0 | — | — | — |
| initial | unconditional\_ab/short | mask25 | 4.10634 | 0.0961538 | 0 | 55 | 52 | 3 |
| initial | unconditional\_ab/short | mask50 | 4.061 | 0.0612245 | 0 | 103 | 98 | 5 |
| initial | unconditional\_ab/short | mask75 | 4.02084 | 0.0567376 | 0 | 148 | 141 | 7 |
| initial | unconditional\_ab/short | mask100 | 3.95762 | 0.0625 | 0 | 200 | 192 | 8 |
| initial | unconditional\_ab/short | mask\_mean | 4.03645 | 0.069154 | 0 | — | — | — |
| initial | unconditional\_single/long | mask25 | 3.41733 | 0.0820896 | 0 | 269 | 268 | 1 |
| initial | unconditional\_single/long | mask50 | 3.40878 | 0.0533333 | 0 | 527 | 525 | 2 |
| initial | unconditional\_single/long | mask75 | 3.53938 | 0.00516796 | 0 | 778 | 774 | 4 |
| initial | unconditional\_single/long | mask100 | 3.60678 | 0 | 0 | 1024 | 1020 | 4 |
| initial | unconditional\_single/long | mask\_mean | 3.49307 | 0.0351477 | 0 | — | — | — |
| initial | unconditional\_single/medium | mask25 | 3.85475 | 0 | 0 | 110 | 109 | 1 |
| initial | unconditional\_single/medium | mask50 | 3.72801 | 0 | 0 | 234 | 232 | 2 |
| initial | unconditional\_single/medium | mask75 | 3.72224 | 0 | 0 | 371 | 367 | 4 |
| initial | unconditional\_single/medium | mask100 | 3.73258 | 0 | 0 | 512 | 508 | 4 |
| initial | unconditional\_single/medium | mask\_mean | 3.75939 | 0 | 0 | — | — | — |
| initial | unconditional\_single/short | mask25 | 3.47104 | 0 | 0 | 28 | 26 | 2 |
| initial | unconditional\_single/short | mask50 | 3.6937 | 0 | 0 | 52 | 50 | 2 |
| initial | unconditional\_single/short | mask75 | 3.49417 | 0 | 0 | 80 | 77 | 3 |
| initial | unconditional\_single/short | mask100 | 3.44943 | 0 | 0 | 100 | 96 | 4 |
| initial | unconditional\_single/short | mask\_mean | 3.52708 | 0 | 0 | — | — | — |
| initial | unconditional\_tcr/long | mask25 | 4.12367 | 0.0368932 | 0 | 516 | 515 | 1 |
| initial | unconditional\_tcr/long | mask50 | 4.32329 | 0.0303951 | 0 | 989 | 987 | 2 |
| initial | unconditional\_tcr/long | mask75 | 4.35912 | 0.0302234 | 0 | 1527 | 1522 | 5 |
| initial | unconditional\_tcr/long | mask100 | 4.13527 | 0.0313726 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_tcr/long | mask\_mean | 4.23534 | 0.0322211 | 0 | — | — | — |
| initial | unconditional\_tcr/medium | mask25 | 4.32901 | 0 | — | 247 | 247 | 0 |
| initial | unconditional\_tcr/medium | mask50 | 4.40907 | 0 | 0 | 500 | 497 | 3 |
| initial | unconditional\_tcr/medium | mask75 | 4.36242 | 0 | 0 | 744 | 738 | 6 |
| initial | unconditional\_tcr/medium | mask100 | 4.26923 | 0 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_tcr/medium | mask\_mean | 4.34243 | 0 | 0 | — | — | — |
| initial | unconditional\_tcr/short | mask25 | 4.34405 | 0.075 | — | 40 | 40 | 0 |
| initial | unconditional\_tcr/short | mask50 | 4.15658 | 0.0777778 | 0 | 93 | 90 | 3 |
| initial | unconditional\_tcr/short | mask75 | 4.14406 | 0.0882353 | 0 | 142 | 136 | 6 |
| initial | unconditional\_tcr/short | mask100 | 4.06085 | 0.09375 | 0 | 200 | 192 | 8 |
| initial | unconditional\_tcr/short | mask\_mean | 4.17639 | 0.0836908 | 0 | — | — | — |
| latest | ab\_antigen/long | mask25 | 1.55937e-05 | 1 | 1 | 530 | 526 | 4 |
| latest | ab\_antigen/long | mask50 | 2.13624e-05 | 1 | 1 | 1019 | 1013 | 6 |
| latest | ab\_antigen/long | mask75 | 7.21071e-05 | 1 | 1 | 1538 | 1532 | 6 |
| latest | ab\_antigen/long | mask100 | 0.0216593 | 0.994118 | 1 | 2048 | 2040 | 8 |
| latest | ab\_antigen/long | mask\_mean | 0.0054421 | 0.998529 | 1 | — | — | — |
| latest | ab\_antigen/medium | mask25 | 0.000261215 | 1 | 1 | 246 | 243 | 3 |
| latest | ab\_antigen/medium | mask50 | 0.00013029 | 1 | 1 | 507 | 502 | 5 |
| latest | ab\_antigen/medium | mask75 | 0.000157465 | 1 | 1 | 765 | 758 | 7 |
| latest | ab\_antigen/medium | mask100 | 0.0313744 | 0.989173 | 1 | 1024 | 1016 | 8 |
| latest | ab\_antigen/medium | mask\_mean | 0.00798085 | 0.997293 | 1 | — | — | — |
| latest | ab\_antigen/short | mask25 | 9.75302e-06 | 1 | 1 | 38 | 34 | 4 |
| latest | ab\_antigen/short | mask50 | 1.60894e-05 | 1 | 1 | 93 | 88 | 5 |
| latest | ab\_antigen/short | mask75 | 0.015036 | 0.992857 | 1 | 145 | 140 | 5 |
| latest | ab\_antigen/short | mask100 | 0.00973679 | 0.994792 | 1 | 200 | 192 | 8 |
| latest | ab\_antigen/short | mask\_mean | 0.00619965 | 0.996912 | 1 | — | — | — |
| latest | tcr\_peptide/long | mask25 | 4.98665e-06 | 1 | 1 | 466 | 465 | 1 |
| latest | tcr\_peptide/long | mask50 | 3.89433e-06 | 1 | 1 | 985 | 981 | 4 |
| latest | tcr\_peptide/long | mask75 | 2.18569e-06 | 1 | 1 | 1484 | 1478 | 6 |
| latest | tcr\_peptide/long | mask100 | 0.0148805 | 0.997059 | 1 | 2048 | 2040 | 8 |
| latest | tcr\_peptide/long | mask\_mean | 0.00372288 | 0.999265 | 1 | — | — | — |
| latest | tcr\_peptide/medium | mask25 | 5.0628e-06 | 1 | 1 | 242 | 241 | 1 |
| latest | tcr\_peptide/medium | mask50 | 5.54456e-06 | 1 | 1 | 506 | 502 | 4 |
| latest | tcr\_peptide/medium | mask75 | 6.18079e-05 | 1 | 1 | 767 | 760 | 7 |
| latest | tcr\_peptide/medium | mask100 | 0.0139024 | 0.995079 | 1 | 1024 | 1016 | 8 |
| latest | tcr\_peptide/medium | mask\_mean | 0.0034937 | 0.99877 | 1 | — | — | — |
| latest | tcr\_peptide/short | mask25 | 6.9991e-05 | 1 | 1 | 51 | 48 | 3 |
| latest | tcr\_peptide/short | mask50 | 4.65537e-05 | 1 | 1 | 110 | 104 | 6 |
| latest | tcr\_peptide/short | mask75 | 1.83075e-05 | 1 | 1 | 144 | 137 | 7 |
| latest | tcr\_peptide/short | mask100 | 0.000112102 | 1 | 1 | 200 | 192 | 8 |
| latest | tcr\_peptide/short | mask\_mean | 6.17386e-05 | 1 | 1 | — | — | — |
| latest | tcr\_pmhc/long | mask25 | 2.36671e-05 | 1 | 1 | 540 | 536 | 4 |
| latest | tcr\_pmhc/long | mask50 | 2.17775e-05 | 1 | 1 | 1038 | 1034 | 4 |
| latest | tcr\_pmhc/long | mask75 | 3.06678e-05 | 1 | 1 | 1562 | 1554 | 8 |
| latest | tcr\_pmhc/long | mask100 | 0.0445882 | 0.984314 | 1 | 2048 | 2040 | 8 |
| latest | tcr\_pmhc/long | mask\_mean | 0.0111661 | 0.996078 | 1 | — | — | — |
| latest | tcr\_pmhc/medium | mask25 | 7.86007e-06 | 1 | 1 | 244 | 242 | 2 |
| latest | tcr\_pmhc/medium | mask50 | 1.10128e-05 | 1 | 1 | 484 | 479 | 5 |
| latest | tcr\_pmhc/medium | mask75 | 8.67967e-05 | 1 | 1 | 777 | 769 | 8 |
| latest | tcr\_pmhc/medium | mask100 | 0.0388402 | 0.988189 | 1 | 1024 | 1016 | 8 |
| latest | tcr\_pmhc/medium | mask\_mean | 0.00973648 | 0.997047 | 1 | — | — | — |
| latest | tcr\_pmhc/short | mask25 | 5.74768e-06 | 1 | 1 | 51 | 48 | 3 |
| latest | tcr\_pmhc/short | mask50 | 7.53502e-06 | 1 | 1 | 86 | 83 | 3 |
| latest | tcr\_pmhc/short | mask75 | 0.0245882 | 0.986395 | 1 | 152 | 147 | 5 |
| latest | tcr\_pmhc/short | mask100 | 0.0842455 | 0.96875 | 1 | 200 | 192 | 8 |
| latest | tcr\_pmhc/short | mask\_mean | 0.0272118 | 0.988786 | 1 | — | — | — |
| latest | unconditional\_ab/long | mask25 | 1.20867e-05 | 1 | 1 | 538 | 535 | 3 |
| latest | unconditional\_ab/long | mask50 | 1.19384e-05 | 1 | 1 | 1050 | 1046 | 4 |
| latest | unconditional\_ab/long | mask75 | 0.000107359 | 1 | 1 | 1545 | 1538 | 7 |
| latest | unconditional\_ab/long | mask100 | 1.64974 | 0.332353 | 1 | 2048 | 2040 | 8 |
| latest | unconditional\_ab/long | mask\_mean | 0.412467 | 0.833088 | 1 | — | — | — |
| latest | unconditional\_ab/medium | mask25 | 0.00010567 | 1 | 1 | 256 | 253 | 3 |
| latest | unconditional\_ab/medium | mask50 | 9.12267e-05 | 1 | 1 | 512 | 508 | 4 |
| latest | unconditional\_ab/medium | mask75 | 0.000105661 | 1 | 1 | 768 | 762 | 6 |
| latest | unconditional\_ab/medium | mask100 | 1.60254 | 0.41437 | 1 | 1024 | 1016 | 8 |
| latest | unconditional\_ab/medium | mask\_mean | 0.400712 | 0.853593 | 1 | — | — | — |
| latest | unconditional\_ab/short | mask25 | 4.6619e-05 | 1 | 1 | 55 | 52 | 3 |
| latest | unconditional\_ab/short | mask50 | 0.149755 | 0.908163 | 1 | 103 | 98 | 5 |
| latest | unconditional\_ab/short | mask75 | 0.432276 | 0.858156 | 1 | 148 | 141 | 7 |
| latest | unconditional\_ab/short | mask100 | 1.38394 | 0.369792 | 1 | 200 | 192 | 8 |
| latest | unconditional\_ab/short | mask\_mean | 0.491505 | 0.784028 | 1 | — | — | — |
| latest | unconditional\_single/long | mask25 | 4.21831e-05 | 1 | 1 | 269 | 268 | 1 |
| latest | unconditional\_single/long | mask50 | 1.87694e-05 | 1 | 1 | 527 | 525 | 2 |
| latest | unconditional\_single/long | mask75 | 0.208346 | 0.958656 | 1 | 778 | 774 | 4 |
| latest | unconditional\_single/long | mask100 | 1.59328 | 0.353922 | 1 | 1024 | 1020 | 4 |
| latest | unconditional\_single/long | mask\_mean | 0.450422 | 0.828144 | 1 | — | — | — |
| latest | unconditional\_single/medium | mask25 | 8.62619e-05 | 1 | 1 | 110 | 109 | 1 |
| latest | unconditional\_single/medium | mask50 | 8.16369e-05 | 1 | 1 | 234 | 232 | 2 |
| latest | unconditional\_single/medium | mask75 | 0.0054546 | 0.997275 | 1 | 371 | 367 | 4 |
| latest | unconditional\_single/medium | mask100 | 1.64819 | 0.374016 | 1 | 512 | 508 | 4 |
| latest | unconditional\_single/medium | mask\_mean | 0.413453 | 0.842823 | 1 | — | — | — |
| latest | unconditional\_single/short | mask25 | 0.000485477 | 1 | 1 | 28 | 26 | 2 |
| latest | unconditional\_single/short | mask50 | 0.290592 | 0.86 | 1 | 52 | 50 | 2 |
| latest | unconditional\_single/short | mask75 | 1.02339 | 0.597403 | 1 | 80 | 77 | 3 |
| latest | unconditional\_single/short | mask100 | 1.27485 | 0.385417 | 1 | 100 | 96 | 4 |
| latest | unconditional\_single/short | mask\_mean | 0.647328 | 0.710705 | 1 | — | — | — |
| latest | unconditional\_tcr/long | mask25 | 3.28997e-06 | 1 | 1 | 516 | 515 | 1 |
| latest | unconditional\_tcr/long | mask50 | 4.48598e-06 | 1 | 1 | 989 | 987 | 2 |
| latest | unconditional\_tcr/long | mask75 | 2.5103e-06 | 1 | 1 | 1527 | 1522 | 5 |
| latest | unconditional\_tcr/long | mask100 | 1.69075 | 0.376471 | 1 | 2048 | 2040 | 8 |
| latest | unconditional\_tcr/long | mask\_mean | 0.42269 | 0.844118 | 1 | — | — | — |
| latest | unconditional\_tcr/medium | mask25 | 1.19987e-05 | 1 | — | 247 | 247 | 0 |
| latest | unconditional\_tcr/medium | mask50 | 2.97682e-06 | 1 | 1 | 500 | 497 | 3 |
| latest | unconditional\_tcr/medium | mask75 | 8.20702e-06 | 1 | 1 | 744 | 738 | 6 |
| latest | unconditional\_tcr/medium | mask100 | 1.58663 | 0.363189 | 1 | 1024 | 1016 | 8 |
| latest | unconditional\_tcr/medium | mask\_mean | 0.396662 | 0.840797 | 1 | — | — | — |
| latest | unconditional\_tcr/short | mask25 | 6.43123e-06 | 1 | — | 40 | 40 | 0 |
| latest | unconditional\_tcr/short | mask50 | 0.000301017 | 1 | 1 | 93 | 90 | 3 |
| latest | unconditional\_tcr/short | mask75 | 0.276692 | 0.926471 | 1 | 142 | 136 | 6 |
| latest | unconditional\_tcr/short | mask100 | 1.25267 | 0.5 | 1 | 200 | 192 | 8 |
| latest | unconditional\_tcr/short | mask\_mean | 0.382417 | 0.856618 | 1 | — | — | — |
| train\_probe | ab\_antigen/long | mask25 | 6.29964e-06 | 1 | — | 530 | 530 | 0 |
| train\_probe | ab\_antigen/long | mask50 | 1.24069e-05 | 1 | 1 | 1034 | 1031 | 3 |
| train\_probe | ab\_antigen/long | mask75 | 1.41417e-05 | 1 | 1 | 1563 | 1557 | 6 |
| train\_probe | ab\_antigen/long | mask100 | 0.0172297 | 0.993627 | 1 | 2048 | 2040 | 8 |
| train\_probe | ab\_antigen/long | mask\_mean | 0.00431565 | 0.998407 | 1 | — | — | — |
| train\_probe | ab\_antigen/medium | mask25 | 0.000151185 | 1 | 1 | 275 | 273 | 2 |
| train\_probe | ab\_antigen/medium | mask50 | 6.86337e-05 | 1 | 1 | 542 | 538 | 4 |
| train\_probe | ab\_antigen/medium | mask75 | 3.06936e-05 | 1 | 1 | 781 | 776 | 5 |
| train\_probe | ab\_antigen/medium | mask100 | 0.00366448 | 0.999016 | 1 | 1024 | 1016 | 8 |
| train\_probe | ab\_antigen/medium | mask\_mean | 0.000978747 | 0.999754 | 1 | — | — | — |
| train\_probe | ab\_antigen/short | mask25 | 1.43757e-05 | 1 | 1 | 52 | 47 | 5 |
| train\_probe | ab\_antigen/short | mask50 | 2.08624e-05 | 1 | 1 | 109 | 102 | 7 |
| train\_probe | ab\_antigen/short | mask75 | 0.00207266 | 1 | 1 | 155 | 148 | 7 |
| train\_probe | ab\_antigen/short | mask100 | 0.0109229 | 0.994792 | 1 | 200 | 192 | 8 |
| train\_probe | ab\_antigen/short | mask\_mean | 0.00325769 | 0.998698 | 1 | — | — | — |
| train\_probe | tcr\_peptide/long | mask25 | 1.66644e-05 | 1 | 1 | 495 | 494 | 1 |
| train\_probe | tcr\_peptide/long | mask50 | 1.69121e-05 | 1 | 1 | 1033 | 1028 | 5 |
| train\_probe | tcr\_peptide/long | mask75 | 1.26518e-05 | 1 | 1 | 1530 | 1522 | 8 |
| train\_probe | tcr\_peptide/long | mask100 | 0.00229442 | 0.99902 | 1 | 2048 | 2040 | 8 |
| train\_probe | tcr\_peptide/long | mask\_mean | 0.000585163 | 0.999755 | 1 | — | — | — |
| train\_probe | tcr\_peptide/medium | mask25 | 8.53115e-06 | 1 | — | 252 | 252 | 0 |
| train\_probe | tcr\_peptide/medium | mask50 | 1.00676e-05 | 1 | 1 | 517 | 513 | 4 |
| train\_probe | tcr\_peptide/medium | mask75 | 3.06385e-05 | 1 | 1 | 760 | 754 | 6 |
| train\_probe | tcr\_peptide/medium | mask100 | 0.0322372 | 0.989173 | 1 | 1024 | 1016 | 8 |
| train\_probe | tcr\_peptide/medium | mask\_mean | 0.0080716 | 0.997293 | 1 | — | — | — |
| train\_probe | tcr\_peptide/short | mask25 | 1.54874e-05 | 1 | 1 | 51 | 46 | 5 |
| train\_probe | tcr\_peptide/short | mask50 | 3.99615e-05 | 1 | 1 | 95 | 90 | 5 |
| train\_probe | tcr\_peptide/short | mask75 | 6.61867e-05 | 1 | 1 | 142 | 136 | 6 |
| train\_probe | tcr\_peptide/short | mask100 | 0.00030042 | 1 | 1 | 200 | 192 | 8 |
| train\_probe | tcr\_peptide/short | mask\_mean | 0.000105514 | 1 | 1 | — | — | — |
| train\_probe | tcr\_pmhc/long | mask25 | 1.20374e-05 | 1 | 1 | 526 | 522 | 4 |
| train\_probe | tcr\_pmhc/long | mask50 | 7.03691e-06 | 1 | 1 | 1036 | 1031 | 5 |
| train\_probe | tcr\_pmhc/long | mask75 | 1.06897e-05 | 1 | 1 | 1557 | 1551 | 6 |
| train\_probe | tcr\_pmhc/long | mask100 | 0.12078 | 0.971079 | 1 | 2048 | 2040 | 8 |
| train\_probe | tcr\_pmhc/long | mask\_mean | 0.0302024 | 0.99277 | 1 | — | — | — |
| train\_probe | tcr\_pmhc/medium | mask25 | 5.65104e-06 | 1 | 1 | 276 | 273 | 3 |
| train\_probe | tcr\_pmhc/medium | mask50 | 9.37655e-06 | 1 | 1 | 549 | 541 | 8 |
| train\_probe | tcr\_pmhc/medium | mask75 | 8.34178e-06 | 1 | 1 | 793 | 785 | 8 |
| train\_probe | tcr\_pmhc/medium | mask100 | 0.00667305 | 0.998031 | 1 | 1024 | 1016 | 8 |
| train\_probe | tcr\_pmhc/medium | mask\_mean | 0.0016741 | 0.999508 | 1 | — | — | — |
| train\_probe | tcr\_pmhc/short | mask25 | 6.27272e-06 | 1 | 1 | 50 | 48 | 2 |
| train\_probe | tcr\_pmhc/short | mask50 | 7.8654e-06 | 1 | 1 | 103 | 100 | 3 |
| train\_probe | tcr\_pmhc/short | mask75 | 7.78604e-05 | 1 | 1 | 152 | 145 | 7 |
| train\_probe | tcr\_pmhc/short | mask100 | 0.00323137 | 1 | 1 | 200 | 192 | 8 |
| train\_probe | tcr\_pmhc/short | mask\_mean | 0.000830841 | 1 | 1 | — | — | — |
| train\_probe | unconditional\_ab/long | mask25 | 1.42479e-05 | 1 | 1 | 509 | 506 | 3 |
| train\_probe | unconditional\_ab/long | mask50 | 2.3024e-05 | 1 | 1 | 1016 | 1010 | 6 |
| train\_probe | unconditional\_ab/long | mask75 | 3.33447e-05 | 1 | 1 | 1569 | 1561 | 8 |
| train\_probe | unconditional\_ab/long | mask100 | 1.61111 | 0.269608 | 1 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_ab/long | mask\_mean | 0.402795 | 0.817402 | 1 | — | — | — |
| train\_probe | unconditional\_ab/medium | mask25 | 0.000187356 | 1 | 1 | 243 | 241 | 2 |
| train\_probe | unconditional\_ab/medium | mask50 | 1.70538e-05 | 1 | 1 | 521 | 516 | 5 |
| train\_probe | unconditional\_ab/medium | mask75 | 3.80103e-05 | 1 | 1 | 766 | 761 | 5 |
| train\_probe | unconditional\_ab/medium | mask100 | 1.67646 | 0.351378 | 1 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_ab/medium | mask\_mean | 0.419175 | 0.837844 | 1 | — | — | — |
| train\_probe | unconditional\_ab/short | mask25 | 3.33918e-05 | 1 | 1 | 53 | 52 | 1 |
| train\_probe | unconditional\_ab/short | mask50 | 0.243937 | 0.942308 | 1 | 106 | 104 | 2 |
| train\_probe | unconditional\_ab/short | mask75 | 0.54788 | 0.778481 | 1 | 163 | 158 | 5 |
| train\_probe | unconditional\_ab/short | mask100 | 1.51457 | 0.338542 | 1 | 200 | 192 | 8 |
| train\_probe | unconditional\_ab/short | mask\_mean | 0.576604 | 0.764833 | 1 | — | — | — |
| train\_probe | unconditional\_single/long | mask25 | 3.23863e-05 | 1 | 1 | 245 | 244 | 1 |
| train\_probe | unconditional\_single/long | mask50 | 2.72568e-05 | 1 | 1 | 492 | 491 | 1 |
| train\_probe | unconditional\_single/long | mask75 | 0.00315664 | 1 | 1 | 771 | 768 | 3 |
| train\_probe | unconditional\_single/long | mask100 | 1.64586 | 0.291176 | 1 | 1024 | 1020 | 4 |
| train\_probe | unconditional\_single/long | mask\_mean | 0.41227 | 0.822794 | 1 | — | — | — |
| train\_probe | unconditional\_single/medium | mask25 | 2.26061e-05 | 1 | 1 | 132 | 130 | 2 |
| train\_probe | unconditional\_single/medium | mask50 | 1.74341e-05 | 1 | 1 | 255 | 252 | 3 |
| train\_probe | unconditional\_single/medium | mask75 | 6.39063e-05 | 1 | 1 | 382 | 379 | 3 |
| train\_probe | unconditional\_single/medium | mask100 | 1.56653 | 0.346457 | 1 | 512 | 508 | 4 |
| train\_probe | unconditional\_single/medium | mask\_mean | 0.391657 | 0.836614 | 1 | — | — | — |
| train\_probe | unconditional\_single/short | mask25 | 0.00184497 | 1 | 1 | 28 | 26 | 2 |
| train\_probe | unconditional\_single/short | mask50 | 0.351942 | 0.836735 | 1 | 52 | 49 | 3 |
| train\_probe | unconditional\_single/short | mask75 | 0.937831 | 0.614286 | 1 | 73 | 70 | 3 |
| train\_probe | unconditional\_single/short | mask100 | 1.24078 | 0.4375 | 1 | 100 | 96 | 4 |
| train\_probe | unconditional\_single/short | mask\_mean | 0.6331 | 0.72213 | 1 | — | — | — |
| train\_probe | unconditional\_tcr/long | mask25 | 4.44018e-06 | 1 | — | 525 | 525 | 0 |
| train\_probe | unconditional\_tcr/long | mask50 | 3.52091e-06 | 1 | 1 | 1048 | 1045 | 3 |
| train\_probe | unconditional\_tcr/long | mask75 | 1.53003e-05 | 1 | 1 | 1585 | 1580 | 5 |
| train\_probe | unconditional\_tcr/long | mask100 | 1.61281 | 0.376471 | 1 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_tcr/long | mask\_mean | 0.403208 | 0.844118 | 1 | — | — | — |
| train\_probe | unconditional\_tcr/medium | mask25 | 3.09759e-06 | 1 | — | 261 | 261 | 0 |
| train\_probe | unconditional\_tcr/medium | mask50 | 0.000102878 | 1 | 1 | 498 | 497 | 1 |
| train\_probe | unconditional\_tcr/medium | mask75 | 0.000123509 | 1 | 1 | 755 | 751 | 4 |
| train\_probe | unconditional\_tcr/medium | mask100 | 1.44502 | 0.367126 | 1 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_tcr/medium | mask\_mean | 0.361314 | 0.841781 | 1 | — | — | — |
| train\_probe | unconditional\_tcr/short | mask25 | 2.34562e-05 | 1 | — | 46 | 46 | 0 |
| train\_probe | unconditional\_tcr/short | mask50 | 0.371023 | 0.891304 | 1 | 97 | 92 | 5 |
| train\_probe | unconditional\_tcr/short | mask75 | 0.428264 | 0.842466 | 1 | 152 | 146 | 6 |
| train\_probe | unconditional\_tcr/short | mask100 | 1.47142 | 0.317708 | 1 | 200 | 192 | 8 |
| train\_probe | unconditional\_tcr/short | mask\_mean | 0.567682 | 0.76287 | 1 | — | — | — |

### 1790803305054490194\_r22\_refined\_fixed\_gate\_72a52fe0：分层 mask 指标

源码目录：/vepfs-mlp2/c20250601/251105016/project/dllm\_test/output/immune\_research\_releases/r20\_r27\_20261001；完成/目标 step：500/500。

kind=synthetic\_generalization\_diagnostic；dataset_kind=deterministic\_artificial\_sequences；biological_quality_claim=False。

| 时点 | mask level | 指标 | 值 |
| --- | --- | --- | --- |
| initial | mask25 | mask\_ratio | 0.25 |
| initial | mask25 | selected\_count | 4467 |
| initial | mask25 | aa\_count | 4428 |
| initial | mask25 | terminal\_count | 39 |
| initial | mask25 | loss | 3.94671 |
| initial | mask25 | aa\_loss | 3.9454 |
| initial | mask25 | aa\_accuracy | 0.0546522 |
| initial | mask25 | terminal\_loss | 4.09557 |
| initial | mask25 | terminal\_accuracy | 0 |
| initial | mask50 | mask\_ratio | 0.5 |
| initial | mask50 | selected\_count | 8888 |
| initial | mask50 | aa\_count | 8819 |
| initial | mask50 | terminal\_count | 69 |
| initial | mask50 | loss | 4.10037 |
| initial | mask50 | aa\_loss | 4.10079 |
| initial | mask50 | aa\_accuracy | 0.0520467 |
| initial | mask50 | terminal\_loss | 4.04707 |
| initial | mask50 | terminal\_accuracy | 0 |
| initial | mask75 | mask\_ratio | 0.75 |
| initial | mask75 | selected\_count | 13437 |
| initial | mask75 | aa\_count | 13330 |
| initial | mask75 | terminal\_count | 107 |
| initial | mask75 | loss | 4.20503 |
| initial | mask75 | aa\_loss | 4.20545 |
| initial | mask75 | aa\_accuracy | 0.0533383 |
| initial | mask75 | terminal\_loss | 4.15238 |
| initial | mask75 | terminal\_accuracy | 0 |
| initial | mask100 | mask\_ratio | 1 |
| initial | mask100 | selected\_count | 17996 |
| initial | mask100 | aa\_count | 17864 |
| initial | mask100 | terminal\_count | 132 |
| initial | mask100 | loss | 4.1714 |
| initial | mask100 | aa\_loss | 4.17087 |
| initial | mask100 | aa\_accuracy | 0.0558665 |
| initial | mask100 | terminal\_loss | 4.24331 |
| initial | mask100 | terminal\_accuracy | 0 |
| initial | mask\_mean | loss | 4.10588 |
| initial | mask\_mean | aa\_loss | 4.10563 |
| initial | mask\_mean | terminal\_loss | 4.13458 |
| initial | mask\_mean | aa\_accuracy | 0.053976 |
| initial | mask\_mean | terminal\_accuracy | 0 |
| latest | mask25 | mask\_ratio | 0.25 |
| latest | mask25 | selected\_count | 4467 |
| latest | mask25 | aa\_count | 4428 |
| latest | mask25 | terminal\_count | 39 |
| latest | mask25 | loss | 0.000181152 |
| latest | mask25 | aa\_loss | 0.000181163 |
| latest | mask25 | aa\_accuracy | 1 |
| latest | mask25 | terminal\_loss | 0.000179855 |
| latest | mask25 | terminal\_accuracy | 1 |
| latest | mask50 | mask\_ratio | 0.5 |
| latest | mask50 | selected\_count | 8888 |
| latest | mask50 | aa\_count | 8819 |
| latest | mask50 | terminal\_count | 69 |
| latest | mask50 | loss | 0.00355179 |
| latest | mask50 | aa\_loss | 0.00357886 |
| latest | mask50 | aa\_accuracy | 0.998526 |
| latest | mask50 | terminal\_loss | 9.15699e-05 |
| latest | mask50 | terminal\_accuracy | 1 |
| latest | mask75 | mask\_ratio | 0.75 |
| latest | mask75 | selected\_count | 13437 |
| latest | mask75 | aa\_count | 13330 |
| latest | mask75 | terminal\_count | 107 |
| latest | mask75 | loss | 0.0186032 |
| latest | mask75 | aa\_loss | 0.0187517 |
| latest | mask75 | aa\_accuracy | 0.992348 |
| latest | mask75 | terminal\_loss | 0.000105749 |
| latest | mask75 | terminal\_accuracy | 1 |
| latest | mask100 | mask\_ratio | 1 |
| latest | mask100 | selected\_count | 17996 |
| latest | mask100 | aa\_count | 17864 |
| latest | mask100 | terminal\_count | 132 |
| latest | mask100 | loss | 0.737844 |
| latest | mask100 | aa\_loss | 0.743226 |
| latest | mask100 | aa\_accuracy | 0.701747 |
| latest | mask100 | terminal\_loss | 0.0093884 |
| latest | mask100 | terminal\_accuracy | 1 |
| latest | mask\_mean | loss | 0.190045 |
| latest | mask\_mean | aa\_loss | 0.191435 |
| latest | mask\_mean | terminal\_loss | 0.00244139 |
| latest | mask\_mean | aa\_accuracy | 0.923155 |
| latest | mask\_mean | terminal\_accuracy | 1 |

评测曲线（原始 step，四档等权均值）：

| step | mean loss | mean AA acc | mean end acc | gradient norm |
| --- | --- | --- | --- | --- |
| 25 | 1.85058 | 0.323122 | 0 | 4.19729 |
| 50 | 1.42035 | 0.489157 | 0 | 3.40979 |
| 75 | 0.818033 | 0.733361 | 0 | 2.38623 |
| 100 | 0.574691 | 0.80419 | 0.403514 | 2.12596 |
| 125 | 0.506755 | 0.823968 | 0.58654 | 2.30675 |
| 150 | 0.427905 | 0.854909 | 0.69289 | 1.68831 |
| 175 | 0.395642 | 0.855498 | 0.91913 | 1.65028 |
| 200 | 0.406314 | 0.857769 | 0.977273 | 1.61295 |
| 225 | 0.344624 | 0.876295 | 0.997664 | 1.76982 |
| 250 | 0.296889 | 0.886638 | 0.893491 | 1.16777 |
| 275 | 0.288644 | 0.885907 | 0.969697 | 1.324 |
| 300 | 0.290525 | 0.891989 | 0.94697 | 0.501226 |
| 325 | 0.272606 | 0.899907 | 0.979331 | 0.580972 |
| 350 | 0.269551 | 0.897994 | 0.977273 | 0.772213 |
| 375 | 0.234286 | 0.907599 | 0.965909 | 0.680651 |
| 400 | 0.217519 | 0.918133 | 0.969697 | 1.11391 |
| 425 | 0.215949 | 0.915218 | 1 | 0.721673 |
| 450 | 0.208097 | 0.921339 | 0.934886 | 0.443432 |
| 475 | 0.193285 | 0.924805 | 1 | 0.36771 |
| 500 | 0.190045 | 0.923155 | 1 | 0.328577 |

有条件/无条件曲线（每档按token计数合并，再平均四档；缺失分组不推断数值）：

| step | 子集 | mean loss | mean AA | mean end | Full-mask AA | Full-mask end | Full-mask AA count | Full-mask end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | conditional | 4.17776 | 0.0591387 | 0 | 0.0632184 | 0 | 9744 | 72 |
| initial | unconditional | 4.02 | 0.0478017 | 0 | 0.0470443 | 0 | 8120 | 60 |
| 25 | conditional | 1.79185 | 0.332756 | 0 | 0.321736 | 0 | 9744 | 72 |
| 25 | unconditional | 1.92091 | 0.311614 | 0 | 0.286207 | 0 | 8120 | 60 |
| 50 | conditional | 1.343 | 0.502087 | 0 | 0.3086 | 0 | 9744 | 72 |
| 50 | unconditional | 1.51296 | 0.473831 | 0 | 0.292365 | 0 | 8120 | 60 |
| 75 | conditional | 0.730081 | 0.753338 | 0 | 0.319479 | 0 | 9744 | 72 |
| 75 | unconditional | 0.923259 | 0.709575 | 0 | 0.289409 | 0 | 8120 | 60 |
| 100 | conditional | 0.522549 | 0.813806 | 0.45919 | 0.352114 | 0.111111 | 9744 | 72 |
| 100 | unconditional | 0.637212 | 0.79267 | 0.315939 | 0.291379 | 0.0666667 | 8120 | 60 |
| 125 | conditional | 0.456859 | 0.839895 | 0.588717 | 0.393473 | 0.236111 | 9744 | 72 |
| 125 | unconditional | 0.566597 | 0.804869 | 0.580688 | 0.280296 | 0.333333 | 8120 | 60 |
| 150 | conditional | 0.380023 | 0.873248 | 0.746454 | 0.514265 | 0.444444 | 9744 | 72 |
| 150 | unconditional | 0.48535 | 0.83291 | 0.627249 | 0.373153 | 0.133333 | 8120 | 60 |
| 175 | conditional | 0.328693 | 0.8861 | 0.979167 | 0.557779 | 0.916667 | 9744 | 72 |
| 175 | unconditional | 0.475962 | 0.818782 | 0.841634 | 0.311207 | 0.6 | 8120 | 60 |
| 200 | conditional | 0.365983 | 0.882025 | 1 | 0.536638 | 1 | 9744 | 72 |
| 200 | unconditional | 0.454694 | 0.828669 | 0.95 | 0.339532 | 0.8 | 8120 | 60 |
| 225 | conditional | 0.264914 | 0.907869 | 1 | 0.638855 | 1 | 9744 | 72 |
| 225 | unconditional | 0.44027 | 0.838411 | 0.994792 | 0.376108 | 1 | 8120 | 60 |
| 250 | conditional | 0.200615 | 0.923342 | 0.965694 | 0.695915 | 0.902778 | 9744 | 72 |
| 250 | unconditional | 0.412413 | 0.842599 | 0.80506 | 0.392118 | 0.333333 | 8120 | 60 |
| 275 | conditional | 0.18001 | 0.933673 | 1 | 0.737172 | 1 | 9744 | 72 |
| 275 | unconditional | 0.418996 | 0.828593 | 0.933333 | 0.33436 | 0.733333 | 8120 | 60 |
| 300 | conditional | 0.185438 | 0.937559 | 1 | 0.754721 | 1 | 9744 | 72 |
| 300 | unconditional | 0.416627 | 0.83731 | 0.883333 | 0.367488 | 0.533333 | 8120 | 60 |
| 325 | conditional | 0.151974 | 0.947786 | 0.990575 | 0.793001 | 0.986111 | 9744 | 72 |
| 325 | unconditional | 0.417357 | 0.842456 | 0.966667 | 0.390887 | 0.866667 | 8120 | 60 |
| 350 | conditional | 0.132929 | 0.95632 | 1 | 0.827073 | 1 | 9744 | 72 |
| 350 | unconditional | 0.433493 | 0.828006 | 0.95 | 0.333498 | 0.8 | 8120 | 60 |
| 375 | conditional | 0.0999246 | 0.964683 | 0.993056 | 0.860119 | 0.972222 | 9744 | 72 |
| 375 | unconditional | 0.395519 | 0.839099 | 0.933333 | 0.371921 | 0.733333 | 8120 | 60 |
| 400 | conditional | 0.0843534 | 0.969048 | 1 | 0.877566 | 1 | 9744 | 72 |
| 400 | unconditional | 0.377312 | 0.857038 | 0.933333 | 0.443842 | 0.733333 | 8120 | 60 |
| 425 | conditional | 0.0651325 | 0.978157 | 1 | 0.912767 | 1 | 9744 | 72 |
| 425 | unconditional | 0.396929 | 0.839694 | 1 | 0.381404 | 1 | 8120 | 60 |
| 450 | conditional | 0.0500578 | 0.984943 | 0.950851 | 0.940887 | 0.861111 | 9744 | 72 |
| 450 | unconditional | 0.397746 | 0.845013 | 0.916667 | 0.400616 | 0.666667 | 8120 | 60 |
| 475 | conditional | 0.0350871 | 0.987761 | 1 | 0.951868 | 1 | 9744 | 72 |
| 475 | unconditional | 0.383118 | 0.849261 | 1 | 0.410837 | 1 | 8120 | 60 |
| 500 | conditional | 0.0205729 | 0.993227 | 1 | 0.972906 | 1 | 9744 | 72 |
| 500 | unconditional | 0.39341 | 0.839071 | 1 | 0.376355 | 1 | 8120 | 60 |

人工数据 manifest：d40bbbbc1c4294eed843d88bcdd0cf193886de6c2d7e9db9069bbf86e931c022；精确 target sequence 重叠=0；精确 record 重叠=0。

跨 radius32 覆盖：{&quot;possible&quot;: true, &quot;train&quot;: true, &quot;valid&quot;: true}。生成规则共享是诊断设计的一部分，不能当作真实下游去重证书。


按任务/链长分组（全局 level 按 token 计数汇总，下面保留各组计数；不同组不得直接比较未经加权的总 loss）：

| 时点 | group/length | mask level | loss | AA acc | end acc | selected count | AA count | end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | ab\_antigen/long | mask25 | 4.02088 | 0.0095057 | 0 | 530 | 526 | 4 |
| initial | ab\_antigen/long | mask50 | 4.33442 | 0.011846 | 0 | 1019 | 1013 | 6 |
| initial | ab\_antigen/long | mask75 | 4.57156 | 0.0221932 | 0 | 1538 | 1532 | 6 |
| initial | ab\_antigen/long | mask100 | 4.47569 | 0.0303922 | 0 | 2048 | 2040 | 8 |
| initial | ab\_antigen/long | mask\_mean | 4.35064 | 0.0184843 | 0 | — | — | — |
| initial | ab\_antigen/medium | mask25 | 4.39362 | 0.0452675 | 0 | 246 | 243 | 3 |
| initial | ab\_antigen/medium | mask50 | 4.39276 | 0.0657371 | 0 | 507 | 502 | 5 |
| initial | ab\_antigen/medium | mask75 | 4.38496 | 0.0620053 | 0 | 765 | 758 | 7 |
| initial | ab\_antigen/medium | mask100 | 4.3142 | 0.0610236 | 0 | 1024 | 1016 | 8 |
| initial | ab\_antigen/medium | mask\_mean | 4.37138 | 0.0585084 | 0 | — | — | — |
| initial | ab\_antigen/short | mask25 | 4.68396 | 0.0588235 | 0 | 38 | 34 | 4 |
| initial | ab\_antigen/short | mask50 | 4.45858 | 0.0795455 | 0 | 93 | 88 | 5 |
| initial | ab\_antigen/short | mask75 | 4.38017 | 0.0785714 | 0 | 145 | 140 | 5 |
| initial | ab\_antigen/short | mask100 | 4.30045 | 0.09375 | 0 | 200 | 192 | 8 |
| initial | ab\_antigen/short | mask\_mean | 4.45579 | 0.0776726 | 0 | — | — | — |
| initial | tcr\_peptide/long | mask25 | 3.7745 | 0.105376 | 0 | 466 | 465 | 1 |
| initial | tcr\_peptide/long | mask50 | 3.9799 | 0.114169 | 0 | 985 | 981 | 4 |
| initial | tcr\_peptide/long | mask75 | 4.09007 | 0.118403 | 0 | 1484 | 1478 | 6 |
| initial | tcr\_peptide/long | mask100 | 4.07857 | 0.12451 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_peptide/long | mask\_mean | 3.98076 | 0.115615 | 0 | — | — | — |
| initial | tcr\_peptide/medium | mask25 | 4.26187 | 0.0580913 | 0 | 242 | 241 | 1 |
| initial | tcr\_peptide/medium | mask50 | 4.24434 | 0.0537849 | 0 | 506 | 502 | 4 |
| initial | tcr\_peptide/medium | mask75 | 4.19083 | 0.0578947 | 0 | 767 | 760 | 7 |
| initial | tcr\_peptide/medium | mask100 | 4.09311 | 0.0610236 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_peptide/medium | mask\_mean | 4.19754 | 0.0576986 | 0 | — | — | — |
| initial | tcr\_peptide/short | mask25 | 3.96389 | 0.104167 | 0 | 51 | 48 | 3 |
| initial | tcr\_peptide/short | mask50 | 3.88309 | 0.0865385 | 0 | 110 | 104 | 6 |
| initial | tcr\_peptide/short | mask75 | 3.91848 | 0.0875912 | 0 | 144 | 137 | 7 |
| initial | tcr\_peptide/short | mask100 | 3.92809 | 0.09375 | 0 | 200 | 192 | 8 |
| initial | tcr\_peptide/short | mask\_mean | 3.92339 | 0.0930116 | 0 | — | — | — |
| initial | tcr\_pmhc/long | mask25 | 3.54116 | 0.0391791 | 0 | 540 | 536 | 4 |
| initial | tcr\_pmhc/long | mask50 | 3.78469 | 0.0357834 | 0 | 1038 | 1034 | 4 |
| initial | tcr\_pmhc/long | mask75 | 4.09161 | 0.032175 | 0 | 1562 | 1554 | 8 |
| initial | tcr\_pmhc/long | mask100 | 4.2881 | 0.0313726 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_pmhc/long | mask\_mean | 3.92639 | 0.0346275 | 0 | — | — | — |
| initial | tcr\_pmhc/medium | mask25 | 4.3718 | 0.0909091 | 0 | 244 | 242 | 2 |
| initial | tcr\_pmhc/medium | mask50 | 4.4696 | 0.0793319 | 0 | 484 | 479 | 5 |
| initial | tcr\_pmhc/medium | mask75 | 4.54578 | 0.0702211 | 0 | 777 | 769 | 8 |
| initial | tcr\_pmhc/medium | mask100 | 4.55209 | 0.0629921 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_pmhc/medium | mask\_mean | 4.48482 | 0.0758636 | 0 | — | — | — |
| initial | tcr\_pmhc/short | mask25 | 4.52547 | 0.0416667 | 0 | 51 | 48 | 3 |
| initial | tcr\_pmhc/short | mask50 | 4.53172 | 0.0481928 | 0 | 86 | 83 | 3 |
| initial | tcr\_pmhc/short | mask75 | 4.37626 | 0.0680272 | 0 | 152 | 147 | 5 |
| initial | tcr\_pmhc/short | mask100 | 4.39867 | 0.0625 | 0 | 200 | 192 | 8 |
| initial | tcr\_pmhc/short | mask\_mean | 4.45803 | 0.0550967 | 0 | — | — | — |
| initial | unconditional\_ab/long | mask25 | 3.74204 | 0.0672897 | 0 | 538 | 535 | 3 |
| initial | unconditional\_ab/long | mask50 | 3.94065 | 0.0554493 | 0 | 1050 | 1046 | 4 |
| initial | unconditional\_ab/long | mask75 | 4.06046 | 0.0630689 | 0 | 1545 | 1538 | 7 |
| initial | unconditional\_ab/long | mask100 | 3.98622 | 0.0627451 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_ab/long | mask\_mean | 3.93234 | 0.0621383 | 0 | — | — | — |
| initial | unconditional\_ab/medium | mask25 | 4.16708 | 0.13834 | 0 | 256 | 253 | 3 |
| initial | unconditional\_ab/medium | mask50 | 4.26042 | 0.147638 | 0 | 512 | 508 | 4 |
| initial | unconditional\_ab/medium | mask75 | 4.22652 | 0.149606 | 0 | 768 | 762 | 6 |
| initial | unconditional\_ab/medium | mask100 | 4.1788 | 0.15748 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_ab/medium | mask\_mean | 4.20821 | 0.148266 | 0 | — | — | — |
| initial | unconditional\_ab/short | mask25 | 4.0033 | 0.0961538 | 0 | 55 | 52 | 3 |
| initial | unconditional\_ab/short | mask50 | 3.92814 | 0.0612245 | 0 | 103 | 98 | 5 |
| initial | unconditional\_ab/short | mask75 | 3.90161 | 0.0567376 | 0 | 148 | 141 | 7 |
| initial | unconditional\_ab/short | mask100 | 3.86558 | 0.0625 | 0 | 200 | 192 | 8 |
| initial | unconditional\_ab/short | mask\_mean | 3.92466 | 0.069154 | 0 | — | — | — |
| initial | unconditional\_single/long | mask25 | 3.36228 | 0.0485075 | 0 | 269 | 268 | 1 |
| initial | unconditional\_single/long | mask50 | 3.37615 | 0.00761905 | 0 | 527 | 525 | 2 |
| initial | unconditional\_single/long | mask75 | 3.4834 | 0 | 0 | 778 | 774 | 4 |
| initial | unconditional\_single/long | mask100 | 3.57495 | 0 | 0 | 1024 | 1020 | 4 |
| initial | unconditional\_single/long | mask\_mean | 3.4492 | 0.0140316 | 0 | — | — | — |
| initial | unconditional\_single/medium | mask25 | 3.80027 | 0 | 0 | 110 | 109 | 1 |
| initial | unconditional\_single/medium | mask50 | 3.61845 | 0 | 0 | 234 | 232 | 2 |
| initial | unconditional\_single/medium | mask75 | 3.63511 | 0 | 0 | 371 | 367 | 4 |
| initial | unconditional\_single/medium | mask100 | 3.62178 | 0 | 0 | 512 | 508 | 4 |
| initial | unconditional\_single/medium | mask\_mean | 3.6689 | 0 | 0 | — | — | — |
| initial | unconditional\_single/short | mask25 | 3.31903 | 0 | 0 | 28 | 26 | 2 |
| initial | unconditional\_single/short | mask50 | 3.46652 | 0 | 0 | 52 | 50 | 2 |
| initial | unconditional\_single/short | mask75 | 3.30701 | 0 | 0 | 80 | 77 | 3 |
| initial | unconditional\_single/short | mask100 | 3.31712 | 0 | 0 | 100 | 96 | 4 |
| initial | unconditional\_single/short | mask\_mean | 3.35242 | 0 | 0 | — | — | — |
| initial | unconditional\_tcr/long | mask25 | 4.04654 | 0.0368932 | 0 | 516 | 515 | 1 |
| initial | unconditional\_tcr/long | mask50 | 4.34307 | 0.0303951 | 0 | 989 | 987 | 2 |
| initial | unconditional\_tcr/long | mask75 | 4.44465 | 0.0282523 | 0 | 1527 | 1522 | 5 |
| initial | unconditional\_tcr/long | mask100 | 4.25195 | 0.0313726 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_tcr/long | mask\_mean | 4.27155 | 0.0317283 | 0 | — | — | — |
| initial | unconditional\_tcr/medium | mask25 | 4.29023 | 0 | — | 247 | 247 | 0 |
| initial | unconditional\_tcr/medium | mask50 | 4.39133 | 0 | 0 | 500 | 497 | 3 |
| initial | unconditional\_tcr/medium | mask75 | 4.36145 | 0 | 0 | 744 | 738 | 6 |
| initial | unconditional\_tcr/medium | mask100 | 4.27926 | 0 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_tcr/medium | mask\_mean | 4.33057 | 0 | 0 | — | — | — |
| initial | unconditional\_tcr/short | mask25 | 4.36457 | 0.075 | — | 40 | 40 | 0 |
| initial | unconditional\_tcr/short | mask50 | 4.17515 | 0.0777778 | 0 | 93 | 90 | 3 |
| initial | unconditional\_tcr/short | mask75 | 4.1806 | 0.0882353 | 0 | 142 | 136 | 6 |
| initial | unconditional\_tcr/short | mask100 | 4.0935 | 0.09375 | 0 | 200 | 192 | 8 |
| initial | unconditional\_tcr/short | mask\_mean | 4.20345 | 0.0836908 | 0 | — | — | — |
| latest | ab\_antigen/long | mask25 | 0.000928122 | 1 | 1 | 530 | 526 | 4 |
| latest | ab\_antigen/long | mask50 | 5.18905e-05 | 1 | 1 | 1019 | 1013 | 6 |
| latest | ab\_antigen/long | mask75 | 4.45864e-05 | 1 | 1 | 1538 | 1532 | 6 |
| latest | ab\_antigen/long | mask100 | 0.0880328 | 0.966177 | 1 | 2048 | 2040 | 8 |
| latest | ab\_antigen/long | mask\_mean | 0.0222643 | 0.991544 | 1 | — | — | — |
| latest | ab\_antigen/medium | mask25 | 1.04945e-05 | 1 | 1 | 246 | 243 | 3 |
| latest | ab\_antigen/medium | mask50 | 1.30756e-05 | 1 | 1 | 507 | 502 | 5 |
| latest | ab\_antigen/medium | mask75 | 1.07981e-05 | 1 | 1 | 765 | 758 | 7 |
| latest | ab\_antigen/medium | mask100 | 0.0267078 | 0.991142 | 1 | 1024 | 1016 | 8 |
| latest | ab\_antigen/medium | mask\_mean | 0.00668554 | 0.997785 | 1 | — | — | — |
| latest | ab\_antigen/short | mask25 | 6.07645e-06 | 1 | 1 | 38 | 34 | 4 |
| latest | ab\_antigen/short | mask50 | 1.30136e-05 | 1 | 1 | 93 | 88 | 5 |
| latest | ab\_antigen/short | mask75 | 0.0067728 | 1 | 1 | 145 | 140 | 5 |
| latest | ab\_antigen/short | mask100 | 0.000677164 | 1 | 1 | 200 | 192 | 8 |
| latest | ab\_antigen/short | mask\_mean | 0.00186726 | 1 | 1 | — | — | — |
| latest | tcr\_peptide/long | mask25 | 1.57322e-05 | 1 | 1 | 466 | 465 | 1 |
| latest | tcr\_peptide/long | mask50 | 2.08184e-05 | 1 | 1 | 985 | 981 | 4 |
| latest | tcr\_peptide/long | mask75 | 6.01649e-06 | 1 | 1 | 1484 | 1478 | 6 |
| latest | tcr\_peptide/long | mask100 | 0.121031 | 0.967647 | 1 | 2048 | 2040 | 8 |
| latest | tcr\_peptide/long | mask\_mean | 0.0302685 | 0.991912 | 1 | — | — | — |
| latest | tcr\_peptide/medium | mask25 | 5.93517e-06 | 1 | 1 | 242 | 241 | 1 |
| latest | tcr\_peptide/medium | mask50 | 4.85301e-06 | 1 | 1 | 506 | 502 | 4 |
| latest | tcr\_peptide/medium | mask75 | 4.54522e-05 | 1 | 1 | 767 | 760 | 7 |
| latest | tcr\_peptide/medium | mask100 | 0.0566531 | 0.982283 | 1 | 1024 | 1016 | 8 |
| latest | tcr\_peptide/medium | mask\_mean | 0.0141773 | 0.995571 | 1 | — | — | — |
| latest | tcr\_peptide/short | mask25 | 3.27135e-05 | 1 | 1 | 51 | 48 | 3 |
| latest | tcr\_peptide/short | mask50 | 0.0010201 | 1 | 1 | 110 | 104 | 6 |
| latest | tcr\_peptide/short | mask75 | 0.00101486 | 1 | 1 | 144 | 137 | 7 |
| latest | tcr\_peptide/short | mask100 | 0.0026696 | 1 | 1 | 200 | 192 | 8 |
| latest | tcr\_peptide/short | mask\_mean | 0.00118432 | 1 | 1 | — | — | — |
| latest | tcr\_pmhc/long | mask25 | 1.28746e-05 | 1 | 1 | 540 | 536 | 4 |
| latest | tcr\_pmhc/long | mask50 | 1.40625e-05 | 1 | 1 | 1038 | 1034 | 4 |
| latest | tcr\_pmhc/long | mask75 | 2.33357e-05 | 1 | 1 | 1562 | 1554 | 8 |
| latest | tcr\_pmhc/long | mask100 | 0.0821511 | 0.971079 | 1 | 2048 | 2040 | 8 |
| latest | tcr\_pmhc/long | mask\_mean | 0.0205503 | 0.99277 | 1 | — | — | — |
| latest | tcr\_pmhc/medium | mask25 | 9.02356e-06 | 1 | 1 | 244 | 242 | 2 |
| latest | tcr\_pmhc/medium | mask50 | 4.90854e-05 | 1 | 1 | 484 | 479 | 5 |
| latest | tcr\_pmhc/medium | mask75 | 0.000314297 | 1 | 1 | 777 | 769 | 8 |
| latest | tcr\_pmhc/medium | mask100 | 0.104374 | 0.961614 | 1 | 1024 | 1016 | 8 |
| latest | tcr\_pmhc/medium | mask\_mean | 0.0261867 | 0.990404 | 1 | — | — | — |
| latest | tcr\_pmhc/short | mask25 | 0.000144773 | 1 | 1 | 51 | 48 | 3 |
| latest | tcr\_pmhc/short | mask50 | 0.00011841 | 1 | 1 | 86 | 83 | 3 |
| latest | tcr\_pmhc/short | mask75 | 0.000429724 | 1 | 1 | 152 | 147 | 5 |
| latest | tcr\_pmhc/short | mask100 | 0.0684956 | 0.979167 | 1 | 200 | 192 | 8 |
| latest | tcr\_pmhc/short | mask\_mean | 0.0172971 | 0.994792 | 1 | — | — | — |
| latest | unconditional\_ab/long | mask25 | 7.78404e-05 | 1 | 1 | 538 | 535 | 3 |
| latest | unconditional\_ab/long | mask50 | 8.35865e-05 | 1 | 1 | 1050 | 1046 | 4 |
| latest | unconditional\_ab/long | mask75 | 0.000265132 | 1 | 1 | 1545 | 1538 | 7 |
| latest | unconditional\_ab/long | mask100 | 1.56984 | 0.340196 | 1 | 2048 | 2040 | 8 |
| latest | unconditional\_ab/long | mask\_mean | 0.392566 | 0.835049 | 1 | — | — | — |
| latest | unconditional\_ab/medium | mask25 | 0.000790844 | 1 | 1 | 256 | 253 | 3 |
| latest | unconditional\_ab/medium | mask50 | 0.00108348 | 1 | 1 | 512 | 508 | 4 |
| latest | unconditional\_ab/medium | mask75 | 0.000489133 | 1 | 1 | 768 | 762 | 6 |
| latest | unconditional\_ab/medium | mask100 | 1.51989 | 0.409449 | 1 | 1024 | 1016 | 8 |
| latest | unconditional\_ab/medium | mask\_mean | 0.380564 | 0.852362 | 1 | — | — | — |
| latest | unconditional\_ab/short | mask25 | 3.53219e-05 | 1 | 1 | 55 | 52 | 3 |
| latest | unconditional\_ab/short | mask50 | 0.142605 | 0.938775 | 1 | 103 | 98 | 5 |
| latest | unconditional\_ab/short | mask75 | 0.463896 | 0.815603 | 1 | 148 | 141 | 7 |
| latest | unconditional\_ab/short | mask100 | 1.24227 | 0.34375 | 1 | 200 | 192 | 8 |
| latest | unconditional\_ab/short | mask\_mean | 0.462203 | 0.774532 | 1 | — | — | — |
| latest | unconditional\_single/long | mask25 | 1.31675e-05 | 1 | 1 | 269 | 268 | 1 |
| latest | unconditional\_single/long | mask50 | 6.78508e-06 | 1 | 1 | 527 | 525 | 2 |
| latest | unconditional\_single/long | mask75 | 0.0904984 | 0.959948 | 1 | 778 | 774 | 4 |
| latest | unconditional\_single/long | mask100 | 1.52468 | 0.359804 | 1 | 1024 | 1020 | 4 |
| latest | unconditional\_single/long | mask\_mean | 0.4038 | 0.829938 | 1 | — | — | — |
| latest | unconditional\_single/medium | mask25 | 3.56432e-06 | 1 | 1 | 110 | 109 | 1 |
| latest | unconditional\_single/medium | mask50 | 6.17254e-06 | 1 | 1 | 234 | 232 | 2 |
| latest | unconditional\_single/medium | mask75 | 0.00492308 | 1 | 1 | 371 | 367 | 4 |
| latest | unconditional\_single/medium | mask100 | 1.56895 | 0.411417 | 1 | 512 | 508 | 4 |
| latest | unconditional\_single/medium | mask\_mean | 0.393471 | 0.852854 | 1 | — | — | — |
| latest | unconditional\_single/short | mask25 | 0.00098327 | 1 | 1 | 28 | 26 | 2 |
| latest | unconditional\_single/short | mask50 | 0.306384 | 0.86 | 1 | 52 | 50 | 2 |
| latest | unconditional\_single/short | mask75 | 0.941682 | 0.571429 | 1 | 80 | 77 | 3 |
| latest | unconditional\_single/short | mask100 | 1.35253 | 0.375 | 1 | 100 | 96 | 4 |
| latest | unconditional\_single/short | mask\_mean | 0.650394 | 0.701607 | 1 | — | — | — |
| latest | unconditional\_tcr/long | mask25 | 9.03592e-06 | 1 | 1 | 516 | 515 | 1 |
| latest | unconditional\_tcr/long | mask50 | 5.24629e-06 | 1 | 1 | 989 | 987 | 2 |
| latest | unconditional\_tcr/long | mask75 | 1.0958e-05 | 1 | 1 | 1527 | 1522 | 5 |
| latest | unconditional\_tcr/long | mask100 | 1.52754 | 0.395588 | 1 | 2048 | 2040 | 8 |
| latest | unconditional\_tcr/long | mask\_mean | 0.381891 | 0.848897 | 1 | — | — | — |
| latest | unconditional\_tcr/medium | mask25 | 1.74478e-05 | 1 | — | 247 | 247 | 0 |
| latest | unconditional\_tcr/medium | mask50 | 6.38786e-06 | 1 | 1 | 500 | 497 | 3 |
| latest | unconditional\_tcr/medium | mask75 | 2.03336e-05 | 1 | 1 | 744 | 738 | 6 |
| latest | unconditional\_tcr/medium | mask100 | 1.53442 | 0.356299 | 1 | 1024 | 1016 | 8 |
| latest | unconditional\_tcr/medium | mask\_mean | 0.383615 | 0.839075 | 1 | — | — | — |
| latest | unconditional\_tcr/short | mask25 | 2.00158e-05 | 1 | — | 40 | 40 | 0 |
| latest | unconditional\_tcr/short | mask50 | 0.000512385 | 1 | 1 | 93 | 90 | 3 |
| latest | unconditional\_tcr/short | mask75 | 0.220663 | 0.911765 | 1 | 142 | 136 | 6 |
| latest | unconditional\_tcr/short | mask100 | 1.27946 | 0.515625 | 1 | 200 | 192 | 8 |
| latest | unconditional\_tcr/short | mask\_mean | 0.375164 | 0.856847 | 1 | — | — | — |
| train\_probe | ab\_antigen/long | mask25 | 7.10174e-05 | 1 | — | 530 | 530 | 0 |
| train\_probe | ab\_antigen/long | mask50 | 2.82627e-05 | 1 | 1 | 1034 | 1031 | 3 |
| train\_probe | ab\_antigen/long | mask75 | 0.000456136 | 1 | 1 | 1563 | 1557 | 6 |
| train\_probe | ab\_antigen/long | mask100 | 0.1052 | 0.955882 | 1 | 2048 | 2040 | 8 |
| train\_probe | ab\_antigen/long | mask\_mean | 0.0264389 | 0.988971 | 1 | — | — | — |
| train\_probe | ab\_antigen/medium | mask25 | 7.6678e-05 | 1 | 1 | 275 | 273 | 2 |
| train\_probe | ab\_antigen/medium | mask50 | 5.16423e-05 | 1 | 1 | 542 | 538 | 4 |
| train\_probe | ab\_antigen/medium | mask75 | 3.84601e-05 | 1 | 1 | 781 | 776 | 5 |
| train\_probe | ab\_antigen/medium | mask100 | 0.00217197 | 1 | 1 | 1024 | 1016 | 8 |
| train\_probe | ab\_antigen/medium | mask\_mean | 0.000584687 | 1 | 1 | — | — | — |
| train\_probe | ab\_antigen/short | mask25 | 4.74538e-06 | 1 | 1 | 52 | 47 | 5 |
| train\_probe | ab\_antigen/short | mask50 | 5.39238e-05 | 1 | 1 | 109 | 102 | 7 |
| train\_probe | ab\_antigen/short | mask75 | 0.00970907 | 0.993243 | 1 | 155 | 148 | 7 |
| train\_probe | ab\_antigen/short | mask100 | 0.000735025 | 1 | 1 | 200 | 192 | 8 |
| train\_probe | ab\_antigen/short | mask\_mean | 0.00262569 | 0.998311 | 1 | — | — | — |
| train\_probe | tcr\_peptide/long | mask25 | 4.32853e-06 | 1 | 1 | 495 | 494 | 1 |
| train\_probe | tcr\_peptide/long | mask50 | 3.55232e-06 | 1 | 1 | 1033 | 1028 | 5 |
| train\_probe | tcr\_peptide/long | mask75 | 5.66239e-06 | 1 | 1 | 1530 | 1522 | 8 |
| train\_probe | tcr\_peptide/long | mask100 | 0.0689358 | 0.978431 | 1 | 2048 | 2040 | 8 |
| train\_probe | tcr\_peptide/long | mask\_mean | 0.0172373 | 0.994608 | 1 | — | — | — |
| train\_probe | tcr\_peptide/medium | mask25 | 4.26402e-06 | 1 | — | 252 | 252 | 0 |
| train\_probe | tcr\_peptide/medium | mask50 | 3.58591e-06 | 1 | 1 | 517 | 513 | 4 |
| train\_probe | tcr\_peptide/medium | mask75 | 1.74506e-05 | 1 | 1 | 760 | 754 | 6 |
| train\_probe | tcr\_peptide/medium | mask100 | 0.044915 | 0.983268 | 1 | 1024 | 1016 | 8 |
| train\_probe | tcr\_peptide/medium | mask\_mean | 0.0112351 | 0.995817 | 1 | — | — | — |
| train\_probe | tcr\_peptide/short | mask25 | 3.55055e-06 | 1 | 1 | 51 | 46 | 5 |
| train\_probe | tcr\_peptide/short | mask50 | 8.35963e-05 | 1 | 1 | 95 | 90 | 5 |
| train\_probe | tcr\_peptide/short | mask75 | 0.000140023 | 1 | 1 | 142 | 136 | 6 |
| train\_probe | tcr\_peptide/short | mask100 | 0.000465218 | 1 | 1 | 200 | 192 | 8 |
| train\_probe | tcr\_peptide/short | mask\_mean | 0.000173097 | 1 | 1 | — | — | — |
| train\_probe | tcr\_pmhc/long | mask25 | 6.0689e-06 | 1 | 1 | 526 | 522 | 4 |
| train\_probe | tcr\_pmhc/long | mask50 | 4.84398e-06 | 1 | 1 | 1036 | 1031 | 5 |
| train\_probe | tcr\_pmhc/long | mask75 | 6.47332e-06 | 1 | 1 | 1557 | 1551 | 6 |
| train\_probe | tcr\_pmhc/long | mask100 | 0.176903 | 0.959804 | 1 | 2048 | 2040 | 8 |
| train\_probe | tcr\_pmhc/long | mask\_mean | 0.04423 | 0.989951 | 1 | — | — | — |
| train\_probe | tcr\_pmhc/medium | mask25 | 5.73212e-05 | 1 | 1 | 276 | 273 | 3 |
| train\_probe | tcr\_pmhc/medium | mask50 | 1.78304e-05 | 1 | 1 | 549 | 541 | 8 |
| train\_probe | tcr\_pmhc/medium | mask75 | 7.44627e-05 | 1 | 1 | 793 | 785 | 8 |
| train\_probe | tcr\_pmhc/medium | mask100 | 0.00848624 | 0.999016 | 1 | 1024 | 1016 | 8 |
| train\_probe | tcr\_pmhc/medium | mask\_mean | 0.00215896 | 0.999754 | 1 | — | — | — |
| train\_probe | tcr\_pmhc/short | mask25 | 9.91553e-06 | 1 | 1 | 50 | 48 | 2 |
| train\_probe | tcr\_pmhc/short | mask50 | 2.38708e-05 | 1 | 1 | 103 | 100 | 3 |
| train\_probe | tcr\_pmhc/short | mask75 | 8.54087e-05 | 1 | 1 | 152 | 145 | 7 |
| train\_probe | tcr\_pmhc/short | mask100 | 0.00291153 | 1 | 1 | 200 | 192 | 8 |
| train\_probe | tcr\_pmhc/short | mask\_mean | 0.000757681 | 1 | 1 | — | — | — |
| train\_probe | unconditional\_ab/long | mask25 | 3.09038e-05 | 1 | 1 | 509 | 506 | 3 |
| train\_probe | unconditional\_ab/long | mask50 | 1.20439e-05 | 1 | 1 | 1016 | 1010 | 6 |
| train\_probe | unconditional\_ab/long | mask75 | 1.54071e-05 | 1 | 1 | 1569 | 1561 | 8 |
| train\_probe | unconditional\_ab/long | mask100 | 1.54242 | 0.291176 | 1 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_ab/long | mask\_mean | 0.385619 | 0.822794 | 1 | — | — | — |
| train\_probe | unconditional\_ab/medium | mask25 | 0.00250369 | 1 | 1 | 243 | 241 | 2 |
| train\_probe | unconditional\_ab/medium | mask50 | 0.00084706 | 1 | 1 | 521 | 516 | 5 |
| train\_probe | unconditional\_ab/medium | mask75 | 0.00157723 | 1 | 1 | 766 | 761 | 5 |
| train\_probe | unconditional\_ab/medium | mask100 | 1.59217 | 0.346457 | 1 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_ab/medium | mask\_mean | 0.399275 | 0.836614 | 1 | — | — | — |
| train\_probe | unconditional\_ab/short | mask25 | 2.53376e-05 | 1 | 1 | 53 | 52 | 1 |
| train\_probe | unconditional\_ab/short | mask50 | 0.280131 | 0.942308 | 1 | 106 | 104 | 2 |
| train\_probe | unconditional\_ab/short | mask75 | 0.594464 | 0.746835 | 1 | 163 | 158 | 5 |
| train\_probe | unconditional\_ab/short | mask100 | 1.50493 | 0.3125 | 1 | 200 | 192 | 8 |
| train\_probe | unconditional\_ab/short | mask\_mean | 0.594888 | 0.750411 | 1 | — | — | — |
| train\_probe | unconditional\_single/long | mask25 | 3.54749e-05 | 1 | 1 | 245 | 244 | 1 |
| train\_probe | unconditional\_single/long | mask50 | 2.16026e-05 | 1 | 1 | 492 | 491 | 1 |
| train\_probe | unconditional\_single/long | mask75 | 0.000701738 | 1 | 1 | 771 | 768 | 3 |
| train\_probe | unconditional\_single/long | mask100 | 1.66601 | 0.295098 | 1 | 1024 | 1020 | 4 |
| train\_probe | unconditional\_single/long | mask\_mean | 0.416692 | 0.823775 | 1 | — | — | — |
| train\_probe | unconditional\_single/medium | mask25 | 3.08585e-06 | 1 | 1 | 132 | 130 | 2 |
| train\_probe | unconditional\_single/medium | mask50 | 9.80217e-06 | 1 | 1 | 255 | 252 | 3 |
| train\_probe | unconditional\_single/medium | mask75 | 4.78722e-05 | 1 | 1 | 382 | 379 | 3 |
| train\_probe | unconditional\_single/medium | mask100 | 1.51628 | 0.370079 | 1 | 512 | 508 | 4 |
| train\_probe | unconditional\_single/medium | mask\_mean | 0.379084 | 0.84252 | 1 | — | — | — |
| train\_probe | unconditional\_single/short | mask25 | 0.00198687 | 1 | 1 | 28 | 26 | 2 |
| train\_probe | unconditional\_single/short | mask50 | 0.393266 | 0.857143 | 1 | 52 | 49 | 3 |
| train\_probe | unconditional\_single/short | mask75 | 0.876605 | 0.628571 | 1 | 73 | 70 | 3 |
| train\_probe | unconditional\_single/short | mask100 | 1.23503 | 0.4375 | 1 | 100 | 96 | 4 |
| train\_probe | unconditional\_single/short | mask\_mean | 0.626721 | 0.730804 | 1 | — | — | — |
| train\_probe | unconditional\_tcr/long | mask25 | 1.26962e-05 | 1 | — | 525 | 525 | 0 |
| train\_probe | unconditional\_tcr/long | mask50 | 2.09094e-05 | 1 | 1 | 1048 | 1045 | 3 |
| train\_probe | unconditional\_tcr/long | mask75 | 1.50544e-05 | 1 | 1 | 1585 | 1580 | 5 |
| train\_probe | unconditional\_tcr/long | mask100 | 1.59059 | 0.346569 | 1 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_tcr/long | mask\_mean | 0.397659 | 0.836642 | 1 | — | — | — |
| train\_probe | unconditional\_tcr/medium | mask25 | 3.78209e-06 | 1 | — | 261 | 261 | 0 |
| train\_probe | unconditional\_tcr/medium | mask50 | 8.52167e-06 | 1 | 1 | 498 | 497 | 1 |
| train\_probe | unconditional\_tcr/medium | mask75 | 8.82638e-05 | 1 | 1 | 755 | 751 | 4 |
| train\_probe | unconditional\_tcr/medium | mask100 | 1.50836 | 0.383858 | 1 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_tcr/medium | mask\_mean | 0.377114 | 0.845965 | 1 | — | — | — |
| train\_probe | unconditional\_tcr/short | mask25 | 5.3485e-05 | 1 | — | 46 | 46 | 0 |
| train\_probe | unconditional\_tcr/short | mask50 | 0.330939 | 0.869565 | 1 | 97 | 92 | 5 |
| train\_probe | unconditional\_tcr/short | mask75 | 0.481848 | 0.828767 | 1 | 152 | 146 | 6 |
| train\_probe | unconditional\_tcr/short | mask100 | 1.45477 | 0.3125 | 1 | 200 | 192 | 8 |
| train\_probe | unconditional\_tcr/short | mask\_mean | 0.566903 | 0.752708 | 1 | — | — | — |

### 1790803305057942502\_r23\_refined\_layerscale\_56d8467a：分层 mask 指标

源码目录：/vepfs-mlp2/c20250601/251105016/project/dllm\_test/output/immune\_research\_releases/r20\_r27\_20261001；完成/目标 step：500/500。

kind=synthetic\_generalization\_diagnostic；dataset_kind=deterministic\_artificial\_sequences；biological_quality_claim=False。

| 时点 | mask level | 指标 | 值 |
| --- | --- | --- | --- |
| initial | mask25 | mask\_ratio | 0.25 |
| initial | mask25 | selected\_count | 4467 |
| initial | mask25 | aa\_count | 4428 |
| initial | mask25 | terminal\_count | 39 |
| initial | mask25 | loss | 3.81518 |
| initial | mask25 | aa\_loss | 3.8107 |
| initial | mask25 | aa\_accuracy | 0.0589431 |
| initial | mask25 | terminal\_loss | 4.32412 |
| initial | mask25 | terminal\_accuracy | 0 |
| initial | mask50 | mask\_ratio | 0.5 |
| initial | mask50 | selected\_count | 8888 |
| initial | mask50 | aa\_count | 8819 |
| initial | mask50 | terminal\_count | 69 |
| initial | mask50 | loss | 3.85226 |
| initial | mask50 | aa\_loss | 3.8493 |
| initial | mask50 | aa\_accuracy | 0.0426352 |
| initial | mask50 | terminal\_loss | 4.23099 |
| initial | mask50 | terminal\_accuracy | 0 |
| initial | mask75 | mask\_ratio | 0.75 |
| initial | mask75 | selected\_count | 13437 |
| initial | mask75 | aa\_count | 13330 |
| initial | mask75 | terminal\_count | 107 |
| initial | mask75 | loss | 3.83517 |
| initial | mask75 | aa\_loss | 3.83171 |
| initial | mask75 | aa\_accuracy | 0.0306827 |
| initial | mask75 | terminal\_loss | 4.26675 |
| initial | mask75 | terminal\_accuracy | 0 |
| initial | mask100 | mask\_ratio | 1 |
| initial | mask100 | selected\_count | 17996 |
| initial | mask100 | aa\_count | 17864 |
| initial | mask100 | terminal\_count | 132 |
| initial | mask100 | loss | 3.80145 |
| initial | mask100 | aa\_loss | 3.79761 |
| initial | mask100 | aa\_accuracy | 0.0301164 |
| initial | mask100 | terminal\_loss | 4.32137 |
| initial | mask100 | terminal\_accuracy | 0 |
| initial | mask\_mean | loss | 3.82602 |
| initial | mask\_mean | aa\_loss | 3.82233 |
| initial | mask\_mean | terminal\_loss | 4.28581 |
| initial | mask\_mean | aa\_accuracy | 0.0405944 |
| initial | mask\_mean | terminal\_accuracy | 0 |
| latest | mask25 | mask\_ratio | 0.25 |
| latest | mask25 | selected\_count | 4467 |
| latest | mask25 | aa\_count | 4428 |
| latest | mask25 | terminal\_count | 39 |
| latest | mask25 | loss | 4.96193e-05 |
| latest | mask25 | aa\_loss | 3.99527e-05 |
| latest | mask25 | aa\_accuracy | 1 |
| latest | mask25 | terminal\_loss | 0.00114714 |
| latest | mask25 | terminal\_accuracy | 1 |
| latest | mask50 | mask\_ratio | 0.5 |
| latest | mask50 | selected\_count | 8888 |
| latest | mask50 | aa\_count | 8819 |
| latest | mask50 | terminal\_count | 69 |
| latest | mask50 | loss | 0.00387918 |
| latest | mask50 | aa\_loss | 0.00390083 |
| latest | mask50 | aa\_accuracy | 0.998299 |
| latest | mask50 | terminal\_loss | 0.00111181 |
| latest | mask50 | terminal\_accuracy | 1 |
| latest | mask75 | mask\_ratio | 0.75 |
| latest | mask75 | selected\_count | 13437 |
| latest | mask75 | aa\_count | 13330 |
| latest | mask75 | terminal\_count | 107 |
| latest | mask75 | loss | 0.0256688 |
| latest | mask75 | aa\_loss | 0.0258537 |
| latest | mask75 | aa\_accuracy | 0.992723 |
| latest | mask75 | terminal\_loss | 0.00263575 |
| latest | mask75 | terminal\_accuracy | 1 |
| latest | mask100 | mask\_ratio | 1 |
| latest | mask100 | selected\_count | 17996 |
| latest | mask100 | aa\_count | 17864 |
| latest | mask100 | terminal\_count | 132 |
| latest | mask100 | loss | 0.733126 |
| latest | mask100 | aa\_loss | 0.737866 |
| latest | mask100 | aa\_accuracy | 0.702642 |
| latest | mask100 | terminal\_loss | 0.0916121 |
| latest | mask100 | terminal\_accuracy | 0.969697 |
| latest | mask\_mean | loss | 0.190681 |
| latest | mask\_mean | aa\_loss | 0.191915 |
| latest | mask\_mean | terminal\_loss | 0.0241267 |
| latest | mask\_mean | aa\_accuracy | 0.923416 |
| latest | mask\_mean | terminal\_accuracy | 0.992424 |

评测曲线（原始 step，四档等权均值）：

| step | mean loss | mean AA acc | mean end acc | gradient norm |
| --- | --- | --- | --- | --- |
| 25 | 1.84578 | 0.320364 | 0 | 1.56676 |
| 50 | 1.18137 | 0.596592 | 0 | 4.12637 |
| 75 | 0.640915 | 0.802672 | 0.0100334 | 3.51836 |
| 100 | 0.596186 | 0.813556 | 0.1791 | 3.44498 |
| 125 | 0.517689 | 0.820537 | 0.736127 | 2.2088 |
| 150 | 0.421129 | 0.858524 | 0.89701 | 0.762092 |
| 175 | 0.405974 | 0.85302 | 0.992424 | 1.86009 |
| 200 | 0.357602 | 0.865901 | 0.984848 | 0.915907 |
| 225 | 0.326915 | 0.872595 | 0.935164 | 1.20638 |
| 250 | 0.273447 | 0.90144 | 0.977273 | 0.848537 |
| 275 | 0.266903 | 0.897458 | 0.969697 | 1.08088 |
| 300 | 0.269096 | 0.903595 | 0.967361 | 0.331419 |
| 325 | 0.2288 | 0.914326 | 0.984848 | 0.33998 |
| 350 | 0.195889 | 0.921756 | 1 | 0.454376 |
| 375 | 0.213232 | 0.917388 | 0.977273 | 0.329231 |
| 400 | 0.192935 | 0.931409 | 1 | 0.427316 |
| 425 | 0.201471 | 0.919584 | 1 | 0.415494 |
| 450 | 0.183567 | 0.929813 | 1 | 0.208101 |
| 475 | 0.18264 | 0.928077 | 0.998106 | 0.369751 |
| 500 | 0.190681 | 0.923416 | 0.992424 | 0.351923 |

有条件/无条件曲线（每档按token计数合并，再平均四档；缺失分组不推断数值）：

| step | 子集 | mean loss | mean AA | mean end | Full-mask AA | Full-mask end | Full-mask AA count | Full-mask end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | conditional | 3.79117 | 0.0444692 | 0 | 0.0298645 | 0 | 9744 | 72 |
| initial | unconditional | 3.86732 | 0.0360518 | 0 | 0.0304187 | 0 | 8120 | 60 |
| 25 | conditional | 1.79854 | 0.317915 | 0 | 0.309011 | 0 | 9744 | 72 |
| 25 | unconditional | 1.90253 | 0.323239 | 0 | 0.314901 | 0 | 8120 | 60 |
| 50 | conditional | 1.06985 | 0.623722 | 0 | 0.306137 | 0 | 9744 | 72 |
| 50 | unconditional | 1.31467 | 0.564438 | 0 | 0.302709 | 0 | 8120 | 60 |
| 75 | conditional | 0.56786 | 0.80702 | 0.01 | 0.317734 | 0 | 9744 | 72 |
| 75 | unconditional | 0.728548 | 0.797453 | 0.00925926 | 0.310468 | 0 | 8120 | 60 |
| 100 | conditional | 0.517468 | 0.81821 | 0.182924 | 0.332102 | 0.0555556 | 9744 | 72 |
| 100 | unconditional | 0.690574 | 0.808 | 0.181961 | 0.313177 | 0 | 8120 | 60 |
| 125 | conditional | 0.439278 | 0.843896 | 0.772823 | 0.394807 | 0.486111 | 9744 | 72 |
| 125 | unconditional | 0.611749 | 0.792514 | 0.692907 | 0.211084 | 0.266667 | 8120 | 60 |
| 150 | conditional | 0.371121 | 0.875475 | 0.922846 | 0.516318 | 0.708333 | 9744 | 72 |
| 150 | unconditional | 0.481125 | 0.83819 | 0.859458 | 0.380911 | 0.666667 | 8120 | 60 |
| 175 | conditional | 0.338754 | 0.875655 | 1 | 0.5117 | 1 | 9744 | 72 |
| 175 | unconditional | 0.486624 | 0.825865 | 0.983333 | 0.330049 | 0.933333 | 8120 | 60 |
| 200 | conditional | 0.288324 | 0.899154 | 1 | 0.600677 | 1 | 9744 | 72 |
| 200 | unconditional | 0.440729 | 0.826003 | 0.966667 | 0.327463 | 0.866667 | 8120 | 60 |
| 225 | conditional | 0.244979 | 0.90711 | 0.996528 | 0.631876 | 0.986111 | 9744 | 72 |
| 225 | unconditional | 0.425232 | 0.831184 | 0.861458 | 0.348522 | 0.466667 | 8120 | 60 |
| 250 | conditional | 0.165308 | 0.942996 | 1 | 0.773707 | 1 | 9744 | 72 |
| 250 | unconditional | 0.403211 | 0.851577 | 0.95 | 0.429434 | 0.8 | 8120 | 60 |
| 275 | conditional | 0.124715 | 0.952979 | 1 | 0.812603 | 1 | 9744 | 72 |
| 275 | unconditional | 0.437521 | 0.830838 | 0.933333 | 0.345074 | 0.733333 | 8120 | 60 |
| 300 | conditional | 0.125369 | 0.962288 | 1 | 0.851704 | 1 | 9744 | 72 |
| 300 | unconditional | 0.441563 | 0.833166 | 0.928125 | 0.355542 | 0.733333 | 8120 | 60 |
| 325 | conditional | 0.0666793 | 0.978582 | 1 | 0.915846 | 1 | 9744 | 72 |
| 325 | unconditional | 0.423336 | 0.837222 | 0.966667 | 0.369951 | 0.866667 | 8120 | 60 |
| 350 | conditional | 0.0284103 | 0.991448 | 1 | 0.965928 | 1 | 9744 | 72 |
| 350 | unconditional | 0.396857 | 0.838129 | 1 | 0.371675 | 1 | 8120 | 60 |
| 375 | conditional | 0.0449308 | 0.983852 | 1 | 0.935755 | 1 | 9744 | 72 |
| 375 | unconditional | 0.415194 | 0.837628 | 0.95 | 0.368596 | 0.8 | 8120 | 60 |
| 400 | conditional | 0.0229073 | 0.993894 | 1 | 0.975575 | 1 | 9744 | 72 |
| 400 | unconditional | 0.396965 | 0.856428 | 1 | 0.443966 | 1 | 8120 | 60 |
| 425 | conditional | 0.0283135 | 0.990737 | 1 | 0.963362 | 1 | 9744 | 72 |
| 425 | unconditional | 0.409257 | 0.834201 | 1 | 0.360837 | 1 | 8120 | 60 |
| 450 | conditional | 0.00931331 | 0.997126 | 1 | 0.988506 | 1 | 9744 | 72 |
| 450 | unconditional | 0.392666 | 0.849038 | 1 | 0.414901 | 1 | 8120 | 60 |
| 475 | conditional | 0.0155099 | 0.994431 | 0.996528 | 0.978346 | 0.986111 | 9744 | 72 |
| 475 | unconditional | 0.383191 | 0.848454 | 1 | 0.412069 | 1 | 8120 | 60 |
| 500 | conditional | 0.00795607 | 0.996767 | 1 | 0.987069 | 1 | 9744 | 72 |
| 500 | unconditional | 0.409951 | 0.835398 | 0.983333 | 0.36133 | 0.933333 | 8120 | 60 |

人工数据 manifest：d40bbbbc1c4294eed843d88bcdd0cf193886de6c2d7e9db9069bbf86e931c022；精确 target sequence 重叠=0；精确 record 重叠=0。

跨 radius32 覆盖：{&quot;possible&quot;: true, &quot;train&quot;: true, &quot;valid&quot;: true}。生成规则共享是诊断设计的一部分，不能当作真实下游去重证书。


按任务/链长分组（全局 level 按 token 计数汇总，下面保留各组计数；不同组不得直接比较未经加权的总 loss）：

| 时点 | group/length | mask level | loss | AA acc | end acc | selected count | AA count | end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | ab\_antigen/long | mask25 | 4.10241 | 0 | 0 | 530 | 526 | 4 |
| initial | ab\_antigen/long | mask50 | 4.14621 | 0 | 0 | 1019 | 1013 | 6 |
| initial | ab\_antigen/long | mask75 | 4.10335 | 0 | 0 | 1538 | 1532 | 6 |
| initial | ab\_antigen/long | mask100 | 4.03516 | 0 | 0 | 2048 | 2040 | 8 |
| initial | ab\_antigen/long | mask\_mean | 4.09678 | 0 | 0 | — | — | — |
| initial | ab\_antigen/medium | mask25 | 4.1109 | 0 | 0 | 246 | 243 | 3 |
| initial | ab\_antigen/medium | mask50 | 4.0947 | 0 | 0 | 507 | 502 | 5 |
| initial | ab\_antigen/medium | mask75 | 4.07023 | 0 | 0 | 765 | 758 | 7 |
| initial | ab\_antigen/medium | mask100 | 4.03235 | 0 | 0 | 1024 | 1016 | 8 |
| initial | ab\_antigen/medium | mask\_mean | 4.07704 | 0 | 0 | — | — | — |
| initial | ab\_antigen/short | mask25 | 3.94668 | 0.0588235 | 0 | 38 | 34 | 4 |
| initial | ab\_antigen/short | mask50 | 3.8665 | 0.0454545 | 0 | 93 | 88 | 5 |
| initial | ab\_antigen/short | mask75 | 3.78397 | 0.05 | 0 | 145 | 140 | 5 |
| initial | ab\_antigen/short | mask100 | 3.7871 | 0.046875 | 0 | 200 | 192 | 8 |
| initial | ab\_antigen/short | mask\_mean | 3.84606 | 0.0502883 | 0 | — | — | — |
| initial | tcr\_peptide/long | mask25 | 3.70198 | 0.0258065 | 0 | 466 | 465 | 1 |
| initial | tcr\_peptide/long | mask50 | 3.77676 | 0.011213 | 0 | 985 | 981 | 4 |
| initial | tcr\_peptide/long | mask75 | 3.77699 | 0.0121786 | 0 | 1484 | 1478 | 6 |
| initial | tcr\_peptide/long | mask100 | 3.76535 | 0.0151961 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_peptide/long | mask\_mean | 3.75527 | 0.0160985 | 0 | — | — | — |
| initial | tcr\_peptide/medium | mask25 | 3.97216 | 0.00829876 | 0 | 242 | 241 | 1 |
| initial | tcr\_peptide/medium | mask50 | 3.90908 | 0.00996016 | 0 | 506 | 502 | 4 |
| initial | tcr\_peptide/medium | mask75 | 3.85215 | 0.0118421 | 0 | 767 | 760 | 7 |
| initial | tcr\_peptide/medium | mask100 | 3.78247 | 0.0147638 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_peptide/medium | mask\_mean | 3.87897 | 0.0112162 | 0 | — | — | — |
| initial | tcr\_peptide/short | mask25 | 3.64878 | 0.0625 | 0 | 51 | 48 | 3 |
| initial | tcr\_peptide/short | mask50 | 3.61911 | 0.0865385 | 0 | 110 | 104 | 6 |
| initial | tcr\_peptide/short | mask75 | 3.66048 | 0.0656934 | 0 | 144 | 137 | 7 |
| initial | tcr\_peptide/short | mask100 | 3.65702 | 0.046875 | 0 | 200 | 192 | 8 |
| initial | tcr\_peptide/short | mask\_mean | 3.64635 | 0.0654017 | 0 | — | — | — |
| initial | tcr\_pmhc/long | mask25 | 3.19962 | 0.266791 | 0 | 540 | 536 | 4 |
| initial | tcr\_pmhc/long | mask50 | 3.30848 | 0.185687 | 0 | 1038 | 1034 | 4 |
| initial | tcr\_pmhc/long | mask75 | 3.37315 | 0.117761 | 0 | 1562 | 1554 | 8 |
| initial | tcr\_pmhc/long | mask100 | 3.41892 | 0.109804 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_pmhc/long | mask\_mean | 3.32504 | 0.170011 | 0 | — | — | — |
| initial | tcr\_pmhc/medium | mask25 | 3.82185 | 0.0165289 | 0 | 244 | 242 | 2 |
| initial | tcr\_pmhc/medium | mask50 | 3.84317 | 0.00626305 | 0 | 484 | 479 | 5 |
| initial | tcr\_pmhc/medium | mask75 | 3.85431 | 0 | 0 | 777 | 769 | 8 |
| initial | tcr\_pmhc/medium | mask100 | 3.8816 | 0 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_pmhc/medium | mask\_mean | 3.85023 | 0.00569799 | 0 | — | — | — |
| initial | tcr\_pmhc/short | mask25 | 3.76194 | 0 | 0 | 51 | 48 | 3 |
| initial | tcr\_pmhc/short | mask50 | 3.81972 | 0.0120482 | 0 | 86 | 83 | 3 |
| initial | tcr\_pmhc/short | mask75 | 3.80688 | 0.0136054 | 0 | 152 | 147 | 5 |
| initial | tcr\_pmhc/short | mask100 | 3.83313 | 0.015625 | 0 | 200 | 192 | 8 |
| initial | tcr\_pmhc/short | mask\_mean | 3.80542 | 0.0103197 | 0 | — | — | — |
| initial | unconditional\_ab/long | mask25 | 3.9701 | 0.0616822 | 0 | 538 | 535 | 3 |
| initial | unconditional\_ab/long | mask50 | 3.92095 | 0.0602294 | 0 | 1050 | 1046 | 4 |
| initial | unconditional\_ab/long | mask75 | 3.88596 | 0.0318596 | 0 | 1545 | 1538 | 7 |
| initial | unconditional\_ab/long | mask100 | 3.79481 | 0.0308824 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_ab/long | mask\_mean | 3.89296 | 0.0461634 | 0 | — | — | — |
| initial | unconditional\_ab/medium | mask25 | 3.7778 | 0.0592885 | 0 | 256 | 253 | 3 |
| initial | unconditional\_ab/medium | mask50 | 3.8424 | 0.0393701 | 0 | 512 | 508 | 4 |
| initial | unconditional\_ab/medium | mask75 | 3.8188 | 0.0354331 | 0 | 768 | 762 | 6 |
| initial | unconditional\_ab/medium | mask100 | 3.7911 | 0.0305118 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_ab/medium | mask\_mean | 3.80753 | 0.0411509 | 0 | — | — | — |
| initial | unconditional\_ab/short | mask25 | 3.85957 | 0.0576923 | 0 | 55 | 52 | 3 |
| initial | unconditional\_ab/short | mask50 | 3.84337 | 0.0306122 | 0 | 103 | 98 | 5 |
| initial | unconditional\_ab/short | mask75 | 3.80081 | 0.035461 | 0 | 148 | 141 | 7 |
| initial | unconditional\_ab/short | mask100 | 3.75868 | 0.03125 | 0 | 200 | 192 | 8 |
| initial | unconditional\_ab/short | mask\_mean | 3.8156 | 0.0387539 | 0 | — | — | — |
| initial | unconditional\_single/long | mask25 | 3.94339 | 0.0485075 | 0 | 269 | 268 | 1 |
| initial | unconditional\_single/long | mask50 | 4.00614 | 0.0514286 | 0 | 527 | 525 | 2 |
| initial | unconditional\_single/long | mask75 | 3.96455 | 0.0516796 | 0 | 778 | 774 | 4 |
| initial | unconditional\_single/long | mask100 | 3.88226 | 0.0617647 | 0 | 1024 | 1020 | 4 |
| initial | unconditional\_single/long | mask\_mean | 3.94909 | 0.0533451 | 0 | — | — | — |
| initial | unconditional\_single/medium | mask25 | 4.15354 | 0.0366972 | 0 | 110 | 109 | 1 |
| initial | unconditional\_single/medium | mask50 | 4.00297 | 0.0258621 | 0 | 234 | 232 | 2 |
| initial | unconditional\_single/medium | mask75 | 3.95779 | 0.0245232 | 0 | 371 | 367 | 4 |
| initial | unconditional\_single/medium | mask100 | 3.90136 | 0.0295276 | 0 | 512 | 508 | 4 |
| initial | unconditional\_single/medium | mask\_mean | 4.00392 | 0.0291525 | 0 | — | — | — |
| initial | unconditional\_single/short | mask25 | 3.41472 | 0.269231 | 0 | 28 | 26 | 2 |
| initial | unconditional\_single/short | mask50 | 3.5961 | 0.18 | 0 | 52 | 50 | 2 |
| initial | unconditional\_single/short | mask75 | 3.42945 | 0.207792 | 0 | 80 | 77 | 3 |
| initial | unconditional\_single/short | mask100 | 3.41298 | 0.21875 | 0 | 100 | 96 | 4 |
| initial | unconditional\_single/short | mask\_mean | 3.46331 | 0.218943 | 0 | — | — | — |
| initial | unconditional\_tcr/long | mask25 | 3.81511 | 0.0291262 | 0 | 516 | 515 | 1 |
| initial | unconditional\_tcr/long | mask50 | 3.90053 | 0.0151976 | 0 | 989 | 987 | 2 |
| initial | unconditional\_tcr/long | mask75 | 3.85577 | 0.0157687 | 0 | 1527 | 1522 | 5 |
| initial | unconditional\_tcr/long | mask100 | 3.77771 | 0.0156863 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_tcr/long | mask\_mean | 3.83728 | 0.0189447 | 0 | — | — | — |
| initial | unconditional\_tcr/medium | mask25 | 3.74311 | 0.0202429 | — | 247 | 247 | 0 |
| initial | unconditional\_tcr/medium | mask50 | 3.86175 | 0.0160966 | 0 | 500 | 497 | 3 |
| initial | unconditional\_tcr/medium | mask75 | 3.85697 | 0.0149051 | 0 | 744 | 738 | 6 |
| initial | unconditional\_tcr/medium | mask100 | 3.8777 | 0.015748 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_tcr/medium | mask\_mean | 3.83488 | 0.0167482 | 0 | — | — | — |
| initial | unconditional\_tcr/short | mask25 | 4.01228 | 0 | — | 40 | 40 | 0 |
| initial | unconditional\_tcr/short | mask50 | 3.82482 | 0 | 0 | 93 | 90 | 3 |
| initial | unconditional\_tcr/short | mask75 | 3.85293 | 0 | 0 | 142 | 136 | 6 |
| initial | unconditional\_tcr/short | mask100 | 3.86799 | 0 | 0 | 200 | 192 | 8 |
| initial | unconditional\_tcr/short | mask\_mean | 3.8895 | 0 | 0 | — | — | — |
| latest | ab\_antigen/long | mask25 | 2.04127e-05 | 1 | 1 | 530 | 526 | 4 |
| latest | ab\_antigen/long | mask50 | 1.46229e-05 | 1 | 1 | 1019 | 1013 | 6 |
| latest | ab\_antigen/long | mask75 | 1.27785e-05 | 1 | 1 | 1538 | 1532 | 6 |
| latest | ab\_antigen/long | mask100 | 0.00863552 | 0.99902 | 1 | 2048 | 2040 | 8 |
| latest | ab\_antigen/long | mask\_mean | 0.00217083 | 0.999755 | 1 | — | — | — |
| latest | ab\_antigen/medium | mask25 | 0.000152184 | 1 | 1 | 246 | 243 | 3 |
| latest | ab\_antigen/medium | mask50 | 7.24102e-05 | 1 | 1 | 507 | 502 | 5 |
| latest | ab\_antigen/medium | mask75 | 6.35791e-05 | 1 | 1 | 765 | 758 | 7 |
| latest | ab\_antigen/medium | mask100 | 0.00610353 | 1 | 1 | 1024 | 1016 | 8 |
| latest | ab\_antigen/medium | mask\_mean | 0.00159792 | 1 | 1 | — | — | — |
| latest | ab\_antigen/short | mask25 | 4.65991e-05 | 1 | 1 | 38 | 34 | 4 |
| latest | ab\_antigen/short | mask50 | 4.84554e-05 | 1 | 1 | 93 | 88 | 5 |
| latest | ab\_antigen/short | mask75 | 0.00079628 | 1 | 1 | 145 | 140 | 5 |
| latest | ab\_antigen/short | mask100 | 0.0126079 | 0.994792 | 1 | 200 | 192 | 8 |
| latest | ab\_antigen/short | mask\_mean | 0.00337481 | 0.998698 | 1 | — | — | — |
| latest | tcr\_peptide/long | mask25 | 5.5916e-06 | 1 | 1 | 466 | 465 | 1 |
| latest | tcr\_peptide/long | mask50 | 5.89662e-06 | 1 | 1 | 985 | 981 | 4 |
| latest | tcr\_peptide/long | mask75 | 5.56004e-06 | 1 | 1 | 1484 | 1478 | 6 |
| latest | tcr\_peptide/long | mask100 | 0.0499596 | 0.978431 | 1 | 2048 | 2040 | 8 |
| latest | tcr\_peptide/long | mask\_mean | 0.0124942 | 0.994608 | 1 | — | — | — |
| latest | tcr\_peptide/medium | mask25 | 2.34936e-05 | 1 | 1 | 242 | 241 | 1 |
| latest | tcr\_peptide/medium | mask50 | 8.35905e-06 | 1 | 1 | 506 | 502 | 4 |
| latest | tcr\_peptide/medium | mask75 | 0.00122206 | 1 | 1 | 767 | 760 | 7 |
| latest | tcr\_peptide/medium | mask100 | 0.0563622 | 0.974409 | 1 | 1024 | 1016 | 8 |
| latest | tcr\_peptide/medium | mask\_mean | 0.014404 | 0.993602 | 1 | — | — | — |
| latest | tcr\_peptide/short | mask25 | 1.44256e-05 | 1 | 1 | 51 | 48 | 3 |
| latest | tcr\_peptide/short | mask50 | 4.22886e-05 | 1 | 1 | 110 | 104 | 6 |
| latest | tcr\_peptide/short | mask75 | 0.000355353 | 1 | 1 | 144 | 137 | 7 |
| latest | tcr\_peptide/short | mask100 | 0.0203488 | 0.984375 | 1 | 200 | 192 | 8 |
| latest | tcr\_peptide/short | mask\_mean | 0.00519022 | 0.996094 | 1 | — | — | — |
| latest | tcr\_pmhc/long | mask25 | 3.9162e-05 | 1 | 1 | 540 | 536 | 4 |
| latest | tcr\_pmhc/long | mask50 | 1.10865e-05 | 1 | 1 | 1038 | 1034 | 4 |
| latest | tcr\_pmhc/long | mask75 | 2.26526e-05 | 1 | 1 | 1562 | 1554 | 8 |
| latest | tcr\_pmhc/long | mask100 | 0.00757439 | 0.99951 | 1 | 2048 | 2040 | 8 |
| latest | tcr\_pmhc/long | mask\_mean | 0.00191182 | 0.999877 | 1 | — | — | — |
| latest | tcr\_pmhc/medium | mask25 | 9.37061e-06 | 1 | 1 | 244 | 242 | 2 |
| latest | tcr\_pmhc/medium | mask50 | 8.94077e-06 | 1 | 1 | 484 | 479 | 5 |
| latest | tcr\_pmhc/medium | mask75 | 1.26907e-05 | 1 | 1 | 777 | 769 | 8 |
| latest | tcr\_pmhc/medium | mask100 | 0.0990336 | 0.951772 | 1 | 1024 | 1016 | 8 |
| latest | tcr\_pmhc/medium | mask\_mean | 0.0247661 | 0.987943 | 1 | — | — | — |
| latest | tcr\_pmhc/short | mask25 | 2.61054e-05 | 1 | 1 | 51 | 48 | 3 |
| latest | tcr\_pmhc/short | mask50 | 2.97403e-05 | 1 | 1 | 86 | 83 | 3 |
| latest | tcr\_pmhc/short | mask75 | 0.00721494 | 1 | 1 | 152 | 147 | 5 |
| latest | tcr\_pmhc/short | mask100 | 0.00636936 | 1 | 1 | 200 | 192 | 8 |
| latest | tcr\_pmhc/short | mask\_mean | 0.00341004 | 1 | 1 | — | — | — |
| latest | unconditional\_ab/long | mask25 | 2.16255e-05 | 1 | 1 | 538 | 535 | 3 |
| latest | unconditional\_ab/long | mask50 | 1.69146e-05 | 1 | 1 | 1050 | 1046 | 4 |
| latest | unconditional\_ab/long | mask75 | 1.73259e-05 | 1 | 1 | 1545 | 1538 | 7 |
| latest | unconditional\_ab/long | mask100 | 1.63553 | 0.336765 | 1 | 2048 | 2040 | 8 |
| latest | unconditional\_ab/long | mask\_mean | 0.408897 | 0.834191 | 1 | — | — | — |
| latest | unconditional\_ab/medium | mask25 | 3.42507e-05 | 1 | 1 | 256 | 253 | 3 |
| latest | unconditional\_ab/medium | mask50 | 5.61449e-05 | 1 | 1 | 512 | 508 | 4 |
| latest | unconditional\_ab/medium | mask75 | 9.75655e-05 | 1 | 1 | 768 | 762 | 6 |
| latest | unconditional\_ab/medium | mask100 | 1.54252 | 0.40748 | 1 | 1024 | 1016 | 8 |
| latest | unconditional\_ab/medium | mask\_mean | 0.385677 | 0.85187 | 1 | — | — | — |
| latest | unconditional\_ab/short | mask25 | 3.23113e-05 | 1 | 1 | 55 | 52 | 3 |
| latest | unconditional\_ab/short | mask50 | 0.155255 | 0.938775 | 1 | 103 | 98 | 5 |
| latest | unconditional\_ab/short | mask75 | 0.451808 | 0.822695 | 1 | 148 | 141 | 7 |
| latest | unconditional\_ab/short | mask100 | 1.31944 | 0.359375 | 1 | 200 | 192 | 8 |
| latest | unconditional\_ab/short | mask\_mean | 0.481634 | 0.780211 | 1 | — | — | — |
| latest | unconditional\_single/long | mask25 | 5.95829e-05 | 1 | 1 | 269 | 268 | 1 |
| latest | unconditional\_single/long | mask50 | 2.58782e-05 | 1 | 1 | 527 | 525 | 2 |
| latest | unconditional\_single/long | mask75 | 0.201296 | 0.958656 | 1 | 778 | 774 | 4 |
| latest | unconditional\_single/long | mask100 | 1.54127 | 0.354902 | 1 | 1024 | 1020 | 4 |
| latest | unconditional\_single/long | mask\_mean | 0.435662 | 0.82839 | 1 | — | — | — |
| latest | unconditional\_single/medium | mask25 | 0.000538209 | 1 | 1 | 110 | 109 | 1 |
| latest | unconditional\_single/medium | mask50 | 0.000734989 | 1 | 1 | 234 | 232 | 2 |
| latest | unconditional\_single/medium | mask75 | 0.00731111 | 1 | 1 | 371 | 367 | 4 |
| latest | unconditional\_single/medium | mask100 | 1.71721 | 0.332677 | 1 | 512 | 508 | 4 |
| latest | unconditional\_single/medium | mask\_mean | 0.431448 | 0.833169 | 1 | — | — | — |
| latest | unconditional\_single/short | mask25 | 0.00121645 | 1 | 1 | 28 | 26 | 2 |
| latest | unconditional\_single/short | mask50 | 0.343904 | 0.82 | 1 | 52 | 50 | 2 |
| latest | unconditional\_single/short | mask75 | 0.948213 | 0.61039 | 1 | 80 | 77 | 3 |
| latest | unconditional\_single/short | mask100 | 1.23574 | 0.40625 | 1 | 100 | 96 | 4 |
| latest | unconditional\_single/short | mask\_mean | 0.632268 | 0.70916 | 1 | — | — | — |
| latest | unconditional\_tcr/long | mask25 | 7.06818e-06 | 1 | 1 | 516 | 515 | 1 |
| latest | unconditional\_tcr/long | mask50 | 5.46938e-06 | 1 | 1 | 989 | 987 | 2 |
| latest | unconditional\_tcr/long | mask75 | 5.45412e-05 | 1 | 1 | 1527 | 1522 | 5 |
| latest | unconditional\_tcr/long | mask100 | 1.55745 | 0.379412 | 1 | 2048 | 2040 | 8 |
| latest | unconditional\_tcr/long | mask\_mean | 0.38938 | 0.844853 | 1 | — | — | — |
| latest | unconditional\_tcr/medium | mask25 | 9.76059e-06 | 1 | — | 247 | 247 | 0 |
| latest | unconditional\_tcr/medium | mask50 | 4.82686e-06 | 1 | 1 | 500 | 497 | 3 |
| latest | unconditional\_tcr/medium | mask75 | 2.2865e-05 | 1 | 1 | 744 | 738 | 6 |
| latest | unconditional\_tcr/medium | mask100 | 1.60192 | 0.348425 | 1 | 1024 | 1016 | 8 |
| latest | unconditional\_tcr/medium | mask\_mean | 0.400489 | 0.837106 | 1 | — | — | — |
| latest | unconditional\_tcr/short | mask25 | 7.528e-06 | 1 | — | 40 | 40 | 0 |
| latest | unconditional\_tcr/short | mask50 | 0.00295368 | 1 | 1 | 93 | 90 | 3 |
| latest | unconditional\_tcr/short | mask75 | 0.284097 | 0.926471 | 1 | 142 | 136 | 6 |
| latest | unconditional\_tcr/short | mask100 | 1.40252 | 0.34375 | 0.5 | 200 | 192 | 8 |
| latest | unconditional\_tcr/short | mask\_mean | 0.422394 | 0.817555 | 0.833333 | — | — | — |
| train\_probe | ab\_antigen/long | mask25 | 1.9888e-05 | 1 | — | 530 | 530 | 0 |
| train\_probe | ab\_antigen/long | mask50 | 2.18978e-05 | 1 | 1 | 1034 | 1031 | 3 |
| train\_probe | ab\_antigen/long | mask75 | 8.2782e-06 | 1 | 1 | 1563 | 1557 | 6 |
| train\_probe | ab\_antigen/long | mask100 | 0.00364158 | 1 | 1 | 2048 | 2040 | 8 |
| train\_probe | ab\_antigen/long | mask\_mean | 0.00092291 | 1 | 1 | — | — | — |
| train\_probe | ab\_antigen/medium | mask25 | 2.22317e-05 | 1 | 1 | 275 | 273 | 2 |
| train\_probe | ab\_antigen/medium | mask50 | 2.66025e-05 | 1 | 1 | 542 | 538 | 4 |
| train\_probe | ab\_antigen/medium | mask75 | 2.13486e-05 | 1 | 1 | 781 | 776 | 5 |
| train\_probe | ab\_antigen/medium | mask100 | 0.00850689 | 0.998031 | 1 | 1024 | 1016 | 8 |
| train\_probe | ab\_antigen/medium | mask\_mean | 0.00214427 | 0.999508 | 1 | — | — | — |
| train\_probe | ab\_antigen/short | mask25 | 8.73282e-05 | 1 | 1 | 52 | 47 | 5 |
| train\_probe | ab\_antigen/short | mask50 | 7.96247e-05 | 1 | 1 | 109 | 102 | 7 |
| train\_probe | ab\_antigen/short | mask75 | 0.00174809 | 1 | 1 | 155 | 148 | 7 |
| train\_probe | ab\_antigen/short | mask100 | 0.00323528 | 1 | 1 | 200 | 192 | 8 |
| train\_probe | ab\_antigen/short | mask\_mean | 0.00128758 | 1 | 1 | — | — | — |
| train\_probe | tcr\_peptide/long | mask25 | 7.18884e-06 | 1 | 1 | 495 | 494 | 1 |
| train\_probe | tcr\_peptide/long | mask50 | 0.000223025 | 1 | 1 | 1033 | 1028 | 5 |
| train\_probe | tcr\_peptide/long | mask75 | 0.000100277 | 1 | 1 | 1530 | 1522 | 8 |
| train\_probe | tcr\_peptide/long | mask100 | 0.0145056 | 0.99902 | 1 | 2048 | 2040 | 8 |
| train\_probe | tcr\_peptide/long | mask\_mean | 0.00370903 | 0.999755 | 1 | — | — | — |
| train\_probe | tcr\_peptide/medium | mask25 | 5.90446e-06 | 1 | — | 252 | 252 | 0 |
| train\_probe | tcr\_peptide/medium | mask50 | 5.68345e-06 | 1 | 1 | 517 | 513 | 4 |
| train\_probe | tcr\_peptide/medium | mask75 | 8.30905e-06 | 1 | 1 | 760 | 754 | 6 |
| train\_probe | tcr\_peptide/medium | mask100 | 0.0714821 | 0.974409 | 1 | 1024 | 1016 | 8 |
| train\_probe | tcr\_peptide/medium | mask\_mean | 0.0178755 | 0.993602 | 1 | — | — | — |
| train\_probe | tcr\_peptide/short | mask25 | 4.9497e-05 | 1 | 1 | 51 | 46 | 5 |
| train\_probe | tcr\_peptide/short | mask50 | 7.87713e-05 | 1 | 1 | 95 | 90 | 5 |
| train\_probe | tcr\_peptide/short | mask75 | 0.00333308 | 1 | 1 | 142 | 136 | 6 |
| train\_probe | tcr\_peptide/short | mask100 | 0.0404183 | 0.973958 | 1 | 200 | 192 | 8 |
| train\_probe | tcr\_peptide/short | mask\_mean | 0.0109699 | 0.99349 | 1 | — | — | — |
| train\_probe | tcr\_pmhc/long | mask25 | 4.80952e-06 | 1 | 1 | 526 | 522 | 4 |
| train\_probe | tcr\_pmhc/long | mask50 | 3.59201e-06 | 1 | 1 | 1036 | 1031 | 5 |
| train\_probe | tcr\_pmhc/long | mask75 | 7.05631e-06 | 1 | 1 | 1557 | 1551 | 6 |
| train\_probe | tcr\_pmhc/long | mask100 | 0.0128787 | 0.994608 | 1 | 2048 | 2040 | 8 |
| train\_probe | tcr\_pmhc/long | mask\_mean | 0.00322353 | 0.998652 | 1 | — | — | — |
| train\_probe | tcr\_pmhc/medium | mask25 | 8.66364e-06 | 1 | 1 | 276 | 273 | 3 |
| train\_probe | tcr\_pmhc/medium | mask50 | 7.60232e-06 | 1 | 1 | 549 | 541 | 8 |
| train\_probe | tcr\_pmhc/medium | mask75 | 1.94605e-05 | 1 | 1 | 793 | 785 | 8 |
| train\_probe | tcr\_pmhc/medium | mask100 | 0.00400324 | 1 | 1 | 1024 | 1016 | 8 |
| train\_probe | tcr\_pmhc/medium | mask\_mean | 0.00100974 | 1 | 1 | — | — | — |
| train\_probe | tcr\_pmhc/short | mask25 | 2.78434e-05 | 1 | 1 | 50 | 48 | 2 |
| train\_probe | tcr\_pmhc/short | mask50 | 2.13186e-05 | 1 | 1 | 103 | 100 | 3 |
| train\_probe | tcr\_pmhc/short | mask75 | 9.00028e-05 | 1 | 1 | 152 | 145 | 7 |
| train\_probe | tcr\_pmhc/short | mask100 | 0.00455086 | 1 | 1 | 200 | 192 | 8 |
| train\_probe | tcr\_pmhc/short | mask\_mean | 0.00117251 | 1 | 1 | — | — | — |
| train\_probe | unconditional\_ab/long | mask25 | 4.88744e-05 | 1 | 1 | 509 | 506 | 3 |
| train\_probe | unconditional\_ab/long | mask50 | 1.20161e-05 | 1 | 1 | 1016 | 1010 | 6 |
| train\_probe | unconditional\_ab/long | mask75 | 1.17512e-05 | 1 | 1 | 1569 | 1561 | 8 |
| train\_probe | unconditional\_ab/long | mask100 | 1.39354 | 0.401471 | 1 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_ab/long | mask\_mean | 0.348404 | 0.850368 | 1 | — | — | — |
| train\_probe | unconditional\_ab/medium | mask25 | 2.11766e-05 | 1 | 1 | 243 | 241 | 2 |
| train\_probe | unconditional\_ab/medium | mask50 | 2.28558e-05 | 1 | 1 | 521 | 516 | 5 |
| train\_probe | unconditional\_ab/medium | mask75 | 4.74702e-05 | 1 | 1 | 766 | 761 | 5 |
| train\_probe | unconditional\_ab/medium | mask100 | 1.57468 | 0.383858 | 1 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_ab/medium | mask\_mean | 0.393694 | 0.845965 | 1 | — | — | — |
| train\_probe | unconditional\_ab/short | mask25 | 1.84138e-05 | 1 | 1 | 53 | 52 | 1 |
| train\_probe | unconditional\_ab/short | mask50 | 0.198432 | 0.942308 | 1 | 106 | 104 | 2 |
| train\_probe | unconditional\_ab/short | mask75 | 0.544621 | 0.778481 | 1 | 163 | 158 | 5 |
| train\_probe | unconditional\_ab/short | mask100 | 1.49194 | 0.328125 | 1 | 200 | 192 | 8 |
| train\_probe | unconditional\_ab/short | mask\_mean | 0.558753 | 0.762228 | 1 | — | — | — |
| train\_probe | unconditional\_single/long | mask25 | 7.76183e-05 | 1 | 1 | 245 | 244 | 1 |
| train\_probe | unconditional\_single/long | mask50 | 3.76172e-05 | 1 | 1 | 492 | 491 | 1 |
| train\_probe | unconditional\_single/long | mask75 | 0.00598125 | 1 | 1 | 771 | 768 | 3 |
| train\_probe | unconditional\_single/long | mask100 | 1.56907 | 0.323529 | 1 | 1024 | 1020 | 4 |
| train\_probe | unconditional\_single/long | mask\_mean | 0.393792 | 0.830882 | 1 | — | — | — |
| train\_probe | unconditional\_single/medium | mask25 | 5.35678e-05 | 1 | 1 | 132 | 130 | 2 |
| train\_probe | unconditional\_single/medium | mask50 | 4.29311e-05 | 1 | 1 | 255 | 252 | 3 |
| train\_probe | unconditional\_single/medium | mask75 | 4.83962e-05 | 1 | 1 | 382 | 379 | 3 |
| train\_probe | unconditional\_single/medium | mask100 | 1.45195 | 0.360236 | 1 | 512 | 508 | 4 |
| train\_probe | unconditional\_single/medium | mask\_mean | 0.363023 | 0.840059 | 1 | — | — | — |
| train\_probe | unconditional\_single/short | mask25 | 0.00704578 | 1 | 1 | 28 | 26 | 2 |
| train\_probe | unconditional\_single/short | mask50 | 0.281963 | 0.938775 | 1 | 52 | 49 | 3 |
| train\_probe | unconditional\_single/short | mask75 | 0.861704 | 0.642857 | 1 | 73 | 70 | 3 |
| train\_probe | unconditional\_single/short | mask100 | 1.24246 | 0.46875 | 1 | 100 | 96 | 4 |
| train\_probe | unconditional\_single/short | mask\_mean | 0.598293 | 0.762596 | 1 | — | — | — |
| train\_probe | unconditional\_tcr/long | mask25 | 7.01307e-06 | 1 | — | 525 | 525 | 0 |
| train\_probe | unconditional\_tcr/long | mask50 | 4.62343e-06 | 1 | 1 | 1048 | 1045 | 3 |
| train\_probe | unconditional\_tcr/long | mask75 | 2.85168e-06 | 1 | 1 | 1585 | 1580 | 5 |
| train\_probe | unconditional\_tcr/long | mask100 | 1.50548 | 0.404902 | 1 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_tcr/long | mask\_mean | 0.376374 | 0.851225 | 1 | — | — | — |
| train\_probe | unconditional\_tcr/medium | mask25 | 3.56026e-06 | 1 | — | 261 | 261 | 0 |
| train\_probe | unconditional\_tcr/medium | mask50 | 1.79476e-05 | 1 | 1 | 498 | 497 | 1 |
| train\_probe | unconditional\_tcr/medium | mask75 | 5.40559e-05 | 1 | 1 | 755 | 751 | 4 |
| train\_probe | unconditional\_tcr/medium | mask100 | 1.53606 | 0.40748 | 1 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_tcr/medium | mask\_mean | 0.384034 | 0.85187 | 1 | — | — | — |
| train\_probe | unconditional\_tcr/short | mask25 | 1.84871e-05 | 1 | — | 46 | 46 | 0 |
| train\_probe | unconditional\_tcr/short | mask50 | 0.282938 | 0.880435 | 1 | 97 | 92 | 5 |
| train\_probe | unconditional\_tcr/short | mask75 | 0.418031 | 0.787671 | 1 | 152 | 146 | 6 |
| train\_probe | unconditional\_tcr/short | mask100 | 1.40533 | 0.28125 | 0.5 | 200 | 192 | 8 |
| train\_probe | unconditional\_tcr/short | mask\_mean | 0.526579 | 0.737339 | 0.833333 | — | — | — |

### 1790803305061375040\_r24\_refined\_fixed\_scale\_4edeef26：分层 mask 指标

源码目录：/vepfs-mlp2/c20250601/251105016/project/dllm\_test/output/immune\_research\_releases/r20\_r27\_20261001；完成/目标 step：500/500。

kind=synthetic\_generalization\_diagnostic；dataset_kind=deterministic\_artificial\_sequences；biological_quality_claim=False。

| 时点 | mask level | 指标 | 值 |
| --- | --- | --- | --- |
| initial | mask25 | mask\_ratio | 0.25 |
| initial | mask25 | selected\_count | 4467 |
| initial | mask25 | aa\_count | 4428 |
| initial | mask25 | terminal\_count | 39 |
| initial | mask25 | loss | 3.81518 |
| initial | mask25 | aa\_loss | 3.8107 |
| initial | mask25 | aa\_accuracy | 0.0589431 |
| initial | mask25 | terminal\_loss | 4.32412 |
| initial | mask25 | terminal\_accuracy | 0 |
| initial | mask50 | mask\_ratio | 0.5 |
| initial | mask50 | selected\_count | 8888 |
| initial | mask50 | aa\_count | 8819 |
| initial | mask50 | terminal\_count | 69 |
| initial | mask50 | loss | 3.85226 |
| initial | mask50 | aa\_loss | 3.8493 |
| initial | mask50 | aa\_accuracy | 0.0426352 |
| initial | mask50 | terminal\_loss | 4.23099 |
| initial | mask50 | terminal\_accuracy | 0 |
| initial | mask75 | mask\_ratio | 0.75 |
| initial | mask75 | selected\_count | 13437 |
| initial | mask75 | aa\_count | 13330 |
| initial | mask75 | terminal\_count | 107 |
| initial | mask75 | loss | 3.83517 |
| initial | mask75 | aa\_loss | 3.83171 |
| initial | mask75 | aa\_accuracy | 0.0306827 |
| initial | mask75 | terminal\_loss | 4.26675 |
| initial | mask75 | terminal\_accuracy | 0 |
| initial | mask100 | mask\_ratio | 1 |
| initial | mask100 | selected\_count | 17996 |
| initial | mask100 | aa\_count | 17864 |
| initial | mask100 | terminal\_count | 132 |
| initial | mask100 | loss | 3.80145 |
| initial | mask100 | aa\_loss | 3.79761 |
| initial | mask100 | aa\_accuracy | 0.0301164 |
| initial | mask100 | terminal\_loss | 4.32137 |
| initial | mask100 | terminal\_accuracy | 0 |
| initial | mask\_mean | loss | 3.82602 |
| initial | mask\_mean | aa\_loss | 3.82233 |
| initial | mask\_mean | terminal\_loss | 4.28581 |
| initial | mask\_mean | aa\_accuracy | 0.0405944 |
| initial | mask\_mean | terminal\_accuracy | 0 |
| latest | mask25 | mask\_ratio | 0.25 |
| latest | mask25 | selected\_count | 4467 |
| latest | mask25 | aa\_count | 4428 |
| latest | mask25 | terminal\_count | 39 |
| latest | mask25 | loss | 7.56128e-05 |
| latest | mask25 | aa\_loss | 7.27824e-05 |
| latest | mask25 | aa\_accuracy | 1 |
| latest | mask25 | terminal\_loss | 0.000396982 |
| latest | mask25 | terminal\_accuracy | 1 |
| latest | mask50 | mask\_ratio | 0.5 |
| latest | mask50 | selected\_count | 8888 |
| latest | mask50 | aa\_count | 8819 |
| latest | mask50 | terminal\_count | 69 |
| latest | mask50 | loss | 0.0038419 |
| latest | mask50 | aa\_loss | 0.00387024 |
| latest | mask50 | aa\_accuracy | 0.998413 |
| latest | mask50 | terminal\_loss | 0.00022013 |
| latest | mask50 | terminal\_accuracy | 1 |
| latest | mask75 | mask\_ratio | 0.75 |
| latest | mask75 | selected\_count | 13437 |
| latest | mask75 | aa\_count | 13330 |
| latest | mask75 | terminal\_count | 107 |
| latest | mask75 | loss | 0.021001 |
| latest | mask75 | aa\_loss | 0.0211673 |
| latest | mask75 | aa\_accuracy | 0.993023 |
| latest | mask75 | terminal\_loss | 0.000283086 |
| latest | mask75 | terminal\_accuracy | 1 |
| latest | mask100 | mask\_ratio | 1 |
| latest | mask100 | selected\_count | 17996 |
| latest | mask100 | aa\_count | 17864 |
| latest | mask100 | terminal\_count | 132 |
| latest | mask100 | loss | 0.738684 |
| latest | mask100 | aa\_loss | 0.744116 |
| latest | mask100 | aa\_accuracy | 0.702922 |
| latest | mask100 | terminal\_loss | 0.00353359 |
| latest | mask100 | terminal\_accuracy | 1 |
| latest | mask\_mean | loss | 0.190901 |
| latest | mask\_mean | aa\_loss | 0.192307 |
| latest | mask\_mean | terminal\_loss | 0.00110845 |
| latest | mask\_mean | aa\_accuracy | 0.923589 |
| latest | mask\_mean | terminal\_accuracy | 1 |

评测曲线（原始 step，四档等权均值）：

| step | mean loss | mean AA acc | mean end acc | gradient norm |
| --- | --- | --- | --- | --- |
| 25 | 1.84536 | 0.321042 | 0 | 1.5845 |
| 50 | 1.2077 | 0.591339 | 0 | 3.71078 |
| 75 | 0.649878 | 0.796136 | 0.0136566 | 3.37973 |
| 100 | 0.539848 | 0.815866 | 0.627291 | 3.36578 |
| 125 | 0.4764 | 0.831241 | 0.973207 | 3.25106 |
| 150 | 0.440977 | 0.852544 | 0.964015 | 1.15607 |
| 175 | 0.40024 | 0.857636 | 0.980176 | 1.57194 |
| 200 | 0.442032 | 0.852406 | 0.955997 | 1.23229 |
| 225 | 0.369909 | 0.869585 | 0.959785 | 0.890233 |
| 250 | 0.27003 | 0.904256 | 0.965909 | 0.568307 |
| 275 | 0.280005 | 0.898566 | 0.952209 | 0.830804 |
| 300 | 0.24875 | 0.902935 | 0.967361 | 0.314503 |
| 325 | 0.215316 | 0.918701 | 0.977273 | 0.587494 |
| 350 | 0.200461 | 0.923008 | 1 | 0.541464 |
| 375 | 0.219239 | 0.915951 | 1 | 0.321179 |
| 400 | 0.191695 | 0.924196 | 1 | 0.611299 |
| 425 | 0.188222 | 0.923661 | 1 | 0.470993 |
| 450 | 0.187089 | 0.925309 | 1 | 0.311281 |
| 475 | 0.178409 | 0.923802 | 0.99053 | 0.206338 |
| 500 | 0.190901 | 0.923589 | 1 | 0.330913 |

有条件/无条件曲线（每档按token计数合并，再平均四档；缺失分组不推断数值）：

| step | 子集 | mean loss | mean AA | mean end | Full-mask AA | Full-mask end | Full-mask AA count | Full-mask end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | conditional | 3.79117 | 0.0444692 | 0 | 0.0298645 | 0 | 9744 | 72 |
| initial | unconditional | 3.86732 | 0.0360518 | 0 | 0.0304187 | 0 | 8120 | 60 |
| 25 | conditional | 1.79789 | 0.31892 | 0 | 0.310242 | 0 | 9744 | 72 |
| 25 | unconditional | 1.90239 | 0.323527 | 0 | 0.314901 | 0 | 8120 | 60 |
| 50 | conditional | 1.11528 | 0.612396 | 0 | 0.307677 | 0 | 9744 | 72 |
| 50 | unconditional | 1.31833 | 0.566385 | 0 | 0.314901 | 0 | 8120 | 60 |
| 75 | conditional | 0.605427 | 0.795763 | 0.0219048 | 0.289409 | 0 | 9744 | 72 |
| 75 | unconditional | 0.703184 | 0.796574 | 0 | 0.313424 | 0 | 8120 | 60 |
| 100 | conditional | 0.461428 | 0.827964 | 0.667922 | 0.360632 | 0.305556 | 9744 | 72 |
| 100 | unconditional | 0.633909 | 0.801366 | 0.576025 | 0.272783 | 0.0666667 | 8120 | 60 |
| 125 | conditional | 0.421587 | 0.849522 | 0.989583 | 0.41759 | 0.958333 | 9744 | 72 |
| 125 | unconditional | 0.54216 | 0.809314 | 0.952199 | 0.272167 | 0.866667 | 8120 | 60 |
| 150 | conditional | 0.420012 | 0.867475 | 0.975694 | 0.48204 | 0.902778 | 9744 | 72 |
| 150 | unconditional | 0.466128 | 0.83463 | 0.95 | 0.364039 | 0.8 | 8120 | 60 |
| 175 | conditional | 0.331617 | 0.886035 | 1 | 0.549774 | 1 | 9744 | 72 |
| 175 | unconditional | 0.482583 | 0.823565 | 0.95625 | 0.32069 | 0.866667 | 8120 | 60 |
| 200 | conditional | 0.387634 | 0.873746 | 0.993056 | 0.507184 | 0.972222 | 9744 | 72 |
| 200 | unconditional | 0.507304 | 0.8268 | 0.911458 | 0.331897 | 0.666667 | 8120 | 60 |
| 225 | conditional | 0.273415 | 0.905844 | 1 | 0.625718 | 1 | 9744 | 72 |
| 225 | unconditional | 0.48569 | 0.826082 | 0.911458 | 0.32968 | 0.666667 | 8120 | 60 |
| 250 | conditional | 0.163925 | 0.946808 | 0.993056 | 0.792898 | 0.972222 | 9744 | 72 |
| 250 | unconditional | 0.397357 | 0.853192 | 0.933333 | 0.433005 | 0.733333 | 8120 | 60 |
| 275 | conditional | 0.114657 | 0.959366 | 1 | 0.838362 | 1 | 9744 | 72 |
| 275 | unconditional | 0.478418 | 0.825611 | 0.894792 | 0.326478 | 0.6 | 8120 | 60 |
| 300 | conditional | 0.117151 | 0.957825 | 1 | 0.834257 | 1 | 9744 | 72 |
| 300 | unconditional | 0.406663 | 0.837069 | 0.928125 | 0.362069 | 0.733333 | 8120 | 60 |
| 325 | conditional | 0.0429493 | 0.986308 | 1 | 0.945505 | 1 | 9744 | 72 |
| 325 | unconditional | 0.422131 | 0.837584 | 0.95 | 0.373276 | 0.8 | 8120 | 60 |
| 350 | conditional | 0.0235831 | 0.995424 | 1 | 0.981835 | 1 | 9744 | 72 |
| 350 | unconditional | 0.412708 | 0.836113 | 1 | 0.366256 | 1 | 8120 | 60 |
| 375 | conditional | 0.0639454 | 0.978628 | 1 | 0.914512 | 1 | 9744 | 72 |
| 375 | unconditional | 0.405588 | 0.840742 | 1 | 0.38436 | 1 | 8120 | 60 |
| 400 | conditional | 0.0197752 | 0.994303 | 1 | 0.977422 | 1 | 9744 | 72 |
| 400 | unconditional | 0.397997 | 0.840067 | 1 | 0.380172 | 1 | 8120 | 60 |
| 425 | conditional | 0.0109088 | 0.996819 | 1 | 0.987274 | 1 | 9744 | 72 |
| 425 | unconditional | 0.400995 | 0.835874 | 1 | 0.367611 | 1 | 8120 | 60 |
| 450 | conditional | 0.00277616 | 0.999546 | 1 | 0.998461 | 1 | 9744 | 72 |
| 450 | unconditional | 0.40826 | 0.836227 | 1 | 0.366133 | 1 | 8120 | 60 |
| 475 | conditional | 0.00607078 | 0.998101 | 0.996528 | 0.992406 | 0.986111 | 9744 | 72 |
| 475 | unconditional | 0.385209 | 0.834646 | 0.983333 | 0.353202 | 0.933333 | 8120 | 60 |
| 500 | conditional | 0.00487879 | 0.998546 | 1 | 0.994458 | 1 | 9744 | 72 |
| 500 | unconditional | 0.414124 | 0.833645 | 1 | 0.353079 | 1 | 8120 | 60 |

人工数据 manifest：d40bbbbc1c4294eed843d88bcdd0cf193886de6c2d7e9db9069bbf86e931c022；精确 target sequence 重叠=0；精确 record 重叠=0。

跨 radius32 覆盖：{&quot;possible&quot;: true, &quot;train&quot;: true, &quot;valid&quot;: true}。生成规则共享是诊断设计的一部分，不能当作真实下游去重证书。


按任务/链长分组（全局 level 按 token 计数汇总，下面保留各组计数；不同组不得直接比较未经加权的总 loss）：

| 时点 | group/length | mask level | loss | AA acc | end acc | selected count | AA count | end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | ab\_antigen/long | mask25 | 4.10241 | 0 | 0 | 530 | 526 | 4 |
| initial | ab\_antigen/long | mask50 | 4.14621 | 0 | 0 | 1019 | 1013 | 6 |
| initial | ab\_antigen/long | mask75 | 4.10335 | 0 | 0 | 1538 | 1532 | 6 |
| initial | ab\_antigen/long | mask100 | 4.03516 | 0 | 0 | 2048 | 2040 | 8 |
| initial | ab\_antigen/long | mask\_mean | 4.09678 | 0 | 0 | — | — | — |
| initial | ab\_antigen/medium | mask25 | 4.1109 | 0 | 0 | 246 | 243 | 3 |
| initial | ab\_antigen/medium | mask50 | 4.0947 | 0 | 0 | 507 | 502 | 5 |
| initial | ab\_antigen/medium | mask75 | 4.07023 | 0 | 0 | 765 | 758 | 7 |
| initial | ab\_antigen/medium | mask100 | 4.03235 | 0 | 0 | 1024 | 1016 | 8 |
| initial | ab\_antigen/medium | mask\_mean | 4.07704 | 0 | 0 | — | — | — |
| initial | ab\_antigen/short | mask25 | 3.94668 | 0.0588235 | 0 | 38 | 34 | 4 |
| initial | ab\_antigen/short | mask50 | 3.8665 | 0.0454545 | 0 | 93 | 88 | 5 |
| initial | ab\_antigen/short | mask75 | 3.78397 | 0.05 | 0 | 145 | 140 | 5 |
| initial | ab\_antigen/short | mask100 | 3.7871 | 0.046875 | 0 | 200 | 192 | 8 |
| initial | ab\_antigen/short | mask\_mean | 3.84606 | 0.0502883 | 0 | — | — | — |
| initial | tcr\_peptide/long | mask25 | 3.70198 | 0.0258065 | 0 | 466 | 465 | 1 |
| initial | tcr\_peptide/long | mask50 | 3.77676 | 0.011213 | 0 | 985 | 981 | 4 |
| initial | tcr\_peptide/long | mask75 | 3.77699 | 0.0121786 | 0 | 1484 | 1478 | 6 |
| initial | tcr\_peptide/long | mask100 | 3.76535 | 0.0151961 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_peptide/long | mask\_mean | 3.75527 | 0.0160985 | 0 | — | — | — |
| initial | tcr\_peptide/medium | mask25 | 3.97216 | 0.00829876 | 0 | 242 | 241 | 1 |
| initial | tcr\_peptide/medium | mask50 | 3.90908 | 0.00996016 | 0 | 506 | 502 | 4 |
| initial | tcr\_peptide/medium | mask75 | 3.85215 | 0.0118421 | 0 | 767 | 760 | 7 |
| initial | tcr\_peptide/medium | mask100 | 3.78247 | 0.0147638 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_peptide/medium | mask\_mean | 3.87897 | 0.0112162 | 0 | — | — | — |
| initial | tcr\_peptide/short | mask25 | 3.64878 | 0.0625 | 0 | 51 | 48 | 3 |
| initial | tcr\_peptide/short | mask50 | 3.61911 | 0.0865385 | 0 | 110 | 104 | 6 |
| initial | tcr\_peptide/short | mask75 | 3.66048 | 0.0656934 | 0 | 144 | 137 | 7 |
| initial | tcr\_peptide/short | mask100 | 3.65702 | 0.046875 | 0 | 200 | 192 | 8 |
| initial | tcr\_peptide/short | mask\_mean | 3.64635 | 0.0654017 | 0 | — | — | — |
| initial | tcr\_pmhc/long | mask25 | 3.19962 | 0.266791 | 0 | 540 | 536 | 4 |
| initial | tcr\_pmhc/long | mask50 | 3.30848 | 0.185687 | 0 | 1038 | 1034 | 4 |
| initial | tcr\_pmhc/long | mask75 | 3.37315 | 0.117761 | 0 | 1562 | 1554 | 8 |
| initial | tcr\_pmhc/long | mask100 | 3.41892 | 0.109804 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_pmhc/long | mask\_mean | 3.32504 | 0.170011 | 0 | — | — | — |
| initial | tcr\_pmhc/medium | mask25 | 3.82185 | 0.0165289 | 0 | 244 | 242 | 2 |
| initial | tcr\_pmhc/medium | mask50 | 3.84317 | 0.00626305 | 0 | 484 | 479 | 5 |
| initial | tcr\_pmhc/medium | mask75 | 3.85431 | 0 | 0 | 777 | 769 | 8 |
| initial | tcr\_pmhc/medium | mask100 | 3.8816 | 0 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_pmhc/medium | mask\_mean | 3.85023 | 0.00569799 | 0 | — | — | — |
| initial | tcr\_pmhc/short | mask25 | 3.76194 | 0 | 0 | 51 | 48 | 3 |
| initial | tcr\_pmhc/short | mask50 | 3.81972 | 0.0120482 | 0 | 86 | 83 | 3 |
| initial | tcr\_pmhc/short | mask75 | 3.80688 | 0.0136054 | 0 | 152 | 147 | 5 |
| initial | tcr\_pmhc/short | mask100 | 3.83313 | 0.015625 | 0 | 200 | 192 | 8 |
| initial | tcr\_pmhc/short | mask\_mean | 3.80542 | 0.0103197 | 0 | — | — | — |
| initial | unconditional\_ab/long | mask25 | 3.9701 | 0.0616822 | 0 | 538 | 535 | 3 |
| initial | unconditional\_ab/long | mask50 | 3.92095 | 0.0602294 | 0 | 1050 | 1046 | 4 |
| initial | unconditional\_ab/long | mask75 | 3.88596 | 0.0318596 | 0 | 1545 | 1538 | 7 |
| initial | unconditional\_ab/long | mask100 | 3.79481 | 0.0308824 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_ab/long | mask\_mean | 3.89296 | 0.0461634 | 0 | — | — | — |
| initial | unconditional\_ab/medium | mask25 | 3.7778 | 0.0592885 | 0 | 256 | 253 | 3 |
| initial | unconditional\_ab/medium | mask50 | 3.8424 | 0.0393701 | 0 | 512 | 508 | 4 |
| initial | unconditional\_ab/medium | mask75 | 3.8188 | 0.0354331 | 0 | 768 | 762 | 6 |
| initial | unconditional\_ab/medium | mask100 | 3.7911 | 0.0305118 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_ab/medium | mask\_mean | 3.80753 | 0.0411509 | 0 | — | — | — |
| initial | unconditional\_ab/short | mask25 | 3.85957 | 0.0576923 | 0 | 55 | 52 | 3 |
| initial | unconditional\_ab/short | mask50 | 3.84337 | 0.0306122 | 0 | 103 | 98 | 5 |
| initial | unconditional\_ab/short | mask75 | 3.80081 | 0.035461 | 0 | 148 | 141 | 7 |
| initial | unconditional\_ab/short | mask100 | 3.75868 | 0.03125 | 0 | 200 | 192 | 8 |
| initial | unconditional\_ab/short | mask\_mean | 3.8156 | 0.0387539 | 0 | — | — | — |
| initial | unconditional\_single/long | mask25 | 3.94339 | 0.0485075 | 0 | 269 | 268 | 1 |
| initial | unconditional\_single/long | mask50 | 4.00614 | 0.0514286 | 0 | 527 | 525 | 2 |
| initial | unconditional\_single/long | mask75 | 3.96455 | 0.0516796 | 0 | 778 | 774 | 4 |
| initial | unconditional\_single/long | mask100 | 3.88226 | 0.0617647 | 0 | 1024 | 1020 | 4 |
| initial | unconditional\_single/long | mask\_mean | 3.94909 | 0.0533451 | 0 | — | — | — |
| initial | unconditional\_single/medium | mask25 | 4.15354 | 0.0366972 | 0 | 110 | 109 | 1 |
| initial | unconditional\_single/medium | mask50 | 4.00297 | 0.0258621 | 0 | 234 | 232 | 2 |
| initial | unconditional\_single/medium | mask75 | 3.95779 | 0.0245232 | 0 | 371 | 367 | 4 |
| initial | unconditional\_single/medium | mask100 | 3.90136 | 0.0295276 | 0 | 512 | 508 | 4 |
| initial | unconditional\_single/medium | mask\_mean | 4.00392 | 0.0291525 | 0 | — | — | — |
| initial | unconditional\_single/short | mask25 | 3.41472 | 0.269231 | 0 | 28 | 26 | 2 |
| initial | unconditional\_single/short | mask50 | 3.5961 | 0.18 | 0 | 52 | 50 | 2 |
| initial | unconditional\_single/short | mask75 | 3.42945 | 0.207792 | 0 | 80 | 77 | 3 |
| initial | unconditional\_single/short | mask100 | 3.41298 | 0.21875 | 0 | 100 | 96 | 4 |
| initial | unconditional\_single/short | mask\_mean | 3.46331 | 0.218943 | 0 | — | — | — |
| initial | unconditional\_tcr/long | mask25 | 3.81511 | 0.0291262 | 0 | 516 | 515 | 1 |
| initial | unconditional\_tcr/long | mask50 | 3.90053 | 0.0151976 | 0 | 989 | 987 | 2 |
| initial | unconditional\_tcr/long | mask75 | 3.85577 | 0.0157687 | 0 | 1527 | 1522 | 5 |
| initial | unconditional\_tcr/long | mask100 | 3.77771 | 0.0156863 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_tcr/long | mask\_mean | 3.83728 | 0.0189447 | 0 | — | — | — |
| initial | unconditional\_tcr/medium | mask25 | 3.74311 | 0.0202429 | — | 247 | 247 | 0 |
| initial | unconditional\_tcr/medium | mask50 | 3.86175 | 0.0160966 | 0 | 500 | 497 | 3 |
| initial | unconditional\_tcr/medium | mask75 | 3.85697 | 0.0149051 | 0 | 744 | 738 | 6 |
| initial | unconditional\_tcr/medium | mask100 | 3.8777 | 0.015748 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_tcr/medium | mask\_mean | 3.83488 | 0.0167482 | 0 | — | — | — |
| initial | unconditional\_tcr/short | mask25 | 4.01228 | 0 | — | 40 | 40 | 0 |
| initial | unconditional\_tcr/short | mask50 | 3.82482 | 0 | 0 | 93 | 90 | 3 |
| initial | unconditional\_tcr/short | mask75 | 3.85293 | 0 | 0 | 142 | 136 | 6 |
| initial | unconditional\_tcr/short | mask100 | 3.86799 | 0 | 0 | 200 | 192 | 8 |
| initial | unconditional\_tcr/short | mask\_mean | 3.8895 | 0 | 0 | — | — | — |
| latest | ab\_antigen/long | mask25 | 2.71254e-05 | 1 | 1 | 530 | 526 | 4 |
| latest | ab\_antigen/long | mask50 | 3.18169e-05 | 1 | 1 | 1019 | 1013 | 6 |
| latest | ab\_antigen/long | mask75 | 6.33671e-06 | 1 | 1 | 1538 | 1532 | 6 |
| latest | ab\_antigen/long | mask100 | 0.00173181 | 1 | 1 | 2048 | 2040 | 8 |
| latest | ab\_antigen/long | mask\_mean | 0.000449272 | 1 | 1 | — | — | — |
| latest | ab\_antigen/medium | mask25 | 6.25424e-05 | 1 | 1 | 246 | 243 | 3 |
| latest | ab\_antigen/medium | mask50 | 3.07109e-05 | 1 | 1 | 507 | 502 | 5 |
| latest | ab\_antigen/medium | mask75 | 1.55668e-05 | 1 | 1 | 765 | 758 | 7 |
| latest | ab\_antigen/medium | mask100 | 0.00169534 | 1 | 1 | 1024 | 1016 | 8 |
| latest | ab\_antigen/medium | mask\_mean | 0.000451041 | 1 | 1 | — | — | — |
| latest | ab\_antigen/short | mask25 | 4.51814e-05 | 1 | 1 | 38 | 34 | 4 |
| latest | ab\_antigen/short | mask50 | 2.99765e-05 | 1 | 1 | 93 | 88 | 5 |
| latest | ab\_antigen/short | mask75 | 0.000106687 | 1 | 1 | 145 | 140 | 5 |
| latest | ab\_antigen/short | mask100 | 0.00182615 | 1 | 1 | 200 | 192 | 8 |
| latest | ab\_antigen/short | mask\_mean | 0.000501999 | 1 | 1 | — | — | — |
| latest | tcr\_peptide/long | mask25 | 6.41547e-06 | 1 | 1 | 466 | 465 | 1 |
| latest | tcr\_peptide/long | mask50 | 6.73514e-06 | 1 | 1 | 985 | 981 | 4 |
| latest | tcr\_peptide/long | mask75 | 4.62455e-06 | 1 | 1 | 1484 | 1478 | 6 |
| latest | tcr\_peptide/long | mask100 | 0.0104621 | 0.998039 | 1 | 2048 | 2040 | 8 |
| latest | tcr\_peptide/long | mask\_mean | 0.00261998 | 0.99951 | 1 | — | — | — |
| latest | tcr\_peptide/medium | mask25 | 5.54153e-06 | 1 | 1 | 242 | 241 | 1 |
| latest | tcr\_peptide/medium | mask50 | 6.37287e-06 | 1 | 1 | 506 | 502 | 4 |
| latest | tcr\_peptide/medium | mask75 | 0.000216739 | 1 | 1 | 767 | 760 | 7 |
| latest | tcr\_peptide/medium | mask100 | 0.0115391 | 0.996063 | 1 | 1024 | 1016 | 8 |
| latest | tcr\_peptide/medium | mask\_mean | 0.00294194 | 0.999016 | 1 | — | — | — |
| latest | tcr\_peptide/short | mask25 | 6.43811e-05 | 1 | 1 | 51 | 48 | 3 |
| latest | tcr\_peptide/short | mask50 | 6.79189e-05 | 1 | 1 | 110 | 104 | 6 |
| latest | tcr\_peptide/short | mask75 | 8.61917e-05 | 1 | 1 | 144 | 137 | 7 |
| latest | tcr\_peptide/short | mask100 | 0.00290994 | 1 | 1 | 200 | 192 | 8 |
| latest | tcr\_peptide/short | mask\_mean | 0.000782108 | 1 | 1 | — | — | — |
| latest | tcr\_pmhc/long | mask25 | 1.04366e-05 | 1 | 1 | 540 | 536 | 4 |
| latest | tcr\_pmhc/long | mask50 | 8.4388e-06 | 1 | 1 | 1038 | 1034 | 4 |
| latest | tcr\_pmhc/long | mask75 | 3.69999e-06 | 1 | 1 | 1562 | 1554 | 8 |
| latest | tcr\_pmhc/long | mask100 | 0.0627475 | 0.980392 | 1 | 2048 | 2040 | 8 |
| latest | tcr\_pmhc/long | mask\_mean | 0.0156925 | 0.995098 | 1 | — | — | — |
| latest | tcr\_pmhc/medium | mask25 | 9.99339e-06 | 1 | 1 | 244 | 242 | 2 |
| latest | tcr\_pmhc/medium | mask50 | 7.44738e-06 | 1 | 1 | 484 | 479 | 5 |
| latest | tcr\_pmhc/medium | mask75 | 2.12736e-05 | 1 | 1 | 777 | 769 | 8 |
| latest | tcr\_pmhc/medium | mask100 | 0.00807647 | 0.997047 | 1 | 1024 | 1016 | 8 |
| latest | tcr\_pmhc/medium | mask\_mean | 0.0020288 | 0.999262 | 1 | — | — | — |
| latest | tcr\_pmhc/short | mask25 | 0.000138923 | 1 | 1 | 51 | 48 | 3 |
| latest | tcr\_pmhc/short | mask50 | 0.000223634 | 1 | 1 | 86 | 83 | 3 |
| latest | tcr\_pmhc/short | mask75 | 0.0393459 | 0.986395 | 1 | 152 | 147 | 5 |
| latest | tcr\_pmhc/short | mask100 | 0.0327755 | 0.984375 | 1 | 200 | 192 | 8 |
| latest | tcr\_pmhc/short | mask\_mean | 0.018121 | 0.992692 | 1 | — | — | — |
| latest | unconditional\_ab/long | mask25 | 8.00109e-05 | 1 | 1 | 538 | 535 | 3 |
| latest | unconditional\_ab/long | mask50 | 1.12598e-05 | 1 | 1 | 1050 | 1046 | 4 |
| latest | unconditional\_ab/long | mask75 | 1.91801e-05 | 1 | 1 | 1545 | 1538 | 7 |
| latest | unconditional\_ab/long | mask100 | 1.59194 | 0.323529 | 1 | 2048 | 2040 | 8 |
| latest | unconditional\_ab/long | mask\_mean | 0.398013 | 0.830882 | 1 | — | — | — |
| latest | unconditional\_ab/medium | mask25 | 0.000159563 | 1 | 1 | 256 | 253 | 3 |
| latest | unconditional\_ab/medium | mask50 | 7.4319e-05 | 1 | 1 | 512 | 508 | 4 |
| latest | unconditional\_ab/medium | mask75 | 4.62248e-05 | 1 | 1 | 768 | 762 | 6 |
| latest | unconditional\_ab/medium | mask100 | 1.57996 | 0.327756 | 1 | 1024 | 1016 | 8 |
| latest | unconditional\_ab/medium | mask\_mean | 0.39506 | 0.831939 | 1 | — | — | — |
| latest | unconditional\_ab/short | mask25 | 0.0017573 | 1 | 1 | 55 | 52 | 3 |
| latest | unconditional\_ab/short | mask50 | 0.128973 | 0.94898 | 1 | 103 | 98 | 5 |
| latest | unconditional\_ab/short | mask75 | 0.424616 | 0.886525 | 1 | 148 | 141 | 7 |
| latest | unconditional\_ab/short | mask100 | 1.19532 | 0.453125 | 1 | 200 | 192 | 8 |
| latest | unconditional\_ab/short | mask\_mean | 0.437667 | 0.822157 | 1 | — | — | — |
| latest | unconditional\_single/long | mask25 | 0.000248726 | 1 | 1 | 269 | 268 | 1 |
| latest | unconditional\_single/long | mask50 | 2.25465e-05 | 1 | 1 | 527 | 525 | 2 |
| latest | unconditional\_single/long | mask75 | 0.114828 | 0.958656 | 1 | 778 | 774 | 4 |
| latest | unconditional\_single/long | mask100 | 1.56433 | 0.371569 | 1 | 1024 | 1020 | 4 |
| latest | unconditional\_single/long | mask\_mean | 0.419856 | 0.832556 | 1 | — | — | — |
| latest | unconditional\_single/medium | mask25 | 2.2226e-05 | 1 | 1 | 110 | 109 | 1 |
| latest | unconditional\_single/medium | mask50 | 1.25543e-05 | 1 | 1 | 234 | 232 | 2 |
| latest | unconditional\_single/medium | mask75 | 0.00384662 | 1 | 1 | 371 | 367 | 4 |
| latest | unconditional\_single/medium | mask100 | 1.766 | 0.356299 | 1 | 512 | 508 | 4 |
| latest | unconditional\_single/medium | mask\_mean | 0.44247 | 0.839075 | 1 | — | — | — |
| latest | unconditional\_single/short | mask25 | 0.000962288 | 1 | 1 | 28 | 26 | 2 |
| latest | unconditional\_single/short | mask50 | 0.364028 | 0.82 | 1 | 52 | 50 | 2 |
| latest | unconditional\_single/short | mask75 | 1.03412 | 0.532468 | 1 | 80 | 77 | 3 |
| latest | unconditional\_single/short | mask100 | 1.32309 | 0.34375 | 1 | 100 | 96 | 4 |
| latest | unconditional\_single/short | mask\_mean | 0.680551 | 0.674054 | 1 | — | — | — |
| latest | unconditional\_tcr/long | mask25 | 9.23799e-06 | 1 | 1 | 516 | 515 | 1 |
| latest | unconditional\_tcr/long | mask50 | 4.5721e-06 | 1 | 1 | 989 | 987 | 2 |
| latest | unconditional\_tcr/long | mask75 | 6.56957e-06 | 1 | 1 | 1527 | 1522 | 5 |
| latest | unconditional\_tcr/long | mask100 | 1.62397 | 0.364216 | 1 | 2048 | 2040 | 8 |
| latest | unconditional\_tcr/long | mask\_mean | 0.405997 | 0.841054 | 1 | — | — | — |
| latest | unconditional\_tcr/medium | mask25 | 5.76044e-06 | 1 | — | 247 | 247 | 0 |
| latest | unconditional\_tcr/medium | mask50 | 2.81089e-06 | 1 | 1 | 500 | 497 | 3 |
| latest | unconditional\_tcr/medium | mask75 | 6.11735e-06 | 1 | 1 | 744 | 738 | 6 |
| latest | unconditional\_tcr/medium | mask100 | 1.70974 | 0.353346 | 1 | 1024 | 1016 | 8 |
| latest | unconditional\_tcr/medium | mask\_mean | 0.42744 | 0.838337 | 1 | — | — | — |
| latest | unconditional\_tcr/short | mask25 | 1.19707e-05 | 1 | — | 40 | 40 | 0 |
| latest | unconditional\_tcr/short | mask50 | 0.0189547 | 1 | 1 | 93 | 90 | 3 |
| latest | unconditional\_tcr/short | mask75 | 0.278513 | 0.948529 | 1 | 142 | 136 | 6 |
| latest | unconditional\_tcr/short | mask100 | 1.39136 | 0.479167 | 1 | 200 | 192 | 8 |
| latest | unconditional\_tcr/short | mask\_mean | 0.422211 | 0.856924 | 1 | — | — | — |
| train\_probe | ab\_antigen/long | mask25 | 4.2324e-05 | 1 | — | 530 | 530 | 0 |
| train\_probe | ab\_antigen/long | mask50 | 0.000235199 | 1 | 1 | 1034 | 1031 | 3 |
| train\_probe | ab\_antigen/long | mask75 | 1.00949e-05 | 1 | 1 | 1563 | 1557 | 6 |
| train\_probe | ab\_antigen/long | mask100 | 0.0266475 | 0.988726 | 1 | 2048 | 2040 | 8 |
| train\_probe | ab\_antigen/long | mask\_mean | 0.00673378 | 0.997181 | 1 | — | — | — |
| train\_probe | ab\_antigen/medium | mask25 | 5.35551e-05 | 1 | 1 | 275 | 273 | 2 |
| train\_probe | ab\_antigen/medium | mask50 | 6.58588e-05 | 1 | 1 | 542 | 538 | 4 |
| train\_probe | ab\_antigen/medium | mask75 | 5.03382e-05 | 1 | 1 | 781 | 776 | 5 |
| train\_probe | ab\_antigen/medium | mask100 | 0.00101019 | 1 | 1 | 1024 | 1016 | 8 |
| train\_probe | ab\_antigen/medium | mask\_mean | 0.000294986 | 1 | 1 | — | — | — |
| train\_probe | ab\_antigen/short | mask25 | 4.71513e-05 | 1 | 1 | 52 | 47 | 5 |
| train\_probe | ab\_antigen/short | mask50 | 5.79832e-05 | 1 | 1 | 109 | 102 | 7 |
| train\_probe | ab\_antigen/short | mask75 | 0.000133525 | 1 | 1 | 155 | 148 | 7 |
| train\_probe | ab\_antigen/short | mask100 | 0.000254981 | 1 | 1 | 200 | 192 | 8 |
| train\_probe | ab\_antigen/short | mask\_mean | 0.00012341 | 1 | 1 | — | — | — |
| train\_probe | tcr\_peptide/long | mask25 | 1.16297e-05 | 1 | 1 | 495 | 494 | 1 |
| train\_probe | tcr\_peptide/long | mask50 | 6.50394e-06 | 1 | 1 | 1033 | 1028 | 5 |
| train\_probe | tcr\_peptide/long | mask75 | 1.11784e-05 | 1 | 1 | 1530 | 1522 | 8 |
| train\_probe | tcr\_peptide/long | mask100 | 0.00818157 | 1 | 1 | 2048 | 2040 | 8 |
| train\_probe | tcr\_peptide/long | mask\_mean | 0.00205272 | 1 | 1 | — | — | — |
| train\_probe | tcr\_peptide/medium | mask25 | 6.15335e-06 | 1 | — | 252 | 252 | 0 |
| train\_probe | tcr\_peptide/medium | mask50 | 5.35783e-06 | 1 | 1 | 517 | 513 | 4 |
| train\_probe | tcr\_peptide/medium | mask75 | 2.53177e-05 | 1 | 1 | 760 | 754 | 6 |
| train\_probe | tcr\_peptide/medium | mask100 | 0.0288835 | 0.990157 | 1 | 1024 | 1016 | 8 |
| train\_probe | tcr\_peptide/medium | mask\_mean | 0.00723008 | 0.997539 | 1 | — | — | — |
| train\_probe | tcr\_peptide/short | mask25 | 4.17891e-05 | 1 | 1 | 51 | 46 | 5 |
| train\_probe | tcr\_peptide/short | mask50 | 5.88879e-05 | 1 | 1 | 95 | 90 | 5 |
| train\_probe | tcr\_peptide/short | mask75 | 0.000174006 | 1 | 1 | 142 | 136 | 6 |
| train\_probe | tcr\_peptide/short | mask100 | 0.00091396 | 1 | 1 | 200 | 192 | 8 |
| train\_probe | tcr\_peptide/short | mask\_mean | 0.000297161 | 1 | 1 | — | — | — |
| train\_probe | tcr\_pmhc/long | mask25 | 4.50737e-06 | 1 | 1 | 526 | 522 | 4 |
| train\_probe | tcr\_pmhc/long | mask50 | 4.86838e-06 | 1 | 1 | 1036 | 1031 | 5 |
| train\_probe | tcr\_pmhc/long | mask75 | 3.86613e-06 | 1 | 1 | 1557 | 1551 | 6 |
| train\_probe | tcr\_pmhc/long | mask100 | 0.104246 | 0.951471 | 1 | 2048 | 2040 | 8 |
| train\_probe | tcr\_pmhc/long | mask\_mean | 0.0260647 | 0.987868 | 1 | — | — | — |
| train\_probe | tcr\_pmhc/medium | mask25 | 4.67069e-06 | 1 | 1 | 276 | 273 | 3 |
| train\_probe | tcr\_pmhc/medium | mask50 | 4.71764e-06 | 1 | 1 | 549 | 541 | 8 |
| train\_probe | tcr\_pmhc/medium | mask75 | 6.10672e-06 | 1 | 1 | 793 | 785 | 8 |
| train\_probe | tcr\_pmhc/medium | mask100 | 0.00172878 | 1 | 1 | 1024 | 1016 | 8 |
| train\_probe | tcr\_pmhc/medium | mask\_mean | 0.000436069 | 1 | 1 | — | — | — |
| train\_probe | tcr\_pmhc/short | mask25 | 1.39137e-05 | 1 | 1 | 50 | 48 | 2 |
| train\_probe | tcr\_pmhc/short | mask50 | 4.17509e-05 | 1 | 1 | 103 | 100 | 3 |
| train\_probe | tcr\_pmhc/short | mask75 | 0.000338508 | 1 | 1 | 152 | 145 | 7 |
| train\_probe | tcr\_pmhc/short | mask100 | 0.0155264 | 0.989583 | 1 | 200 | 192 | 8 |
| train\_probe | tcr\_pmhc/short | mask\_mean | 0.00398014 | 0.997396 | 1 | — | — | — |
| train\_probe | unconditional\_ab/long | mask25 | 6.02156e-05 | 1 | 1 | 509 | 506 | 3 |
| train\_probe | unconditional\_ab/long | mask50 | 8.31622e-06 | 1 | 1 | 1016 | 1010 | 6 |
| train\_probe | unconditional\_ab/long | mask75 | 1.06429e-05 | 1 | 1 | 1569 | 1561 | 8 |
| train\_probe | unconditional\_ab/long | mask100 | 1.27468 | 0.510784 | 1 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_ab/long | mask\_mean | 0.318689 | 0.877696 | 1 | — | — | — |
| train\_probe | unconditional\_ab/medium | mask25 | 3.36593e-05 | 1 | 1 | 243 | 241 | 2 |
| train\_probe | unconditional\_ab/medium | mask50 | 2.53071e-05 | 1 | 1 | 521 | 516 | 5 |
| train\_probe | unconditional\_ab/medium | mask75 | 4.23236e-05 | 1 | 1 | 766 | 761 | 5 |
| train\_probe | unconditional\_ab/medium | mask100 | 1.56476 | 0.397638 | 1 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_ab/medium | mask\_mean | 0.391216 | 0.849409 | 1 | — | — | — |
| train\_probe | unconditional\_ab/short | mask25 | 0.000372577 | 1 | 1 | 53 | 52 | 1 |
| train\_probe | unconditional\_ab/short | mask50 | 0.141742 | 0.942308 | 1 | 106 | 104 | 2 |
| train\_probe | unconditional\_ab/short | mask75 | 0.506147 | 0.778481 | 1 | 163 | 158 | 5 |
| train\_probe | unconditional\_ab/short | mask100 | 1.41298 | 0.401042 | 1 | 200 | 192 | 8 |
| train\_probe | unconditional\_ab/short | mask\_mean | 0.51531 | 0.780458 | 1 | — | — | — |
| train\_probe | unconditional\_single/long | mask25 | 0.000517123 | 1 | 1 | 245 | 244 | 1 |
| train\_probe | unconditional\_single/long | mask50 | 0.000113234 | 1 | 1 | 492 | 491 | 1 |
| train\_probe | unconditional\_single/long | mask75 | 0.000478497 | 1 | 1 | 771 | 768 | 3 |
| train\_probe | unconditional\_single/long | mask100 | 1.66278 | 0.333333 | 1 | 1024 | 1020 | 4 |
| train\_probe | unconditional\_single/long | mask\_mean | 0.415971 | 0.833333 | 1 | — | — | — |
| train\_probe | unconditional\_single/medium | mask25 | 2.80549e-05 | 1 | 1 | 132 | 130 | 2 |
| train\_probe | unconditional\_single/medium | mask50 | 3.83643e-05 | 1 | 1 | 255 | 252 | 3 |
| train\_probe | unconditional\_single/medium | mask75 | 0.000180623 | 1 | 1 | 382 | 379 | 3 |
| train\_probe | unconditional\_single/medium | mask100 | 1.51841 | 0.320866 | 1 | 512 | 508 | 4 |
| train\_probe | unconditional\_single/medium | mask\_mean | 0.379664 | 0.830217 | 1 | — | — | — |
| train\_probe | unconditional\_single/short | mask25 | 0.00505514 | 1 | 1 | 28 | 26 | 2 |
| train\_probe | unconditional\_single/short | mask50 | 0.269185 | 0.857143 | 1 | 52 | 49 | 3 |
| train\_probe | unconditional\_single/short | mask75 | 0.777475 | 0.657143 | 1 | 73 | 70 | 3 |
| train\_probe | unconditional\_single/short | mask100 | 1.22341 | 0.40625 | 1 | 100 | 96 | 4 |
| train\_probe | unconditional\_single/short | mask\_mean | 0.56878 | 0.730134 | 1 | — | — | — |
| train\_probe | unconditional\_tcr/long | mask25 | 5.75516e-06 | 1 | — | 525 | 525 | 0 |
| train\_probe | unconditional\_tcr/long | mask50 | 3.03837e-06 | 1 | 1 | 1048 | 1045 | 3 |
| train\_probe | unconditional\_tcr/long | mask75 | 7.01727e-06 | 1 | 1 | 1585 | 1580 | 5 |
| train\_probe | unconditional\_tcr/long | mask100 | 1.53122 | 0.422059 | 1 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_tcr/long | mask\_mean | 0.38281 | 0.855515 | 1 | — | — | — |
| train\_probe | unconditional\_tcr/medium | mask25 | 1.69998e-06 | 1 | — | 261 | 261 | 0 |
| train\_probe | unconditional\_tcr/medium | mask50 | 4.66606e-06 | 1 | 1 | 498 | 497 | 1 |
| train\_probe | unconditional\_tcr/medium | mask75 | 3.08302e-05 | 1 | 1 | 755 | 751 | 4 |
| train\_probe | unconditional\_tcr/medium | mask100 | 1.53451 | 0.404528 | 1 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_tcr/medium | mask\_mean | 0.383637 | 0.851132 | 1 | — | — | — |
| train\_probe | unconditional\_tcr/short | mask25 | 2.99231e-05 | 1 | — | 46 | 46 | 0 |
| train\_probe | unconditional\_tcr/short | mask50 | 0.430572 | 0.869565 | 1 | 97 | 92 | 5 |
| train\_probe | unconditional\_tcr/short | mask75 | 0.596686 | 0.815068 | 1 | 152 | 146 | 6 |
| train\_probe | unconditional\_tcr/short | mask100 | 1.50301 | 0.354167 | 1 | 200 | 192 | 8 |
| train\_probe | unconditional\_tcr/short | mask\_mean | 0.632573 | 0.7597 | 1 | — | — | — |

### 1790803305064798754\_r25\_refined\_late\_fusion\_5cd5e385：分层 mask 指标

源码目录：/vepfs-mlp2/c20250601/251105016/project/dllm\_test/output/immune\_research\_releases/r20\_r27\_20261001；完成/目标 step：500/500。

kind=synthetic\_generalization\_diagnostic；dataset_kind=deterministic\_artificial\_sequences；biological_quality_claim=False。

| 时点 | mask level | 指标 | 值 |
| --- | --- | --- | --- |
| initial | mask25 | mask\_ratio | 0.25 |
| initial | mask25 | selected\_count | 4467 |
| initial | mask25 | aa\_count | 4428 |
| initial | mask25 | terminal\_count | 39 |
| initial | mask25 | loss | 3.94605 |
| initial | mask25 | aa\_loss | 3.94325 |
| initial | mask25 | aa\_accuracy | 0.0372629 |
| initial | mask25 | terminal\_loss | 4.26441 |
| initial | mask25 | terminal\_accuracy | 0 |
| initial | mask50 | mask\_ratio | 0.5 |
| initial | mask50 | selected\_count | 8888 |
| initial | mask50 | aa\_count | 8819 |
| initial | mask50 | terminal\_count | 69 |
| initial | mask50 | loss | 4.03979 |
| initial | mask50 | aa\_loss | 4.0364 |
| initial | mask50 | aa\_accuracy | 0.0413879 |
| initial | mask50 | terminal\_loss | 4.47271 |
| initial | mask50 | terminal\_accuracy | 0 |
| initial | mask75 | mask\_ratio | 0.75 |
| initial | mask75 | selected\_count | 13437 |
| initial | mask75 | aa\_count | 13330 |
| initial | mask75 | terminal\_count | 107 |
| initial | mask75 | loss | 4.09853 |
| initial | mask75 | aa\_loss | 4.09435 |
| initial | mask75 | aa\_accuracy | 0.0528132 |
| initial | mask75 | terminal\_loss | 4.61916 |
| initial | mask75 | terminal\_accuracy | 0 |
| initial | mask100 | mask\_ratio | 1 |
| initial | mask100 | selected\_count | 17996 |
| initial | mask100 | aa\_count | 17864 |
| initial | mask100 | terminal\_count | 132 |
| initial | mask100 | loss | 4.08227 |
| initial | mask100 | aa\_loss | 4.07792 |
| initial | mask100 | aa\_accuracy | 0.0315719 |
| initial | mask100 | terminal\_loss | 4.67092 |
| initial | mask100 | terminal\_accuracy | 0 |
| initial | mask\_mean | loss | 4.04166 |
| initial | mask\_mean | aa\_loss | 4.03798 |
| initial | mask\_mean | terminal\_loss | 4.5068 |
| initial | mask\_mean | aa\_accuracy | 0.040759 |
| initial | mask\_mean | terminal\_accuracy | 0 |
| latest | mask25 | mask\_ratio | 0.25 |
| latest | mask25 | selected\_count | 4467 |
| latest | mask25 | aa\_count | 4428 |
| latest | mask25 | terminal\_count | 39 |
| latest | mask25 | loss | 0.000234005 |
| latest | mask25 | aa\_loss | 0.000150864 |
| latest | mask25 | aa\_accuracy | 1 |
| latest | mask25 | terminal\_loss | 0.00967371 |
| latest | mask25 | terminal\_accuracy | 1 |
| latest | mask50 | mask\_ratio | 0.5 |
| latest | mask50 | selected\_count | 8888 |
| latest | mask50 | aa\_count | 8819 |
| latest | mask50 | terminal\_count | 69 |
| latest | mask50 | loss | 0.00437842 |
| latest | mask50 | aa\_loss | 0.00437564 |
| latest | mask50 | aa\_accuracy | 0.998413 |
| latest | mask50 | terminal\_loss | 0.00473364 |
| latest | mask50 | terminal\_accuracy | 1 |
| latest | mask75 | mask\_ratio | 0.75 |
| latest | mask75 | selected\_count | 13437 |
| latest | mask75 | aa\_count | 13330 |
| latest | mask75 | terminal\_count | 107 |
| latest | mask75 | loss | 0.0167021 |
| latest | mask75 | aa\_loss | 0.0168232 |
| latest | mask75 | aa\_accuracy | 0.993773 |
| latest | mask75 | terminal\_loss | 0.00161709 |
| latest | mask75 | terminal\_accuracy | 1 |
| latest | mask100 | mask\_ratio | 1 |
| latest | mask100 | selected\_count | 17996 |
| latest | mask100 | aa\_count | 17864 |
| latest | mask100 | terminal\_count | 132 |
| latest | mask100 | loss | 0.841605 |
| latest | mask100 | aa\_loss | 0.847171 |
| latest | mask100 | aa\_accuracy | 0.665417 |
| latest | mask100 | terminal\_loss | 0.0883177 |
| latest | mask100 | terminal\_accuracy | 0.969697 |
| latest | mask\_mean | loss | 0.21573 |
| latest | mask\_mean | aa\_loss | 0.21713 |
| latest | mask\_mean | terminal\_loss | 0.0260855 |
| latest | mask\_mean | aa\_accuracy | 0.914401 |
| latest | mask\_mean | terminal\_accuracy | 0.992424 |

评测曲线（原始 step，四档等权均值）：

| step | mean loss | mean AA acc | mean end acc | gradient norm |
| --- | --- | --- | --- | --- |
| 25 | 1.89873 | 0.311368 | 0 | 2.06952 |
| 50 | 1.51997 | 0.455006 | 0 | 1.81002 |
| 75 | 0.786989 | 0.726315 | 0.527519 | 2.3938 |
| 100 | 0.533884 | 0.796634 | 0.859454 | 3.12544 |
| 125 | 0.488581 | 0.823189 | 0.947585 | 2.87509 |
| 150 | 0.444366 | 0.836147 | 0.853784 | 1.56165 |
| 175 | 0.411866 | 0.844408 | 0.967684 | 2.42332 |
| 200 | 0.412167 | 0.849816 | 1 | 1.08288 |
| 225 | 0.387582 | 0.860055 | 0.98167 | 1.79001 |
| 250 | 0.330797 | 0.87215 | 0.992424 | 0.825315 |
| 275 | 0.344498 | 0.868857 | 0.984848 | 0.750903 |
| 300 | 0.326219 | 0.876259 | 0.967239 | 0.292812 |
| 325 | 0.31463 | 0.880339 | 0.962121 | 0.382345 |
| 350 | 0.286612 | 0.890116 | 0.992424 | 0.589581 |
| 375 | 0.267462 | 0.897582 | 0.971148 | 0.693421 |
| 400 | 0.259769 | 0.904252 | 0.992424 | 0.526668 |
| 425 | 0.255513 | 0.902654 | 1 | 0.427743 |
| 450 | 0.242044 | 0.906954 | 1 | 0.704324 |
| 475 | 0.217088 | 0.915061 | 0.984848 | 0.432728 |
| 500 | 0.21573 | 0.914401 | 0.992424 | 0.261576 |

有条件/无条件曲线（每档按token计数合并，再平均四档；缺失分组不推断数值）：

| step | 子集 | mean loss | mean AA | mean end | Full-mask AA | Full-mask end | Full-mask AA count | Full-mask end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | conditional | 4.10076 | 0.0341571 | 0 | 0.0332512 | 0 | 9744 | 72 |
| initial | unconditional | 3.97121 | 0.0486268 | 0 | 0.0295567 | 0 | 8120 | 60 |
| 25 | conditional | 1.87798 | 0.307129 | 0 | 0.278838 | 0 | 9744 | 72 |
| 25 | unconditional | 1.92347 | 0.316475 | 0 | 0.314901 | 0 | 8120 | 60 |
| 50 | conditional | 1.47197 | 0.463663 | 0 | 0.306342 | 0 | 9744 | 72 |
| 50 | unconditional | 1.57724 | 0.444736 | 0 | 0.314901 | 0 | 8120 | 60 |
| 75 | conditional | 0.734682 | 0.735352 | 0.562561 | 0.309729 | 0.916667 | 9744 | 72 |
| 75 | unconditional | 0.849598 | 0.715567 | 0.485632 | 0.303571 | 0.8 | 8120 | 60 |
| 100 | conditional | 0.481945 | 0.808606 | 0.912224 | 0.319684 | 0.902778 | 9744 | 72 |
| 100 | unconditional | 0.596171 | 0.782273 | 0.789087 | 0.231158 | 0.866667 | 8120 | 60 |
| 125 | conditional | 0.439121 | 0.833391 | 0.953858 | 0.385263 | 1 | 9744 | 72 |
| 125 | unconditional | 0.547915 | 0.810951 | 0.945023 | 0.298399 | 1 | 8120 | 60 |
| 150 | conditional | 0.37691 | 0.860474 | 0.911358 | 0.471675 | 0.75 | 9744 | 72 |
| 150 | unconditional | 0.525259 | 0.806983 | 0.775165 | 0.283251 | 0.6 | 8120 | 60 |
| 175 | conditional | 0.347764 | 0.871426 | 0.99 | 0.504208 | 1 | 9744 | 72 |
| 175 | unconditional | 0.488736 | 0.81201 | 0.936541 | 0.285961 | 0.933333 | 8120 | 60 |
| 200 | conditional | 0.352477 | 0.871197 | 1 | 0.499384 | 1 | 9744 | 72 |
| 200 | unconditional | 0.483774 | 0.824169 | 1 | 0.320197 | 1 | 8120 | 60 |
| 225 | conditional | 0.319488 | 0.889103 | 1 | 0.562808 | 1 | 9744 | 72 |
| 225 | unconditional | 0.469269 | 0.825214 | 0.953208 | 0.328079 | 1 | 8120 | 60 |
| 250 | conditional | 0.26709 | 0.898336 | 1 | 0.599959 | 1 | 9744 | 72 |
| 250 | unconditional | 0.407237 | 0.840732 | 0.983333 | 0.385222 | 0.933333 | 8120 | 60 |
| 275 | conditional | 0.234952 | 0.910616 | 1 | 0.645833 | 1 | 9744 | 72 |
| 275 | unconditional | 0.475936 | 0.818756 | 0.966667 | 0.299631 | 0.866667 | 8120 | 60 |
| 300 | conditional | 0.25924 | 0.90476 | 1 | 0.624282 | 1 | 9744 | 72 |
| 300 | unconditional | 0.406584 | 0.842062 | 0.922884 | 0.385591 | 0.8 | 8120 | 60 |
| 325 | conditional | 0.214526 | 0.922034 | 1 | 0.689655 | 1 | 9744 | 72 |
| 325 | unconditional | 0.434748 | 0.830308 | 0.916667 | 0.340148 | 0.666667 | 8120 | 60 |
| 350 | conditional | 0.17683 | 0.933813 | 1 | 0.73594 | 1 | 9744 | 72 |
| 350 | unconditional | 0.418346 | 0.837681 | 0.983333 | 0.367241 | 0.933333 | 8120 | 60 |
| 375 | conditional | 0.145527 | 0.946343 | 0.993056 | 0.785509 | 0.972222 | 9744 | 72 |
| 375 | unconditional | 0.413778 | 0.839074 | 0.944792 | 0.374384 | 0.8 | 8120 | 60 |
| 400 | conditional | 0.153899 | 0.94222 | 1 | 0.769704 | 1 | 9744 | 72 |
| 400 | unconditional | 0.386806 | 0.858692 | 0.983333 | 0.449138 | 0.933333 | 8120 | 60 |
| 425 | conditional | 0.139554 | 0.947403 | 1 | 0.790435 | 1 | 9744 | 72 |
| 425 | unconditional | 0.394659 | 0.848959 | 1 | 0.412438 | 1 | 8120 | 60 |
| 450 | conditional | 0.10367 | 0.963892 | 1 | 0.856117 | 1 | 9744 | 72 |
| 450 | unconditional | 0.408089 | 0.838631 | 1 | 0.374261 | 1 | 8120 | 60 |
| 475 | conditional | 0.0733801 | 0.974163 | 1 | 0.897065 | 1 | 9744 | 72 |
| 475 | unconditional | 0.389533 | 0.84414 | 0.966667 | 0.390764 | 0.866667 | 8120 | 60 |
| 500 | conditional | 0.0626954 | 0.977542 | 1 | 0.910304 | 1 | 9744 | 72 |
| 500 | unconditional | 0.399366 | 0.838634 | 0.983333 | 0.371552 | 0.933333 | 8120 | 60 |

人工数据 manifest：d40bbbbc1c4294eed843d88bcdd0cf193886de6c2d7e9db9069bbf86e931c022；精确 target sequence 重叠=0；精确 record 重叠=0。

跨 radius32 覆盖：{&quot;possible&quot;: true, &quot;train&quot;: true, &quot;valid&quot;: true}。生成规则共享是诊断设计的一部分，不能当作真实下游去重证书。


按任务/链长分组（全局 level 按 token 计数汇总，下面保留各组计数；不同组不得直接比较未经加权的总 loss）：

| 时点 | group/length | mask level | loss | AA acc | end acc | selected count | AA count | end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | ab\_antigen/long | mask25 | 4.10376 | 0.026616 | 0 | 530 | 526 | 4 |
| initial | ab\_antigen/long | mask50 | 4.30155 | 0.0266535 | 0 | 1019 | 1013 | 6 |
| initial | ab\_antigen/long | mask75 | 4.40906 | 0.0261097 | 0 | 1538 | 1532 | 6 |
| initial | ab\_antigen/long | mask100 | 4.34778 | 0.0156863 | 0 | 2048 | 2040 | 8 |
| initial | ab\_antigen/long | mask\_mean | 4.29054 | 0.0237664 | 0 | — | — | — |
| initial | ab\_antigen/medium | mask25 | 4.45436 | 0.0452675 | 0 | 246 | 243 | 3 |
| initial | ab\_antigen/medium | mask50 | 4.46488 | 0.0278884 | 0 | 507 | 502 | 5 |
| initial | ab\_antigen/medium | mask75 | 4.49795 | 0.0277045 | 0 | 765 | 758 | 7 |
| initial | ab\_antigen/medium | mask100 | 4.45486 | 0.0295276 | 0 | 1024 | 1016 | 8 |
| initial | ab\_antigen/medium | mask\_mean | 4.46801 | 0.032597 | 0 | — | — | — |
| initial | ab\_antigen/short | mask25 | 4.69282 | 0.0294118 | 0 | 38 | 34 | 4 |
| initial | ab\_antigen/short | mask50 | 4.39749 | 0.0340909 | 0 | 93 | 88 | 5 |
| initial | ab\_antigen/short | mask75 | 4.33879 | 0.0357143 | 0 | 145 | 140 | 5 |
| initial | ab\_antigen/short | mask100 | 4.28528 | 0.015625 | 0 | 200 | 192 | 8 |
| initial | ab\_antigen/short | mask\_mean | 4.4286 | 0.0287105 | 0 | — | — | — |
| initial | tcr\_peptide/long | mask25 | 3.81168 | 0.0451613 | 0 | 466 | 465 | 1 |
| initial | tcr\_peptide/long | mask50 | 3.89662 | 0.0458716 | 0 | 985 | 981 | 4 |
| initial | tcr\_peptide/long | mask75 | 3.96553 | 0.0500677 | 0 | 1484 | 1478 | 6 |
| initial | tcr\_peptide/long | mask100 | 3.98102 | 0.0470588 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_peptide/long | mask\_mean | 3.91372 | 0.0470398 | 0 | — | — | — |
| initial | tcr\_peptide/medium | mask25 | 3.96461 | 0.0373444 | 0 | 242 | 241 | 1 |
| initial | tcr\_peptide/medium | mask50 | 3.94159 | 0.0517928 | 0 | 506 | 502 | 4 |
| initial | tcr\_peptide/medium | mask75 | 3.88725 | 0.0513158 | 0 | 767 | 760 | 7 |
| initial | tcr\_peptide/medium | mask100 | 3.83488 | 0.0472441 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_peptide/medium | mask\_mean | 3.90708 | 0.0469243 | 0 | — | — | — |
| initial | tcr\_peptide/short | mask25 | 3.86838 | 0.0416667 | 0 | 51 | 48 | 3 |
| initial | tcr\_peptide/short | mask50 | 3.6784 | 0.0961538 | 0 | 110 | 104 | 6 |
| initial | tcr\_peptide/short | mask75 | 3.75761 | 0.0875912 | 0 | 144 | 137 | 7 |
| initial | tcr\_peptide/short | mask100 | 3.77943 | 0.09375 | 0 | 200 | 192 | 8 |
| initial | tcr\_peptide/short | mask\_mean | 3.77096 | 0.0797904 | 0 | — | — | — |
| initial | tcr\_pmhc/long | mask25 | 3.6662 | 0.0261194 | 0 | 540 | 536 | 4 |
| initial | tcr\_pmhc/long | mask50 | 3.87821 | 0.0367505 | 0 | 1038 | 1034 | 4 |
| initial | tcr\_pmhc/long | mask75 | 4.0134 | 0.0212355 | 0 | 1562 | 1554 | 8 |
| initial | tcr\_pmhc/long | mask100 | 4.05544 | 0.0308824 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_pmhc/long | mask\_mean | 3.90331 | 0.0287469 | 0 | — | — | — |
| initial | tcr\_pmhc/medium | mask25 | 4.19948 | 0.0123967 | 0 | 244 | 242 | 2 |
| initial | tcr\_pmhc/medium | mask50 | 4.2887 | 0.0250522 | 0 | 484 | 479 | 5 |
| initial | tcr\_pmhc/medium | mask75 | 4.35228 | 0.0312094 | 0 | 777 | 769 | 8 |
| initial | tcr\_pmhc/medium | mask100 | 4.34503 | 0.0305118 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_pmhc/medium | mask\_mean | 4.29637 | 0.0247925 | 0 | — | — | — |
| initial | tcr\_pmhc/short | mask25 | 4.38912 | 0.0208333 | 0 | 51 | 48 | 3 |
| initial | tcr\_pmhc/short | mask50 | 4.37921 | 0.0240964 | 0 | 86 | 83 | 3 |
| initial | tcr\_pmhc/short | mask75 | 4.18772 | 0.0204082 | 0 | 152 | 147 | 5 |
| initial | tcr\_pmhc/short | mask100 | 4.20912 | 0.015625 | 0 | 200 | 192 | 8 |
| initial | tcr\_pmhc/short | mask\_mean | 4.29129 | 0.0202407 | 0 | — | — | — |
| initial | unconditional\_ab/long | mask25 | 3.95361 | 0.0373832 | 0 | 538 | 535 | 3 |
| initial | unconditional\_ab/long | mask50 | 4.10352 | 0.0334608 | 0 | 1050 | 1046 | 4 |
| initial | unconditional\_ab/long | mask75 | 4.21672 | 0.0318596 | 0 | 1545 | 1538 | 7 |
| initial | unconditional\_ab/long | mask100 | 4.16261 | 0.0308824 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_ab/long | mask\_mean | 4.10912 | 0.0333965 | 0 | — | — | — |
| initial | unconditional\_ab/medium | mask25 | 4.21185 | 0.0948617 | 0 | 256 | 253 | 3 |
| initial | unconditional\_ab/medium | mask50 | 4.29512 | 0.0807087 | 0 | 512 | 508 | 4 |
| initial | unconditional\_ab/medium | mask75 | 4.2954 | 0.0748032 | 0 | 768 | 762 | 6 |
| initial | unconditional\_ab/medium | mask100 | 4.24567 | 0.0147638 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_ab/medium | mask\_mean | 4.26201 | 0.0662843 | 0 | — | — | — |
| initial | unconditional\_ab/short | mask25 | 3.86841 | 0.0576923 | 0 | 55 | 52 | 3 |
| initial | unconditional\_ab/short | mask50 | 3.89384 | 0.0714286 | 0 | 103 | 98 | 5 |
| initial | unconditional\_ab/short | mask75 | 3.85869 | 0.106383 | 0 | 148 | 141 | 7 |
| initial | unconditional\_ab/short | mask100 | 3.84764 | 0.125 | 0 | 200 | 192 | 8 |
| initial | unconditional\_ab/short | mask\_mean | 3.86715 | 0.090126 | 0 | — | — | — |
| initial | unconditional\_single/long | mask25 | 3.42299 | 0.0485075 | 0 | 269 | 268 | 1 |
| initial | unconditional\_single/long | mask50 | 3.36982 | 0.0838095 | 0 | 527 | 525 | 2 |
| initial | unconditional\_single/long | mask75 | 3.34001 | 0.22739 | 0 | 778 | 774 | 4 |
| initial | unconditional\_single/long | mask100 | 3.37679 | 0 | 0 | 1024 | 1020 | 4 |
| initial | unconditional\_single/long | mask\_mean | 3.3774 | 0.0899268 | 0 | — | — | — |
| initial | unconditional\_single/medium | mask25 | 3.71585 | 0.0366972 | 0 | 110 | 109 | 1 |
| initial | unconditional\_single/medium | mask50 | 3.50228 | 0.0818966 | 0 | 234 | 232 | 2 |
| initial | unconditional\_single/medium | mask75 | 3.51101 | 0.19346 | 0 | 371 | 367 | 4 |
| initial | unconditional\_single/medium | mask100 | 3.50481 | 0 | 0 | 512 | 508 | 4 |
| initial | unconditional\_single/medium | mask\_mean | 3.55849 | 0.0780136 | 0 | — | — | — |
| initial | unconditional\_single/short | mask25 | 3.10833 | 0 | 0 | 28 | 26 | 2 |
| initial | unconditional\_single/short | mask50 | 3.28518 | 0 | 0 | 52 | 50 | 2 |
| initial | unconditional\_single/short | mask75 | 3.10215 | 0 | 0 | 80 | 77 | 3 |
| initial | unconditional\_single/short | mask100 | 3.10264 | 0 | 0 | 100 | 96 | 4 |
| initial | unconditional\_single/short | mask\_mean | 3.14957 | 0 | 0 | — | — | — |
| initial | unconditional\_tcr/long | mask25 | 3.86257 | 0.0194175 | 0 | 516 | 515 | 1 |
| initial | unconditional\_tcr/long | mask50 | 3.99619 | 0.0182371 | 0 | 989 | 987 | 2 |
| initial | unconditional\_tcr/long | mask75 | 4.09982 | 0.0157687 | 0 | 1527 | 1522 | 5 |
| initial | unconditional\_tcr/long | mask100 | 4.10163 | 0.0151961 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_tcr/long | mask\_mean | 4.01505 | 0.0171548 | 0 | — | — | — |
| initial | unconditional\_tcr/medium | mask25 | 4.14186 | 0.0566802 | — | 247 | 247 | 0 |
| initial | unconditional\_tcr/medium | mask50 | 4.26738 | 0.0382294 | 0 | 500 | 497 | 3 |
| initial | unconditional\_tcr/medium | mask75 | 4.23126 | 0.0745257 | 0 | 744 | 738 | 6 |
| initial | unconditional\_tcr/medium | mask100 | 4.18619 | 0.0935039 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_tcr/medium | mask\_mean | 4.20667 | 0.0657348 | 0 | — | — | — |
| initial | unconditional\_tcr/short | mask25 | 4.14871 | 0.025 | — | 40 | 40 | 0 |
| initial | unconditional\_tcr/short | mask50 | 4.01632 | 0.0555556 | 0 | 93 | 90 | 3 |
| initial | unconditional\_tcr/short | mask75 | 4.10977 | 0.0441176 | 0 | 142 | 136 | 6 |
| initial | unconditional\_tcr/short | mask100 | 4.08673 | 0.0625 | 0 | 200 | 192 | 8 |
| initial | unconditional\_tcr/short | mask\_mean | 4.09038 | 0.0467933 | 0 | — | — | — |
| latest | ab\_antigen/long | mask25 | 0.000121886 | 1 | 1 | 530 | 526 | 4 |
| latest | ab\_antigen/long | mask50 | 5.08045e-05 | 1 | 1 | 1019 | 1013 | 6 |
| latest | ab\_antigen/long | mask75 | 2.98864e-05 | 1 | 1 | 1538 | 1532 | 6 |
| latest | ab\_antigen/long | mask100 | 0.398454 | 0.865686 | 1 | 2048 | 2040 | 8 |
| latest | ab\_antigen/long | mask\_mean | 0.0996641 | 0.966422 | 1 | — | — | — |
| latest | ab\_antigen/medium | mask25 | 0.000257524 | 1 | 1 | 246 | 243 | 3 |
| latest | ab\_antigen/medium | mask50 | 0.000344316 | 1 | 1 | 507 | 502 | 5 |
| latest | ab\_antigen/medium | mask75 | 0.00112521 | 1 | 1 | 765 | 758 | 7 |
| latest | ab\_antigen/medium | mask100 | 0.181242 | 0.939961 | 1 | 1024 | 1016 | 8 |
| latest | ab\_antigen/medium | mask\_mean | 0.0457422 | 0.98499 | 1 | — | — | — |
| latest | ab\_antigen/short | mask25 | 0.000100986 | 1 | 1 | 38 | 34 | 4 |
| latest | ab\_antigen/short | mask50 | 9.93875e-05 | 1 | 1 | 93 | 88 | 5 |
| latest | ab\_antigen/short | mask75 | 0.0127674 | 0.992857 | 1 | 145 | 140 | 5 |
| latest | ab\_antigen/short | mask100 | 0.0100672 | 1 | 1 | 200 | 192 | 8 |
| latest | ab\_antigen/short | mask\_mean | 0.00575875 | 0.998214 | 1 | — | — | — |
| latest | tcr\_peptide/long | mask25 | 7.04304e-05 | 1 | 1 | 466 | 465 | 1 |
| latest | tcr\_peptide/long | mask50 | 4.01162e-05 | 1 | 1 | 985 | 981 | 4 |
| latest | tcr\_peptide/long | mask75 | 2.54667e-05 | 1 | 1 | 1484 | 1478 | 6 |
| latest | tcr\_peptide/long | mask100 | 0.452616 | 0.82598 | 1 | 2048 | 2040 | 8 |
| latest | tcr\_peptide/long | mask\_mean | 0.113188 | 0.956495 | 1 | — | — | — |
| latest | tcr\_peptide/medium | mask25 | 2.49687e-05 | 1 | 1 | 242 | 241 | 1 |
| latest | tcr\_peptide/medium | mask50 | 2.32281e-05 | 1 | 1 | 506 | 502 | 4 |
| latest | tcr\_peptide/medium | mask75 | 0.000120002 | 1 | 1 | 767 | 760 | 7 |
| latest | tcr\_peptide/medium | mask100 | 0.192127 | 0.929134 | 1 | 1024 | 1016 | 8 |
| latest | tcr\_peptide/medium | mask\_mean | 0.0480738 | 0.982283 | 1 | — | — | — |
| latest | tcr\_peptide/short | mask25 | 3.61883e-05 | 1 | 1 | 51 | 48 | 3 |
| latest | tcr\_peptide/short | mask50 | 0.000882873 | 1 | 1 | 110 | 104 | 6 |
| latest | tcr\_peptide/short | mask75 | 0.000694357 | 1 | 1 | 144 | 137 | 7 |
| latest | tcr\_peptide/short | mask100 | 0.00499334 | 1 | 1 | 200 | 192 | 8 |
| latest | tcr\_peptide/short | mask\_mean | 0.00165169 | 1 | 1 | — | — | — |
| latest | tcr\_pmhc/long | mask25 | 8.07662e-05 | 1 | 1 | 540 | 536 | 4 |
| latest | tcr\_pmhc/long | mask50 | 4.64826e-05 | 1 | 1 | 1038 | 1034 | 4 |
| latest | tcr\_pmhc/long | mask75 | 2.25026e-05 | 1 | 1 | 1562 | 1554 | 8 |
| latest | tcr\_pmhc/long | mask100 | 0.120618 | 0.959804 | 1 | 2048 | 2040 | 8 |
| latest | tcr\_pmhc/long | mask\_mean | 0.030192 | 0.989951 | 1 | — | — | — |
| latest | tcr\_pmhc/medium | mask25 | 4.52639e-05 | 1 | 1 | 244 | 242 | 2 |
| latest | tcr\_pmhc/medium | mask50 | 3.25141e-05 | 1 | 1 | 484 | 479 | 5 |
| latest | tcr\_pmhc/medium | mask75 | 4.33422e-05 | 1 | 1 | 777 | 769 | 8 |
| latest | tcr\_pmhc/medium | mask100 | 0.0727764 | 0.970472 | 1 | 1024 | 1016 | 8 |
| latest | tcr\_pmhc/medium | mask\_mean | 0.0182244 | 0.992618 | 1 | — | — | — |
| latest | tcr\_pmhc/short | mask25 | 6.99799e-05 | 1 | 1 | 51 | 48 | 3 |
| latest | tcr\_pmhc/short | mask50 | 0.000113516 | 1 | 1 | 86 | 83 | 3 |
| latest | tcr\_pmhc/short | mask75 | 0.0124895 | 1 | 1 | 152 | 147 | 5 |
| latest | tcr\_pmhc/short | mask100 | 0.0164374 | 1 | 1 | 200 | 192 | 8 |
| latest | tcr\_pmhc/short | mask\_mean | 0.00727759 | 1 | 1 | — | — | — |
| latest | unconditional\_ab/long | mask25 | 5.66003e-05 | 1 | 1 | 538 | 535 | 3 |
| latest | unconditional\_ab/long | mask50 | 3.15984e-05 | 1 | 1 | 1050 | 1046 | 4 |
| latest | unconditional\_ab/long | mask75 | 1.84977e-05 | 1 | 1 | 1545 | 1538 | 7 |
| latest | unconditional\_ab/long | mask100 | 1.57599 | 0.348039 | 1 | 2048 | 2040 | 8 |
| latest | unconditional\_ab/long | mask\_mean | 0.394024 | 0.83701 | 1 | — | — | — |
| latest | unconditional\_ab/medium | mask25 | 0.000126883 | 1 | 1 | 256 | 253 | 3 |
| latest | unconditional\_ab/medium | mask50 | 0.000347626 | 1 | 1 | 512 | 508 | 4 |
| latest | unconditional\_ab/medium | mask75 | 8.93153e-05 | 1 | 1 | 768 | 762 | 6 |
| latest | unconditional\_ab/medium | mask100 | 1.5579 | 0.389764 | 0.5 | 1024 | 1016 | 8 |
| latest | unconditional\_ab/medium | mask\_mean | 0.389616 | 0.847441 | 0.875 | — | — | — |
| latest | unconditional\_ab/short | mask25 | 0.00116565 | 1 | 1 | 55 | 52 | 3 |
| latest | unconditional\_ab/short | mask50 | 0.206512 | 0.928571 | 1 | 103 | 98 | 5 |
| latest | unconditional\_ab/short | mask75 | 0.408208 | 0.843972 | 1 | 148 | 141 | 7 |
| latest | unconditional\_ab/short | mask100 | 1.198 | 0.4375 | 1 | 200 | 192 | 8 |
| latest | unconditional\_ab/short | mask\_mean | 0.453472 | 0.802511 | 1 | — | — | — |
| latest | unconditional\_single/long | mask25 | 6.95586e-05 | 1 | 1 | 269 | 268 | 1 |
| latest | unconditional\_single/long | mask50 | 3.87167e-05 | 1 | 1 | 527 | 525 | 2 |
| latest | unconditional\_single/long | mask75 | 0.0594903 | 0.975452 | 1 | 778 | 774 | 4 |
| latest | unconditional\_single/long | mask100 | 1.61252 | 0.382353 | 1 | 1024 | 1020 | 4 |
| latest | unconditional\_single/long | mask\_mean | 0.418029 | 0.839451 | 1 | — | — | — |
| latest | unconditional\_single/medium | mask25 | 0.00560206 | 1 | 1 | 110 | 109 | 1 |
| latest | unconditional\_single/medium | mask50 | 0.0095208 | 0.99569 | 1 | 234 | 232 | 2 |
| latest | unconditional\_single/medium | mask75 | 0.00192653 | 1 | 1 | 371 | 367 | 4 |
| latest | unconditional\_single/medium | mask100 | 1.71636 | 0.356299 | 1 | 512 | 508 | 4 |
| latest | unconditional\_single/medium | mask\_mean | 0.433352 | 0.837997 | 1 | — | — | — |
| latest | unconditional\_single/short | mask25 | 0.000672155 | 1 | 1 | 28 | 26 | 2 |
| latest | unconditional\_single/short | mask50 | 0.279524 | 0.88 | 1 | 52 | 50 | 2 |
| latest | unconditional\_single/short | mask75 | 0.924273 | 0.597403 | 1 | 80 | 77 | 3 |
| latest | unconditional\_single/short | mask100 | 1.39108 | 0.375 | 1 | 100 | 96 | 4 |
| latest | unconditional\_single/short | mask\_mean | 0.648887 | 0.713101 | 1 | — | — | — |
| latest | unconditional\_tcr/long | mask25 | 3.84868e-05 | 1 | 1 | 516 | 515 | 1 |
| latest | unconditional\_tcr/long | mask50 | 2.26362e-05 | 1 | 1 | 989 | 987 | 2 |
| latest | unconditional\_tcr/long | mask75 | 1.23225e-05 | 1 | 1 | 1527 | 1522 | 5 |
| latest | unconditional\_tcr/long | mask100 | 1.54368 | 0.376471 | 1 | 2048 | 2040 | 8 |
| latest | unconditional\_tcr/long | mask\_mean | 0.385937 | 0.844118 | 1 | — | — | — |
| latest | unconditional\_tcr/medium | mask25 | 4.81553e-05 | 1 | — | 247 | 247 | 0 |
| latest | unconditional\_tcr/medium | mask50 | 2.2687e-05 | 1 | 1 | 500 | 497 | 3 |
| latest | unconditional\_tcr/medium | mask75 | 9.38536e-05 | 1 | 1 | 744 | 738 | 6 |
| latest | unconditional\_tcr/medium | mask100 | 1.51017 | 0.346457 | 1 | 1024 | 1016 | 8 |
| latest | unconditional\_tcr/medium | mask\_mean | 0.377584 | 0.836614 | 1 | — | — | — |
| latest | unconditional\_tcr/short | mask25 | 5.04942e-05 | 1 | — | 40 | 40 | 0 |
| latest | unconditional\_tcr/short | mask50 | 0.00170413 | 1 | 1 | 93 | 90 | 3 |
| latest | unconditional\_tcr/short | mask75 | 0.267113 | 0.926471 | 1 | 142 | 136 | 6 |
| latest | unconditional\_tcr/short | mask100 | 1.26436 | 0.520833 | 1 | 200 | 192 | 8 |
| latest | unconditional\_tcr/short | mask\_mean | 0.383306 | 0.861826 | 1 | — | — | — |
| train\_probe | ab\_antigen/long | mask25 | 6.43996e-05 | 1 | — | 530 | 530 | 0 |
| train\_probe | ab\_antigen/long | mask50 | 5.95883e-05 | 1 | 1 | 1034 | 1031 | 3 |
| train\_probe | ab\_antigen/long | mask75 | 3.31688e-05 | 1 | 1 | 1563 | 1557 | 6 |
| train\_probe | ab\_antigen/long | mask100 | 0.453475 | 0.831373 | 1 | 2048 | 2040 | 8 |
| train\_probe | ab\_antigen/long | mask\_mean | 0.113408 | 0.957843 | 1 | — | — | — |
| train\_probe | ab\_antigen/medium | mask25 | 0.00112505 | 1 | 1 | 275 | 273 | 2 |
| train\_probe | ab\_antigen/medium | mask50 | 0.000407443 | 1 | 1 | 542 | 538 | 4 |
| train\_probe | ab\_antigen/medium | mask75 | 0.000182869 | 1 | 1 | 781 | 776 | 5 |
| train\_probe | ab\_antigen/medium | mask100 | 0.134903 | 0.955709 | 1 | 1024 | 1016 | 8 |
| train\_probe | ab\_antigen/medium | mask\_mean | 0.0341546 | 0.988927 | 1 | — | — | — |
| train\_probe | ab\_antigen/short | mask25 | 4.42332e-05 | 1 | 1 | 52 | 47 | 5 |
| train\_probe | ab\_antigen/short | mask50 | 0.000323797 | 1 | 1 | 109 | 102 | 7 |
| train\_probe | ab\_antigen/short | mask75 | 0.0026622 | 1 | 1 | 155 | 148 | 7 |
| train\_probe | ab\_antigen/short | mask100 | 0.0067471 | 1 | 1 | 200 | 192 | 8 |
| train\_probe | ab\_antigen/short | mask\_mean | 0.00244433 | 1 | 1 | — | — | — |
| train\_probe | tcr\_peptide/long | mask25 | 4.58827e-05 | 1 | 1 | 495 | 494 | 1 |
| train\_probe | tcr\_peptide/long | mask50 | 3.78893e-05 | 1 | 1 | 1033 | 1028 | 5 |
| train\_probe | tcr\_peptide/long | mask75 | 3.44041e-05 | 1 | 1 | 1530 | 1522 | 8 |
| train\_probe | tcr\_peptide/long | mask100 | 0.626229 | 0.770098 | 1 | 2048 | 2040 | 8 |
| train\_probe | tcr\_peptide/long | mask\_mean | 0.156587 | 0.942525 | 1 | — | — | — |
| train\_probe | tcr\_peptide/medium | mask25 | 2.44397e-05 | 1 | — | 252 | 252 | 0 |
| train\_probe | tcr\_peptide/medium | mask50 | 1.60689e-05 | 1 | 1 | 517 | 513 | 4 |
| train\_probe | tcr\_peptide/medium | mask75 | 6.13156e-05 | 1 | 1 | 760 | 754 | 6 |
| train\_probe | tcr\_peptide/medium | mask100 | 0.204111 | 0.927165 | 1 | 1024 | 1016 | 8 |
| train\_probe | tcr\_peptide/medium | mask\_mean | 0.0510531 | 0.981791 | 1 | — | — | — |
| train\_probe | tcr\_peptide/short | mask25 | 0.00016969 | 1 | 1 | 51 | 46 | 5 |
| train\_probe | tcr\_peptide/short | mask50 | 0.000185025 | 1 | 1 | 95 | 90 | 5 |
| train\_probe | tcr\_peptide/short | mask75 | 0.00190365 | 1 | 1 | 142 | 136 | 6 |
| train\_probe | tcr\_peptide/short | mask100 | 0.00414107 | 1 | 1 | 200 | 192 | 8 |
| train\_probe | tcr\_peptide/short | mask\_mean | 0.00159986 | 1 | 1 | — | — | — |
| train\_probe | tcr\_pmhc/long | mask25 | 8.20522e-05 | 1 | 1 | 526 | 522 | 4 |
| train\_probe | tcr\_pmhc/long | mask50 | 4.90566e-05 | 1 | 1 | 1036 | 1031 | 5 |
| train\_probe | tcr\_pmhc/long | mask75 | 2.50596e-05 | 1 | 1 | 1557 | 1551 | 6 |
| train\_probe | tcr\_pmhc/long | mask100 | 0.209963 | 0.932843 | 1 | 2048 | 2040 | 8 |
| train\_probe | tcr\_pmhc/long | mask\_mean | 0.0525297 | 0.983211 | 1 | — | — | — |
| train\_probe | tcr\_pmhc/medium | mask25 | 7.54916e-05 | 1 | 1 | 276 | 273 | 3 |
| train\_probe | tcr\_pmhc/medium | mask50 | 5.2802e-05 | 1 | 1 | 549 | 541 | 8 |
| train\_probe | tcr\_pmhc/medium | mask75 | 2.79665e-05 | 1 | 1 | 793 | 785 | 8 |
| train\_probe | tcr\_pmhc/medium | mask100 | 0.0347214 | 0.992126 | 1 | 1024 | 1016 | 8 |
| train\_probe | tcr\_pmhc/medium | mask\_mean | 0.00871942 | 0.998031 | 1 | — | — | — |
| train\_probe | tcr\_pmhc/short | mask25 | 4.83035e-05 | 1 | 1 | 50 | 48 | 2 |
| train\_probe | tcr\_pmhc/short | mask50 | 7.07022e-05 | 1 | 1 | 103 | 100 | 3 |
| train\_probe | tcr\_pmhc/short | mask75 | 0.000304147 | 1 | 1 | 152 | 145 | 7 |
| train\_probe | tcr\_pmhc/short | mask100 | 0.00355071 | 1 | 1 | 200 | 192 | 8 |
| train\_probe | tcr\_pmhc/short | mask\_mean | 0.000993465 | 1 | 1 | — | — | — |
| train\_probe | unconditional\_ab/long | mask25 | 7.62273e-05 | 1 | 1 | 509 | 506 | 3 |
| train\_probe | unconditional\_ab/long | mask50 | 4.36199e-05 | 1 | 1 | 1016 | 1010 | 6 |
| train\_probe | unconditional\_ab/long | mask75 | 0.00231122 | 0.998719 | 1 | 1569 | 1561 | 8 |
| train\_probe | unconditional\_ab/long | mask100 | 1.48937 | 0.355882 | 1 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_ab/long | mask\_mean | 0.37295 | 0.83865 | 1 | — | — | — |
| train\_probe | unconditional\_ab/medium | mask25 | 0.000115654 | 1 | 1 | 243 | 241 | 2 |
| train\_probe | unconditional\_ab/medium | mask50 | 0.000412712 | 1 | 1 | 521 | 516 | 5 |
| train\_probe | unconditional\_ab/medium | mask75 | 0.000144047 | 1 | 1 | 766 | 761 | 5 |
| train\_probe | unconditional\_ab/medium | mask100 | 1.57383 | 0.330709 | 0.5 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_ab/medium | mask\_mean | 0.393626 | 0.832677 | 0.875 | — | — | — |
| train\_probe | unconditional\_ab/short | mask25 | 0.000183498 | 1 | 1 | 53 | 52 | 1 |
| train\_probe | unconditional\_ab/short | mask50 | 0.259646 | 0.942308 | 1 | 106 | 104 | 2 |
| train\_probe | unconditional\_ab/short | mask75 | 0.557391 | 0.765823 | 1 | 163 | 158 | 5 |
| train\_probe | unconditional\_ab/short | mask100 | 1.45988 | 0.34375 | 1 | 200 | 192 | 8 |
| train\_probe | unconditional\_ab/short | mask\_mean | 0.569274 | 0.76297 | 1 | — | — | — |
| train\_probe | unconditional\_single/long | mask25 | 7.01058e-05 | 1 | 1 | 245 | 244 | 1 |
| train\_probe | unconditional\_single/long | mask50 | 3.18677e-05 | 1 | 1 | 492 | 491 | 1 |
| train\_probe | unconditional\_single/long | mask75 | 3.75457e-05 | 1 | 1 | 771 | 768 | 3 |
| train\_probe | unconditional\_single/long | mask100 | 1.49902 | 0.376471 | 1 | 1024 | 1020 | 4 |
| train\_probe | unconditional\_single/long | mask\_mean | 0.374791 | 0.844118 | 1 | — | — | — |
| train\_probe | unconditional\_single/medium | mask25 | 0.000784679 | 1 | 1 | 132 | 130 | 2 |
| train\_probe | unconditional\_single/medium | mask50 | 0.000152177 | 1 | 1 | 255 | 252 | 3 |
| train\_probe | unconditional\_single/medium | mask75 | 0.000170654 | 1 | 1 | 382 | 379 | 3 |
| train\_probe | unconditional\_single/medium | mask100 | 1.53255 | 0.344488 | 1 | 512 | 508 | 4 |
| train\_probe | unconditional\_single/medium | mask\_mean | 0.383415 | 0.836122 | 1 | — | — | — |
| train\_probe | unconditional\_single/short | mask25 | 0.00325425 | 1 | 1 | 28 | 26 | 2 |
| train\_probe | unconditional\_single/short | mask50 | 0.370667 | 0.877551 | 1 | 52 | 49 | 3 |
| train\_probe | unconditional\_single/short | mask75 | 0.83938 | 0.642857 | 1 | 73 | 70 | 3 |
| train\_probe | unconditional\_single/short | mask100 | 1.21577 | 0.4375 | 1 | 100 | 96 | 4 |
| train\_probe | unconditional\_single/short | mask\_mean | 0.607267 | 0.739477 | 1 | — | — | — |
| train\_probe | unconditional\_tcr/long | mask25 | 2.35792e-05 | 1 | — | 525 | 525 | 0 |
| train\_probe | unconditional\_tcr/long | mask50 | 1.63051e-05 | 1 | 1 | 1048 | 1045 | 3 |
| train\_probe | unconditional\_tcr/long | mask75 | 1.83358e-05 | 1 | 1 | 1585 | 1580 | 5 |
| train\_probe | unconditional\_tcr/long | mask100 | 1.57722 | 0.366667 | 1 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_tcr/long | mask\_mean | 0.394318 | 0.841667 | 1 | — | — | — |
| train\_probe | unconditional\_tcr/medium | mask25 | 3.24416e-05 | 1 | — | 261 | 261 | 0 |
| train\_probe | unconditional\_tcr/medium | mask50 | 7.69195e-05 | 1 | 1 | 498 | 497 | 1 |
| train\_probe | unconditional\_tcr/medium | mask75 | 8.39964e-05 | 1 | 1 | 755 | 751 | 4 |
| train\_probe | unconditional\_tcr/medium | mask100 | 1.47087 | 0.350394 | 1 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_tcr/medium | mask\_mean | 0.367765 | 0.837598 | 1 | — | — | — |
| train\_probe | unconditional\_tcr/short | mask25 | 0.000190124 | 1 | — | 46 | 46 | 0 |
| train\_probe | unconditional\_tcr/short | mask50 | 0.274367 | 0.923913 | 1 | 97 | 92 | 5 |
| train\_probe | unconditional\_tcr/short | mask75 | 0.410568 | 0.856164 | 1 | 152 | 146 | 6 |
| train\_probe | unconditional\_tcr/short | mask100 | 1.38607 | 0.307292 | 1 | 200 | 192 | 8 |
| train\_probe | unconditional\_tcr/short | mask\_mean | 0.5178 | 0.771842 | 1 | — | — | — |

### 1790803305068196112\_r26\_refined\_local\_global\_5bf50a1c：分层 mask 指标

源码目录：/vepfs-mlp2/c20250601/251105016/project/dllm\_test/output/immune\_research\_releases/r20\_r27\_20261001；完成/目标 step：500/500。

kind=synthetic\_generalization\_diagnostic；dataset_kind=deterministic\_artificial\_sequences；biological_quality_claim=False。

| 时点 | mask level | 指标 | 值 |
| --- | --- | --- | --- |
| initial | mask25 | mask\_ratio | 0.25 |
| initial | mask25 | selected\_count | 4467 |
| initial | mask25 | aa\_count | 4428 |
| initial | mask25 | terminal\_count | 39 |
| initial | mask25 | loss | 3.89278 |
| initial | mask25 | aa\_loss | 3.88838 |
| initial | mask25 | aa\_accuracy | 0.0350045 |
| initial | mask25 | terminal\_loss | 4.39224 |
| initial | mask25 | terminal\_accuracy | 0 |
| initial | mask50 | mask\_ratio | 0.5 |
| initial | mask50 | selected\_count | 8888 |
| initial | mask50 | aa\_count | 8819 |
| initial | mask50 | terminal\_count | 69 |
| initial | mask50 | loss | 4.00888 |
| initial | mask50 | aa\_loss | 4.00457 |
| initial | mask50 | aa\_accuracy | 0.035038 |
| initial | mask50 | terminal\_loss | 4.55927 |
| initial | mask50 | terminal\_accuracy | 0 |
| initial | mask75 | mask\_ratio | 0.75 |
| initial | mask75 | selected\_count | 13437 |
| initial | mask75 | aa\_count | 13330 |
| initial | mask75 | terminal\_count | 107 |
| initial | mask75 | loss | 4.07546 |
| initial | mask75 | aa\_loss | 4.07043 |
| initial | mask75 | aa\_accuracy | 0.0447862 |
| initial | mask75 | terminal\_loss | 4.70148 |
| initial | mask75 | terminal\_accuracy | 0 |
| initial | mask100 | mask\_ratio | 1 |
| initial | mask100 | selected\_count | 17996 |
| initial | mask100 | aa\_count | 17864 |
| initial | mask100 | terminal\_count | 132 |
| initial | mask100 | loss | 4.07558 |
| initial | mask100 | aa\_loss | 4.07093 |
| initial | mask100 | aa\_accuracy | 0.0292208 |
| initial | mask100 | terminal\_loss | 4.7046 |
| initial | mask100 | terminal\_accuracy | 0 |
| initial | mask\_mean | loss | 4.01317 |
| initial | mask\_mean | aa\_loss | 4.00858 |
| initial | mask\_mean | terminal\_loss | 4.58939 |
| initial | mask\_mean | aa\_accuracy | 0.0360124 |
| initial | mask\_mean | terminal\_accuracy | 0 |
| latest | mask25 | mask\_ratio | 0.25 |
| latest | mask25 | selected\_count | 4467 |
| latest | mask25 | aa\_count | 4428 |
| latest | mask25 | terminal\_count | 39 |
| latest | mask25 | loss | 4.51852e-05 |
| latest | mask25 | aa\_loss | 3.57709e-05 |
| latest | mask25 | aa\_accuracy | 1 |
| latest | mask25 | terminal\_loss | 0.00111408 |
| latest | mask25 | terminal\_accuracy | 1 |
| latest | mask50 | mask\_ratio | 0.5 |
| latest | mask50 | selected\_count | 8888 |
| latest | mask50 | aa\_count | 8819 |
| latest | mask50 | terminal\_count | 69 |
| latest | mask50 | loss | 0.00412023 |
| latest | mask50 | aa\_loss | 0.00415063 |
| latest | mask50 | aa\_accuracy | 0.998186 |
| latest | mask50 | terminal\_loss | 0.000235315 |
| latest | mask50 | terminal\_accuracy | 1 |
| latest | mask75 | mask\_ratio | 0.75 |
| latest | mask75 | selected\_count | 13437 |
| latest | mask75 | aa\_count | 13330 |
| latest | mask75 | terminal\_count | 107 |
| latest | mask75 | loss | 0.0215626 |
| latest | mask75 | aa\_loss | 0.0217344 |
| latest | mask75 | aa\_accuracy | 0.993023 |
| latest | mask75 | terminal\_loss | 0.000166454 |
| latest | mask75 | terminal\_accuracy | 1 |
| latest | mask100 | mask\_ratio | 1 |
| latest | mask100 | selected\_count | 17996 |
| latest | mask100 | aa\_count | 17864 |
| latest | mask100 | terminal\_count | 132 |
| latest | mask100 | loss | 0.749581 |
| latest | mask100 | aa\_loss | 0.755081 |
| latest | mask100 | aa\_accuracy | 0.704098 |
| latest | mask100 | terminal\_loss | 0.005193 |
| latest | mask100 | terminal\_accuracy | 1 |
| latest | mask\_mean | loss | 0.193827 |
| latest | mask\_mean | aa\_loss | 0.195251 |
| latest | mask\_mean | terminal\_loss | 0.00167721 |
| latest | mask\_mean | aa\_accuracy | 0.923827 |
| latest | mask\_mean | terminal\_accuracy | 1 |

评测曲线（原始 step，四档等权均值）：

| step | mean loss | mean AA acc | mean end acc | gradient norm |
| --- | --- | --- | --- | --- |
| 25 | 1.76833 | 0.363794 | 0 | 2.39835 |
| 50 | 0.989363 | 0.675248 | 0 | 2.31231 |
| 75 | 0.682895 | 0.774061 | 0.270463 | 3.28464 |
| 100 | 0.518047 | 0.811465 | 0.810016 | 2.84128 |
| 125 | 0.442792 | 0.837268 | 0.995327 | 2.85562 |
| 150 | 0.38494 | 0.857739 | 0.954103 | 1.24418 |
| 175 | 0.383167 | 0.860651 | 0.984848 | 1.05538 |
| 200 | 0.336445 | 0.878107 | 1 | 0.758794 |
| 225 | 0.265697 | 0.896989 | 0.944633 | 0.702105 |
| 250 | 0.252088 | 0.906191 | 1 | 0.459424 |
| 275 | 0.246308 | 0.912019 | 0.952209 | 0.916468 |
| 300 | 0.198492 | 0.92678 | 0.969697 | 0.254451 |
| 325 | 0.196214 | 0.923481 | 0.945076 | 0.286821 |
| 350 | 0.190651 | 0.924468 | 0.969697 | 0.389259 |
| 375 | 0.183209 | 0.925041 | 0.954545 | 0.759866 |
| 400 | 0.185626 | 0.9301 | 1 | 0.28013 |
| 425 | 0.188098 | 0.925636 | 1 | 0.408123 |
| 450 | 0.183675 | 0.928178 | 0.994318 | 0.225123 |
| 475 | 0.183305 | 0.929594 | 1 | 0.203428 |
| 500 | 0.193827 | 0.923827 | 1 | 0.3012 |

有条件/无条件曲线（每档按token计数合并，再平均四档；缺失分组不推断数值）：

| step | 子集 | mean loss | mean AA | mean end | Full-mask AA | Full-mask end | Full-mask AA count | Full-mask end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | conditional | 4.08659 | 0.0313433 | 0 | 0.0339696 | 0 | 9744 | 72 |
| initial | unconditional | 3.92552 | 0.0415568 | 0 | 0.0235222 | 0 | 8120 | 60 |
| 25 | conditional | 1.74062 | 0.369605 | 0 | 0.282635 | 0 | 9744 | 72 |
| 25 | unconditional | 1.80139 | 0.356951 | 0 | 0.314901 | 0 | 8120 | 60 |
| 50 | conditional | 0.949409 | 0.679515 | 0 | 0.313629 | 0 | 9744 | 72 |
| 50 | unconditional | 1.0374 | 0.67015 | 0 | 0.315517 | 0 | 8120 | 60 |
| 75 | conditional | 0.609528 | 0.785091 | 0.331639 | 0.335283 | 0.375 | 9744 | 72 |
| 75 | unconditional | 0.770843 | 0.760841 | 0.191567 | 0.268719 | 0.133333 | 8120 | 60 |
| 100 | conditional | 0.45527 | 0.831737 | 0.866852 | 0.385776 | 0.861111 | 9744 | 72 |
| 100 | unconditional | 0.593335 | 0.787155 | 0.727811 | 0.225123 | 0.733333 | 8120 | 60 |
| 125 | conditional | 0.386094 | 0.862213 | 1 | 0.471162 | 1 | 9744 | 72 |
| 125 | unconditional | 0.51082 | 0.807328 | 0.989583 | 0.259113 | 1 | 8120 | 60 |
| 150 | conditional | 0.307085 | 0.888471 | 0.975694 | 0.561884 | 0.902778 | 9744 | 72 |
| 150 | unconditional | 0.478361 | 0.820865 | 0.928125 | 0.304187 | 0.733333 | 8120 | 60 |
| 175 | conditional | 0.294021 | 0.890026 | 1 | 0.568452 | 1 | 9744 | 72 |
| 175 | unconditional | 0.490131 | 0.825407 | 0.966667 | 0.327956 | 0.866667 | 8120 | 60 |
| 200 | conditional | 0.253849 | 0.918079 | 1 | 0.6797 | 1 | 9744 | 72 |
| 200 | unconditional | 0.435559 | 0.830142 | 1 | 0.341133 | 1 | 8120 | 60 |
| 225 | conditional | 0.140474 | 0.949687 | 1 | 0.803366 | 1 | 9744 | 72 |
| 225 | unconditional | 0.415962 | 0.833756 | 0.878125 | 0.354434 | 0.533333 | 8120 | 60 |
| 250 | conditional | 0.117555 | 0.955718 | 1 | 0.825842 | 1 | 9744 | 72 |
| 250 | unconditional | 0.413526 | 0.846759 | 1 | 0.406281 | 1 | 8120 | 60 |
| 275 | conditional | 0.0811126 | 0.979464 | 1 | 0.919438 | 1 | 9744 | 72 |
| 275 | unconditional | 0.444539 | 0.83109 | 0.894792 | 0.342611 | 0.6 | 8120 | 60 |
| 300 | conditional | 0.0297205 | 0.991181 | 1 | 0.965825 | 1 | 9744 | 72 |
| 300 | unconditional | 0.401007 | 0.849504 | 0.933333 | 0.414286 | 0.733333 | 8120 | 60 |
| 325 | conditional | 0.0208342 | 0.993979 | 0.996528 | 0.976191 | 0.986111 | 9744 | 72 |
| 325 | unconditional | 0.406666 | 0.838886 | 0.883333 | 0.376108 | 0.533333 | 8120 | 60 |
| 350 | conditional | 0.0125629 | 0.99657 | 1 | 0.986556 | 1 | 9744 | 72 |
| 350 | unconditional | 0.404351 | 0.837948 | 0.933333 | 0.367734 | 0.733333 | 8120 | 60 |
| 375 | conditional | 0.00294255 | 0.99923 | 1 | 0.996921 | 1 | 9744 | 72 |
| 375 | unconditional | 0.399522 | 0.836018 | 0.9 | 0.360345 | 0.6 | 8120 | 60 |
| 400 | conditional | 0.011762 | 0.997203 | 1 | 0.988814 | 1 | 9744 | 72 |
| 400 | unconditional | 0.394257 | 0.84958 | 1 | 0.418227 | 1 | 8120 | 60 |
| 425 | conditional | 0.0113551 | 0.997647 | 1 | 0.991277 | 1 | 9744 | 72 |
| 425 | unconditional | 0.400186 | 0.839229 | 1 | 0.381527 | 1 | 8120 | 60 |
| 450 | conditional | 0.00556625 | 0.998383 | 0.989583 | 0.994356 | 0.958333 | 9744 | 72 |
| 450 | unconditional | 0.397402 | 0.843934 | 1 | 0.397044 | 1 | 8120 | 60 |
| 475 | conditional | 0.0166447 | 0.994807 | 1 | 0.980603 | 1 | 9744 | 72 |
| 475 | unconditional | 0.383293 | 0.85134 | 1 | 0.422291 | 1 | 8120 | 60 |
| 500 | conditional | 0.0124598 | 0.995612 | 1 | 0.983272 | 1 | 9744 | 72 |
| 500 | unconditional | 0.411465 | 0.837688 | 1 | 0.369089 | 1 | 8120 | 60 |

人工数据 manifest：d40bbbbc1c4294eed843d88bcdd0cf193886de6c2d7e9db9069bbf86e931c022；精确 target sequence 重叠=0；精确 record 重叠=0。

跨 radius32 覆盖：{&quot;possible&quot;: true, &quot;train&quot;: true, &quot;valid&quot;: true}。生成规则共享是诊断设计的一部分，不能当作真实下游去重证书。


按任务/链长分组（全局 level 按 token 计数汇总，下面保留各组计数；不同组不得直接比较未经加权的总 loss）：

| 时点 | group/length | mask level | loss | AA acc | end acc | selected count | AA count | end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | ab\_antigen/long | mask25 | 4.24973 | 0.0190114 | 0 | 530 | 526 | 4 |
| initial | ab\_antigen/long | mask50 | 4.38742 | 0.0227048 | 0 | 1019 | 1013 | 6 |
| initial | ab\_antigen/long | mask75 | 4.39272 | 0.0267624 | 0 | 1538 | 1532 | 6 |
| initial | ab\_antigen/long | mask100 | 4.27875 | 0.0264706 | 0 | 2048 | 2040 | 8 |
| initial | ab\_antigen/long | mask\_mean | 4.32715 | 0.0237373 | 0 | — | — | — |
| initial | ab\_antigen/medium | mask25 | 4.25892 | 0.00823045 | 0 | 246 | 243 | 3 |
| initial | ab\_antigen/medium | mask50 | 4.34305 | 0.0159363 | 0 | 507 | 502 | 5 |
| initial | ab\_antigen/medium | mask75 | 4.4131 | 0.0422164 | 0 | 765 | 758 | 7 |
| initial | ab\_antigen/medium | mask100 | 4.39726 | 0.0305118 | 0 | 1024 | 1016 | 8 |
| initial | ab\_antigen/medium | mask\_mean | 4.35308 | 0.0242237 | 0 | — | — | — |
| initial | ab\_antigen/short | mask25 | 4.42935 | 0.0588235 | 0 | 38 | 34 | 4 |
| initial | ab\_antigen/short | mask50 | 4.26535 | 0.0340909 | 0 | 93 | 88 | 5 |
| initial | ab\_antigen/short | mask75 | 4.24109 | 0.0428571 | 0 | 145 | 140 | 5 |
| initial | ab\_antigen/short | mask100 | 4.20677 | 0.046875 | 0 | 200 | 192 | 8 |
| initial | ab\_antigen/short | mask\_mean | 4.28564 | 0.0456616 | 0 | — | — | — |
| initial | tcr\_peptide/long | mask25 | 3.75574 | 0.0473118 | 0 | 466 | 465 | 1 |
| initial | tcr\_peptide/long | mask50 | 3.86912 | 0.0417941 | 0 | 985 | 981 | 4 |
| initial | tcr\_peptide/long | mask75 | 3.97539 | 0.0575101 | 0 | 1484 | 1478 | 6 |
| initial | tcr\_peptide/long | mask100 | 4.05611 | 0.0622549 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_peptide/long | mask\_mean | 3.91409 | 0.0522177 | 0 | — | — | — |
| initial | tcr\_peptide/medium | mask25 | 3.83075 | 0.0290456 | 0 | 242 | 241 | 1 |
| initial | tcr\_peptide/medium | mask50 | 3.90211 | 0.0239044 | 0 | 506 | 502 | 4 |
| initial | tcr\_peptide/medium | mask75 | 3.94099 | 0.0276316 | 0 | 767 | 760 | 7 |
| initial | tcr\_peptide/medium | mask100 | 3.89363 | 0.0305118 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_peptide/medium | mask\_mean | 3.89187 | 0.0277734 | 0 | — | — | — |
| initial | tcr\_peptide/short | mask25 | 3.7955 | 0.0625 | 0 | 51 | 48 | 3 |
| initial | tcr\_peptide/short | mask50 | 3.6999 | 0.0480769 | 0 | 110 | 104 | 6 |
| initial | tcr\_peptide/short | mask75 | 3.76546 | 0.0510949 | 0 | 144 | 137 | 7 |
| initial | tcr\_peptide/short | mask100 | 3.7947 | 0.046875 | 0 | 200 | 192 | 8 |
| initial | tcr\_peptide/short | mask\_mean | 3.76389 | 0.0521367 | 0 | — | — | — |
| initial | tcr\_pmhc/long | mask25 | 3.64688 | 0.0317164 | 0 | 540 | 536 | 4 |
| initial | tcr\_pmhc/long | mask50 | 3.87128 | 0.0135397 | 0 | 1038 | 1034 | 4 |
| initial | tcr\_pmhc/long | mask75 | 4.05111 | 0.0135135 | 0 | 1562 | 1554 | 8 |
| initial | tcr\_pmhc/long | mask100 | 4.16231 | 0.0156863 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_pmhc/long | mask\_mean | 3.9329 | 0.018614 | 0 | — | — | — |
| initial | tcr\_pmhc/medium | mask25 | 3.91128 | 0.0371901 | 0 | 244 | 242 | 2 |
| initial | tcr\_pmhc/medium | mask50 | 4.13464 | 0.0375783 | 0 | 484 | 479 | 5 |
| initial | tcr\_pmhc/medium | mask75 | 4.33153 | 0.0351105 | 0 | 777 | 769 | 8 |
| initial | tcr\_pmhc/medium | mask100 | 4.36849 | 0.0314961 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_pmhc/medium | mask\_mean | 4.18648 | 0.0353437 | 0 | — | — | — |
| initial | tcr\_pmhc/short | mask25 | 4.16854 | 0.0416667 | 0 | 51 | 48 | 3 |
| initial | tcr\_pmhc/short | mask50 | 4.28146 | 0.0361446 | 0 | 86 | 83 | 3 |
| initial | tcr\_pmhc/short | mask75 | 4.21468 | 0.0408163 | 0 | 152 | 147 | 5 |
| initial | tcr\_pmhc/short | mask100 | 4.19548 | 0.03125 | 0 | 200 | 192 | 8 |
| initial | tcr\_pmhc/short | mask\_mean | 4.21504 | 0.0374694 | 0 | — | — | — |
| initial | unconditional\_ab/long | mask25 | 4.01602 | 0.0429907 | 0 | 538 | 535 | 3 |
| initial | unconditional\_ab/long | mask50 | 4.10145 | 0.0650096 | 0 | 1050 | 1046 | 4 |
| initial | unconditional\_ab/long | mask75 | 4.15189 | 0.0591678 | 0 | 1545 | 1538 | 7 |
| initial | unconditional\_ab/long | mask100 | 4.10116 | 0.0313726 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_ab/long | mask\_mean | 4.09263 | 0.0496351 | 0 | — | — | — |
| initial | unconditional\_ab/medium | mask25 | 3.9054 | 0.083004 | 0 | 256 | 253 | 3 |
| initial | unconditional\_ab/medium | mask50 | 4.09812 | 0.0826772 | 0 | 512 | 508 | 4 |
| initial | unconditional\_ab/medium | mask75 | 4.16791 | 0.0813648 | 0 | 768 | 762 | 6 |
| initial | unconditional\_ab/medium | mask100 | 4.17491 | 0.0787402 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_ab/medium | mask\_mean | 4.08659 | 0.0814465 | 0 | — | — | — |
| initial | unconditional\_ab/short | mask25 | 3.8656 | 0.0384615 | 0 | 55 | 52 | 3 |
| initial | unconditional\_ab/short | mask50 | 3.90127 | 0.0408163 | 0 | 103 | 98 | 5 |
| initial | unconditional\_ab/short | mask75 | 3.87111 | 0.0212766 | 0 | 148 | 141 | 7 |
| initial | unconditional\_ab/short | mask100 | 3.84389 | 0.03125 | 0 | 200 | 192 | 8 |
| initial | unconditional\_ab/short | mask\_mean | 3.87047 | 0.0329511 | 0 | — | — | — |
| initial | unconditional\_single/long | mask25 | 3.41248 | 0.0485075 | 0 | 269 | 268 | 1 |
| initial | unconditional\_single/long | mask50 | 3.39562 | 0.0761905 | 0 | 527 | 525 | 2 |
| initial | unconditional\_single/long | mask75 | 3.36794 | 0.193798 | 0 | 778 | 774 | 4 |
| initial | unconditional\_single/long | mask100 | 3.38543 | 0 | 0 | 1024 | 1020 | 4 |
| initial | unconditional\_single/long | mask\_mean | 3.39037 | 0.0796241 | 0 | — | — | — |
| initial | unconditional\_single/medium | mask25 | 3.7779 | 0.0366972 | 0 | 110 | 109 | 1 |
| initial | unconditional\_single/medium | mask50 | 3.60855 | 0.0258621 | 0 | 234 | 232 | 2 |
| initial | unconditional\_single/medium | mask75 | 3.57466 | 0.0435967 | 0 | 371 | 367 | 4 |
| initial | unconditional\_single/medium | mask100 | 3.51609 | 0 | 0 | 512 | 508 | 4 |
| initial | unconditional\_single/medium | mask\_mean | 3.6193 | 0.026539 | 0 | — | — | — |
| initial | unconditional\_single/short | mask25 | 3.17399 | 0.230769 | 0 | 28 | 26 | 2 |
| initial | unconditional\_single/short | mask50 | 3.39984 | 0 | 0 | 52 | 50 | 2 |
| initial | unconditional\_single/short | mask75 | 3.17481 | 0 | 0 | 80 | 77 | 3 |
| initial | unconditional\_single/short | mask100 | 3.12259 | 0 | 0 | 100 | 96 | 4 |
| initial | unconditional\_single/short | mask\_mean | 3.21781 | 0.0576923 | 0 | — | — | — |
| initial | unconditional\_tcr/long | mask25 | 3.89262 | 0.0213592 | 0 | 516 | 515 | 1 |
| initial | unconditional\_tcr/long | mask50 | 3.99228 | 0.0192503 | 0 | 989 | 987 | 2 |
| initial | unconditional\_tcr/long | mask75 | 4.04526 | 0.0151117 | 0 | 1527 | 1522 | 5 |
| initial | unconditional\_tcr/long | mask100 | 4.0541 | 0.0156863 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_tcr/long | mask\_mean | 3.99607 | 0.0178519 | 0 | — | — | — |
| initial | unconditional\_tcr/medium | mask25 | 3.83404 | 0 | — | 247 | 247 | 0 |
| initial | unconditional\_tcr/medium | mask50 | 4.08607 | 0 | 0 | 500 | 497 | 3 |
| initial | unconditional\_tcr/medium | mask75 | 4.12407 | 0 | 0 | 744 | 738 | 6 |
| initial | unconditional\_tcr/medium | mask100 | 4.11728 | 0 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_tcr/medium | mask\_mean | 4.04036 | 0 | 0 | — | — | — |
| initial | unconditional\_tcr/short | mask25 | 4.06521 | 0.025 | — | 40 | 40 | 0 |
| initial | unconditional\_tcr/short | mask50 | 4.00589 | 0.0333333 | 0 | 93 | 90 | 3 |
| initial | unconditional\_tcr/short | mask75 | 4.09862 | 0.0441176 | 0 | 142 | 136 | 6 |
| initial | unconditional\_tcr/short | mask100 | 4.03079 | 0.046875 | 0 | 200 | 192 | 8 |
| initial | unconditional\_tcr/short | mask\_mean | 4.05013 | 0.0373315 | 0 | — | — | — |
| latest | ab\_antigen/long | mask25 | 4.91526e-05 | 1 | 1 | 530 | 526 | 4 |
| latest | ab\_antigen/long | mask50 | 2.30262e-05 | 1 | 1 | 1019 | 1013 | 6 |
| latest | ab\_antigen/long | mask75 | 1.28021e-05 | 1 | 1 | 1538 | 1532 | 6 |
| latest | ab\_antigen/long | mask100 | 0.0952125 | 0.959804 | 1 | 2048 | 2040 | 8 |
| latest | ab\_antigen/long | mask\_mean | 0.0238244 | 0.989951 | 1 | — | — | — |
| latest | ab\_antigen/medium | mask25 | 9.48125e-05 | 1 | 1 | 246 | 243 | 3 |
| latest | ab\_antigen/medium | mask50 | 4.30706e-05 | 1 | 1 | 507 | 502 | 5 |
| latest | ab\_antigen/medium | mask75 | 3.43583e-05 | 1 | 1 | 765 | 758 | 7 |
| latest | ab\_antigen/medium | mask100 | 0.000691348 | 1 | 1 | 1024 | 1016 | 8 |
| latest | ab\_antigen/medium | mask\_mean | 0.000215897 | 1 | 1 | — | — | — |
| latest | ab\_antigen/short | mask25 | 8.03073e-06 | 1 | 1 | 38 | 34 | 4 |
| latest | ab\_antigen/short | mask50 | 1.18421e-05 | 1 | 1 | 93 | 88 | 5 |
| latest | ab\_antigen/short | mask75 | 0.00100871 | 1 | 1 | 145 | 140 | 5 |
| latest | ab\_antigen/short | mask100 | 0.00107817 | 1 | 1 | 200 | 192 | 8 |
| latest | ab\_antigen/short | mask\_mean | 0.000526686 | 1 | 1 | — | — | — |
| latest | tcr\_peptide/long | mask25 | 2.39428e-05 | 1 | 1 | 466 | 465 | 1 |
| latest | tcr\_peptide/long | mask50 | 1.18537e-05 | 1 | 1 | 985 | 981 | 4 |
| latest | tcr\_peptide/long | mask75 | 8.23629e-06 | 1 | 1 | 1484 | 1478 | 6 |
| latest | tcr\_peptide/long | mask100 | 0.0858669 | 0.971569 | 1 | 2048 | 2040 | 8 |
| latest | tcr\_peptide/long | mask\_mean | 0.0214777 | 0.992892 | 1 | — | — | — |
| latest | tcr\_peptide/medium | mask25 | 2.36292e-05 | 1 | 1 | 242 | 241 | 1 |
| latest | tcr\_peptide/medium | mask50 | 3.34135e-05 | 1 | 1 | 506 | 502 | 4 |
| latest | tcr\_peptide/medium | mask75 | 1.67326e-05 | 1 | 1 | 767 | 760 | 7 |
| latest | tcr\_peptide/medium | mask100 | 0.010229 | 0.996063 | 1 | 1024 | 1016 | 8 |
| latest | tcr\_peptide/medium | mask\_mean | 0.0025757 | 0.999016 | 1 | — | — | — |
| latest | tcr\_peptide/short | mask25 | 3.60663e-06 | 1 | 1 | 51 | 48 | 3 |
| latest | tcr\_peptide/short | mask50 | 1.16732e-05 | 1 | 1 | 110 | 104 | 6 |
| latest | tcr\_peptide/short | mask75 | 1.16703e-05 | 1 | 1 | 144 | 137 | 7 |
| latest | tcr\_peptide/short | mask100 | 0.000188329 | 1 | 1 | 200 | 192 | 8 |
| latest | tcr\_peptide/short | mask\_mean | 5.38198e-05 | 1 | 1 | — | — | — |
| latest | tcr\_pmhc/long | mask25 | 3.85402e-05 | 1 | 1 | 540 | 536 | 4 |
| latest | tcr\_pmhc/long | mask50 | 1.0206e-05 | 1 | 1 | 1038 | 1034 | 4 |
| latest | tcr\_pmhc/long | mask75 | 9.86043e-06 | 1 | 1 | 1562 | 1554 | 8 |
| latest | tcr\_pmhc/long | mask100 | 0.0297225 | 0.994608 | 1 | 2048 | 2040 | 8 |
| latest | tcr\_pmhc/long | mask\_mean | 0.00744528 | 0.998652 | 1 | — | — | — |
| latest | tcr\_pmhc/medium | mask25 | 4.40697e-05 | 1 | 1 | 244 | 242 | 2 |
| latest | tcr\_pmhc/medium | mask50 | 2.82502e-05 | 1 | 1 | 484 | 479 | 5 |
| latest | tcr\_pmhc/medium | mask75 | 2.25475e-05 | 1 | 1 | 777 | 769 | 8 |
| latest | tcr\_pmhc/medium | mask100 | 0.00529588 | 0.999016 | 1 | 1024 | 1016 | 8 |
| latest | tcr\_pmhc/medium | mask\_mean | 0.00134769 | 0.999754 | 1 | — | — | — |
| latest | tcr\_pmhc/short | mask25 | 1.04972e-05 | 1 | 1 | 51 | 48 | 3 |
| latest | tcr\_pmhc/short | mask50 | 2.09086e-05 | 1 | 1 | 86 | 83 | 3 |
| latest | tcr\_pmhc/short | mask75 | 0.119244 | 0.959184 | 1 | 152 | 147 | 5 |
| latest | tcr\_pmhc/short | mask100 | 0.077162 | 0.963542 | 1 | 200 | 192 | 8 |
| latest | tcr\_pmhc/short | mask\_mean | 0.0491094 | 0.980681 | 1 | — | — | — |
| latest | unconditional\_ab/long | mask25 | 3.0646e-05 | 1 | 1 | 538 | 535 | 3 |
| latest | unconditional\_ab/long | mask50 | 1.28292e-05 | 1 | 1 | 1050 | 1046 | 4 |
| latest | unconditional\_ab/long | mask75 | 0.000150123 | 1 | 1 | 1545 | 1538 | 7 |
| latest | unconditional\_ab/long | mask100 | 1.58222 | 0.359314 | 1 | 2048 | 2040 | 8 |
| latest | unconditional\_ab/long | mask\_mean | 0.395602 | 0.839828 | 1 | — | — | — |
| latest | unconditional\_ab/medium | mask25 | 4.8635e-05 | 1 | 1 | 256 | 253 | 3 |
| latest | unconditional\_ab/medium | mask50 | 1.59495e-05 | 1 | 1 | 512 | 508 | 4 |
| latest | unconditional\_ab/medium | mask75 | 1.69072e-05 | 1 | 1 | 768 | 762 | 6 |
| latest | unconditional\_ab/medium | mask100 | 1.62287 | 0.329724 | 1 | 1024 | 1016 | 8 |
| latest | unconditional\_ab/medium | mask\_mean | 0.405739 | 0.832431 | 1 | — | — | — |
| latest | unconditional\_ab/short | mask25 | 0.00017961 | 1 | 1 | 55 | 52 | 3 |
| latest | unconditional\_ab/short | mask50 | 0.191826 | 0.918367 | 1 | 103 | 98 | 5 |
| latest | unconditional\_ab/short | mask75 | 0.406722 | 0.87234 | 1 | 148 | 141 | 7 |
| latest | unconditional\_ab/short | mask100 | 1.17948 | 0.46875 | 1 | 200 | 192 | 8 |
| latest | unconditional\_ab/short | mask\_mean | 0.444551 | 0.814864 | 1 | — | — | — |
| latest | unconditional\_single/long | mask25 | 3.14672e-05 | 1 | 1 | 269 | 268 | 1 |
| latest | unconditional\_single/long | mask50 | 2.13538e-05 | 1 | 1 | 527 | 525 | 2 |
| latest | unconditional\_single/long | mask75 | 0.139271 | 0.958656 | 1 | 778 | 774 | 4 |
| latest | unconditional\_single/long | mask100 | 1.63658 | 0.377451 | 1 | 1024 | 1020 | 4 |
| latest | unconditional\_single/long | mask\_mean | 0.443977 | 0.834027 | 1 | — | — | — |
| latest | unconditional\_single/medium | mask25 | 3.07348e-05 | 1 | 1 | 110 | 109 | 1 |
| latest | unconditional\_single/medium | mask50 | 1.14935e-05 | 1 | 1 | 234 | 232 | 2 |
| latest | unconditional\_single/medium | mask75 | 0.000161109 | 1 | 1 | 371 | 367 | 4 |
| latest | unconditional\_single/medium | mask100 | 1.70586 | 0.372047 | 1 | 512 | 508 | 4 |
| latest | unconditional\_single/medium | mask\_mean | 0.426515 | 0.843012 | 1 | — | — | — |
| latest | unconditional\_single/short | mask25 | 0.000522205 | 1 | 1 | 28 | 26 | 2 |
| latest | unconditional\_single/short | mask50 | 0.3198 | 0.84 | 1 | 52 | 50 | 2 |
| latest | unconditional\_single/short | mask75 | 0.932685 | 0.597403 | 1 | 80 | 77 | 3 |
| latest | unconditional\_single/short | mask100 | 1.33621 | 0.385417 | 1 | 100 | 96 | 4 |
| latest | unconditional\_single/short | mask\_mean | 0.647306 | 0.705705 | 1 | — | — | — |
| latest | unconditional\_tcr/long | mask25 | 4.53771e-05 | 1 | 1 | 516 | 515 | 1 |
| latest | unconditional\_tcr/long | mask50 | 1.7594e-05 | 1 | 1 | 989 | 987 | 2 |
| latest | unconditional\_tcr/long | mask75 | 2.00529e-05 | 1 | 1 | 1527 | 1522 | 5 |
| latest | unconditional\_tcr/long | mask100 | 1.58873 | 0.380392 | 1 | 2048 | 2040 | 8 |
| latest | unconditional\_tcr/long | mask\_mean | 0.397203 | 0.845098 | 1 | — | — | — |
| latest | unconditional\_tcr/medium | mask25 | 5.67773e-05 | 1 | — | 247 | 247 | 0 |
| latest | unconditional\_tcr/medium | mask50 | 4.13938e-05 | 1 | 1 | 500 | 497 | 3 |
| latest | unconditional\_tcr/medium | mask75 | 0.000121982 | 1 | 1 | 744 | 738 | 6 |
| latest | unconditional\_tcr/medium | mask100 | 1.65625 | 0.350394 | 1 | 1024 | 1016 | 8 |
| latest | unconditional\_tcr/medium | mask\_mean | 0.414119 | 0.837598 | 1 | — | — | — |
| latest | unconditional\_tcr/short | mask25 | 7.00943e-06 | 1 | — | 40 | 40 | 0 |
| latest | unconditional\_tcr/short | mask50 | 0.000612442 | 1 | 1 | 93 | 90 | 3 |
| latest | unconditional\_tcr/short | mask75 | 0.195577 | 0.955882 | 1 | 142 | 136 | 6 |
| latest | unconditional\_tcr/short | mask100 | 1.27374 | 0.5 | 1 | 200 | 192 | 8 |
| latest | unconditional\_tcr/short | mask\_mean | 0.367484 | 0.863971 | 1 | — | — | — |
| train\_probe | ab\_antigen/long | mask25 | 4.07638e-05 | 1 | — | 530 | 530 | 0 |
| train\_probe | ab\_antigen/long | mask50 | 3.11925e-05 | 1 | 1 | 1034 | 1031 | 3 |
| train\_probe | ab\_antigen/long | mask75 | 1.67684e-05 | 1 | 1 | 1563 | 1557 | 6 |
| train\_probe | ab\_antigen/long | mask100 | 0.0129114 | 0.996569 | 1 | 2048 | 2040 | 8 |
| train\_probe | ab\_antigen/long | mask\_mean | 0.00325002 | 0.999142 | 1 | — | — | — |
| train\_probe | ab\_antigen/medium | mask25 | 0.00012438 | 1 | 1 | 275 | 273 | 2 |
| train\_probe | ab\_antigen/medium | mask50 | 2.08289e-05 | 1 | 1 | 542 | 538 | 4 |
| train\_probe | ab\_antigen/medium | mask75 | 2.00767e-05 | 1 | 1 | 781 | 776 | 5 |
| train\_probe | ab\_antigen/medium | mask100 | 0.000358949 | 1 | 1 | 1024 | 1016 | 8 |
| train\_probe | ab\_antigen/medium | mask\_mean | 0.000131059 | 1 | 1 | — | — | — |
| train\_probe | ab\_antigen/short | mask25 | 2.17716e-05 | 1 | 1 | 52 | 47 | 5 |
| train\_probe | ab\_antigen/short | mask50 | 3.65031e-05 | 1 | 1 | 109 | 102 | 7 |
| train\_probe | ab\_antigen/short | mask75 | 0.000273015 | 1 | 1 | 155 | 148 | 7 |
| train\_probe | ab\_antigen/short | mask100 | 0.00096703 | 1 | 1 | 200 | 192 | 8 |
| train\_probe | ab\_antigen/short | mask\_mean | 0.00032458 | 1 | 1 | — | — | — |
| train\_probe | tcr\_peptide/long | mask25 | 2.49002e-05 | 1 | 1 | 495 | 494 | 1 |
| train\_probe | tcr\_peptide/long | mask50 | 1.34017e-05 | 1 | 1 | 1033 | 1028 | 5 |
| train\_probe | tcr\_peptide/long | mask75 | 9.27218e-06 | 1 | 1 | 1530 | 1522 | 8 |
| train\_probe | tcr\_peptide/long | mask100 | 0.0700946 | 0.973529 | 1 | 2048 | 2040 | 8 |
| train\_probe | tcr\_peptide/long | mask\_mean | 0.0175355 | 0.993382 | 1 | — | — | — |
| train\_probe | tcr\_peptide/medium | mask25 | 2.19536e-05 | 1 | — | 252 | 252 | 0 |
| train\_probe | tcr\_peptide/medium | mask50 | 1.47882e-05 | 1 | 1 | 517 | 513 | 4 |
| train\_probe | tcr\_peptide/medium | mask75 | 1.01752e-05 | 1 | 1 | 760 | 754 | 6 |
| train\_probe | tcr\_peptide/medium | mask100 | 0.0216118 | 0.991142 | 1 | 1024 | 1016 | 8 |
| train\_probe | tcr\_peptide/medium | mask\_mean | 0.00541468 | 0.997785 | 1 | — | — | — |
| train\_probe | tcr\_peptide/short | mask25 | 1.08429e-05 | 1 | 1 | 51 | 46 | 5 |
| train\_probe | tcr\_peptide/short | mask50 | 7.59778e-06 | 1 | 1 | 95 | 90 | 5 |
| train\_probe | tcr\_peptide/short | mask75 | 0.000174454 | 1 | 1 | 142 | 136 | 6 |
| train\_probe | tcr\_peptide/short | mask100 | 0.00513299 | 1 | 1 | 200 | 192 | 8 |
| train\_probe | tcr\_peptide/short | mask\_mean | 0.00133147 | 1 | 1 | — | — | — |
| train\_probe | tcr\_pmhc/long | mask25 | 2.18232e-05 | 1 | 1 | 526 | 522 | 4 |
| train\_probe | tcr\_pmhc/long | mask50 | 1.25806e-05 | 1 | 1 | 1036 | 1031 | 5 |
| train\_probe | tcr\_pmhc/long | mask75 | 8.03774e-06 | 1 | 1 | 1557 | 1551 | 6 |
| train\_probe | tcr\_pmhc/long | mask100 | 0.0306987 | 0.990686 | 1 | 2048 | 2040 | 8 |
| train\_probe | tcr\_pmhc/long | mask\_mean | 0.00768529 | 0.997672 | 1 | — | — | — |
| train\_probe | tcr\_pmhc/medium | mask25 | 4.29372e-05 | 1 | 1 | 276 | 273 | 3 |
| train\_probe | tcr\_pmhc/medium | mask50 | 2.19371e-05 | 1 | 1 | 549 | 541 | 8 |
| train\_probe | tcr\_pmhc/medium | mask75 | 2.34293e-05 | 1 | 1 | 793 | 785 | 8 |
| train\_probe | tcr\_pmhc/medium | mask100 | 0.0029552 | 0.999016 | 1 | 1024 | 1016 | 8 |
| train\_probe | tcr\_pmhc/medium | mask\_mean | 0.000760877 | 0.999754 | 1 | — | — | — |
| train\_probe | tcr\_pmhc/short | mask25 | 2.6606e-05 | 1 | 1 | 50 | 48 | 2 |
| train\_probe | tcr\_pmhc/short | mask50 | 1.80635e-05 | 1 | 1 | 103 | 100 | 3 |
| train\_probe | tcr\_pmhc/short | mask75 | 0.000190211 | 1 | 1 | 152 | 145 | 7 |
| train\_probe | tcr\_pmhc/short | mask100 | 0.0111984 | 0.994792 | 1 | 200 | 192 | 8 |
| train\_probe | tcr\_pmhc/short | mask\_mean | 0.00285833 | 0.998698 | 1 | — | — | — |
| train\_probe | unconditional\_ab/long | mask25 | 0.000114762 | 1 | 1 | 509 | 506 | 3 |
| train\_probe | unconditional\_ab/long | mask50 | 3.85198e-05 | 1 | 1 | 1016 | 1010 | 6 |
| train\_probe | unconditional\_ab/long | mask75 | 1.73193e-05 | 1 | 1 | 1569 | 1561 | 8 |
| train\_probe | unconditional\_ab/long | mask100 | 1.40955 | 0.410294 | 1 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_ab/long | mask\_mean | 0.352429 | 0.852574 | 1 | — | — | — |
| train\_probe | unconditional\_ab/medium | mask25 | 4.5213e-05 | 1 | 1 | 243 | 241 | 2 |
| train\_probe | unconditional\_ab/medium | mask50 | 2.89406e-05 | 1 | 1 | 521 | 516 | 5 |
| train\_probe | unconditional\_ab/medium | mask75 | 4.12352e-05 | 1 | 1 | 766 | 761 | 5 |
| train\_probe | unconditional\_ab/medium | mask100 | 1.7248 | 0.316929 | 1 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_ab/medium | mask\_mean | 0.43123 | 0.829232 | 1 | — | — | — |
| train\_probe | unconditional\_ab/short | mask25 | 2.70378e-05 | 1 | 1 | 53 | 52 | 1 |
| train\_probe | unconditional\_ab/short | mask50 | 0.227015 | 0.942308 | 1 | 106 | 104 | 2 |
| train\_probe | unconditional\_ab/short | mask75 | 0.583921 | 0.740506 | 1 | 163 | 158 | 5 |
| train\_probe | unconditional\_ab/short | mask100 | 1.35713 | 0.385417 | 1 | 200 | 192 | 8 |
| train\_probe | unconditional\_ab/short | mask\_mean | 0.542024 | 0.767058 | 1 | — | — | — |
| train\_probe | unconditional\_single/long | mask25 | 8.19369e-05 | 1 | 1 | 245 | 244 | 1 |
| train\_probe | unconditional\_single/long | mask50 | 4.36183e-05 | 1 | 1 | 492 | 491 | 1 |
| train\_probe | unconditional\_single/long | mask75 | 0.00901483 | 0.997396 | 1 | 771 | 768 | 3 |
| train\_probe | unconditional\_single/long | mask100 | 1.52727 | 0.371569 | 1 | 1024 | 1020 | 4 |
| train\_probe | unconditional\_single/long | mask\_mean | 0.384102 | 0.842241 | 1 | — | — | — |
| train\_probe | unconditional\_single/medium | mask25 | 2.66741e-05 | 1 | 1 | 132 | 130 | 2 |
| train\_probe | unconditional\_single/medium | mask50 | 6.06584e-05 | 1 | 1 | 255 | 252 | 3 |
| train\_probe | unconditional\_single/medium | mask75 | 3.92132e-05 | 1 | 1 | 382 | 379 | 3 |
| train\_probe | unconditional\_single/medium | mask100 | 1.59947 | 0.340551 | 1 | 512 | 508 | 4 |
| train\_probe | unconditional\_single/medium | mask\_mean | 0.3999 | 0.835138 | 1 | — | — | — |
| train\_probe | unconditional\_single/short | mask25 | 0.00057006 | 1 | 1 | 28 | 26 | 2 |
| train\_probe | unconditional\_single/short | mask50 | 0.291224 | 0.877551 | 1 | 52 | 49 | 3 |
| train\_probe | unconditional\_single/short | mask75 | 0.791696 | 0.628571 | 1 | 73 | 70 | 3 |
| train\_probe | unconditional\_single/short | mask100 | 1.17184 | 0.46875 | 1 | 100 | 96 | 4 |
| train\_probe | unconditional\_single/short | mask\_mean | 0.563832 | 0.743718 | 1 | — | — | — |
| train\_probe | unconditional\_tcr/long | mask25 | 3.65491e-05 | 1 | — | 525 | 525 | 0 |
| train\_probe | unconditional\_tcr/long | mask50 | 1.48932e-05 | 1 | 1 | 1048 | 1045 | 3 |
| train\_probe | unconditional\_tcr/long | mask75 | 6.34137e-06 | 1 | 1 | 1585 | 1580 | 5 |
| train\_probe | unconditional\_tcr/long | mask100 | 1.59399 | 0.4 | 1 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_tcr/long | mask\_mean | 0.398513 | 0.85 | 1 | — | — | — |
| train\_probe | unconditional\_tcr/medium | mask25 | 1.28562e-05 | 1 | — | 261 | 261 | 0 |
| train\_probe | unconditional\_tcr/medium | mask50 | 7.92612e-06 | 1 | 1 | 498 | 497 | 1 |
| train\_probe | unconditional\_tcr/medium | mask75 | 1.257e-05 | 1 | 1 | 755 | 751 | 4 |
| train\_probe | unconditional\_tcr/medium | mask100 | 1.49134 | 0.389764 | 1 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_tcr/medium | mask\_mean | 0.372843 | 0.847441 | 1 | — | — | — |
| train\_probe | unconditional\_tcr/short | mask25 | 2.74881e-05 | 1 | — | 46 | 46 | 0 |
| train\_probe | unconditional\_tcr/short | mask50 | 0.24176 | 0.934783 | 1 | 97 | 92 | 5 |
| train\_probe | unconditional\_tcr/short | mask75 | 0.488564 | 0.863014 | 1 | 152 | 146 | 6 |
| train\_probe | unconditional\_tcr/short | mask100 | 1.40655 | 0.3125 | 1 | 200 | 192 | 8 |
| train\_probe | unconditional\_tcr/short | mask\_mean | 0.534226 | 0.777574 | 1 | — | — | — |

### 1790803305071555086\_r27\_refined\_value\_residual\_52f013e5：分层 mask 指标

源码目录：/vepfs-mlp2/c20250601/251105016/project/dllm\_test/output/immune\_research\_releases/r20\_r27\_20261001；完成/目标 step：500/500。

kind=synthetic\_generalization\_diagnostic；dataset_kind=deterministic\_artificial\_sequences；biological_quality_claim=False。

| 时点 | mask level | 指标 | 值 |
| --- | --- | --- | --- |
| initial | mask25 | mask\_ratio | 0.25 |
| initial | mask25 | selected\_count | 4467 |
| initial | mask25 | aa\_count | 4428 |
| initial | mask25 | terminal\_count | 39 |
| initial | mask25 | loss | 3.90573 |
| initial | mask25 | aa\_loss | 3.89496 |
| initial | mask25 | aa\_accuracy | 0.0643631 |
| initial | mask25 | terminal\_loss | 5.12936 |
| initial | mask25 | terminal\_accuracy | 0 |
| initial | mask50 | mask\_ratio | 0.5 |
| initial | mask50 | selected\_count | 8888 |
| initial | mask50 | aa\_count | 8819 |
| initial | mask50 | terminal\_count | 69 |
| initial | mask50 | loss | 4.11095 |
| initial | mask50 | aa\_loss | 4.10142 |
| initial | mask50 | aa\_accuracy | 0.0666742 |
| initial | mask50 | terminal\_loss | 5.32885 |
| initial | mask50 | terminal\_accuracy | 0 |
| initial | mask75 | mask\_ratio | 0.75 |
| initial | mask75 | selected\_count | 13437 |
| initial | mask75 | aa\_count | 13330 |
| initial | mask75 | terminal\_count | 107 |
| initial | mask75 | loss | 4.25822 |
| initial | mask75 | aa\_loss | 4.24812 |
| initial | mask75 | aa\_accuracy | 0.0588897 |
| initial | mask75 | terminal\_loss | 5.51605 |
| initial | mask75 | terminal\_accuracy | 0 |
| initial | mask100 | mask\_ratio | 1 |
| initial | mask100 | selected\_count | 17996 |
| initial | mask100 | aa\_count | 17864 |
| initial | mask100 | terminal\_count | 132 |
| initial | mask100 | loss | 4.24321 |
| initial | mask100 | aa\_loss | 4.23335 |
| initial | mask100 | aa\_accuracy | 0.0667824 |
| initial | mask100 | terminal\_loss | 5.57776 |
| initial | mask100 | terminal\_accuracy | 0 |
| initial | mask\_mean | loss | 4.12953 |
| initial | mask\_mean | aa\_loss | 4.11946 |
| initial | mask\_mean | terminal\_loss | 5.388 |
| initial | mask\_mean | aa\_accuracy | 0.0641774 |
| initial | mask\_mean | terminal\_accuracy | 0 |
| latest | mask25 | mask\_ratio | 0.25 |
| latest | mask25 | selected\_count | 4467 |
| latest | mask25 | aa\_count | 4428 |
| latest | mask25 | terminal\_count | 39 |
| latest | mask25 | loss | 4.30967e-05 |
| latest | mask25 | aa\_loss | 3.97885e-05 |
| latest | mask25 | aa\_accuracy | 1 |
| latest | mask25 | terminal\_loss | 0.000418704 |
| latest | mask25 | terminal\_accuracy | 1 |
| latest | mask50 | mask\_ratio | 0.5 |
| latest | mask50 | selected\_count | 8888 |
| latest | mask50 | aa\_count | 8819 |
| latest | mask50 | terminal\_count | 69 |
| latest | mask50 | loss | 0.00393787 |
| latest | mask50 | aa\_loss | 0.00396046 |
| latest | mask50 | aa\_accuracy | 0.998639 |
| latest | mask50 | terminal\_loss | 0.00105112 |
| latest | mask50 | terminal\_accuracy | 1 |
| latest | mask75 | mask\_ratio | 0.75 |
| latest | mask75 | selected\_count | 13437 |
| latest | mask75 | aa\_count | 13330 |
| latest | mask75 | terminal\_count | 107 |
| latest | mask75 | loss | 0.0187197 |
| latest | mask75 | aa\_loss | 0.0188594 |
| latest | mask75 | aa\_accuracy | 0.993548 |
| latest | mask75 | terminal\_loss | 0.00131372 |
| latest | mask75 | terminal\_accuracy | 1 |
| latest | mask100 | mask\_ratio | 1 |
| latest | mask100 | selected\_count | 17996 |
| latest | mask100 | aa\_count | 17864 |
| latest | mask100 | terminal\_count | 132 |
| latest | mask100 | loss | 0.74735 |
| latest | mask100 | aa\_loss | 0.752513 |
| latest | mask100 | aa\_accuracy | 0.705889 |
| latest | mask100 | terminal\_loss | 0.0486353 |
| latest | mask100 | terminal\_accuracy | 0.969697 |
| latest | mask\_mean | loss | 0.192513 |
| latest | mask\_mean | aa\_loss | 0.193843 |
| latest | mask\_mean | terminal\_loss | 0.0128547 |
| latest | mask\_mean | aa\_accuracy | 0.924519 |
| latest | mask\_mean | terminal\_accuracy | 0.992424 |

评测曲线（原始 step，四档等权均值）：

| step | mean loss | mean AA acc | mean end acc | gradient norm |
| --- | --- | --- | --- | --- |
| 25 | 1.81904 | 0.326881 | 0 | 3.94664 |
| 50 | 1.03998 | 0.651924 | 0 | 3.41808 |
| 75 | 0.648404 | 0.787072 | 0.0392977 | 3.33625 |
| 100 | 0.51894 | 0.816014 | 0.589938 | 2.15494 |
| 125 | 0.516287 | 0.814012 | 0.932109 | 2.73508 |
| 150 | 0.453238 | 0.840757 | 0.930658 | 1.58996 |
| 175 | 0.487104 | 0.84291 | 0.985415 | 2.09064 |
| 200 | 0.410901 | 0.852874 | 0.992424 | 2.80852 |
| 225 | 0.383593 | 0.863611 | 0.980176 | 3.86038 |
| 250 | 0.332888 | 0.87169 | 1 | 1.13685 |
| 275 | 0.341477 | 0.876881 | 0.937624 | 1.52439 |
| 300 | 0.290659 | 0.889225 | 0.974494 | 0.664522 |
| 325 | 0.322163 | 0.891189 | 0.94697 | 0.938915 |
| 350 | 0.254295 | 0.901932 | 0.9726 | 1.71408 |
| 375 | 0.250706 | 0.904412 | 1 | 1.0117 |
| 400 | 0.235941 | 0.907688 | 1 | 0.708821 |
| 425 | 0.209025 | 0.921571 | 1 | 0.678494 |
| 450 | 0.209859 | 0.920153 | 0.967361 | 0.6482 |
| 475 | 0.20603 | 0.923685 | 0.954545 | 1.95877 |
| 500 | 0.192513 | 0.924519 | 0.992424 | 0.499902 |

有条件/无条件曲线（每档按token计数合并，再平均四档；缺失分组不推断数值）：

| step | 子集 | mean loss | mean AA | mean end | Full-mask AA | Full-mask end | Full-mask AA count | Full-mask end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | conditional | 4.19569 | 0.0633803 | 0 | 0.0707102 | 0 | 9744 | 72 |
| initial | unconditional | 4.05032 | 0.065109 | 0 | 0.062069 | 0 | 8120 | 60 |
| 25 | conditional | 1.77652 | 0.324365 | 0 | 0.290127 | 0 | 9744 | 72 |
| 25 | unconditional | 1.87007 | 0.32979 | 0 | 0.283005 | 0 | 8120 | 60 |
| 50 | conditional | 0.968557 | 0.65568 | 0 | 0.301211 | 0 | 9744 | 72 |
| 50 | unconditional | 1.12559 | 0.647461 | 0 | 0.30234 | 0 | 8120 | 60 |
| 75 | conditional | 0.599743 | 0.797 | 0.0519048 | 0.306958 | 0 | 9744 | 72 |
| 75 | unconditional | 0.706784 | 0.775152 | 0.0178571 | 0.227956 | 0 | 8120 | 60 |
| 100 | conditional | 0.447324 | 0.827348 | 0.67453 | 0.357656 | 0.180556 | 9744 | 72 |
| 100 | unconditional | 0.60482 | 0.802427 | 0.468667 | 0.275985 | 0 | 8120 | 60 |
| 125 | conditional | 0.462885 | 0.828997 | 0.986111 | 0.343186 | 0.944444 | 9744 | 72 |
| 125 | unconditional | 0.580328 | 0.796038 | 0.860499 | 0.23202 | 0.733333 | 8120 | 60 |
| 150 | conditional | 0.39009 | 0.862371 | 0.974929 | 0.465004 | 0.916667 | 9744 | 72 |
| 150 | unconditional | 0.529002 | 0.814826 | 0.870916 | 0.291749 | 0.733333 | 8120 | 60 |
| 175 | conditional | 0.461514 | 0.85627 | 0.981874 | 0.438937 | 0.944444 | 9744 | 72 |
| 175 | unconditional | 0.517794 | 0.82688 | 0.989583 | 0.333621 | 1 | 8120 | 60 |
| 200 | conditional | 0.345025 | 0.877043 | 1 | 0.521346 | 1 | 9744 | 72 |
| 200 | unconditional | 0.489948 | 0.823873 | 0.983333 | 0.321921 | 0.933333 | 8120 | 60 |
| 225 | conditional | 0.292397 | 0.901404 | 0.995763 | 0.613814 | 1 | 9744 | 72 |
| 225 | unconditional | 0.493017 | 0.818263 | 0.961458 | 0.297167 | 0.866667 | 8120 | 60 |
| 250 | conditional | 0.248532 | 0.909033 | 1 | 0.638957 | 1 | 9744 | 72 |
| 250 | unconditional | 0.434104 | 0.826886 | 1 | 0.330665 | 1 | 8120 | 60 |
| 275 | conditional | 0.211118 | 0.92243 | 0.986111 | 0.690681 | 0.944444 | 9744 | 72 |
| 275 | unconditional | 0.497876 | 0.822232 | 0.879167 | 0.316502 | 0.6 | 8120 | 60 |
| 300 | conditional | 0.193225 | 0.924192 | 0.989583 | 0.701047 | 0.958333 | 9744 | 72 |
| 300 | unconditional | 0.407576 | 0.847267 | 0.95625 | 0.404926 | 0.866667 | 8120 | 60 |
| 325 | conditional | 0.180528 | 0.935394 | 1 | 0.743227 | 1 | 9744 | 72 |
| 325 | unconditional | 0.492125 | 0.838146 | 0.883333 | 0.374138 | 0.533333 | 8120 | 60 |
| 350 | conditional | 0.11395 | 0.957444 | 1 | 0.830049 | 1 | 9744 | 72 |
| 350 | unconditional | 0.422705 | 0.835322 | 0.939583 | 0.363424 | 0.8 | 8120 | 60 |
| 375 | conditional | 0.103403 | 0.962643 | 1 | 0.850985 | 1 | 9744 | 72 |
| 375 | unconditional | 0.427463 | 0.834539 | 1 | 0.355665 | 1 | 8120 | 60 |
| 400 | conditional | 0.0820159 | 0.973804 | 1 | 0.895628 | 1 | 9744 | 72 |
| 400 | unconditional | 0.420645 | 0.828354 | 1 | 0.335714 | 1 | 8120 | 60 |
| 425 | conditional | 0.0464217 | 0.988933 | 1 | 0.956281 | 1 | 9744 | 72 |
| 425 | unconditional | 0.404144 | 0.840741 | 1 | 0.385345 | 1 | 8120 | 60 |
| 450 | conditional | 0.0355656 | 0.989265 | 0.995763 | 0.958436 | 1 | 9744 | 72 |
| 450 | unconditional | 0.419007 | 0.837221 | 0.933333 | 0.370936 | 0.733333 | 8120 | 60 |
| 475 | conditional | 0.0477555 | 0.985358 | 0.986111 | 0.941708 | 0.944444 | 9744 | 72 |
| 475 | unconditional | 0.395956 | 0.849681 | 0.916667 | 0.420197 | 0.666667 | 8120 | 60 |
| 500 | conditional | 0.0120992 | 0.9961 | 1 | 0.984401 | 1 | 9744 | 72 |
| 500 | unconditional | 0.409006 | 0.838624 | 0.983333 | 0.371675 | 0.933333 | 8120 | 60 |

人工数据 manifest：d40bbbbc1c4294eed843d88bcdd0cf193886de6c2d7e9db9069bbf86e931c022；精确 target sequence 重叠=0；精确 record 重叠=0。

跨 radius32 覆盖：{&quot;possible&quot;: true, &quot;train&quot;: true, &quot;valid&quot;: true}。生成规则共享是诊断设计的一部分，不能当作真实下游去重证书。


按任务/链长分组（全局 level 按 token 计数汇总，下面保留各组计数；不同组不得直接比较未经加权的总 loss）：

| 时点 | group/length | mask level | loss | AA acc | end acc | selected count | AA count | end count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | ab\_antigen/long | mask25 | 4.06387 | 0.0304183 | 0 | 530 | 526 | 4 |
| initial | ab\_antigen/long | mask50 | 4.40229 | 0.0306022 | 0 | 1019 | 1013 | 6 |
| initial | ab\_antigen/long | mask75 | 4.65354 | 0.0306789 | 0 | 1538 | 1532 | 6 |
| initial | ab\_antigen/long | mask100 | 4.64382 | 0.0313726 | 0 | 2048 | 2040 | 8 |
| initial | ab\_antigen/long | mask\_mean | 4.44088 | 0.030768 | 0 | — | — | — |
| initial | ab\_antigen/medium | mask25 | 4.4492 | 0.156379 | 0 | 246 | 243 | 3 |
| initial | ab\_antigen/medium | mask50 | 4.52118 | 0.161355 | 0 | 507 | 502 | 5 |
| initial | ab\_antigen/medium | mask75 | 4.59706 | 0.154354 | 0 | 765 | 758 | 7 |
| initial | ab\_antigen/medium | mask100 | 4.58314 | 0.15748 | 0 | 1024 | 1016 | 8 |
| initial | ab\_antigen/medium | mask\_mean | 4.53764 | 0.157392 | 0 | — | — | — |
| initial | ab\_antigen/short | mask25 | 5.25912 | 0 | 0 | 38 | 34 | 4 |
| initial | ab\_antigen/short | mask50 | 4.90384 | 0.0340909 | 0 | 93 | 88 | 5 |
| initial | ab\_antigen/short | mask75 | 4.77905 | 0.0428571 | 0 | 145 | 140 | 5 |
| initial | ab\_antigen/short | mask100 | 4.70735 | 0.0625 | 0 | 200 | 192 | 8 |
| initial | ab\_antigen/short | mask\_mean | 4.91234 | 0.034862 | 0 | — | — | — |
| initial | tcr\_peptide/long | mask25 | 3.57421 | 0.105376 | 0 | 466 | 465 | 1 |
| initial | tcr\_peptide/long | mask50 | 3.84408 | 0.114169 | 0 | 985 | 981 | 4 |
| initial | tcr\_peptide/long | mask75 | 4.06469 | 0.0784844 | 0 | 1484 | 1478 | 6 |
| initial | tcr\_peptide/long | mask100 | 4.09122 | 0.12451 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_peptide/long | mask\_mean | 3.89355 | 0.105635 | 0 | — | — | — |
| initial | tcr\_peptide/medium | mask25 | 4.0786 | 0.0414938 | 0 | 242 | 241 | 1 |
| initial | tcr\_peptide/medium | mask50 | 4.10604 | 0.0179283 | 0 | 506 | 502 | 4 |
| initial | tcr\_peptide/medium | mask75 | 4.04149 | 0.0355263 | 0 | 767 | 760 | 7 |
| initial | tcr\_peptide/medium | mask100 | 3.93722 | 0.0610236 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_peptide/medium | mask\_mean | 4.04084 | 0.038993 | 0 | — | — | — |
| initial | tcr\_peptide/short | mask25 | 4.078 | 0.0416667 | 0 | 51 | 48 | 3 |
| initial | tcr\_peptide/short | mask50 | 4.01756 | 0.0576923 | 0 | 110 | 104 | 6 |
| initial | tcr\_peptide/short | mask75 | 4.03916 | 0.0656934 | 0 | 144 | 137 | 7 |
| initial | tcr\_peptide/short | mask100 | 4.01962 | 0.09375 | 0 | 200 | 192 | 8 |
| initial | tcr\_peptide/short | mask\_mean | 4.03858 | 0.0647006 | 0 | — | — | — |
| initial | tcr\_pmhc/long | mask25 | 3.45892 | 0.0391791 | 0 | 540 | 536 | 4 |
| initial | tcr\_pmhc/long | mask50 | 3.84629 | 0.0357834 | 0 | 1038 | 1034 | 4 |
| initial | tcr\_pmhc/long | mask75 | 4.25287 | 0.028314 | 0 | 1562 | 1554 | 8 |
| initial | tcr\_pmhc/long | mask100 | 4.43587 | 0.0210784 | 0 | 2048 | 2040 | 8 |
| initial | tcr\_pmhc/long | mask\_mean | 3.99849 | 0.0310887 | 0 | — | — | — |
| initial | tcr\_pmhc/medium | mask25 | 4.24292 | 0.053719 | 0 | 244 | 242 | 2 |
| initial | tcr\_pmhc/medium | mask50 | 4.39061 | 0.039666 | 0 | 484 | 479 | 5 |
| initial | tcr\_pmhc/medium | mask75 | 4.47512 | 0.053316 | 0 | 777 | 769 | 8 |
| initial | tcr\_pmhc/medium | mask100 | 4.46935 | 0.0629921 | 0 | 1024 | 1016 | 8 |
| initial | tcr\_pmhc/medium | mask\_mean | 4.3945 | 0.0524233 | 0 | — | — | — |
| initial | tcr\_pmhc/short | mask25 | 4.46829 | 0.0208333 | 0 | 51 | 48 | 3 |
| initial | tcr\_pmhc/short | mask50 | 4.44788 | 0.0240964 | 0 | 86 | 83 | 3 |
| initial | tcr\_pmhc/short | mask75 | 4.30161 | 0.0612245 | 0 | 152 | 147 | 5 |
| initial | tcr\_pmhc/short | mask100 | 4.29595 | 0.0625 | 0 | 200 | 192 | 8 |
| initial | tcr\_pmhc/short | mask\_mean | 4.37843 | 0.0421636 | 0 | — | — | — |
| initial | unconditional\_ab/long | mask25 | 3.72304 | 0.166355 | 0 | 538 | 535 | 3 |
| initial | unconditional\_ab/long | mask50 | 4.01507 | 0.179732 | 0 | 1050 | 1046 | 4 |
| initial | unconditional\_ab/long | mask75 | 4.18285 | 0.182055 | 0 | 1545 | 1538 | 7 |
| initial | unconditional\_ab/long | mask100 | 4.13411 | 0.187255 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_ab/long | mask\_mean | 4.01377 | 0.178849 | 0 | — | — | — |
| initial | unconditional\_ab/medium | mask25 | 4.52467 | 0.0474308 | 0 | 256 | 253 | 3 |
| initial | unconditional\_ab/medium | mask50 | 4.62929 | 0.0570866 | 0 | 512 | 508 | 4 |
| initial | unconditional\_ab/medium | mask75 | 4.59985 | 0.0629921 | 0 | 768 | 762 | 6 |
| initial | unconditional\_ab/medium | mask100 | 4.55057 | 0.0610236 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_ab/medium | mask\_mean | 4.5761 | 0.0571333 | 0 | — | — | — |
| initial | unconditional\_ab/short | mask25 | 4.32037 | 0.0384615 | 0 | 55 | 52 | 3 |
| initial | unconditional\_ab/short | mask50 | 4.14468 | 0.0408163 | 0 | 103 | 98 | 5 |
| initial | unconditional\_ab/short | mask75 | 4.10821 | 0.0567376 | 0 | 148 | 141 | 7 |
| initial | unconditional\_ab/short | mask100 | 4.07402 | 0.0625 | 0 | 200 | 192 | 8 |
| initial | unconditional\_ab/short | mask\_mean | 4.16182 | 0.0496289 | 0 | — | — | — |
| initial | unconditional\_single/long | mask25 | 3.56576 | 0.0485075 | 0 | 269 | 268 | 1 |
| initial | unconditional\_single/long | mask50 | 3.54172 | 0.0685714 | 0 | 527 | 525 | 2 |
| initial | unconditional\_single/long | mask75 | 3.59429 | 0.0335917 | 0 | 778 | 774 | 4 |
| initial | unconditional\_single/long | mask100 | 3.64062 | 0 | 0 | 1024 | 1020 | 4 |
| initial | unconditional\_single/long | mask\_mean | 3.5856 | 0.0376677 | 0 | — | — | — |
| initial | unconditional\_single/medium | mask25 | 3.96295 | 0 | 0 | 110 | 109 | 1 |
| initial | unconditional\_single/medium | mask50 | 3.81305 | 0 | 0 | 234 | 232 | 2 |
| initial | unconditional\_single/medium | mask75 | 3.80054 | 0 | 0 | 371 | 367 | 4 |
| initial | unconditional\_single/medium | mask100 | 3.77909 | 0 | 0 | 512 | 508 | 4 |
| initial | unconditional\_single/medium | mask\_mean | 3.83891 | 0 | 0 | — | — | — |
| initial | unconditional\_single/short | mask25 | 3.34487 | 0 | 0 | 28 | 26 | 2 |
| initial | unconditional\_single/short | mask50 | 3.54422 | 0 | 0 | 52 | 50 | 2 |
| initial | unconditional\_single/short | mask75 | 3.40152 | 0 | 0 | 80 | 77 | 3 |
| initial | unconditional\_single/short | mask100 | 3.41363 | 0 | 0 | 100 | 96 | 4 |
| initial | unconditional\_single/short | mask\_mean | 3.42606 | 0 | 0 | — | — | — |
| initial | unconditional\_tcr/long | mask25 | 3.66417 | 0.0368932 | 0 | 516 | 515 | 1 |
| initial | unconditional\_tcr/long | mask50 | 3.99255 | 0.029382 | 0 | 989 | 987 | 2 |
| initial | unconditional\_tcr/long | mask75 | 4.20409 | 0 | 0 | 1527 | 1522 | 5 |
| initial | unconditional\_tcr/long | mask100 | 4.05612 | 0.0156863 | 0 | 2048 | 2040 | 8 |
| initial | unconditional\_tcr/long | mask\_mean | 3.97923 | 0.0204904 | 0 | — | — | — |
| initial | unconditional\_tcr/medium | mask25 | 4.30167 | 0 | — | 247 | 247 | 0 |
| initial | unconditional\_tcr/medium | mask50 | 4.41837 | 0 | 0 | 500 | 497 | 3 |
| initial | unconditional\_tcr/medium | mask75 | 4.40098 | 0 | 0 | 744 | 738 | 6 |
| initial | unconditional\_tcr/medium | mask100 | 4.30471 | 0 | 0 | 1024 | 1016 | 8 |
| initial | unconditional\_tcr/medium | mask\_mean | 4.35643 | 0 | 0 | — | — | — |
| initial | unconditional\_tcr/short | mask25 | 4.1561 | 0 | — | 40 | 40 | 0 |
| initial | unconditional\_tcr/short | mask50 | 4.12996 | 0.0222222 | 0 | 93 | 90 | 3 |
| initial | unconditional\_tcr/short | mask75 | 4.13885 | 0.0514706 | 0 | 142 | 136 | 6 |
| initial | unconditional\_tcr/short | mask100 | 4.10155 | 0.0833333 | 0 | 200 | 192 | 8 |
| initial | unconditional\_tcr/short | mask\_mean | 4.13161 | 0.0392565 | 0 | — | — | — |
| latest | ab\_antigen/long | mask25 | 2.38216e-05 | 1 | 1 | 530 | 526 | 4 |
| latest | ab\_antigen/long | mask50 | 2.35876e-05 | 1 | 1 | 1019 | 1013 | 6 |
| latest | ab\_antigen/long | mask75 | 6.28816e-05 | 1 | 1 | 1538 | 1532 | 6 |
| latest | ab\_antigen/long | mask100 | 0.0602754 | 0.982353 | 1 | 2048 | 2040 | 8 |
| latest | ab\_antigen/long | mask\_mean | 0.0150964 | 0.995588 | 1 | — | — | — |
| latest | ab\_antigen/medium | mask25 | 1.06186e-05 | 1 | 1 | 246 | 243 | 3 |
| latest | ab\_antigen/medium | mask50 | 2.80069e-05 | 1 | 1 | 507 | 502 | 5 |
| latest | ab\_antigen/medium | mask75 | 1.5045e-05 | 1 | 1 | 765 | 758 | 7 |
| latest | ab\_antigen/medium | mask100 | 0.185349 | 0.935039 | 1 | 1024 | 1016 | 8 |
| latest | ab\_antigen/medium | mask\_mean | 0.0463506 | 0.98376 | 1 | — | — | — |
| latest | ab\_antigen/short | mask25 | 2.2691e-05 | 1 | 1 | 38 | 34 | 4 |
| latest | ab\_antigen/short | mask50 | 7.6941e-05 | 1 | 1 | 93 | 88 | 5 |
| latest | ab\_antigen/short | mask75 | 0.00112884 | 1 | 1 | 145 | 140 | 5 |
| latest | ab\_antigen/short | mask100 | 0.00202843 | 1 | 1 | 200 | 192 | 8 |
| latest | ab\_antigen/short | mask\_mean | 0.000814224 | 1 | 1 | — | — | — |
| latest | tcr\_peptide/long | mask25 | 2.11605e-05 | 1 | 1 | 466 | 465 | 1 |
| latest | tcr\_peptide/long | mask50 | 1.48455e-05 | 1 | 1 | 985 | 981 | 4 |
| latest | tcr\_peptide/long | mask75 | 1.11947e-05 | 1 | 1 | 1484 | 1478 | 6 |
| latest | tcr\_peptide/long | mask100 | 0.0258732 | 0.993627 | 1 | 2048 | 2040 | 8 |
| latest | tcr\_peptide/long | mask\_mean | 0.00648009 | 0.998407 | 1 | — | — | — |
| latest | tcr\_peptide/medium | mask25 | 1.50723e-05 | 1 | 1 | 242 | 241 | 1 |
| latest | tcr\_peptide/medium | mask50 | 3.0463e-06 | 1 | 1 | 506 | 502 | 4 |
| latest | tcr\_peptide/medium | mask75 | 0.000129634 | 1 | 1 | 767 | 760 | 7 |
| latest | tcr\_peptide/medium | mask100 | 0.0550644 | 0.980315 | 1 | 1024 | 1016 | 8 |
| latest | tcr\_peptide/medium | mask\_mean | 0.013803 | 0.995079 | 1 | — | — | — |
| latest | tcr\_peptide/short | mask25 | 7.94715e-06 | 1 | 1 | 51 | 48 | 3 |
| latest | tcr\_peptide/short | mask50 | 5.93201e-05 | 1 | 1 | 110 | 104 | 6 |
| latest | tcr\_peptide/short | mask75 | 9.31041e-05 | 1 | 1 | 144 | 137 | 7 |
| latest | tcr\_peptide/short | mask100 | 0.000920483 | 1 | 1 | 200 | 192 | 8 |
| latest | tcr\_peptide/short | mask\_mean | 0.000270214 | 1 | 1 | — | — | — |
| latest | tcr\_pmhc/long | mask25 | 4.17276e-05 | 1 | 1 | 540 | 536 | 4 |
| latest | tcr\_pmhc/long | mask50 | 1.15344e-05 | 1 | 1 | 1038 | 1034 | 4 |
| latest | tcr\_pmhc/long | mask75 | 0.000220926 | 1 | 1 | 1562 | 1554 | 8 |
| latest | tcr\_pmhc/long | mask100 | 0.0239666 | 0.991667 | 1 | 2048 | 2040 | 8 |
| latest | tcr\_pmhc/long | mask\_mean | 0.00606019 | 0.997917 | 1 | — | — | — |
| latest | tcr\_pmhc/medium | mask25 | 2.05047e-06 | 1 | 1 | 244 | 242 | 2 |
| latest | tcr\_pmhc/medium | mask50 | 2.76487e-06 | 1 | 1 | 484 | 479 | 5 |
| latest | tcr\_pmhc/medium | mask75 | 8.05568e-06 | 1 | 1 | 777 | 769 | 8 |
| latest | tcr\_pmhc/medium | mask100 | 0.0006113 | 1 | 1 | 1024 | 1016 | 8 |
| latest | tcr\_pmhc/medium | mask\_mean | 0.000156043 | 1 | 1 | — | — | — |
| latest | tcr\_pmhc/short | mask25 | 4.12104e-05 | 1 | 1 | 51 | 48 | 3 |
| latest | tcr\_pmhc/short | mask50 | 4.46559e-05 | 1 | 1 | 86 | 83 | 3 |
| latest | tcr\_pmhc/short | mask75 | 0.00123395 | 1 | 1 | 152 | 147 | 5 |
| latest | tcr\_pmhc/short | mask100 | 0.00245239 | 1 | 1 | 200 | 192 | 8 |
| latest | tcr\_pmhc/short | mask\_mean | 0.000943051 | 1 | 1 | — | — | — |
| latest | unconditional\_ab/long | mask25 | 0.000115958 | 1 | 1 | 538 | 535 | 3 |
| latest | unconditional\_ab/long | mask50 | 1.80611e-05 | 1 | 1 | 1050 | 1046 | 4 |
| latest | unconditional\_ab/long | mask75 | 0.000488177 | 1 | 1 | 1545 | 1538 | 7 |
| latest | unconditional\_ab/long | mask100 | 1.58478 | 0.337255 | 1 | 2048 | 2040 | 8 |
| latest | unconditional\_ab/long | mask\_mean | 0.396352 | 0.834314 | 1 | — | — | — |
| latest | unconditional\_ab/medium | mask25 | 7.43145e-06 | 1 | 1 | 256 | 253 | 3 |
| latest | unconditional\_ab/medium | mask50 | 7.67099e-06 | 1 | 1 | 512 | 508 | 4 |
| latest | unconditional\_ab/medium | mask75 | 2.70885e-05 | 1 | 1 | 768 | 762 | 6 |
| latest | unconditional\_ab/medium | mask100 | 1.64108 | 0.34252 | 0.5 | 1024 | 1016 | 8 |
| latest | unconditional\_ab/medium | mask\_mean | 0.41028 | 0.83563 | 0.875 | — | — | — |
| latest | unconditional\_ab/short | mask25 | 1.21457e-05 | 1 | 1 | 55 | 52 | 3 |
| latest | unconditional\_ab/short | mask50 | 0.192739 | 0.938775 | 1 | 103 | 98 | 5 |
| latest | unconditional\_ab/short | mask75 | 0.372753 | 0.87234 | 1 | 148 | 141 | 7 |
| latest | unconditional\_ab/short | mask100 | 1.26664 | 0.395833 | 1 | 200 | 192 | 8 |
| latest | unconditional\_ab/short | mask\_mean | 0.458036 | 0.801737 | 1 | — | — | — |
| latest | unconditional\_single/long | mask25 | 0.000122521 | 1 | 1 | 269 | 268 | 1 |
| latest | unconditional\_single/long | mask50 | 0.000218884 | 1 | 1 | 527 | 525 | 2 |
| latest | unconditional\_single/long | mask75 | 0.114661 | 0.958656 | 1 | 778 | 774 | 4 |
| latest | unconditional\_single/long | mask100 | 1.56478 | 0.368627 | 1 | 1024 | 1020 | 4 |
| latest | unconditional\_single/long | mask\_mean | 0.419946 | 0.831821 | 1 | — | — | — |
| latest | unconditional\_single/medium | mask25 | 9.14678e-05 | 1 | 1 | 110 | 109 | 1 |
| latest | unconditional\_single/medium | mask50 | 3.561e-05 | 1 | 1 | 234 | 232 | 2 |
| latest | unconditional\_single/medium | mask75 | 0.000107693 | 1 | 1 | 371 | 367 | 4 |
| latest | unconditional\_single/medium | mask100 | 1.65826 | 0.387795 | 1 | 512 | 508 | 4 |
| latest | unconditional\_single/medium | mask\_mean | 0.414624 | 0.846949 | 1 | — | — | — |
| latest | unconditional\_single/short | mask25 | 0.000907569 | 1 | 1 | 28 | 26 | 2 |
| latest | unconditional\_single/short | mask50 | 0.286386 | 0.88 | 1 | 52 | 50 | 2 |
| latest | unconditional\_single/short | mask75 | 0.97385 | 0.61039 | 1 | 80 | 77 | 3 |
| latest | unconditional\_single/short | mask100 | 1.35163 | 0.375 | 1 | 100 | 96 | 4 |
| latest | unconditional\_single/short | mask\_mean | 0.653194 | 0.716347 | 1 | — | — | — |
| latest | unconditional\_tcr/long | mask25 | 6.3756e-06 | 1 | 1 | 516 | 515 | 1 |
| latest | unconditional\_tcr/long | mask50 | 1.30788e-05 | 1 | 1 | 989 | 987 | 2 |
| latest | unconditional\_tcr/long | mask75 | 7.84403e-05 | 1 | 1 | 1527 | 1522 | 5 |
| latest | unconditional\_tcr/long | mask100 | 1.58735 | 0.412255 | 1 | 2048 | 2040 | 8 |
| latest | unconditional\_tcr/long | mask\_mean | 0.396861 | 0.853064 | 1 | — | — | — |
| latest | unconditional\_tcr/medium | mask25 | 1.96183e-06 | 1 | — | 247 | 247 | 0 |
| latest | unconditional\_tcr/medium | mask50 | 8.47587e-06 | 1 | 1 | 500 | 497 | 3 |
| latest | unconditional\_tcr/medium | mask75 | 1.00888e-05 | 1 | 1 | 744 | 738 | 6 |
| latest | unconditional\_tcr/medium | mask100 | 1.65509 | 0.362205 | 1 | 1024 | 1016 | 8 |
| latest | unconditional\_tcr/medium | mask\_mean | 0.413777 | 0.840551 | 1 | — | — | — |
| latest | unconditional\_tcr/short | mask25 | 5.07513e-06 | 1 | — | 40 | 40 | 0 |
| latest | unconditional\_tcr/short | mask50 | 7.14798e-05 | 1 | 1 | 93 | 90 | 3 |
| latest | unconditional\_tcr/short | mask75 | 0.192761 | 0.955882 | 1 | 142 | 136 | 6 |
| latest | unconditional\_tcr/short | mask100 | 1.32131 | 0.458333 | 1 | 200 | 192 | 8 |
| latest | unconditional\_tcr/short | mask\_mean | 0.378536 | 0.853554 | 1 | — | — | — |
| train\_probe | ab\_antigen/long | mask25 | 3.56319e-05 | 1 | — | 530 | 530 | 0 |
| train\_probe | ab\_antigen/long | mask50 | 2.43435e-05 | 1 | 1 | 1034 | 1031 | 3 |
| train\_probe | ab\_antigen/long | mask75 | 2.88112e-05 | 1 | 1 | 1563 | 1557 | 6 |
| train\_probe | ab\_antigen/long | mask100 | 0.425994 | 0.840196 | 1 | 2048 | 2040 | 8 |
| train\_probe | ab\_antigen/long | mask\_mean | 0.106521 | 0.960049 | 1 | — | — | — |
| train\_probe | ab\_antigen/medium | mask25 | 8.54651e-05 | 1 | 1 | 275 | 273 | 2 |
| train\_probe | ab\_antigen/medium | mask50 | 2.6649e-05 | 1 | 1 | 542 | 538 | 4 |
| train\_probe | ab\_antigen/medium | mask75 | 1.0577e-05 | 1 | 1 | 781 | 776 | 5 |
| train\_probe | ab\_antigen/medium | mask100 | 0.0742957 | 0.976378 | 1 | 1024 | 1016 | 8 |
| train\_probe | ab\_antigen/medium | mask\_mean | 0.0186046 | 0.994094 | 1 | — | — | — |
| train\_probe | ab\_antigen/short | mask25 | 2.90348e-05 | 1 | 1 | 52 | 47 | 5 |
| train\_probe | ab\_antigen/short | mask50 | 3.0899e-05 | 1 | 1 | 109 | 102 | 7 |
| train\_probe | ab\_antigen/short | mask75 | 0.00939093 | 1 | 1 | 155 | 148 | 7 |
| train\_probe | ab\_antigen/short | mask100 | 0.00759341 | 1 | 1 | 200 | 192 | 8 |
| train\_probe | ab\_antigen/short | mask\_mean | 0.00426107 | 1 | 1 | — | — | — |
| train\_probe | tcr\_peptide/long | mask25 | 5.78524e-05 | 1 | 1 | 495 | 494 | 1 |
| train\_probe | tcr\_peptide/long | mask50 | 2.10659e-05 | 1 | 1 | 1033 | 1028 | 5 |
| train\_probe | tcr\_peptide/long | mask75 | 1.9938e-05 | 1 | 1 | 1530 | 1522 | 8 |
| train\_probe | tcr\_peptide/long | mask100 | 0.111064 | 0.962255 | 1 | 2048 | 2040 | 8 |
| train\_probe | tcr\_peptide/long | mask\_mean | 0.0277907 | 0.990564 | 1 | — | — | — |
| train\_probe | tcr\_peptide/medium | mask25 | 2.36898e-06 | 1 | — | 252 | 252 | 0 |
| train\_probe | tcr\_peptide/medium | mask50 | 2.52997e-06 | 1 | 1 | 517 | 513 | 4 |
| train\_probe | tcr\_peptide/medium | mask75 | 5.56153e-06 | 1 | 1 | 760 | 754 | 6 |
| train\_probe | tcr\_peptide/medium | mask100 | 0.0399917 | 0.991142 | 1 | 1024 | 1016 | 8 |
| train\_probe | tcr\_peptide/medium | mask\_mean | 0.0100005 | 0.997785 | 1 | — | — | — |
| train\_probe | tcr\_peptide/short | mask25 | 1.18314e-05 | 1 | 1 | 51 | 46 | 5 |
| train\_probe | tcr\_peptide/short | mask50 | 2.43294e-05 | 1 | 1 | 95 | 90 | 5 |
| train\_probe | tcr\_peptide/short | mask75 | 4.35543e-05 | 1 | 1 | 142 | 136 | 6 |
| train\_probe | tcr\_peptide/short | mask100 | 0.000525306 | 1 | 1 | 200 | 192 | 8 |
| train\_probe | tcr\_peptide/short | mask\_mean | 0.000151255 | 1 | 1 | — | — | — |
| train\_probe | tcr\_pmhc/long | mask25 | 8.48737e-06 | 1 | 1 | 526 | 522 | 4 |
| train\_probe | tcr\_pmhc/long | mask50 | 6.56088e-06 | 1 | 1 | 1036 | 1031 | 5 |
| train\_probe | tcr\_pmhc/long | mask75 | 7.41531e-06 | 1 | 1 | 1557 | 1551 | 6 |
| train\_probe | tcr\_pmhc/long | mask100 | 0.00188647 | 1 | 1 | 2048 | 2040 | 8 |
| train\_probe | tcr\_pmhc/long | mask\_mean | 0.000477234 | 1 | 1 | — | — | — |
| train\_probe | tcr\_pmhc/medium | mask25 | 1.08968e-05 | 1 | 1 | 276 | 273 | 3 |
| train\_probe | tcr\_pmhc/medium | mask50 | 4.56489e-06 | 1 | 1 | 549 | 541 | 8 |
| train\_probe | tcr\_pmhc/medium | mask75 | 3.8669e-06 | 1 | 1 | 793 | 785 | 8 |
| train\_probe | tcr\_pmhc/medium | mask100 | 0.00136713 | 1 | 1 | 1024 | 1016 | 8 |
| train\_probe | tcr\_pmhc/medium | mask\_mean | 0.000346615 | 1 | 1 | — | — | — |
| train\_probe | tcr\_pmhc/short | mask25 | 1.1663e-05 | 1 | 1 | 50 | 48 | 2 |
| train\_probe | tcr\_pmhc/short | mask50 | 2.15016e-05 | 1 | 1 | 103 | 100 | 3 |
| train\_probe | tcr\_pmhc/short | mask75 | 0.000188632 | 1 | 1 | 152 | 145 | 7 |
| train\_probe | tcr\_pmhc/short | mask100 | 0.00194292 | 1 | 1 | 200 | 192 | 8 |
| train\_probe | tcr\_pmhc/short | mask\_mean | 0.00054118 | 1 | 1 | — | — | — |
| train\_probe | unconditional\_ab/long | mask25 | 4.515e-05 | 1 | 1 | 509 | 506 | 3 |
| train\_probe | unconditional\_ab/long | mask50 | 2.0219e-05 | 1 | 1 | 1016 | 1010 | 6 |
| train\_probe | unconditional\_ab/long | mask75 | 6.53946e-05 | 1 | 1 | 1569 | 1561 | 8 |
| train\_probe | unconditional\_ab/long | mask100 | 1.56215 | 0.318627 | 1 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_ab/long | mask\_mean | 0.39057 | 0.829657 | 1 | — | — | — |
| train\_probe | unconditional\_ab/medium | mask25 | 5.31818e-05 | 1 | 1 | 243 | 241 | 2 |
| train\_probe | unconditional\_ab/medium | mask50 | 9.71051e-06 | 1 | 1 | 521 | 516 | 5 |
| train\_probe | unconditional\_ab/medium | mask75 | 0.000169626 | 1 | 1 | 766 | 761 | 5 |
| train\_probe | unconditional\_ab/medium | mask100 | 1.69555 | 0.279528 | 0.5 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_ab/medium | mask\_mean | 0.423946 | 0.819882 | 0.875 | — | — | — |
| train\_probe | unconditional\_ab/short | mask25 | 0.000160552 | 1 | 1 | 53 | 52 | 1 |
| train\_probe | unconditional\_ab/short | mask50 | 0.183706 | 0.942308 | 1 | 106 | 104 | 2 |
| train\_probe | unconditional\_ab/short | mask75 | 0.616029 | 0.734177 | 1 | 163 | 158 | 5 |
| train\_probe | unconditional\_ab/short | mask100 | 1.5232 | 0.333333 | 1 | 200 | 192 | 8 |
| train\_probe | unconditional\_ab/short | mask\_mean | 0.580774 | 0.752455 | 1 | — | — | — |
| train\_probe | unconditional\_single/long | mask25 | 7.23976e-05 | 1 | 1 | 245 | 244 | 1 |
| train\_probe | unconditional\_single/long | mask50 | 0.00013505 | 1 | 1 | 492 | 491 | 1 |
| train\_probe | unconditional\_single/long | mask75 | 0.0255379 | 0.992188 | 1 | 771 | 768 | 3 |
| train\_probe | unconditional\_single/long | mask100 | 1.62896 | 0.301961 | 1 | 1024 | 1020 | 4 |
| train\_probe | unconditional\_single/long | mask\_mean | 0.413676 | 0.823537 | 1 | — | — | — |
| train\_probe | unconditional\_single/medium | mask25 | 4.71069e-05 | 1 | 1 | 132 | 130 | 2 |
| train\_probe | unconditional\_single/medium | mask50 | 8.24324e-05 | 1 | 1 | 255 | 252 | 3 |
| train\_probe | unconditional\_single/medium | mask75 | 0.000175305 | 1 | 1 | 382 | 379 | 3 |
| train\_probe | unconditional\_single/medium | mask100 | 1.64618 | 0.309055 | 1 | 512 | 508 | 4 |
| train\_probe | unconditional\_single/medium | mask\_mean | 0.41162 | 0.827264 | 1 | — | — | — |
| train\_probe | unconditional\_single/short | mask25 | 0.000580671 | 1 | 1 | 28 | 26 | 2 |
| train\_probe | unconditional\_single/short | mask50 | 0.364804 | 0.816326 | 1 | 52 | 49 | 3 |
| train\_probe | unconditional\_single/short | mask75 | 0.80293 | 0.614286 | 1 | 73 | 70 | 3 |
| train\_probe | unconditional\_single/short | mask100 | 1.28538 | 0.4375 | 1 | 100 | 96 | 4 |
| train\_probe | unconditional\_single/short | mask\_mean | 0.613424 | 0.717028 | 1 | — | — | — |
| train\_probe | unconditional\_tcr/long | mask25 | 1.54429e-05 | 1 | — | 525 | 525 | 0 |
| train\_probe | unconditional\_tcr/long | mask50 | 7.00301e-06 | 1 | 1 | 1048 | 1045 | 3 |
| train\_probe | unconditional\_tcr/long | mask75 | 5.27012e-06 | 1 | 1 | 1585 | 1580 | 5 |
| train\_probe | unconditional\_tcr/long | mask100 | 1.68671 | 0.286765 | 1 | 2048 | 2040 | 8 |
| train\_probe | unconditional\_tcr/long | mask\_mean | 0.421685 | 0.821691 | 1 | — | — | — |
| train\_probe | unconditional\_tcr/medium | mask25 | 4.6156e-05 | 1 | — | 261 | 261 | 0 |
| train\_probe | unconditional\_tcr/medium | mask50 | 1.31149e-05 | 1 | 1 | 498 | 497 | 1 |
| train\_probe | unconditional\_tcr/medium | mask75 | 9.01159e-06 | 1 | 1 | 755 | 751 | 4 |
| train\_probe | unconditional\_tcr/medium | mask100 | 1.54882 | 0.287402 | 1 | 1024 | 1016 | 8 |
| train\_probe | unconditional\_tcr/medium | mask\_mean | 0.387223 | 0.82185 | 1 | — | — | — |
| train\_probe | unconditional\_tcr/short | mask25 | 2.17163e-06 | 1 | — | 46 | 46 | 0 |
| train\_probe | unconditional\_tcr/short | mask50 | 0.300827 | 0.913043 | 1 | 97 | 92 | 5 |
| train\_probe | unconditional\_tcr/short | mask75 | 0.511622 | 0.821918 | 1 | 152 | 146 | 6 |
| train\_probe | unconditional\_tcr/short | mask100 | 1.55834 | 0.354167 | 1 | 200 | 192 | 8 |
| train\_probe | unconditional\_tcr/short | mask\_mean | 0.592697 | 0.772282 | 1 | — | — | — |

## 判读规则与下一阶段

- 显存扫描中的预期 OOM 档位属于正常上界探索；只有 runner 失败或 completion 不一致才视为执行失败。
- source_changed / profile_summary_hash_mismatch / run_summary_hash_mismatch：源码或结果与证书不一致，优先报告；不自动修改训练代码或重试。
- endpoint_shortcut_signal：AA acc ≤ 0.01 且 end acc ≥ 0.99；短程总 loss 下降不足以证明学会残基。该项是诊断信号，不能直接据此判定最终架构优劣。
- end_not_learned_yet：≥10 步但 end acc=0，保留后续曲线观察，不能等同程序失败。
- 原六条 toy 的100步诊断已结束；下一阶段使用独立 release/profile 的广长度、条件链和独立人工 heldout 诊断，按25步记录并保留同方法不同预算实例。不能把重新初始化称为续训。
- 人工数据的25/500步等结果只验证实现、优化与人工规则迁移；不能替代真实语料清理证书、真实数据验证、下游测试与多个 seed。

## 原六条 toy 全 mask 诊断的可识别性限制

以下只适用于原 R00–R19 六条 toy 协议，不能用于后续扩展人工 heldout。CPU 实际比较全部 7 个 denoise 输入特征：4 条 TCR 在全 mask 后输入完全相同，标签不同；AB/Nb 各自可区分。输入分组：[[0, 1, 2, 3], [4], [5]]。

原 6 条 toy 的 AA-only 准确率理论上限 46/82=56.10%；全部目标 token 联合上限 51/93=54.84%。两种上限分别优化对应指标，不声称可同时达到。

原100步不能以AA/end同时100%为通过标准。后续R20–R27采用广长度和条件链人工诊断，应使用对应fixture的独立可识别性审计，不能沿用旧toy上限。

证据：/vepfs-mlp2/c20250601/251105016/project/dllm\_test/output/immune\_research\_pool\_c20250601\_20260930/monitor/full\_mask\_identifiability.json。

## 扩展人工 heldout 的独立可识别性限制

仅对应fixture manifest SHA=d40bbbbc1c4294eed843d88bcdd0cf193886de6c2d7e9db9069bbf86e931c022。输入确定性argmax上限不等于模型可学习性或生物泛化保证。

| 子集 | full-mask输入组数 | full-mask AA-only上限 | 联合token上限 | 联合argmax最优时end范围 | 四档mean-AA上限 |
| --- | --- | --- | --- | --- | --- |
| all | 39 | 0.820869 | 0.81996 | [0.696969696969697, 0.8484848484848485] | 0.955217 |
| conditional | 36 | 1 | 1 | [1.0, 1.0] | 1 |
| unconditional | 3 | 0.605911 | 0.603912 | [0.3333333333333333, 0.6666666666666666] | 0.901478 |

end范围只限定为同时达到联合argmax最优；terminal单独优化可更高，AA-only与end-only最优不可混同。条件组仍可区分，不能把其end未学习全部归因于无条件输入碰撞。

证据：/vepfs-mlp2/c20250601/251105016/project/dllm\_test/output/immune\_research\_pool\_c20250601\_20260930/monitor/expanded\_identifiability.json；详细方法：/vepfs-mlp2/c20250601/251105016/project/dllm\_test/docs/IMMUNE\_RESEARCH\_EXPANDED\_IDENTIFIABILITY.md。

## 有效变化历史

- 2026-09-30 19:43:56 UTC — r11\_span\_qknorm\_register (1790796212599373446\_r11\_span\_qknorm\_register\_2a307772): pending, step=0/?, alerts=[]; r09\_equal\_source (1790796212592865084\_r09\_equal\_source\_fa295088): pending, step=0/?, alerts=[]; r10\_role\_pair\_bias (1790796212596100361\_r10\_role\_pair\_bias\_10d5f5b5): pending, step=0/?, alerts=[]; r08\_vicreg (1790796212589837668\_r08\_vicreg\_a1da0f51): pending, step=0/?, alerts=[]; r02\_chain\_completion (1790796212571024553\_r02\_chain\_completion\_499888e5): running, step=10/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r04\_qk\_norm (1790796212577263807\_r04\_qk\_norm\_32f94214): running, step=10/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r03\_chain\_balanced\_loss (1790796212574033832\_r03\_chain\_balanced\_loss\_0693e278): running, step=10/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r00\_a4\_control (1790796212564610986\_r00\_a4\_control\_754dd5e6): running, step=10/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r06\_local\_conv7 (1790796212583598406\_r06\_local\_conv7\_78cd3d2d): running, step=10/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r07\_mask\_ratio\_film (1790796212586628536\_r07\_mask\_ratio\_film\_7688a166): running, step=10/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r01\_span\_mask (1790796212567802264\_r01\_span\_mask\_26105bc0): running, step=10/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r05\_register4 (1790796212580387587\_r05\_register4\_3a525f6b): running, step=10/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]
- 2026-09-30 19:45:09 UTC — r02\_chain\_completion (1790796212571024553\_r02\_chain\_completion\_499888e5): running, step=15/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r04\_qk\_norm (1790796212577263807\_r04\_qk\_norm\_32f94214): running, step=15/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r03\_chain\_balanced\_loss (1790796212574033832\_r03\_chain\_balanced\_loss\_0693e278): running, step=15/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r00\_a4\_control (1790796212564610986\_r00\_a4\_control\_754dd5e6): running, step=15/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r06\_local\_conv7 (1790796212583598406\_r06\_local\_conv7\_78cd3d2d): running, step=15/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r07\_mask\_ratio\_film (1790796212586628536\_r07\_mask\_ratio\_film\_7688a166): running, step=15/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r01\_span\_mask (1790796212567802264\_r01\_span\_mask\_26105bc0): running, step=15/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r05\_register4 (1790796212580387587\_r05\_register4\_3a525f6b): running, step=15/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]
- 2026-09-30 19:45:39 UTC — r11\_span\_qknorm\_register (1790796212599373446\_r11\_span\_qknorm\_register\_2a307772): running, step=0/?, alerts=[]; r09\_equal\_source (1790796212592865084\_r09\_equal\_source\_fa295088): running, step=0/?, alerts=[]; r10\_role\_pair\_bias (1790796212596100361\_r10\_role\_pair\_bias\_10d5f5b5): running, step=0/?, alerts=[]; r08\_vicreg (1790796212589837668\_r08\_vicreg\_a1da0f51): running, step=0/?, alerts=[]; r02\_chain\_completion (1790796212571024553\_r02\_chain\_completion\_499888e5): succeeded, step=20/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r04\_qk\_norm (1790796212577263807\_r04\_qk\_norm\_32f94214): succeeded, step=20/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r03\_chain\_balanced\_loss (1790796212574033832\_r03\_chain\_balanced\_loss\_0693e278): succeeded, step=20/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r00\_a4\_control (1790796212564610986\_r00\_a4\_control\_754dd5e6): succeeded, step=20/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r06\_local\_conv7 (1790796212583598406\_r06\_local\_conv7\_78cd3d2d): succeeded, step=20/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r07\_mask\_ratio\_film (1790796212586628536\_r07\_mask\_ratio\_film\_7688a166): succeeded, step=20/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r01\_span\_mask (1790796212567802264\_r01\_span\_mask\_26105bc0): succeeded, step=20/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]
- 2026-09-30 19:46:09 UTC — r05\_register4 (1790796212580387587\_r05\_register4\_3a525f6b): succeeded, step=20/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]
- 2026-09-30 19:50:14 UTC — r09\_equal\_source (1790796212592865084\_r09\_equal\_source\_fa295088): running, step=0/?, alerts=[]
- 2026-09-30 19:50:44 UTC — r11\_span\_qknorm\_register (1790796212599373446\_r11\_span\_qknorm\_register\_2a307772): running, step=0/?, alerts=[]
- 2026-09-30 19:52:15 UTC — r11\_span\_qknorm\_register (1790796212599373446\_r11\_span\_qknorm\_register\_2a307772): running, step=5/?, alerts=[]; r09\_equal\_source (1790796212592865084\_r09\_equal\_source\_fa295088): running, step=5/?, alerts=[]
- 2026-09-30 19:53:46 UTC — r11\_span\_qknorm\_register (1790796212599373446\_r11\_span\_qknorm\_register\_2a307772): running, step=10/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r09\_equal\_source (1790796212592865084\_r09\_equal\_source\_fa295088): running, step=10/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r10\_role\_pair\_bias (1790796212596100361\_r10\_role\_pair\_bias\_10d5f5b5): running, step=0/?, alerts=[]; r08\_vicreg (1790796212589837668\_r08\_vicreg\_a1da0f51): running, step=0/?, alerts=[]
- 2026-09-30 19:55:16 UTC — r11\_span\_qknorm\_register (1790796212599373446\_r11\_span\_qknorm\_register\_2a307772): running, step=15/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r09\_equal\_source (1790796212592865084\_r09\_equal\_source\_fa295088): running, step=15/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]
- 2026-09-30 19:55:47 UTC — r10\_role\_pair\_bias (1790796212596100361\_r10\_role\_pair\_bias\_10d5f5b5): running, step=5/?, alerts=[]
- 2026-09-30 19:56:47 UTC — r08\_vicreg (1790796212589837668\_r08\_vicreg\_a1da0f51): running, step=5/?, alerts=[]; r11\_span\_qknorm\_register (1790796212599373446\_r11\_span\_qknorm\_register\_2a307772): succeeded, step=20/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r09\_equal\_source (1790796212592865084\_r09\_equal\_source\_fa295088): succeeded, step=20/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]
- 2026-09-30 19:57:17 UTC — r10\_role\_pair\_bias (1790796212596100361\_r10\_role\_pair\_bias\_10d5f5b5): running, step=10/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]
- 2026-09-30 19:59:19 UTC — r10\_role\_pair\_bias (1790796212596100361\_r10\_role\_pair\_bias\_10d5f5b5): running, step=15/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]
- 2026-09-30 19:59:49 UTC — r08\_vicreg (1790796212589837668\_r08\_vicreg\_a1da0f51): running, step=10/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]
- 2026-09-30 20:01:20 UTC — r10\_role\_pair\_bias (1790796212596100361\_r10\_role\_pair\_bias\_10d5f5b5): succeeded, step=20/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]
- 2026-09-30 20:02:50 UTC — r08\_vicreg (1790796212589837668\_r08\_vicreg\_a1da0f51): running, step=15/?, alerts=[&#x27;end\_not\_learned\_yet&#x27;]
- 2026-09-30 20:02:52 UTC — r08\_vicreg (1790796212589837668\_r08\_vicreg\_a1da0f51): running, step=15/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r02\_chain\_completion (1790796212571024553\_r02\_chain\_completion\_499888e5): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r04\_qk\_norm (1790796212577263807\_r04\_qk\_norm\_32f94214): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r03\_chain\_balanced\_loss (1790796212574033832\_r03\_chain\_balanced\_loss\_0693e278): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r00\_a4\_control (1790796212564610986\_r00\_a4\_control\_754dd5e6): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r11\_span\_qknorm\_register (1790796212599373446\_r11\_span\_qknorm\_register\_2a307772): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r09\_equal\_source (1790796212592865084\_r09\_equal\_source\_fa295088): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r06\_local\_conv7 (1790796212583598406\_r06\_local\_conv7\_78cd3d2d): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r10\_role\_pair\_bias (1790796212596100361\_r10\_role\_pair\_bias\_10d5f5b5): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r07\_mask\_ratio\_film (1790796212586628536\_r07\_mask\_ratio\_film\_7688a166): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r01\_span\_mask (1790796212567802264\_r01\_span\_mask\_26105bc0): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r05\_register4 (1790796212580387587\_r05\_register4\_3a525f6b): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]
- 2026-09-30 20:05:54 UTC — r08\_vicreg (1790796212589837668\_r08\_vicreg\_a1da0f51): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]
- 2026-09-30 20:08:25 UTC — r02\_chain\_completion (1790796212571024553\_r02\_chain\_completion\_499888e5): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r04\_qk\_norm (1790796212577263807\_r04\_qk\_norm\_32f94214): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r03\_chain\_balanced\_loss (1790796212574033832\_r03\_chain\_balanced\_loss\_0693e278): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r00\_a4\_control (1790796212564610986\_r00\_a4\_control\_754dd5e6): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r11\_span\_qknorm\_register (1790796212599373446\_r11\_span\_qknorm\_register\_2a307772): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r09\_equal\_source (1790796212592865084\_r09\_equal\_source\_fa295088): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r06\_local\_conv7 (1790796212583598406\_r06\_local\_conv7\_78cd3d2d): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r10\_role\_pair\_bias (1790796212596100361\_r10\_role\_pair\_bias\_10d5f5b5): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r07\_mask\_ratio\_film (1790796212586628536\_r07\_mask\_ratio\_film\_7688a166): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r08\_vicreg (1790796212589837668\_r08\_vicreg\_a1da0f51): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r01\_span\_mask (1790796212567802264\_r01\_span\_mask\_26105bc0): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r05\_register4 (1790796212580387587\_r05\_register4\_3a525f6b): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]
- 2026-09-30 20:10:57 UTC — r12\_architecture\_control (1790799044322727901\_r12\_architecture\_control\_37b2b106): running, step=0/100, alerts=[]; r19\_value\_residual (1790799044355869324\_r19\_value\_residual\_3bcfd14a): running, step=0/100, alerts=[]; r13\_head\_output\_gate (1790799044331991663\_r13\_head\_output\_gate\_72718882): running, step=0/100, alerts=[]; r18\_chain\_local\_global (1790799044352237695\_r18\_chain\_local\_global\_a0a72b83): running, step=0/100, alerts=[]; r14\_fixed\_half\_attention (1790799044337931939\_r14\_fixed\_half\_attention\_8643a4ca): running, step=0/100, alerts=[]; r15\_layerscale (1790799044341624231\_r15\_layerscale\_d0115c63): running, step=0/100, alerts=[]; r17\_chain\_late\_fusion (1790799044348827183\_r17\_chain\_late\_fusion\_2a3a70e9): running, step=0/100, alerts=[]; r16\_fixed\_residual\_scale (1790799044345097609\_r16\_fixed\_residual\_scale\_6111de03): running, step=0/100, alerts=[]
- 2026-09-30 20:13:58 UTC — r12\_architecture\_control (1790799044322727901\_r12\_architecture\_control\_37b2b106): running, step=0/100, alerts=[]; r19\_value\_residual (1790799044355869324\_r19\_value\_residual\_3bcfd14a): running, step=0/100, alerts=[]; r13\_head\_output\_gate (1790799044331991663\_r13\_head\_output\_gate\_72718882): running, step=0/100, alerts=[]; r18\_chain\_local\_global (1790799044352237695\_r18\_chain\_local\_global\_a0a72b83): running, step=0/100, alerts=[]; r14\_fixed\_half\_attention (1790799044337931939\_r14\_fixed\_half\_attention\_8643a4ca): running, step=0/100, alerts=[]; r15\_layerscale (1790799044341624231\_r15\_layerscale\_d0115c63): running, step=0/100, alerts=[]; r17\_chain\_late\_fusion (1790799044348827183\_r17\_chain\_late\_fusion\_2a3a70e9): running, step=0/100, alerts=[]; r16\_fixed\_residual\_scale (1790799044345097609\_r16\_fixed\_residual\_scale\_6111de03): running, step=0/100, alerts=[]
- 2026-09-30 20:15:30 UTC — r12\_architecture\_control (1790799044322727901\_r12\_architecture\_control\_37b2b106): running, step=5/100, alerts=[]; r19\_value\_residual (1790799044355869324\_r19\_value\_residual\_3bcfd14a): running, step=5/100, alerts=[]; r13\_head\_output\_gate (1790799044331991663\_r13\_head\_output\_gate\_72718882): running, step=5/100, alerts=[]; r18\_chain\_local\_global (1790799044352237695\_r18\_chain\_local\_global\_a0a72b83): running, step=5/100, alerts=[]; r14\_fixed\_half\_attention (1790799044337931939\_r14\_fixed\_half\_attention\_8643a4ca): running, step=5/100, alerts=[]; r15\_layerscale (1790799044341624231\_r15\_layerscale\_d0115c63): running, step=5/100, alerts=[]; r17\_chain\_late\_fusion (1790799044348827183\_r17\_chain\_late\_fusion\_2a3a70e9): running, step=5/100, alerts=[]; r16\_fixed\_residual\_scale (1790799044345097609\_r16\_fixed\_residual\_scale\_6111de03): running, step=5/100, alerts=[]
- 2026-09-30 20:17:01 UTC — r12\_architecture\_control (1790799044322727901\_r12\_architecture\_control\_37b2b106): running, step=10/100, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r19\_value\_residual (1790799044355869324\_r19\_value\_residual\_3bcfd14a): running, step=10/100, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r13\_head\_output\_gate (1790799044331991663\_r13\_head\_output\_gate\_72718882): running, step=10/100, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r18\_chain\_local\_global (1790799044352237695\_r18\_chain\_local\_global\_a0a72b83): running, step=10/100, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r14\_fixed\_half\_attention (1790799044337931939\_r14\_fixed\_half\_attention\_8643a4ca): running, step=10/100, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r15\_layerscale (1790799044341624231\_r15\_layerscale\_d0115c63): running, step=10/100, alerts=[]; r17\_chain\_late\_fusion (1790799044348827183\_r17\_chain\_late\_fusion\_2a3a70e9): running, step=10/100, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r16\_fixed\_residual\_scale (1790799044345097609\_r16\_fixed\_residual\_scale\_6111de03): running, step=10/100, alerts=[]
- 2026-09-30 20:18:32 UTC — r12\_architecture\_control (1790799044322727901\_r12\_architecture\_control\_37b2b106): running, step=15/100, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r19\_value\_residual (1790799044355869324\_r19\_value\_residual\_3bcfd14a): running, step=15/100, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r13\_head\_output\_gate (1790799044331991663\_r13\_head\_output\_gate\_72718882): running, step=15/100, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r18\_chain\_local\_global (1790799044352237695\_r18\_chain\_local\_global\_a0a72b83): running, step=15/100, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r14\_fixed\_half\_attention (1790799044337931939\_r14\_fixed\_half\_attention\_8643a4ca): running, step=15/100, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r15\_layerscale (1790799044341624231\_r15\_layerscale\_d0115c63): running, step=15/100, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r17\_chain\_late\_fusion (1790799044348827183\_r17\_chain\_late\_fusion\_2a3a70e9): running, step=15/100, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r16\_fixed\_residual\_scale (1790799044345097609\_r16\_fixed\_residual\_scale\_6111de03): running, step=15/100, alerts=[&#x27;end\_not\_learned\_yet&#x27;]
- 2026-09-30 20:20:03 UTC — r12\_architecture\_control (1790799044322727901\_r12\_architecture\_control\_37b2b106): running, step=20/100, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r19\_value\_residual (1790799044355869324\_r19\_value\_residual\_3bcfd14a): running, step=20/100, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r13\_head\_output\_gate (1790799044331991663\_r13\_head\_output\_gate\_72718882): running, step=20/100, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r18\_chain\_local\_global (1790799044352237695\_r18\_chain\_local\_global\_a0a72b83): running, step=20/100, alerts=[]; r14\_fixed\_half\_attention (1790799044337931939\_r14\_fixed\_half\_attention\_8643a4ca): running, step=20/100, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r15\_layerscale (1790799044341624231\_r15\_layerscale\_d0115c63): running, step=20/100, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r17\_chain\_late\_fusion (1790799044348827183\_r17\_chain\_late\_fusion\_2a3a70e9): running, step=20/100, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r16\_fixed\_residual\_scale (1790799044345097609\_r16\_fixed\_residual\_scale\_6111de03): running, step=20/100, alerts=[&#x27;end\_not\_learned\_yet&#x27;]
- 2026-09-30 20:21:34 UTC — r12\_architecture\_control (1790799044322727901\_r12\_architecture\_control\_37b2b106): running, step=25/100, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r19\_value\_residual (1790799044355869324\_r19\_value\_residual\_3bcfd14a): running, step=25/100, alerts=[]; r13\_head\_output\_gate (1790799044331991663\_r13\_head\_output\_gate\_72718882): running, step=25/100, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r18\_chain\_local\_global (1790799044352237695\_r18\_chain\_local\_global\_a0a72b83): running, step=25/100, alerts=[]; r14\_fixed\_half\_attention (1790799044337931939\_r14\_fixed\_half\_attention\_8643a4ca): running, step=25/100, alerts=[]; r15\_layerscale (1790799044341624231\_r15\_layerscale\_d0115c63): running, step=25/100, alerts=[]; r17\_chain\_late\_fusion (1790799044348827183\_r17\_chain\_late\_fusion\_2a3a70e9): running, step=25/100, alerts=[]; r16\_fixed\_residual\_scale (1790799044345097609\_r16\_fixed\_residual\_scale\_6111de03): running, step=25/100, alerts=[]
- 2026-09-30 20:23:05 UTC — r12\_architecture\_control (1790799044322727901\_r12\_architecture\_control\_37b2b106): running, step=30/100, alerts=[]; r19\_value\_residual (1790799044355869324\_r19\_value\_residual\_3bcfd14a): running, step=30/100, alerts=[]; r13\_head\_output\_gate (1790799044331991663\_r13\_head\_output\_gate\_72718882): running, step=30/100, alerts=[]; r18\_chain\_local\_global (1790799044352237695\_r18\_chain\_local\_global\_a0a72b83): running, step=30/100, alerts=[]; r14\_fixed\_half\_attention (1790799044337931939\_r14\_fixed\_half\_attention\_8643a4ca): running, step=30/100, alerts=[]; r15\_layerscale (1790799044341624231\_r15\_layerscale\_d0115c63): running, step=30/100, alerts=[]; r17\_chain\_late\_fusion (1790799044348827183\_r17\_chain\_late\_fusion\_2a3a70e9): running, step=30/100, alerts=[]; r16\_fixed\_residual\_scale (1790799044345097609\_r16\_fixed\_residual\_scale\_6111de03): running, step=30/100, alerts=[]
- 2026-09-30 20:24:37 UTC — r12\_architecture\_control (1790799044322727901\_r12\_architecture\_control\_37b2b106): running, step=35/100, alerts=[]; r19\_value\_residual (1790799044355869324\_r19\_value\_residual\_3bcfd14a): running, step=35/100, alerts=[]; r13\_head\_output\_gate (1790799044331991663\_r13\_head\_output\_gate\_72718882): running, step=35/100, alerts=[]; r18\_chain\_local\_global (1790799044352237695\_r18\_chain\_local\_global\_a0a72b83): running, step=35/100, alerts=[]; r14\_fixed\_half\_attention (1790799044337931939\_r14\_fixed\_half\_attention\_8643a4ca): running, step=35/100, alerts=[]; r15\_layerscale (1790799044341624231\_r15\_layerscale\_d0115c63): running, step=35/100, alerts=[]; r17\_chain\_late\_fusion (1790799044348827183\_r17\_chain\_late\_fusion\_2a3a70e9): running, step=35/100, alerts=[]; r16\_fixed\_residual\_scale (1790799044345097609\_r16\_fixed\_residual\_scale\_6111de03): running, step=35/100, alerts=[]
- 2026-09-30 20:26:16 UTC — r12\_architecture\_control (1790799044322727901\_r12\_architecture\_control\_37b2b106): running, step=40/100, alerts=[]; r19\_value\_residual (1790799044355869324\_r19\_value\_residual\_3bcfd14a): running, step=40/100, alerts=[]; r13\_head\_output\_gate (1790799044331991663\_r13\_head\_output\_gate\_72718882): running, step=40/100, alerts=[]; r18\_chain\_local\_global (1790799044352237695\_r18\_chain\_local\_global\_a0a72b83): running, step=40/100, alerts=[]; r14\_fixed\_half\_attention (1790799044337931939\_r14\_fixed\_half\_attention\_8643a4ca): running, step=40/100, alerts=[]; r15\_layerscale (1790799044341624231\_r15\_layerscale\_d0115c63): running, step=40/100, alerts=[]; r17\_chain\_late\_fusion (1790799044348827183\_r17\_chain\_late\_fusion\_2a3a70e9): running, step=40/100, alerts=[]; r16\_fixed\_residual\_scale (1790799044345097609\_r16\_fixed\_residual\_scale\_6111de03): running, step=40/100, alerts=[]
- 2026-09-30 20:27:47 UTC — r12\_architecture\_control (1790799044322727901\_r12\_architecture\_control\_37b2b106): running, step=45/100, alerts=[]; r19\_value\_residual (1790799044355869324\_r19\_value\_residual\_3bcfd14a): running, step=45/100, alerts=[]; r13\_head\_output\_gate (1790799044331991663\_r13\_head\_output\_gate\_72718882): running, step=45/100, alerts=[]; r18\_chain\_local\_global (1790799044352237695\_r18\_chain\_local\_global\_a0a72b83): running, step=45/100, alerts=[]; r14\_fixed\_half\_attention (1790799044337931939\_r14\_fixed\_half\_attention\_8643a4ca): running, step=45/100, alerts=[]; r15\_layerscale (1790799044341624231\_r15\_layerscale\_d0115c63): running, step=45/100, alerts=[]; r17\_chain\_late\_fusion (1790799044348827183\_r17\_chain\_late\_fusion\_2a3a70e9): running, step=45/100, alerts=[]; r16\_fixed\_residual\_scale (1790799044345097609\_r16\_fixed\_residual\_scale\_6111de03): running, step=45/100, alerts=[]
- 2026-09-30 20:29:18 UTC — r12\_architecture\_control (1790799044322727901\_r12\_architecture\_control\_37b2b106): running, step=50/100, alerts=[]; r19\_value\_residual (1790799044355869324\_r19\_value\_residual\_3bcfd14a): running, step=50/100, alerts=[]; r13\_head\_output\_gate (1790799044331991663\_r13\_head\_output\_gate\_72718882): running, step=50/100, alerts=[]; r18\_chain\_local\_global (1790799044352237695\_r18\_chain\_local\_global\_a0a72b83): running, step=50/100, alerts=[]; r14\_fixed\_half\_attention (1790799044337931939\_r14\_fixed\_half\_attention\_8643a4ca): running, step=50/100, alerts=[]; r15\_layerscale (1790799044341624231\_r15\_layerscale\_d0115c63): running, step=50/100, alerts=[]; r17\_chain\_late\_fusion (1790799044348827183\_r17\_chain\_late\_fusion\_2a3a70e9): running, step=50/100, alerts=[]; r16\_fixed\_residual\_scale (1790799044345097609\_r16\_fixed\_residual\_scale\_6111de03): running, step=50/100, alerts=[]
- 2026-09-30 20:30:50 UTC — r12\_architecture\_control (1790799044322727901\_r12\_architecture\_control\_37b2b106): running, step=55/100, alerts=[]; r19\_value\_residual (1790799044355869324\_r19\_value\_residual\_3bcfd14a): running, step=55/100, alerts=[]; r13\_head\_output\_gate (1790799044331991663\_r13\_head\_output\_gate\_72718882): running, step=55/100, alerts=[]; r18\_chain\_local\_global (1790799044352237695\_r18\_chain\_local\_global\_a0a72b83): running, step=55/100, alerts=[]; r14\_fixed\_half\_attention (1790799044337931939\_r14\_fixed\_half\_attention\_8643a4ca): running, step=55/100, alerts=[]; r15\_layerscale (1790799044341624231\_r15\_layerscale\_d0115c63): running, step=55/100, alerts=[]; r17\_chain\_late\_fusion (1790799044348827183\_r17\_chain\_late\_fusion\_2a3a70e9): running, step=55/100, alerts=[]; r16\_fixed\_residual\_scale (1790799044345097609\_r16\_fixed\_residual\_scale\_6111de03): running, step=55/100, alerts=[]
- 2026-09-30 20:32:21 UTC — r12\_architecture\_control (1790799044322727901\_r12\_architecture\_control\_37b2b106): running, step=60/100, alerts=[]; r19\_value\_residual (1790799044355869324\_r19\_value\_residual\_3bcfd14a): running, step=60/100, alerts=[]; r13\_head\_output\_gate (1790799044331991663\_r13\_head\_output\_gate\_72718882): running, step=60/100, alerts=[]; r18\_chain\_local\_global (1790799044352237695\_r18\_chain\_local\_global\_a0a72b83): running, step=60/100, alerts=[]; r14\_fixed\_half\_attention (1790799044337931939\_r14\_fixed\_half\_attention\_8643a4ca): running, step=60/100, alerts=[]; r15\_layerscale (1790799044341624231\_r15\_layerscale\_d0115c63): running, step=60/100, alerts=[]; r17\_chain\_late\_fusion (1790799044348827183\_r17\_chain\_late\_fusion\_2a3a70e9): running, step=60/100, alerts=[]; r16\_fixed\_residual\_scale (1790799044345097609\_r16\_fixed\_residual\_scale\_6111de03): running, step=60/100, alerts=[]
- 2026-09-30 20:33:52 UTC — r12\_architecture\_control (1790799044322727901\_r12\_architecture\_control\_37b2b106): running, step=65/100, alerts=[]; r19\_value\_residual (1790799044355869324\_r19\_value\_residual\_3bcfd14a): running, step=65/100, alerts=[]; r13\_head\_output\_gate (1790799044331991663\_r13\_head\_output\_gate\_72718882): running, step=65/100, alerts=[]; r18\_chain\_local\_global (1790799044352237695\_r18\_chain\_local\_global\_a0a72b83): running, step=65/100, alerts=[]; r14\_fixed\_half\_attention (1790799044337931939\_r14\_fixed\_half\_attention\_8643a4ca): running, step=65/100, alerts=[]; r15\_layerscale (1790799044341624231\_r15\_layerscale\_d0115c63): running, step=65/100, alerts=[]; r17\_chain\_late\_fusion (1790799044348827183\_r17\_chain\_late\_fusion\_2a3a70e9): running, step=65/100, alerts=[]; r16\_fixed\_residual\_scale (1790799044345097609\_r16\_fixed\_residual\_scale\_6111de03): running, step=65/100, alerts=[]
- 2026-09-30 20:35:24 UTC — r12\_architecture\_control (1790799044322727901\_r12\_architecture\_control\_37b2b106): running, step=70/100, alerts=[]; r19\_value\_residual (1790799044355869324\_r19\_value\_residual\_3bcfd14a): running, step=70/100, alerts=[]; r13\_head\_output\_gate (1790799044331991663\_r13\_head\_output\_gate\_72718882): running, step=70/100, alerts=[]; r18\_chain\_local\_global (1790799044352237695\_r18\_chain\_local\_global\_a0a72b83): running, step=70/100, alerts=[]; r14\_fixed\_half\_attention (1790799044337931939\_r14\_fixed\_half\_attention\_8643a4ca): running, step=70/100, alerts=[]; r15\_layerscale (1790799044341624231\_r15\_layerscale\_d0115c63): running, step=70/100, alerts=[]; r17\_chain\_late\_fusion (1790799044348827183\_r17\_chain\_late\_fusion\_2a3a70e9): running, step=70/100, alerts=[]; r16\_fixed\_residual\_scale (1790799044345097609\_r16\_fixed\_residual\_scale\_6111de03): running, step=70/100, alerts=[]
- 2026-09-30 20:36:55 UTC — r12\_architecture\_control (1790799044322727901\_r12\_architecture\_control\_37b2b106): running, step=75/100, alerts=[]; r19\_value\_residual (1790799044355869324\_r19\_value\_residual\_3bcfd14a): running, step=75/100, alerts=[]; r13\_head\_output\_gate (1790799044331991663\_r13\_head\_output\_gate\_72718882): running, step=75/100, alerts=[]; r18\_chain\_local\_global (1790799044352237695\_r18\_chain\_local\_global\_a0a72b83): running, step=75/100, alerts=[]; r14\_fixed\_half\_attention (1790799044337931939\_r14\_fixed\_half\_attention\_8643a4ca): running, step=75/100, alerts=[]; r15\_layerscale (1790799044341624231\_r15\_layerscale\_d0115c63): running, step=75/100, alerts=[]; r17\_chain\_late\_fusion (1790799044348827183\_r17\_chain\_late\_fusion\_2a3a70e9): running, step=75/100, alerts=[]; r16\_fixed\_residual\_scale (1790799044345097609\_r16\_fixed\_residual\_scale\_6111de03): running, step=75/100, alerts=[]
- 2026-09-30 20:38:26 UTC — r12\_architecture\_control (1790799044322727901\_r12\_architecture\_control\_37b2b106): running, step=80/100, alerts=[]; r19\_value\_residual (1790799044355869324\_r19\_value\_residual\_3bcfd14a): running, step=80/100, alerts=[]; r13\_head\_output\_gate (1790799044331991663\_r13\_head\_output\_gate\_72718882): running, step=80/100, alerts=[]; r18\_chain\_local\_global (1790799044352237695\_r18\_chain\_local\_global\_a0a72b83): running, step=80/100, alerts=[]; r14\_fixed\_half\_attention (1790799044337931939\_r14\_fixed\_half\_attention\_8643a4ca): running, step=80/100, alerts=[]; r15\_layerscale (1790799044341624231\_r15\_layerscale\_d0115c63): running, step=80/100, alerts=[]; r17\_chain\_late\_fusion (1790799044348827183\_r17\_chain\_late\_fusion\_2a3a70e9): running, step=80/100, alerts=[]; r16\_fixed\_residual\_scale (1790799044345097609\_r16\_fixed\_residual\_scale\_6111de03): running, step=80/100, alerts=[]
- 2026-09-30 20:39:58 UTC — r12\_architecture\_control (1790799044322727901\_r12\_architecture\_control\_37b2b106): running, step=85/100, alerts=[]; r19\_value\_residual (1790799044355869324\_r19\_value\_residual\_3bcfd14a): running, step=85/100, alerts=[]; r13\_head\_output\_gate (1790799044331991663\_r13\_head\_output\_gate\_72718882): running, step=85/100, alerts=[]; r18\_chain\_local\_global (1790799044352237695\_r18\_chain\_local\_global\_a0a72b83): running, step=85/100, alerts=[]; r14\_fixed\_half\_attention (1790799044337931939\_r14\_fixed\_half\_attention\_8643a4ca): running, step=85/100, alerts=[]; r15\_layerscale (1790799044341624231\_r15\_layerscale\_d0115c63): running, step=85/100, alerts=[]; r17\_chain\_late\_fusion (1790799044348827183\_r17\_chain\_late\_fusion\_2a3a70e9): running, step=85/100, alerts=[]; r16\_fixed\_residual\_scale (1790799044345097609\_r16\_fixed\_residual\_scale\_6111de03): running, step=85/100, alerts=[]
- 2026-09-30 20:41:29 UTC — r12\_architecture\_control (1790799044322727901\_r12\_architecture\_control\_37b2b106): running, step=90/100, alerts=[]; r19\_value\_residual (1790799044355869324\_r19\_value\_residual\_3bcfd14a): running, step=90/100, alerts=[]; r13\_head\_output\_gate (1790799044331991663\_r13\_head\_output\_gate\_72718882): running, step=90/100, alerts=[]; r18\_chain\_local\_global (1790799044352237695\_r18\_chain\_local\_global\_a0a72b83): running, step=90/100, alerts=[]; r14\_fixed\_half\_attention (1790799044337931939\_r14\_fixed\_half\_attention\_8643a4ca): running, step=90/100, alerts=[]; r15\_layerscale (1790799044341624231\_r15\_layerscale\_d0115c63): running, step=90/100, alerts=[]; r17\_chain\_late\_fusion (1790799044348827183\_r17\_chain\_late\_fusion\_2a3a70e9): running, step=90/100, alerts=[]; r16\_fixed\_residual\_scale (1790799044345097609\_r16\_fixed\_residual\_scale\_6111de03): running, step=90/100, alerts=[]
- 2026-09-30 20:43:01 UTC — r12\_architecture\_control (1790799044322727901\_r12\_architecture\_control\_37b2b106): running, step=95/100, alerts=[]; r19\_value\_residual (1790799044355869324\_r19\_value\_residual\_3bcfd14a): running, step=95/100, alerts=[]; r13\_head\_output\_gate (1790799044331991663\_r13\_head\_output\_gate\_72718882): running, step=95/100, alerts=[]; r18\_chain\_local\_global (1790799044352237695\_r18\_chain\_local\_global\_a0a72b83): running, step=95/100, alerts=[]; r14\_fixed\_half\_attention (1790799044337931939\_r14\_fixed\_half\_attention\_8643a4ca): running, step=95/100, alerts=[]; r15\_layerscale (1790799044341624231\_r15\_layerscale\_d0115c63): running, step=95/100, alerts=[]; r17\_chain\_late\_fusion (1790799044348827183\_r17\_chain\_late\_fusion\_2a3a70e9): running, step=95/100, alerts=[]; r16\_fixed\_residual\_scale (1790799044345097609\_r16\_fixed\_residual\_scale\_6111de03): running, step=95/100, alerts=[]
- 2026-09-30 20:44:32 UTC — r12\_architecture\_control (1790799044322727901\_r12\_architecture\_control\_37b2b106): succeeded, step=100/100, alerts=[]; r19\_value\_residual (1790799044355869324\_r19\_value\_residual\_3bcfd14a): succeeded, step=100/100, alerts=[]; r13\_head\_output\_gate (1790799044331991663\_r13\_head\_output\_gate\_72718882): succeeded, step=100/100, alerts=[]; r18\_chain\_local\_global (1790799044352237695\_r18\_chain\_local\_global\_a0a72b83): succeeded, step=100/100, alerts=[]; r14\_fixed\_half\_attention (1790799044337931939\_r14\_fixed\_half\_attention\_8643a4ca): succeeded, step=100/100, alerts=[]; r15\_layerscale (1790799044341624231\_r15\_layerscale\_d0115c63): succeeded, step=100/100, alerts=[]; r17\_chain\_late\_fusion (1790799044348827183\_r17\_chain\_late\_fusion\_2a3a70e9): succeeded, step=100/100, alerts=[]; r16\_fixed\_residual\_scale (1790799044345097609\_r16\_fixed\_residual\_scale\_6111de03): succeeded, step=100/100, alerts=[]
- 2026-09-30 20:47:05 UTC — r12\_architecture\_control (1790799044322727901\_r12\_architecture\_control\_37b2b106): succeeded, step=100/100, alerts=[]; r19\_value\_residual (1790799044355869324\_r19\_value\_residual\_3bcfd14a): succeeded, step=100/100, alerts=[]; r13\_head\_output\_gate (1790799044331991663\_r13\_head\_output\_gate\_72718882): succeeded, step=100/100, alerts=[]; r18\_chain\_local\_global (1790799044352237695\_r18\_chain\_local\_global\_a0a72b83): succeeded, step=100/100, alerts=[]; r14\_fixed\_half\_attention (1790799044337931939\_r14\_fixed\_half\_attention\_8643a4ca): succeeded, step=100/100, alerts=[]; r15\_layerscale (1790799044341624231\_r15\_layerscale\_d0115c63): succeeded, step=100/100, alerts=[]; r17\_chain\_late\_fusion (1790799044348827183\_r17\_chain\_late\_fusion\_2a3a70e9): succeeded, step=100/100, alerts=[]; r16\_fixed\_residual\_scale (1790799044345097609\_r16\_fixed\_residual\_scale\_6111de03): succeeded, step=100/100, alerts=[]
- 2026-09-30 20:58:30 UTC — r02\_chain\_completion (1790796212571024553\_r02\_chain\_completion\_499888e5): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r12\_architecture\_control (1790799044322727901\_r12\_architecture\_control\_37b2b106): succeeded, step=100/100, alerts=[]; r04\_qk\_norm (1790796212577263807\_r04\_qk\_norm\_32f94214): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r19\_value\_residual (1790799044355869324\_r19\_value\_residual\_3bcfd14a): succeeded, step=100/100, alerts=[]; r13\_head\_output\_gate (1790799044331991663\_r13\_head\_output\_gate\_72718882): succeeded, step=100/100, alerts=[]; r18\_chain\_local\_global (1790799044352237695\_r18\_chain\_local\_global\_a0a72b83): succeeded, step=100/100, alerts=[]; r03\_chain\_balanced\_loss (1790796212574033832\_r03\_chain\_balanced\_loss\_0693e278): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r00\_a4\_control (1790796212564610986\_r00\_a4\_control\_754dd5e6): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r14\_fixed\_half\_attention (1790799044337931939\_r14\_fixed\_half\_attention\_8643a4ca): succeeded, step=100/100, alerts=[]; r15\_layerscale (1790799044341624231\_r15\_layerscale\_d0115c63): succeeded, step=100/100, alerts=[]; r11\_span\_qknorm\_register (1790796212599373446\_r11\_span\_qknorm\_register\_2a307772): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r09\_equal\_source (1790796212592865084\_r09\_equal\_source\_fa295088): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r17\_chain\_late\_fusion (1790799044348827183\_r17\_chain\_late\_fusion\_2a3a70e9): succeeded, step=100/100, alerts=[]; r06\_local\_conv7 (1790796212583598406\_r06\_local\_conv7\_78cd3d2d): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r10\_role\_pair\_bias (1790796212596100361\_r10\_role\_pair\_bias\_10d5f5b5): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r07\_mask\_ratio\_film (1790796212586628536\_r07\_mask\_ratio\_film\_7688a166): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r08\_vicreg (1790796212589837668\_r08\_vicreg\_a1da0f51): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r01\_span\_mask (1790796212567802264\_r01\_span\_mask\_26105bc0): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r16\_fixed\_residual\_scale (1790799044345097609\_r16\_fixed\_residual\_scale\_6111de03): succeeded, step=100/100, alerts=[]; r05\_register4 (1790796212580387587\_r05\_register4\_3a525f6b): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]
- 2026-09-30 21:03:05 UTC — r21\_refined\_head\_gate (1790802160343981887\_r21\_refined\_head\_gate\_80d10a6d): running, step=0/25, alerts=[]; r26\_refined\_local\_global (1790802160358902166\_r26\_refined\_local\_global\_905c9acd): running, step=0/25, alerts=[]; r27\_refined\_value\_residual (1790802160361769452\_r27\_refined\_value\_residual\_0b46ff76): running, step=0/25, alerts=[]; r23\_refined\_layerscale (1790802160350276097\_r23\_refined\_layerscale\_eb29f0d4): running, step=0/25, alerts=[]; r22\_refined\_fixed\_gate (1790802160346881056\_r22\_refined\_fixed\_gate\_263516a3): running, step=0/25, alerts=[]; r20\_refined\_control (1790802160338779408\_r20\_refined\_control\_3b41bffe): running, step=0/25, alerts=[]; r25\_refined\_late\_fusion (1790802160356028347\_r25\_refined\_late\_fusion\_4640596c): running, step=0/25, alerts=[]; r24\_refined\_fixed\_scale (1790802160353131370\_r24\_refined\_fixed\_scale\_c6e0089d): running, step=0/25, alerts=[]
- 2026-09-30 21:07:40 UTC — r21\_refined\_head\_gate (1790802160343981887\_r21\_refined\_head\_gate\_80d10a6d): running, step=0/25, alerts=[]; r26\_refined\_local\_global (1790802160358902166\_r26\_refined\_local\_global\_905c9acd): running, step=0/25, alerts=[]; r27\_refined\_value\_residual (1790802160361769452\_r27\_refined\_value\_residual\_0b46ff76): running, step=0/25, alerts=[]; r23\_refined\_layerscale (1790802160350276097\_r23\_refined\_layerscale\_eb29f0d4): running, step=0/25, alerts=[]; r22\_refined\_fixed\_gate (1790802160346881056\_r22\_refined\_fixed\_gate\_263516a3): running, step=0/25, alerts=[]; r20\_refined\_control (1790802160338779408\_r20\_refined\_control\_3b41bffe): running, step=0/25, alerts=[]; r25\_refined\_late\_fusion (1790802160356028347\_r25\_refined\_late\_fusion\_4640596c): running, step=0/25, alerts=[]; r24\_refined\_fixed\_scale (1790802160353131370\_r24\_refined\_fixed\_scale\_c6e0089d): running, step=0/25, alerts=[]
- 2026-09-30 21:19:27 UTC — r20\_refined\_control (1790802160338779408\_r20\_refined\_control\_3b41bffe): succeeded, step=25/25, alerts=[&#x27;end\_not\_learned\_yet&#x27;]
- 2026-09-30 21:19:58 UTC — r21\_refined\_head\_gate (1790802160343981887\_r21\_refined\_head\_gate\_80d10a6d): succeeded, step=25/25, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r26\_refined\_local\_global (1790802160358902166\_r26\_refined\_local\_global\_905c9acd): succeeded, step=25/25, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r27\_refined\_value\_residual (1790802160361769452\_r27\_refined\_value\_residual\_0b46ff76): succeeded, step=25/25, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r23\_refined\_layerscale (1790802160350276097\_r23\_refined\_layerscale\_eb29f0d4): succeeded, step=25/25, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r22\_refined\_fixed\_gate (1790802160346881056\_r22\_refined\_fixed\_gate\_263516a3): succeeded, step=25/25, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r25\_refined\_late\_fusion (1790802160356028347\_r25\_refined\_late\_fusion\_4640596c): succeeded, step=25/25, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r24\_refined\_fixed\_scale (1790802160353131370\_r24\_refined\_fixed\_scale\_c6e0089d): succeeded, step=25/25, alerts=[&#x27;end\_not\_learned\_yet&#x27;]
- 2026-09-30 21:22:02 UTC — r23\_refined\_layerscale (1790803305057942502\_r23\_refined\_layerscale\_56d8467a): running, step=0/500, alerts=[]; r25\_refined\_late\_fusion (1790803305064798754\_r25\_refined\_late\_fusion\_5cd5e385): running, step=0/500, alerts=[]; r27\_refined\_value\_residual (1790803305071555086\_r27\_refined\_value\_residual\_52f013e5): running, step=0/500, alerts=[]; r22\_refined\_fixed\_gate (1790803305054490194\_r22\_refined\_fixed\_gate\_72a52fe0): running, step=0/500, alerts=[]; r24\_refined\_fixed\_scale (1790803305061375040\_r24\_refined\_fixed\_scale\_4edeef26): running, step=0/500, alerts=[]; r26\_refined\_local\_global (1790803305068196112\_r26\_refined\_local\_global\_5bf50a1c): running, step=0/500, alerts=[]; r20\_refined\_control (1790803305045499169\_r20\_refined\_control\_a37dc5ab): running, step=0/500, alerts=[]; r21\_refined\_head\_gate (1790803305051081571\_r21\_refined\_head\_gate\_9cc7449e): running, step=0/500, alerts=[]
- 2026-09-30 21:26:10 UTC — r23\_refined\_layerscale (1790803305057942502\_r23\_refined\_layerscale\_56d8467a): running, step=0/500, alerts=[]; r25\_refined\_late\_fusion (1790803305064798754\_r25\_refined\_late\_fusion\_5cd5e385): running, step=0/500, alerts=[]; r27\_refined\_value\_residual (1790803305071555086\_r27\_refined\_value\_residual\_52f013e5): running, step=0/500, alerts=[]; r22\_refined\_fixed\_gate (1790803305054490194\_r22\_refined\_fixed\_gate\_72a52fe0): running, step=0/500, alerts=[]; r24\_refined\_fixed\_scale (1790803305061375040\_r24\_refined\_fixed\_scale\_4edeef26): running, step=0/500, alerts=[]; r26\_refined\_local\_global (1790803305068196112\_r26\_refined\_local\_global\_5bf50a1c): running, step=0/500, alerts=[]; r20\_refined\_control (1790803305045499169\_r20\_refined\_control\_a37dc5ab): running, step=0/500, alerts=[]; r21\_refined\_head\_gate (1790803305051081571\_r21\_refined\_head\_gate\_9cc7449e): running, step=0/500, alerts=[]; r02\_chain\_completion (1790796212571024553\_r02\_chain\_completion\_499888e5): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r21\_refined\_head\_gate (1790802160343981887\_r21\_refined\_head\_gate\_80d10a6d): succeeded, step=25/25, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r26\_refined\_local\_global (1790802160358902166\_r26\_refined\_local\_global\_905c9acd): succeeded, step=25/25, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r12\_architecture\_control (1790799044322727901\_r12\_architecture\_control\_37b2b106): succeeded, step=100/100, alerts=[]; r04\_qk\_norm (1790796212577263807\_r04\_qk\_norm\_32f94214): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r19\_value\_residual (1790799044355869324\_r19\_value\_residual\_3bcfd14a): succeeded, step=100/100, alerts=[]; r13\_head\_output\_gate (1790799044331991663\_r13\_head\_output\_gate\_72718882): succeeded, step=100/100, alerts=[]; r27\_refined\_value\_residual (1790802160361769452\_r27\_refined\_value\_residual\_0b46ff76): succeeded, step=25/25, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r23\_refined\_layerscale (1790802160350276097\_r23\_refined\_layerscale\_eb29f0d4): succeeded, step=25/25, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r18\_chain\_local\_global (1790799044352237695\_r18\_chain\_local\_global\_a0a72b83): succeeded, step=100/100, alerts=[]; r03\_chain\_balanced\_loss (1790796212574033832\_r03\_chain\_balanced\_loss\_0693e278): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r00\_a4\_control (1790796212564610986\_r00\_a4\_control\_754dd5e6): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r14\_fixed\_half\_attention (1790799044337931939\_r14\_fixed\_half\_attention\_8643a4ca): succeeded, step=100/100, alerts=[]; r15\_layerscale (1790799044341624231\_r15\_layerscale\_d0115c63): succeeded, step=100/100, alerts=[]; r11\_span\_qknorm\_register (1790796212599373446\_r11\_span\_qknorm\_register\_2a307772): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r22\_refined\_fixed\_gate (1790802160346881056\_r22\_refined\_fixed\_gate\_263516a3): succeeded, step=25/25, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r09\_equal\_source (1790796212592865084\_r09\_equal\_source\_fa295088): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r20\_refined\_control (1790802160338779408\_r20\_refined\_control\_3b41bffe): succeeded, step=25/25, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r25\_refined\_late\_fusion (1790802160356028347\_r25\_refined\_late\_fusion\_4640596c): succeeded, step=25/25, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r17\_chain\_late\_fusion (1790799044348827183\_r17\_chain\_late\_fusion\_2a3a70e9): succeeded, step=100/100, alerts=[]; r06\_local\_conv7 (1790796212583598406\_r06\_local\_conv7\_78cd3d2d): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r10\_role\_pair\_bias (1790796212596100361\_r10\_role\_pair\_bias\_10d5f5b5): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r07\_mask\_ratio\_film (1790796212586628536\_r07\_mask\_ratio\_film\_7688a166): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r08\_vicreg (1790796212589837668\_r08\_vicreg\_a1da0f51): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r01\_span\_mask (1790796212567802264\_r01\_span\_mask\_26105bc0): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r24\_refined\_fixed\_scale (1790802160353131370\_r24\_refined\_fixed\_scale\_c6e0089d): succeeded, step=25/25, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r16\_fixed\_residual\_scale (1790799044345097609\_r16\_fixed\_residual\_scale\_6111de03): succeeded, step=100/100, alerts=[]; r05\_register4 (1790796212580387587\_r05\_register4\_3a525f6b): succeeded, step=20/20, alerts=[&#x27;end\_not\_learned\_yet&#x27;]
- 2026-09-30 21:33:59 UTC — r23\_refined\_layerscale (1790803305057942502\_r23\_refined\_layerscale\_56d8467a): running, step=25/500, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r25\_refined\_late\_fusion (1790803305064798754\_r25\_refined\_late\_fusion\_5cd5e385): running, step=25/500, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r27\_refined\_value\_residual (1790803305071555086\_r27\_refined\_value\_residual\_52f013e5): running, step=25/500, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r22\_refined\_fixed\_gate (1790803305054490194\_r22\_refined\_fixed\_gate\_72a52fe0): running, step=25/500, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r24\_refined\_fixed\_scale (1790803305061375040\_r24\_refined\_fixed\_scale\_4edeef26): running, step=25/500, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r26\_refined\_local\_global (1790803305068196112\_r26\_refined\_local\_global\_5bf50a1c): running, step=25/500, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r20\_refined\_control (1790803305045499169\_r20\_refined\_control\_a37dc5ab): running, step=25/500, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r21\_refined\_head\_gate (1790803305051081571\_r21\_refined\_head\_gate\_9cc7449e): running, step=25/500, alerts=[&#x27;end\_not\_learned\_yet&#x27;]
- 2026-09-30 21:46:05 UTC — r23\_refined\_layerscale (1790803305057942502\_r23\_refined\_layerscale\_56d8467a): running, step=50/500, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r25\_refined\_late\_fusion (1790803305064798754\_r25\_refined\_late\_fusion\_5cd5e385): running, step=50/500, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r27\_refined\_value\_residual (1790803305071555086\_r27\_refined\_value\_residual\_52f013e5): running, step=50/500, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r22\_refined\_fixed\_gate (1790803305054490194\_r22\_refined\_fixed\_gate\_72a52fe0): running, step=50/500, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r24\_refined\_fixed\_scale (1790803305061375040\_r24\_refined\_fixed\_scale\_4edeef26): running, step=50/500, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r26\_refined\_local\_global (1790803305068196112\_r26\_refined\_local\_global\_5bf50a1c): running, step=50/500, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r20\_refined\_control (1790803305045499169\_r20\_refined\_control\_a37dc5ab): running, step=50/500, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r21\_refined\_head\_gate (1790803305051081571\_r21\_refined\_head\_gate\_9cc7449e): running, step=50/500, alerts=[&#x27;end\_not\_learned\_yet&#x27;]
- 2026-09-30 21:57:35 UTC — r20\_refined\_control (1790803305045499169\_r20\_refined\_control\_a37dc5ab): running, step=75/500, alerts=[&#x27;end\_not\_learned\_yet&#x27;]
- 2026-09-30 21:58:07 UTC — r23\_refined\_layerscale (1790803305057942502\_r23\_refined\_layerscale\_56d8467a): running, step=75/500, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r25\_refined\_late\_fusion (1790803305064798754\_r25\_refined\_late\_fusion\_5cd5e385): running, step=75/500, alerts=[]; r27\_refined\_value\_residual (1790803305071555086\_r27\_refined\_value\_residual\_52f013e5): running, step=75/500, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r22\_refined\_fixed\_gate (1790803305054490194\_r22\_refined\_fixed\_gate\_72a52fe0): running, step=75/500, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r24\_refined\_fixed\_scale (1790803305061375040\_r24\_refined\_fixed\_scale\_4edeef26): running, step=75/500, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r26\_refined\_local\_global (1790803305068196112\_r26\_refined\_local\_global\_5bf50a1c): running, step=75/500, alerts=[]; r21\_refined\_head\_gate (1790803305051081571\_r21\_refined\_head\_gate\_9cc7449e): running, step=75/500, alerts=[&#x27;end\_not\_learned\_yet&#x27;]
- 2026-09-30 22:09:38 UTC — r23\_refined\_layerscale (1790803305057942502\_r23\_refined\_layerscale\_56d8467a): running, step=100/500, alerts=[]; r25\_refined\_late\_fusion (1790803305064798754\_r25\_refined\_late\_fusion\_5cd5e385): running, step=100/500, alerts=[]; r27\_refined\_value\_residual (1790803305071555086\_r27\_refined\_value\_residual\_52f013e5): running, step=100/500, alerts=[]; r22\_refined\_fixed\_gate (1790803305054490194\_r22\_refined\_fixed\_gate\_72a52fe0): running, step=100/500, alerts=[]; r24\_refined\_fixed\_scale (1790803305061375040\_r24\_refined\_fixed\_scale\_4edeef26): running, step=100/500, alerts=[]; r26\_refined\_local\_global (1790803305068196112\_r26\_refined\_local\_global\_5bf50a1c): running, step=100/500, alerts=[]; r20\_refined\_control (1790803305045499169\_r20\_refined\_control\_a37dc5ab): running, step=100/500, alerts=[]; r21\_refined\_head\_gate (1790803305051081571\_r21\_refined\_head\_gate\_9cc7449e): running, step=100/500, alerts=[&#x27;end\_not\_learned\_yet&#x27;]
- 2026-09-30 22:21:43 UTC — r23\_refined\_layerscale (1790803305057942502\_r23\_refined\_layerscale\_56d8467a): running, step=125/500, alerts=[]; r25\_refined\_late\_fusion (1790803305064798754\_r25\_refined\_late\_fusion\_5cd5e385): running, step=125/500, alerts=[]; r27\_refined\_value\_residual (1790803305071555086\_r27\_refined\_value\_residual\_52f013e5): running, step=125/500, alerts=[]; r22\_refined\_fixed\_gate (1790803305054490194\_r22\_refined\_fixed\_gate\_72a52fe0): running, step=125/500, alerts=[]; r24\_refined\_fixed\_scale (1790803305061375040\_r24\_refined\_fixed\_scale\_4edeef26): running, step=125/500, alerts=[]; r26\_refined\_local\_global (1790803305068196112\_r26\_refined\_local\_global\_5bf50a1c): running, step=125/500, alerts=[]; r20\_refined\_control (1790803305045499169\_r20\_refined\_control\_a37dc5ab): running, step=125/500, alerts=[]; r21\_refined\_head\_gate (1790803305051081571\_r21\_refined\_head\_gate\_9cc7449e): running, step=125/500, alerts=[]
- 2026-09-30 22:33:49 UTC — r23\_refined\_layerscale (1790803305057942502\_r23\_refined\_layerscale\_56d8467a): running, step=150/500, alerts=[]; r25\_refined\_late\_fusion (1790803305064798754\_r25\_refined\_late\_fusion\_5cd5e385): running, step=150/500, alerts=[]; r27\_refined\_value\_residual (1790803305071555086\_r27\_refined\_value\_residual\_52f013e5): running, step=150/500, alerts=[]; r22\_refined\_fixed\_gate (1790803305054490194\_r22\_refined\_fixed\_gate\_72a52fe0): running, step=150/500, alerts=[]; r24\_refined\_fixed\_scale (1790803305061375040\_r24\_refined\_fixed\_scale\_4edeef26): running, step=150/500, alerts=[]; r26\_refined\_local\_global (1790803305068196112\_r26\_refined\_local\_global\_5bf50a1c): running, step=150/500, alerts=[]; r20\_refined\_control (1790803305045499169\_r20\_refined\_control\_a37dc5ab): running, step=150/500, alerts=[]; r21\_refined\_head\_gate (1790803305051081571\_r21\_refined\_head\_gate\_9cc7449e): running, step=150/500, alerts=[]
- 2026-09-30 22:45:25 UTC — r20\_refined\_control (1790803305045499169\_r20\_refined\_control\_a37dc5ab): running, step=175/500, alerts=[]
- 2026-09-30 22:45:56 UTC — r23\_refined\_layerscale (1790803305057942502\_r23\_refined\_layerscale\_56d8467a): running, step=175/500, alerts=[]; r25\_refined\_late\_fusion (1790803305064798754\_r25\_refined\_late\_fusion\_5cd5e385): running, step=175/500, alerts=[]; r27\_refined\_value\_residual (1790803305071555086\_r27\_refined\_value\_residual\_52f013e5): running, step=175/500, alerts=[]; r22\_refined\_fixed\_gate (1790803305054490194\_r22\_refined\_fixed\_gate\_72a52fe0): running, step=175/500, alerts=[]; r24\_refined\_fixed\_scale (1790803305061375040\_r24\_refined\_fixed\_scale\_4edeef26): running, step=175/500, alerts=[]; r26\_refined\_local\_global (1790803305068196112\_r26\_refined\_local\_global\_5bf50a1c): running, step=175/500, alerts=[]; r21\_refined\_head\_gate (1790803305051081571\_r21\_refined\_head\_gate\_9cc7449e): running, step=175/500, alerts=[]
- 2026-09-30 22:57:32 UTC — r23\_refined\_layerscale (1790803305057942502\_r23\_refined\_layerscale\_56d8467a): running, step=200/500, alerts=[]; r25\_refined\_late\_fusion (1790803305064798754\_r25\_refined\_late\_fusion\_5cd5e385): running, step=200/500, alerts=[]; r27\_refined\_value\_residual (1790803305071555086\_r27\_refined\_value\_residual\_52f013e5): running, step=200/500, alerts=[]; r22\_refined\_fixed\_gate (1790803305054490194\_r22\_refined\_fixed\_gate\_72a52fe0): running, step=200/500, alerts=[]; r24\_refined\_fixed\_scale (1790803305061375040\_r24\_refined\_fixed\_scale\_4edeef26): running, step=200/500, alerts=[]; r26\_refined\_local\_global (1790803305068196112\_r26\_refined\_local\_global\_5bf50a1c): running, step=200/500, alerts=[]; r20\_refined\_control (1790803305045499169\_r20\_refined\_control\_a37dc5ab): running, step=200/500, alerts=[]; r21\_refined\_head\_gate (1790803305051081571\_r21\_refined\_head\_gate\_9cc7449e): running, step=200/500, alerts=[]
- 2026-09-30 23:09:09 UTC — r20\_refined\_control (1790803305045499169\_r20\_refined\_control\_a37dc5ab): running, step=225/500, alerts=[]
- 2026-09-30 23:09:40 UTC — r23\_refined\_layerscale (1790803305057942502\_r23\_refined\_layerscale\_56d8467a): running, step=225/500, alerts=[]; r25\_refined\_late\_fusion (1790803305064798754\_r25\_refined\_late\_fusion\_5cd5e385): running, step=225/500, alerts=[]; r27\_refined\_value\_residual (1790803305071555086\_r27\_refined\_value\_residual\_52f013e5): running, step=225/500, alerts=[]; r22\_refined\_fixed\_gate (1790803305054490194\_r22\_refined\_fixed\_gate\_72a52fe0): running, step=225/500, alerts=[]; r24\_refined\_fixed\_scale (1790803305061375040\_r24\_refined\_fixed\_scale\_4edeef26): running, step=225/500, alerts=[]; r26\_refined\_local\_global (1790803305068196112\_r26\_refined\_local\_global\_5bf50a1c): running, step=225/500, alerts=[]; r21\_refined\_head\_gate (1790803305051081571\_r21\_refined\_head\_gate\_9cc7449e): running, step=225/500, alerts=[]
- 2026-09-30 23:21:20 UTC — r23\_refined\_layerscale (1790803305057942502\_r23\_refined\_layerscale\_56d8467a): running, step=250/500, alerts=[]; r25\_refined\_late\_fusion (1790803305064798754\_r25\_refined\_late\_fusion\_5cd5e385): running, step=250/500, alerts=[]; r27\_refined\_value\_residual (1790803305071555086\_r27\_refined\_value\_residual\_52f013e5): running, step=250/500, alerts=[]; r22\_refined\_fixed\_gate (1790803305054490194\_r22\_refined\_fixed\_gate\_72a52fe0): running, step=250/500, alerts=[]; r24\_refined\_fixed\_scale (1790803305061375040\_r24\_refined\_fixed\_scale\_4edeef26): running, step=250/500, alerts=[]; r26\_refined\_local\_global (1790803305068196112\_r26\_refined\_local\_global\_5bf50a1c): running, step=250/500, alerts=[]; r20\_refined\_control (1790803305045499169\_r20\_refined\_control\_a37dc5ab): running, step=250/500, alerts=[]
- 2026-09-30 23:21:52 UTC — r21\_refined\_head\_gate (1790803305051081571\_r21\_refined\_head\_gate\_9cc7449e): running, step=250/500, alerts=[]
- 2026-09-30 23:33:01 UTC — r24\_refined\_fixed\_scale (1790803305061375040\_r24\_refined\_fixed\_scale\_4edeef26): running, step=275/500, alerts=[]; r20\_refined\_control (1790803305045499169\_r20\_refined\_control\_a37dc5ab): running, step=275/500, alerts=[]
- 2026-09-30 23:33:33 UTC — r23\_refined\_layerscale (1790803305057942502\_r23\_refined\_layerscale\_56d8467a): running, step=275/500, alerts=[]; r25\_refined\_late\_fusion (1790803305064798754\_r25\_refined\_late\_fusion\_5cd5e385): running, step=275/500, alerts=[]; r27\_refined\_value\_residual (1790803305071555086\_r27\_refined\_value\_residual\_52f013e5): running, step=275/500, alerts=[]; r22\_refined\_fixed\_gate (1790803305054490194\_r22\_refined\_fixed\_gate\_72a52fe0): running, step=275/500, alerts=[]; r26\_refined\_local\_global (1790803305068196112\_r26\_refined\_local\_global\_5bf50a1c): running, step=275/500, alerts=[]; r21\_refined\_head\_gate (1790803305051081571\_r21\_refined\_head\_gate\_9cc7449e): running, step=275/500, alerts=[]
- 2026-09-30 23:45:13 UTC — r23\_refined\_layerscale (1790803305057942502\_r23\_refined\_layerscale\_56d8467a): running, step=300/500, alerts=[]; r25\_refined\_late\_fusion (1790803305064798754\_r25\_refined\_late\_fusion\_5cd5e385): running, step=300/500, alerts=[]; r27\_refined\_value\_residual (1790803305071555086\_r27\_refined\_value\_residual\_52f013e5): running, step=300/500, alerts=[]; r22\_refined\_fixed\_gate (1790803305054490194\_r22\_refined\_fixed\_gate\_72a52fe0): running, step=300/500, alerts=[]; r24\_refined\_fixed\_scale (1790803305061375040\_r24\_refined\_fixed\_scale\_4edeef26): running, step=300/500, alerts=[]; r26\_refined\_local\_global (1790803305068196112\_r26\_refined\_local\_global\_5bf50a1c): running, step=300/500, alerts=[]; r20\_refined\_control (1790803305045499169\_r20\_refined\_control\_a37dc5ab): running, step=300/500, alerts=[]; r21\_refined\_head\_gate (1790803305051081571\_r21\_refined\_head\_gate\_9cc7449e): running, step=300/500, alerts=[]
- 2026-09-30 23:56:57 UTC — r22\_refined\_fixed\_gate (1790803305054490194\_r22\_refined\_fixed\_gate\_72a52fe0): running, step=325/500, alerts=[]; r24\_refined\_fixed\_scale (1790803305061375040\_r24\_refined\_fixed\_scale\_4edeef26): running, step=325/500, alerts=[]; r20\_refined\_control (1790803305045499169\_r20\_refined\_control\_a37dc5ab): running, step=325/500, alerts=[]
- 2026-09-30 23:57:29 UTC — r23\_refined\_layerscale (1790803305057942502\_r23\_refined\_layerscale\_56d8467a): running, step=325/500, alerts=[]; r25\_refined\_late\_fusion (1790803305064798754\_r25\_refined\_late\_fusion\_5cd5e385): running, step=325/500, alerts=[]; r27\_refined\_value\_residual (1790803305071555086\_r27\_refined\_value\_residual\_52f013e5): running, step=325/500, alerts=[]; r26\_refined\_local\_global (1790803305068196112\_r26\_refined\_local\_global\_5bf50a1c): running, step=325/500, alerts=[]; r21\_refined\_head\_gate (1790803305051081571\_r21\_refined\_head\_gate\_9cc7449e): running, step=325/500, alerts=[]
- 2026-10-01 00:08:55 UTC — r23\_refined\_layerscale (1790803305057942502\_r23\_refined\_layerscale\_56d8467a): running, step=350/500, alerts=[]; r25\_refined\_late\_fusion (1790803305064798754\_r25\_refined\_late\_fusion\_5cd5e385): running, step=350/500, alerts=[]; r27\_refined\_value\_residual (1790803305071555086\_r27\_refined\_value\_residual\_52f013e5): running, step=350/500, alerts=[]; r22\_refined\_fixed\_gate (1790803305054490194\_r22\_refined\_fixed\_gate\_72a52fe0): running, step=350/500, alerts=[]; r24\_refined\_fixed\_scale (1790803305061375040\_r24\_refined\_fixed\_scale\_4edeef26): running, step=350/500, alerts=[]; r26\_refined\_local\_global (1790803305068196112\_r26\_refined\_local\_global\_5bf50a1c): running, step=350/500, alerts=[]; r20\_refined\_control (1790803305045499169\_r20\_refined\_control\_a37dc5ab): running, step=350/500, alerts=[]
- 2026-10-01 00:09:27 UTC — r21\_refined\_head\_gate (1790803305051081571\_r21\_refined\_head\_gate\_9cc7449e): running, step=350/500, alerts=[]
- 2026-10-01 00:20:40 UTC — r20\_refined\_control (1790803305045499169\_r20\_refined\_control\_a37dc5ab): running, step=375/500, alerts=[]
- 2026-10-01 00:21:12 UTC — r23\_refined\_layerscale (1790803305057942502\_r23\_refined\_layerscale\_56d8467a): running, step=375/500, alerts=[]; r25\_refined\_late\_fusion (1790803305064798754\_r25\_refined\_late\_fusion\_5cd5e385): running, step=375/500, alerts=[]; r27\_refined\_value\_residual (1790803305071555086\_r27\_refined\_value\_residual\_52f013e5): running, step=375/500, alerts=[]; r22\_refined\_fixed\_gate (1790803305054490194\_r22\_refined\_fixed\_gate\_72a52fe0): running, step=375/500, alerts=[]; r24\_refined\_fixed\_scale (1790803305061375040\_r24\_refined\_fixed\_scale\_4edeef26): running, step=375/500, alerts=[]; r26\_refined\_local\_global (1790803305068196112\_r26\_refined\_local\_global\_5bf50a1c): running, step=375/500, alerts=[]; r21\_refined\_head\_gate (1790803305051081571\_r21\_refined\_head\_gate\_9cc7449e): running, step=375/500, alerts=[]
- 2026-10-01 00:32:58 UTC — r23\_refined\_layerscale (1790803305057942502\_r23\_refined\_layerscale\_56d8467a): running, step=400/500, alerts=[]; r25\_refined\_late\_fusion (1790803305064798754\_r25\_refined\_late\_fusion\_5cd5e385): running, step=400/500, alerts=[]; r27\_refined\_value\_residual (1790803305071555086\_r27\_refined\_value\_residual\_52f013e5): running, step=400/500, alerts=[]; r22\_refined\_fixed\_gate (1790803305054490194\_r22\_refined\_fixed\_gate\_72a52fe0): running, step=400/500, alerts=[]; r24\_refined\_fixed\_scale (1790803305061375040\_r24\_refined\_fixed\_scale\_4edeef26): running, step=400/500, alerts=[]; r26\_refined\_local\_global (1790803305068196112\_r26\_refined\_local\_global\_5bf50a1c): running, step=400/500, alerts=[]; r20\_refined\_control (1790803305045499169\_r20\_refined\_control\_a37dc5ab): running, step=400/500, alerts=[]
- 2026-10-01 00:33:31 UTC — r21\_refined\_head\_gate (1790803305051081571\_r21\_refined\_head\_gate\_9cc7449e): running, step=400/500, alerts=[]
- 2026-10-01 00:44:48 UTC — r23\_refined\_layerscale (1790803305057942502\_r23\_refined\_layerscale\_56d8467a): running, step=425/500, alerts=[]; r27\_refined\_value\_residual (1790803305071555086\_r27\_refined\_value\_residual\_52f013e5): running, step=425/500, alerts=[]; r22\_refined\_fixed\_gate (1790803305054490194\_r22\_refined\_fixed\_gate\_72a52fe0): running, step=425/500, alerts=[]; r24\_refined\_fixed\_scale (1790803305061375040\_r24\_refined\_fixed\_scale\_4edeef26): running, step=425/500, alerts=[]; r20\_refined\_control (1790803305045499169\_r20\_refined\_control\_a37dc5ab): running, step=425/500, alerts=[]
- 2026-10-01 00:45:20 UTC — r25\_refined\_late\_fusion (1790803305064798754\_r25\_refined\_late\_fusion\_5cd5e385): running, step=425/500, alerts=[]; r26\_refined\_local\_global (1790803305068196112\_r26\_refined\_local\_global\_5bf50a1c): running, step=425/500, alerts=[]; r21\_refined\_head\_gate (1790803305051081571\_r21\_refined\_head\_gate\_9cc7449e): running, step=425/500, alerts=[]
- 2026-10-01 00:56:37 UTC — r24\_refined\_fixed\_scale (1790803305061375040\_r24\_refined\_fixed\_scale\_4edeef26): running, step=450/500, alerts=[]; r20\_refined\_control (1790803305045499169\_r20\_refined\_control\_a37dc5ab): running, step=450/500, alerts=[]
- 2026-10-01 00:57:09 UTC — r23\_refined\_layerscale (1790803305057942502\_r23\_refined\_layerscale\_56d8467a): running, step=450/500, alerts=[]; r25\_refined\_late\_fusion (1790803305064798754\_r25\_refined\_late\_fusion\_5cd5e385): running, step=450/500, alerts=[]; r27\_refined\_value\_residual (1790803305071555086\_r27\_refined\_value\_residual\_52f013e5): running, step=450/500, alerts=[]; r22\_refined\_fixed\_gate (1790803305054490194\_r22\_refined\_fixed\_gate\_72a52fe0): running, step=450/500, alerts=[]; r26\_refined\_local\_global (1790803305068196112\_r26\_refined\_local\_global\_5bf50a1c): running, step=450/500, alerts=[]; r21\_refined\_head\_gate (1790803305051081571\_r21\_refined\_head\_gate\_9cc7449e): running, step=450/500, alerts=[]
- 2026-10-01 01:08:28 UTC — r20\_refined\_control (1790803305045499169\_r20\_refined\_control\_a37dc5ab): running, step=475/500, alerts=[]
- 2026-10-01 01:09:00 UTC — r23\_refined\_layerscale (1790803305057942502\_r23\_refined\_layerscale\_56d8467a): running, step=475/500, alerts=[]; r25\_refined\_late\_fusion (1790803305064798754\_r25\_refined\_late\_fusion\_5cd5e385): running, step=475/500, alerts=[]; r27\_refined\_value\_residual (1790803305071555086\_r27\_refined\_value\_residual\_52f013e5): running, step=475/500, alerts=[]; r22\_refined\_fixed\_gate (1790803305054490194\_r22\_refined\_fixed\_gate\_72a52fe0): running, step=475/500, alerts=[]; r24\_refined\_fixed\_scale (1790803305061375040\_r24\_refined\_fixed\_scale\_4edeef26): running, step=475/500, alerts=[]; r26\_refined\_local\_global (1790803305068196112\_r26\_refined\_local\_global\_5bf50a1c): running, step=475/500, alerts=[]; r21\_refined\_head\_gate (1790803305051081571\_r21\_refined\_head\_gate\_9cc7449e): running, step=475/500, alerts=[]
- 2026-10-01 01:20:19 UTC — r24\_refined\_fixed\_scale (1790803305061375040\_r24\_refined\_fixed\_scale\_4edeef26): running, step=500/500, alerts=[]; r20\_refined\_control (1790803305045499169\_r20\_refined\_control\_a37dc5ab): succeeded, step=500/500, alerts=[]
- 2026-10-01 01:20:52 UTC — r23\_refined\_layerscale (1790803305057942502\_r23\_refined\_layerscale\_56d8467a): succeeded, step=500/500, alerts=[]; r25\_refined\_late\_fusion (1790803305064798754\_r25\_refined\_late\_fusion\_5cd5e385): succeeded, step=500/500, alerts=[]; r27\_refined\_value\_residual (1790803305071555086\_r27\_refined\_value\_residual\_52f013e5): succeeded, step=500/500, alerts=[]; r22\_refined\_fixed\_gate (1790803305054490194\_r22\_refined\_fixed\_gate\_72a52fe0): succeeded, step=500/500, alerts=[]; r24\_refined\_fixed\_scale (1790803305061375040\_r24\_refined\_fixed\_scale\_4edeef26): succeeded, step=500/500, alerts=[]; r26\_refined\_local\_global (1790803305068196112\_r26\_refined\_local\_global\_5bf50a1c): succeeded, step=500/500, alerts=[]; r21\_refined\_head\_gate (1790803305051081571\_r21\_refined\_head\_gate\_9cc7449e): succeeded, step=500/500, alerts=[]
- 2026-10-01 01:31:12 UTC — r21\_refined\_head\_gate (1790802160343981887\_r21\_refined\_head\_gate\_80d10a6d): succeeded, step=25/25, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r26\_refined\_local\_global (1790802160358902166\_r26\_refined\_local\_global\_905c9acd): succeeded, step=25/25, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r23\_refined\_layerscale (1790803305057942502\_r23\_refined\_layerscale\_56d8467a): succeeded, step=500/500, alerts=[]; r25\_refined\_late\_fusion (1790803305064798754\_r25\_refined\_late\_fusion\_5cd5e385): succeeded, step=500/500, alerts=[]; r27\_refined\_value\_residual (1790803305071555086\_r27\_refined\_value\_residual\_52f013e5): succeeded, step=500/500, alerts=[]; r27\_refined\_value\_residual (1790802160361769452\_r27\_refined\_value\_residual\_0b46ff76): succeeded, step=25/25, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r23\_refined\_layerscale (1790802160350276097\_r23\_refined\_layerscale\_eb29f0d4): succeeded, step=25/25, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r22\_refined\_fixed\_gate (1790803305054490194\_r22\_refined\_fixed\_gate\_72a52fe0): succeeded, step=500/500, alerts=[]; r24\_refined\_fixed\_scale (1790803305061375040\_r24\_refined\_fixed\_scale\_4edeef26): succeeded, step=500/500, alerts=[]; r22\_refined\_fixed\_gate (1790802160346881056\_r22\_refined\_fixed\_gate\_263516a3): succeeded, step=25/25, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r26\_refined\_local\_global (1790803305068196112\_r26\_refined\_local\_global\_5bf50a1c): succeeded, step=500/500, alerts=[]; r20\_refined\_control (1790802160338779408\_r20\_refined\_control\_3b41bffe): succeeded, step=25/25, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r25\_refined\_late\_fusion (1790802160356028347\_r25\_refined\_late\_fusion\_4640596c): succeeded, step=25/25, alerts=[&#x27;end\_not\_learned\_yet&#x27;]; r20\_refined\_control (1790803305045499169\_r20\_refined\_control\_a37dc5ab): succeeded, step=500/500, alerts=[]; r21\_refined\_head\_gate (1790803305051081571\_r21\_refined\_head\_gate\_9cc7449e): succeeded, step=500/500, alerts=[]; r24\_refined\_fixed\_scale (1790802160353131370\_r24\_refined\_fixed\_scale\_c6e0089d): succeeded, step=25/25, alerts=[&#x27;end\_not\_learned\_yet&#x27;]
- 2026-10-01 03:08:38 UTC — d01\_pilot\_runtime\_gpu\_r20\_v1 (1790824088674261408\_d01\_pilot\_runtime\_gpu\_r20\_v1\_9af03156): running, step=None/None, alerts=[]
- 2026-10-01 03:20:36 UTC — d01\_pilot\_runtime\_gpu\_r20\_v1 (1790824088674261408\_d01\_pilot\_runtime\_gpu\_r20\_v1\_9af03156): failed, step=None/None, alerts=[&#x27;trainer\_diagnostic\_execution\_failed&#x27;]
- 2026-10-01 03:56:44 UTC — d01\_pilot\_runtime\_gpu\_r20\_v1 (1790824088674261408\_d01\_pilot\_runtime\_gpu\_r20\_v1\_9af03156): failed, step=None/None, alerts=[&#x27;trainer\_diagnostic\_execution\_failed&#x27;]
- 2026-10-01 04:01:07 UTC — d01\_pilot\_runtime\_gpu\_r20\_v1 (1790827241091679008\_d01\_pilot\_runtime\_gpu\_r20\_v1\_ea0bba4a): running, step=None/None, alerts=[]
- 2026-10-01 04:13:09 UTC — d01\_pilot\_runtime\_gpu\_r20\_v1 (1790827241091679008\_d01\_pilot\_runtime\_gpu\_r20\_v1\_ea0bba4a): failed, step=None/None, alerts=[&#x27;trainer\_diagnostic\_execution\_failed&#x27;]
- 2026-10-01 05:15:02 UTC — d01\_pilot\_runtime\_gpu\_r20\_v1 (1790831680607223314\_d01\_pilot\_runtime\_gpu\_r20\_v1\_48f59959): running, step=None/None, alerts=[]
- 2026-10-01 05:45:19 UTC — d01\_pilot\_runtime\_gpu\_r20\_v1 (1790831680607223314\_d01\_pilot\_runtime\_gpu\_r20\_v1\_48f59959): succeeded, step=None/None, alerts=[]
- 2026-10-01 07:43:42 UTC — r24\_refined\_fixed\_scale (1790840619270196057\_r24\_refined\_fixed\_scale\_e01f3f97): pending, step=0/1, alerts=[]; r26\_refined\_local\_global (1790840619276677050\_r26\_refined\_local\_global\_e113e0e8): pending, step=0/1, alerts=[]; r21\_refined\_head\_gate (1790840619261329817\_r21\_refined\_head\_gate\_c27b5443): running, step=0/1, alerts=[]
- 2026-10-01 07:44:15 UTC — r24\_refined\_fixed\_scale (1790840619270196057\_r24\_refined\_fixed\_scale\_e01f3f97): running, step=0/1, alerts=[]; r26\_refined\_local\_global (1790840619276677050\_r26\_refined\_local\_global\_e113e0e8): running, step=0/1, alerts=[]
- 2026-10-01 08:14:02 UTC — r24\_refined\_fixed\_scale (1790840619270196057\_r24\_refined\_fixed\_scale\_e01f3f97): succeeded, step=None/None, alerts=[&#x27;trainer\_diagnostic\_invalid\_envelope&#x27;, &#x27;trainer\_diagnostic\_runtime\_identity\_mismatch&#x27;]; r21\_refined\_head\_gate (1790840619261329817\_r21\_refined\_head\_gate\_c27b5443): succeeded, step=None/None, alerts=[&#x27;trainer\_diagnostic\_invalid\_envelope&#x27;, &#x27;trainer\_diagnostic\_runtime\_identity\_mismatch&#x27;]; r26\_refined\_local\_global (1790840619276677050\_r26\_refined\_local\_global\_e113e0e8): succeeded, step=None/None, alerts=[&#x27;trainer\_diagnostic\_invalid\_envelope&#x27;, &#x27;trainer\_diagnostic\_runtime\_identity\_mismatch&#x27;]
- 2026-10-02 01:54:16 UTC — d02\_pilot\_seed\_gpu\_r26\_seed42\_v2 (1790906045332754556\_d02\_pilot\_seed\_gpu\_r26\_seed42\_v2\_83205e87): running, step=0/1, alerts=[]; d02\_pilot\_seed\_gpu\_r26\_seed43\_v2 (1790906045339673599\_d02\_pilot\_seed\_gpu\_r26\_seed43\_v2\_19e36eda): running, step=0/1, alerts=[]; d02\_pilot\_seed\_gpu\_r20\_seed43\_v2 (1790906045298475453\_d02\_pilot\_seed\_gpu\_r20\_seed43\_v2\_a9a6be7e): running, step=0/1, alerts=[]; d02\_pilot\_seed\_gpu\_r24\_seed42\_v2 (1790906045319006048\_d02\_pilot\_seed\_gpu\_r24\_seed42\_v2\_57a0529c): running, step=0/1, alerts=[]; d02\_pilot\_seed\_gpu\_r21\_seed43\_v2 (1790906045312184986\_d02\_pilot\_seed\_gpu\_r21\_seed43\_v2\_1e0b6229): running, step=0/1, alerts=[]; d02\_pilot\_seed\_gpu\_r21\_seed42\_v2 (1790906045305519301\_d02\_pilot\_seed\_gpu\_r21\_seed42\_v2\_82caee86): running, step=0/1, alerts=[]; d02\_pilot\_seed\_gpu\_r24\_seed43\_v2 (1790906045325720427\_d02\_pilot\_seed\_gpu\_r24\_seed43\_v2\_32829b7f): running, step=0/1, alerts=[]; d02\_pilot\_seed\_gpu\_r20\_seed42\_v2 (1790906045287527921\_d02\_pilot\_seed\_gpu\_r20\_seed42\_v2\_786cccfe): running, step=0/1, alerts=[]
- 2026-10-02 01:57:04 UTC — d02\_pilot\_seed\_gpu\_r21\_seed43\_v2 (1790906045312184986\_d02\_pilot\_seed\_gpu\_r21\_seed43\_v2\_1e0b6229): failed, step=0/1, alerts=[&#x27;runner\_failure&#x27;]
- 2026-10-02 02:26:55 UTC — d02\_pilot\_seed\_gpu\_r26\_seed42\_v2 (1790906045332754556\_d02\_pilot\_seed\_gpu\_r26\_seed42\_v2\_83205e87): succeeded, step=0/1, alerts=[&#x27;inconsistent\_completion&#x27;]; d02\_pilot\_seed\_gpu\_r26\_seed43\_v2 (1790906045339673599\_d02\_pilot\_seed\_gpu\_r26\_seed43\_v2\_19e36eda): succeeded, step=0/1, alerts=[&#x27;inconsistent\_completion&#x27;]; d02\_pilot\_seed\_gpu\_r20\_seed43\_v2 (1790906045298475453\_d02\_pilot\_seed\_gpu\_r20\_seed43\_v2\_a9a6be7e): succeeded, step=0/1, alerts=[&#x27;inconsistent\_completion&#x27;]; d02\_pilot\_seed\_gpu\_r24\_seed42\_v2 (1790906045319006048\_d02\_pilot\_seed\_gpu\_r24\_seed42\_v2\_57a0529c): succeeded, step=0/1, alerts=[&#x27;inconsistent\_completion&#x27;]; d02\_pilot\_seed\_gpu\_r21\_seed42\_v2 (1790906045305519301\_d02\_pilot\_seed\_gpu\_r21\_seed42\_v2\_82caee86): succeeded, step=0/1, alerts=[&#x27;inconsistent\_completion&#x27;]; d02\_pilot\_seed\_gpu\_r24\_seed43\_v2 (1790906045325720427\_d02\_pilot\_seed\_gpu\_r24\_seed43\_v2\_32829b7f): succeeded, step=0/1, alerts=[&#x27;inconsistent\_completion&#x27;]; d02\_pilot\_seed\_gpu\_r20\_seed42\_v2 (1790906045287527921\_d02\_pilot\_seed\_gpu\_r20\_seed42\_v2\_786cccfe): succeeded, step=0/1, alerts=[&#x27;inconsistent\_completion&#x27;]
- 2026-10-02 03:13:53 UTC — d03\_r21\_seed43\_tiny\_gpu\_observation\_v1 (1790910822468354321\_d03\_r21\_seed43\_tiny\_gpu\_observation\_v1\_58e84271): running, step=0/1, alerts=[]
- 2026-10-02 03:14:27 UTC — d03\_r21\_seed43\_tiny\_gpu\_observation\_v1 (1790910822468354321\_d03\_r21\_seed43\_tiny\_gpu\_observation\_v1\_58e84271): failed, step=0/1, alerts=[&#x27;runner\_failure&#x27;]
- 2026-10-02 03:25:41 UTC — d03\_r21\_seed43\_tiny\_gpu\_observation\_v1 (1790911529399018156\_d03\_r21\_seed43\_tiny\_gpu\_observation\_v1\_6acb659f): running, step=0/1, alerts=[]
- 2026-10-02 03:28:00 UTC — d03\_r21\_seed43\_tiny\_gpu\_observation\_v1 (1790911529399018156\_d03\_r21\_seed43\_tiny\_gpu\_observation\_v1\_6acb659f): failed, step=0/1, alerts=[&#x27;runner\_failure&#x27;]
- 2026-10-02 03:38:40 UTC — d03\_r21\_seed43\_tiny\_gpu\_observation\_v1 (1790912303689399341\_d03\_r21\_seed43\_tiny\_gpu\_observation\_v1\_324e2b1c): running, step=0/1, alerts=[]
- 2026-10-02 03:39:14 UTC — d03\_r21\_seed43\_tiny\_gpu\_observation\_v1 (1790912303689399341\_d03\_r21\_seed43\_tiny\_gpu\_observation\_v1\_324e2b1c): succeeded, step=0/1, alerts=[&#x27;inconsistent\_completion&#x27;]
- 2026-10-02 04:07:46 UTC — d03\_r21\_seed43\_tiny\_gpu\_observation\_v1 (1790914034411779826\_d03\_r21\_seed43\_tiny\_gpu\_observation\_v1\_e950d2a0): succeeded, step=0/1, alerts=[&#x27;inconsistent\_completion&#x27;]
- 2026-10-02 04:31:48 UTC — d03\_r21\_seed43\_tiny\_gpu\_observation\_v1 (1790915494600058674\_d03\_r21\_seed43\_tiny\_gpu\_observation\_v1\_5d224d8a): running, step=0/1, alerts=[]
- 2026-10-02 04:32:21 UTC — d03\_r21\_seed43\_tiny\_gpu\_observation\_v1 (1790915494600058674\_d03\_r21\_seed43\_tiny\_gpu\_observation\_v1\_5d224d8a): succeeded, step=0/1, alerts=[&#x27;inconsistent\_completion&#x27;]
- 2026-10-02 04:59:49 UTC — d03\_r21\_seed43\_tiny\_gpu\_observation\_v1 (1790917182885479822\_d03\_r21\_seed43\_tiny\_gpu\_observation\_v1\_6c5bc4a4): running, step=0/1, alerts=[]
- 2026-10-02 05:00:22 UTC — d03\_r21\_seed43\_tiny\_gpu\_observation\_v1 (1790917182885479822\_d03\_r21\_seed43\_tiny\_gpu\_observation\_v1\_6c5bc4a4): succeeded, step=0/1, alerts=[&#x27;inconsistent\_completion&#x27;]
- 2026-10-02 05:26:03 UTC — d02\_pilot\_seed\_gpu\_r21\_seed43\_v2 (1790918753525278168\_d02\_pilot\_seed\_gpu\_r21\_seed43\_v2\_82aa7438): running, step=0/1, alerts=[]
- 2026-10-02 05:58:26 UTC — d02\_pilot\_seed\_gpu\_r21\_seed43\_v2 (1790918753525278168\_d02\_pilot\_seed\_gpu\_r21\_seed43\_v2\_82aa7438): succeeded, step=0/1, alerts=[&#x27;source\_changed&#x27;, &#x27;inconsistent\_completion&#x27;]
- 2026-10-02 06:37:02 UTC — d04\_r21\_CE\_learning\_v1 (1790923015794991437\_d04\_r21\_CE\_learning\_v1\_4dc72fa7): running, step=0/25, alerts=[]
- 2026-10-02 06:49:54 UTC — d04\_r21\_CE\_learning\_v1 (1790923015794991437\_d04\_r21\_CE\_learning\_v1\_4dc72fa7): succeeded, step=0/25, alerts=[&#x27;inconsistent\_completion&#x27;]
- 2026-10-02 06:57:11 UTC — d04\_r21\_CE\_learning\_v1 (1790924220633707073\_d04\_r21\_CE\_learning\_v1\_dd5651b2): running, step=0/500, alerts=[]
- 2026-10-02 07:42:36 UTC — d05\_r26\_seed42\_learning\_v1 (1790926923188130232\_d05\_r26\_seed42\_learning\_v1\_5c83be2b): running, step=0/25, alerts=[]; d05\_r24\_seed42\_learning\_v1 (1790926923172990505\_d05\_r24\_seed42\_learning\_v1\_85f994fa): running, step=0/25, alerts=[]; d05\_r20\_seed43\_learning\_v1 (1790926923157371090\_d05\_r20\_seed43\_learning\_v1\_4b31a0c1): running, step=0/25, alerts=[]; d05\_r24\_seed43\_learning\_v1 (1790926923180466754\_d05\_r24\_seed43\_learning\_v1\_47ed9473): running, step=0/25, alerts=[]; d05\_r21\_seed42\_learning\_v1 (1790926923165197398\_d05\_r21\_seed42\_learning\_v1\_12e96433): running, step=0/25, alerts=[]; d05\_r26\_seed43\_learning\_v1 (1790926923199830052\_d05\_r26\_seed43\_learning\_v1\_1b590a01): running, step=0/25, alerts=[]; d05\_r20\_seed42\_learning\_v1 (1790926923146311111\_d05\_r20\_seed42\_learning\_v1\_cc8db6a6): running, step=0/25, alerts=[]
- 2026-10-02 07:54:54 UTC — d05\_r26\_seed42\_learning\_v1 (1790926923188130232\_d05\_r26\_seed42\_learning\_v1\_5c83be2b): succeeded, step=0/25, alerts=[&#x27;inconsistent\_completion&#x27;]; d05\_r24\_seed42\_learning\_v1 (1790926923172990505\_d05\_r24\_seed42\_learning\_v1\_85f994fa): succeeded, step=0/25, alerts=[&#x27;inconsistent\_completion&#x27;]; d05\_r20\_seed43\_learning\_v1 (1790926923157371090\_d05\_r20\_seed43\_learning\_v1\_4b31a0c1): succeeded, step=0/25, alerts=[&#x27;inconsistent\_completion&#x27;]; d05\_r24\_seed43\_learning\_v1 (1790926923180466754\_d05\_r24\_seed43\_learning\_v1\_47ed9473): succeeded, step=0/25, alerts=[&#x27;inconsistent\_completion&#x27;]; d05\_r21\_seed42\_learning\_v1 (1790926923165197398\_d05\_r21\_seed42\_learning\_v1\_12e96433): succeeded, step=0/25, alerts=[&#x27;inconsistent\_completion&#x27;]; d05\_r26\_seed43\_learning\_v1 (1790926923199830052\_d05\_r26\_seed43\_learning\_v1\_1b590a01): succeeded, step=0/25, alerts=[&#x27;inconsistent\_completion&#x27;]; d05\_r20\_seed42\_learning\_v1 (1790926923146311111\_d05\_r20\_seed42\_learning\_v1\_cc8db6a6): succeeded, step=0/25, alerts=[&#x27;inconsistent\_completion&#x27;]
- 2026-10-02 08:01:39 UTC — d05\_r20\_seed42\_learning\_v1 (1790928092530413132\_d05\_r20\_seed42\_learning\_v1\_590a927e): running, step=0/500, alerts=[]; d05\_r21\_seed42\_learning\_v1 (1790928092548359830\_d05\_r21\_seed42\_learning\_v1\_c91f2623): running, step=0/500, alerts=[]; d05\_r24\_seed42\_learning\_v1 (1790928092555818254\_d05\_r24\_seed42\_learning\_v1\_36c7b5d6): running, step=0/500, alerts=[]; d05\_r24\_seed43\_learning\_v1 (1790928092563094796\_d05\_r24\_seed43\_learning\_v1\_89ccffc5): running, step=0/500, alerts=[]; d05\_r26\_seed42\_learning\_v1 (1790928092570644812\_d05\_r26\_seed42\_learning\_v1\_15eb88d3): running, step=0/500, alerts=[]; d05\_r26\_seed43\_learning\_v1 (1790928092580810482\_d05\_r26\_seed43\_learning\_v1\_ea041411): running, step=0/500, alerts=[]; d05\_r20\_seed43\_learning\_v1 (1790928092539965903\_d05\_r20\_seed43\_learning\_v1\_582ef7c9): running, step=0/500, alerts=[]
- 2026-10-02 10:56:37 UTC — d04\_r21\_CE\_learning\_v1 (1790924220633707073\_d04\_r21\_CE\_learning\_v1\_dd5651b2): succeeded, step=0/500, alerts=[&#x27;inconsistent\_completion&#x27;]
- 2026-10-02 12:00:35 UTC — d05\_r20\_seed42\_learning\_v1 (1790928092530413132\_d05\_r20\_seed42\_learning\_v1\_590a927e): succeeded, step=0/500, alerts=[&#x27;inconsistent\_completion&#x27;]; d05\_r24\_seed42\_learning\_v1 (1790928092555818254\_d05\_r24\_seed42\_learning\_v1\_36c7b5d6): succeeded, step=0/500, alerts=[&#x27;inconsistent\_completion&#x27;]; d05\_r24\_seed43\_learning\_v1 (1790928092563094796\_d05\_r24\_seed43\_learning\_v1\_89ccffc5): succeeded, step=0/500, alerts=[&#x27;inconsistent\_completion&#x27;]; d05\_r20\_seed43\_learning\_v1 (1790928092539965903\_d05\_r20\_seed43\_learning\_v1\_582ef7c9): succeeded, step=0/500, alerts=[&#x27;inconsistent\_completion&#x27;]
- 2026-10-02 12:01:08 UTC — d05\_r21\_seed42\_learning\_v1 (1790928092548359830\_d05\_r21\_seed42\_learning\_v1\_c91f2623): succeeded, step=0/500, alerts=[&#x27;inconsistent\_completion&#x27;]; d05\_r26\_seed42\_learning\_v1 (1790928092570644812\_d05\_r26\_seed42\_learning\_v1\_15eb88d3): succeeded, step=0/500, alerts=[&#x27;inconsistent\_completion&#x27;]; d05\_r26\_seed43\_learning\_v1 (1790928092580810482\_d05\_r26\_seed43\_learning\_v1\_ea041411): succeeded, step=0/500, alerts=[&#x27;inconsistent\_completion&#x27;]
- 2026-10-02 12:46:03 UTC — g01\_r21\_world8\_member\_5 (1790944282201990944\_g01\_r21\_world8\_member\_5\_2c7fe185): pending, step=0/6, alerts=[]; g01\_r21\_world8\_member\_6 (1790944282201993846\_g01\_r21\_world8\_member\_6\_fb50cd24): pending, step=0/6, alerts=[]; g01\_r21\_world8\_member\_7 (1790944282201997144\_g01\_r21\_world8\_member\_7\_bf02438a): pending, step=0/6, alerts=[]; g01\_r21\_world8\_member\_4 (1790944282201987921\_g01\_r21\_world8\_member\_4\_b0826020): pending, step=0/6, alerts=[]; g01\_r21\_world8\_member\_3 (1790944282201984472\_g01\_r21\_world8\_member\_3\_5371c8df): running, step=0/6, alerts=[]; g01\_r21\_world8\_member\_2 (1790944282201980741\_g01\_r21\_world8\_member\_2\_29d5bdcf): running, step=0/6, alerts=[]; g01\_r21\_world8\_member\_1 (1790944282201974433\_g01\_r21\_world8\_member\_1\_2bce92c7): running, step=0/6, alerts=[]; g01\_r21\_world8\_member\_0 (1790944282201965405\_g01\_r21\_world8\_member\_0\_64116e94): running, step=0/6, alerts=[]
- 2026-10-02 12:46:37 UTC — g01\_r21\_world8\_member\_3 (1790944282201984472\_g01\_r21\_world8\_member\_3\_5371c8df): failed, step=0/6, alerts=[&#x27;runner\_failure&#x27;]; g01\_r21\_world8\_member\_2 (1790944282201980741\_g01\_r21\_world8\_member\_2\_29d5bdcf): failed, step=0/6, alerts=[&#x27;runner\_failure&#x27;]; g01\_r21\_world8\_member\_1 (1790944282201974433\_g01\_r21\_world8\_member\_1\_2bce92c7): failed, step=0/6, alerts=[&#x27;runner\_failure&#x27;]; g01\_r21\_world8\_member\_5 (1790944282201990944\_g01\_r21\_world8\_member\_5\_2c7fe185): failed, step=0/6, alerts=[&#x27;runner\_failure&#x27;]; g01\_r21\_world8\_member\_6 (1790944282201993846\_g01\_r21\_world8\_member\_6\_fb50cd24): failed, step=0/6, alerts=[&#x27;runner\_failure&#x27;]; g01\_r21\_world8\_member\_7 (1790944282201997144\_g01\_r21\_world8\_member\_7\_bf02438a): failed, step=0/6, alerts=[&#x27;runner\_failure&#x27;]; g01\_r21\_world8\_member\_4 (1790944282201987921\_g01\_r21\_world8\_member\_4\_b0826020): failed, step=0/6, alerts=[&#x27;runner\_failure&#x27;]; g01\_r21\_world8\_member\_0 (1790944282201965405\_g01\_r21\_world8\_member\_0\_64116e94): failed, step=0/6, alerts=[&#x27;runner\_failure&#x27;]
- 2026-10-02 14:02:47 UTC — g02\_r21\_world8\_member\_7 (1790948366803137277\_g02\_r21\_world8\_member\_7\_e2a05b6f): running, step=0/6, alerts=[]; g02\_r21\_world8\_member\_1 (1790948366803116904\_g02\_r21\_world8\_member\_1\_54a65390): running, step=0/6, alerts=[]; g02\_r21\_world8\_member\_0 (1790948366803107556\_g02\_r21\_world8\_member\_0\_04be8e50): running, step=0/6, alerts=[]; g02\_r21\_world8\_member\_2 (1790948366803122484\_g02\_r21\_world8\_member\_2\_55767f54): running, step=0/6, alerts=[]; g02\_r21\_world8\_member\_6 (1790948366803134323\_g02\_r21\_world8\_member\_6\_9cc6aa7a): running, step=0/6, alerts=[]; g02\_r21\_world8\_member\_3 (1790948366803125661\_g02\_r21\_world8\_member\_3\_97773210): running, step=0/6, alerts=[]; g02\_r21\_world8\_member\_5 (1790948366803131655\_g02\_r21\_world8\_member\_5\_b5922d3c): running, step=0/6, alerts=[]; g02\_r21\_world8\_member\_4 (1790948366803128800\_g02\_r21\_world8\_member\_4\_3a3aa296): running, step=0/6, alerts=[]
- 2026-10-02 14:03:21 UTC — g02\_r21\_world8\_member\_7 (1790948366803137277\_g02\_r21\_world8\_member\_7\_e2a05b6f): failed, step=0/6, alerts=[&#x27;runner\_failure&#x27;]; g02\_r21\_world8\_member\_1 (1790948366803116904\_g02\_r21\_world8\_member\_1\_54a65390): failed, step=0/6, alerts=[&#x27;runner\_failure&#x27;]; g02\_r21\_world8\_member\_0 (1790948366803107556\_g02\_r21\_world8\_member\_0\_04be8e50): failed, step=0/6, alerts=[&#x27;runner\_failure&#x27;]; g02\_r21\_world8\_member\_2 (1790948366803122484\_g02\_r21\_world8\_member\_2\_55767f54): failed, step=0/6, alerts=[&#x27;runner\_failure&#x27;]; g02\_r21\_world8\_member\_6 (1790948366803134323\_g02\_r21\_world8\_member\_6\_9cc6aa7a): failed, step=0/6, alerts=[&#x27;runner\_failure&#x27;]; g02\_r21\_world8\_member\_3 (1790948366803125661\_g02\_r21\_world8\_member\_3\_97773210): failed, step=0/6, alerts=[&#x27;runner\_failure&#x27;]; g02\_r21\_world8\_member\_5 (1790948366803131655\_g02\_r21\_world8\_member\_5\_b5922d3c): failed, step=0/6, alerts=[&#x27;runner\_failure&#x27;]; g02\_r21\_world8\_member\_4 (1790948366803128800\_g02\_r21\_world8\_member\_4\_3a3aa296): failed, step=0/6, alerts=[&#x27;runner\_failure&#x27;]
- 2026-10-02 15:20:57 UTC — g03\_r21\_world8\_member\_2 (1790953672752455089\_g03\_r21\_world8\_member\_2\_e4c41241): running, step=0/6, alerts=[]; g03\_r21\_world8\_member\_1 (1790953672752451387\_g03\_r21\_world8\_member\_1\_e6edbcbb): running, step=0/6, alerts=[]; g03\_r21\_world8\_member\_5 (1790953672752465889\_g03\_r21\_world8\_member\_5\_7e6275aa): running, step=0/6, alerts=[]; g03\_r21\_world8\_member\_6 (1790953672752469183\_g03\_r21\_world8\_member\_6\_6fcb1ff2): running, step=0/6, alerts=[]; g03\_r21\_world8\_member\_3 (1790953672752458546\_g03\_r21\_world8\_member\_3\_f4bae8db): running, step=0/6, alerts=[]; g03\_r21\_world8\_member\_0 (1790953672752444849\_g03\_r21\_world8\_member\_0\_674b9eed): running, step=0/6, alerts=[]; g03\_r21\_world8\_member\_7 (1790953672752472296\_g03\_r21\_world8\_member\_7\_21a90bb3): running, step=0/6, alerts=[]; g03\_r21\_world8\_member\_4 (1790953672752461949\_g03\_r21\_world8\_member\_4\_0d9e8e48): running, step=0/6, alerts=[]
- 2026-10-02 15:21:31 UTC — g03\_r21\_world8\_member\_2 (1790953672752455089\_g03\_r21\_world8\_member\_2\_e4c41241): failed, step=0/6, alerts=[&#x27;runner\_failure&#x27;]; g03\_r21\_world8\_member\_1 (1790953672752451387\_g03\_r21\_world8\_member\_1\_e6edbcbb): failed, step=0/6, alerts=[&#x27;runner\_failure&#x27;]; g03\_r21\_world8\_member\_5 (1790953672752465889\_g03\_r21\_world8\_member\_5\_7e6275aa): failed, step=0/6, alerts=[&#x27;runner\_failure&#x27;]; g03\_r21\_world8\_member\_6 (1790953672752469183\_g03\_r21\_world8\_member\_6\_6fcb1ff2): failed, step=0/6, alerts=[&#x27;runner\_failure&#x27;]; g03\_r21\_world8\_member\_3 (1790953672752458546\_g03\_r21\_world8\_member\_3\_f4bae8db): failed, step=0/6, alerts=[&#x27;runner\_failure&#x27;]; g03\_r21\_world8\_member\_0 (1790953672752444849\_g03\_r21\_world8\_member\_0\_674b9eed): failed, step=0/6, alerts=[&#x27;runner\_failure&#x27;]; g03\_r21\_world8\_member\_7 (1790953672752472296\_g03\_r21\_world8\_member\_7\_21a90bb3): failed, step=0/6, alerts=[&#x27;runner\_failure&#x27;]; g03\_r21\_world8\_member\_4 (1790953672752461949\_g03\_r21\_world8\_member\_4\_0d9e8e48): failed, step=0/6, alerts=[&#x27;runner\_failure&#x27;]
- 2026-10-02 19:14:36 UTC — p01\_v7\_r20\_seed42 (1790968470149719891\_p01\_v7\_r20\_seed42\_838aa052): running, step=0/3125, alerts=[]; p01\_v7\_r21\_seed42 (1790968470161653699\_p01\_v7\_r21\_seed42\_ccb7e981): running, step=0/3125, alerts=[]
- 2026-10-02 19:15:12 UTC — p01\_v7\_r20\_seed42 (1790968470149719891\_p01\_v7\_r20\_seed42\_838aa052): running, step=0/3125, alerts=[&#x27;source\_changed&#x27;]; p01\_v7\_r21\_seed42 (1790968470161653699\_p01\_v7\_r21\_seed42\_ccb7e981): running, step=0/3125, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-02 19:17:31 UTC — p01\_v7\_r26\_seed43 (1790968636168787261\_p01\_v7\_r26\_seed43\_6a3a3518): running, step=0/3125, alerts=[]; p01\_v7\_r20\_seed43 (1790968636135679016\_p01\_v7\_r20\_seed43\_abf75de1): running, step=0/3125, alerts=[]; p01\_v7\_r24\_seed42 (1790968636111438133\_p01\_v7\_r24\_seed42\_b0d0a2be): running, step=0/3125, alerts=[]; p01\_v7\_r24\_seed43 (1790968636157009559\_p01\_v7\_r24\_seed43\_a9336889): running, step=0/3125, alerts=[]; p01\_v7\_r21\_seed43 (1790968636145868351\_p01\_v7\_r21\_seed43\_3f1b2997): running, step=0/3125, alerts=[]; p01\_v7\_r26\_seed42 (1790968636125595682\_p01\_v7\_r26\_seed42\_be605dad): running, step=0/3125, alerts=[]
- 2026-10-02 19:18:06 UTC — p01\_v7\_r26\_seed43 (1790968636168787261\_p01\_v7\_r26\_seed43\_6a3a3518): running, step=0/3125, alerts=[&#x27;source\_changed&#x27;]; p01\_v7\_r20\_seed43 (1790968636135679016\_p01\_v7\_r20\_seed43\_abf75de1): running, step=0/3125, alerts=[&#x27;source\_changed&#x27;]; p01\_v7\_r24\_seed42 (1790968636111438133\_p01\_v7\_r24\_seed42\_b0d0a2be): running, step=0/3125, alerts=[&#x27;source\_changed&#x27;]; p01\_v7\_r24\_seed43 (1790968636157009559\_p01\_v7\_r24\_seed43\_a9336889): running, step=0/3125, alerts=[&#x27;source\_changed&#x27;]; p01\_v7\_r21\_seed43 (1790968636145868351\_p01\_v7\_r21\_seed43\_3f1b2997): running, step=0/3125, alerts=[&#x27;source\_changed&#x27;]; p01\_v7\_r26\_seed42 (1790968636125595682\_p01\_v7\_r26\_seed42\_be605dad): running, step=0/3125, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-02 20:34:05 UTC — p01\_v7\_r26\_seed43 (1790968636168787261\_p01\_v7\_r26\_seed43\_6a3a3518): running, step=0/3125, alerts=[&#x27;source\_changed&#x27;]; p01\_v7\_r20\_seed43 (1790968636135679016\_p01\_v7\_r20\_seed43\_abf75de1): running, step=0/3125, alerts=[&#x27;source\_changed&#x27;]; p01\_v7\_r20\_seed42 (1790968470149719891\_p01\_v7\_r20\_seed42\_838aa052): running, step=0/3125, alerts=[&#x27;source\_changed&#x27;]; p01\_v7\_r21\_seed42 (1790968470161653699\_p01\_v7\_r21\_seed42\_ccb7e981): running, step=0/3125, alerts=[&#x27;source\_changed&#x27;]; p01\_v7\_r24\_seed42 (1790968636111438133\_p01\_v7\_r24\_seed42\_b0d0a2be): running, step=0/3125, alerts=[&#x27;source\_changed&#x27;]; p01\_v7\_r24\_seed43 (1790968636157009559\_p01\_v7\_r24\_seed43\_a9336889): running, step=0/3125, alerts=[&#x27;source\_changed&#x27;]; p01\_v7\_r21\_seed43 (1790968636145868351\_p01\_v7\_r21\_seed43\_3f1b2997): running, step=0/3125, alerts=[&#x27;source\_changed&#x27;]; p01\_v7\_r26\_seed42 (1790968636125595682\_p01\_v7\_r26\_seed42\_be605dad): running, step=0/3125, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-02 20:35:46 UTC — p01\_v7\_r26\_seed43 (1790968636168787261\_p01\_v7\_r26\_seed43\_6a3a3518): running, step=0/3125, alerts=[&#x27;source\_changed&#x27;]; p01\_v7\_r20\_seed43 (1790968636135679016\_p01\_v7\_r20\_seed43\_abf75de1): running, step=0/3125, alerts=[&#x27;source\_changed&#x27;]; p01\_v7\_r20\_seed42 (1790968470149719891\_p01\_v7\_r20\_seed42\_838aa052): running, step=0/3125, alerts=[&#x27;source\_changed&#x27;]; p01\_v7\_r21\_seed42 (1790968470161653699\_p01\_v7\_r21\_seed42\_ccb7e981): running, step=0/3125, alerts=[&#x27;source\_changed&#x27;]; p01\_v7\_r24\_seed42 (1790968636111438133\_p01\_v7\_r24\_seed42\_b0d0a2be): running, step=0/3125, alerts=[&#x27;source\_changed&#x27;]; p01\_v7\_r24\_seed43 (1790968636157009559\_p01\_v7\_r24\_seed43\_a9336889): running, step=0/3125, alerts=[&#x27;source\_changed&#x27;]; p01\_v7\_r21\_seed43 (1790968636145868351\_p01\_v7\_r21\_seed43\_3f1b2997): running, step=0/3125, alerts=[&#x27;source\_changed&#x27;]; p01\_v7\_r26\_seed42 (1790968636125595682\_p01\_v7\_r26\_seed42\_be605dad): running, step=0/3125, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-02 20:59:15 UTC — p02\_v7\_cv\_segment\_global\_seed42 (1790974728125458819\_p02\_v7\_cv\_segment\_global\_seed42\_fb1caf55): pending, step=0/3750, alerts=[]; p02\_v7\_f101\_seed43 (1790974728298328966\_p02\_v7\_f101\_seed43\_dffcf249): pending, step=0/3750, alerts=[]; p02\_v7\_f000\_seed43 (1790974728147557201\_p02\_v7\_f000\_seed43\_e4b3bc10): pending, step=0/3750, alerts=[]; p02\_v7\_f001\_seed42 (1790974728116054138\_p02\_v7\_f001\_seed42\_e0e1afbe): pending, step=0/3750, alerts=[]; p02\_v7\_f100\_seed42 (1790974728195988843\_p02\_v7\_f100\_seed42\_edb40cf9): pending, step=0/3750, alerts=[]; p02\_v7\_f111\_seed42 (1790974728238311982\_p02\_v7\_f111\_seed42\_1f2408bb): pending, step=0/3750, alerts=[]; p02\_v7\_cv\_segment\_global\_seed43 (1790974728166373677\_p02\_v7\_cv\_segment\_global\_seed43\_8938fe7e): pending, step=0/3750, alerts=[]; p02\_v7\_f100\_seed43 (1790974728268557869\_p02\_v7\_f100\_seed43\_71448331): pending, step=0/3750, alerts=[]; p02\_v7\_fixed\_gate\_control\_seed42 (1790974728248207117\_p02\_v7\_fixed\_gate\_control\_seed42\_121777f2): pending, step=0/3750, alerts=[]; p02\_v7\_f110\_seed43 (1790974728308733270\_p02\_v7\_f110\_seed43\_5dc2a9f1): pending, step=0/3750, alerts=[]; p02\_v7\_cv\_late\_fusion\_seed43 (1790974728176231879\_p02\_v7\_cv\_late\_fusion\_seed43\_a68e3b52): pending, step=0/3750, alerts=[]; p02\_v7\_f010\_seed43 (1790974728258219500\_p02\_v7\_f010\_seed43\_c95a33ba): pending, step=0/3750, alerts=[]; p02\_v7\_f010\_seed42 (1790974728186010249\_p02\_v7\_f010\_seed42\_85e4e2d6): pending, step=0/3750, alerts=[]; p02\_v7\_f011\_seed43 (1790974728285936659\_p02\_v7\_f011\_seed43\_eda28dda): pending, step=0/3750, alerts=[]; p02\_v7\_f101\_seed42 (1790974728218042172\_p02\_v7\_f101\_seed42\_39513f1a): pending, step=0/3750, alerts=[]; p02\_v7\_f111\_seed43 (1790974728318695588\_p02\_v7\_f111\_seed43\_891e6010): pending, step=0/3750, alerts=[]; p02\_v7\_fixed\_gate\_control\_seed43 (1790974728328829268\_p02\_v7\_fixed\_gate\_control\_seed43\_5dae9651): pending, step=0/3750, alerts=[]; p02\_v7\_f011\_seed42 (1790974728206202226\_p02\_v7\_f011\_seed42\_188332f8): pending, step=0/3750, alerts=[]; p02\_v7\_cv\_late\_fusion\_seed42 (1790974728135266312\_p02\_v7\_cv\_late\_fusion\_seed42\_7b904983): pending, step=0/3750, alerts=[]; p02\_v7\_f000\_seed42 (1790974728104139450\_p02\_v7\_f000\_seed42\_9a84aad6): pending, step=0/3750, alerts=[]; p02\_v7\_f001\_seed43 (1790974728156871579\_p02\_v7\_f001\_seed43\_e810d85a): pending, step=0/3750, alerts=[]; p02\_v7\_f110\_seed42 (1790974728228059321\_p02\_v7\_f110\_seed42\_09bd5d48): pending, step=0/3750, alerts=[]
- 2026-10-02 21:45:57 UTC — p02\_v7\_f000\_seed42 (1790974728104139450\_p02\_v7\_f000\_seed42\_9a84aad6): running, step=0/3750, alerts=[]; p01\_v7\_r20\_seed42 (1790968470149719891\_p01\_v7\_r20\_seed42\_838aa052): succeeded, step=3125/3125, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-02 21:46:31 UTC — p02\_v7\_f000\_seed42 (1790974728104139450\_p02\_v7\_f000\_seed42\_9a84aad6): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-02 21:47:05 UTC — p02\_v7\_f001\_seed42 (1790974728116054138\_p02\_v7\_f001\_seed42\_e0e1afbe): running, step=0/3750, alerts=[]; p01\_v7\_r21\_seed42 (1790968470161653699\_p01\_v7\_r21\_seed42\_ccb7e981): succeeded, step=3125/3125, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-02 21:47:38 UTC — p02\_v7\_f001\_seed42 (1790974728116054138\_p02\_v7\_f001\_seed42\_e0e1afbe): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-02 21:48:12 UTC — p02\_v7\_cv\_segment\_global\_seed42 (1790974728125458819\_p02\_v7\_cv\_segment\_global\_seed42\_fb1caf55): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]; p01\_v7\_r24\_seed43 (1790968636157009559\_p01\_v7\_r24\_seed43\_a9336889): running, step=3125/3125, alerts=[&#x27;source\_changed&#x27;]; p02\_v7\_cv\_late\_fusion\_seed42 (1790974728135266312\_p02\_v7\_cv\_late\_fusion\_seed42\_7b904983): running, step=0/3750, alerts=[]; p01\_v7\_r20\_seed43 (1790968636135679016\_p01\_v7\_r20\_seed43\_abf75de1): succeeded, step=3125/3125, alerts=[&#x27;source\_changed&#x27;]; p01\_v7\_r24\_seed42 (1790968636111438133\_p01\_v7\_r24\_seed42\_b0d0a2be): succeeded, step=3125/3125, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-02 21:48:46 UTC — p02\_v7\_f000\_seed43 (1790974728147557201\_p02\_v7\_f000\_seed43\_e4b3bc10): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]; p02\_v7\_cv\_segment\_global\_seed43 (1790974728166373677\_p02\_v7\_cv\_segment\_global\_seed43\_8938fe7e): running, step=0/3750, alerts=[]; p02\_v7\_cv\_late\_fusion\_seed42 (1790974728135266312\_p02\_v7\_cv\_late\_fusion\_seed42\_7b904983): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]; p02\_v7\_f001\_seed43 (1790974728156871579\_p02\_v7\_f001\_seed43\_e810d85a): running, step=0/3750, alerts=[]; p01\_v7\_r26\_seed43 (1790968636168787261\_p01\_v7\_r26\_seed43\_6a3a3518): succeeded, step=3125/3125, alerts=[&#x27;source\_changed&#x27;]; p01\_v7\_r24\_seed43 (1790968636157009559\_p01\_v7\_r24\_seed43\_a9336889): succeeded, step=3125/3125, alerts=[&#x27;source\_changed&#x27;]; p01\_v7\_r26\_seed42 (1790968636125595682\_p01\_v7\_r26\_seed42\_be605dad): succeeded, step=3125/3125, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-02 21:49:20 UTC — p02\_v7\_cv\_segment\_global\_seed43 (1790974728166373677\_p02\_v7\_cv\_segment\_global\_seed43\_8938fe7e): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]; p02\_v7\_f001\_seed43 (1790974728156871579\_p02\_v7\_f001\_seed43\_e810d85a): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-02 21:49:54 UTC — p02\_v7\_cv\_late\_fusion\_seed43 (1790974728176231879\_p02\_v7\_cv\_late\_fusion\_seed43\_a68e3b52): running, step=0/3750, alerts=[]; p01\_v7\_r21\_seed43 (1790968636145868351\_p01\_v7\_r21\_seed43\_3f1b2997): succeeded, step=3125/3125, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-02 21:50:27 UTC — p02\_v7\_cv\_late\_fusion\_seed43 (1790974728176231879\_p02\_v7\_cv\_late\_fusion\_seed43\_a68e3b52): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-02 22:50:36 UTC — p04\_v7\_capacity (1790981433321562451\_p04\_v7\_capacity\_febb28e3): pending, step=0/72, alerts=[]
- 2026-10-03 01:39:42 UTC — p02\_v7\_f000\_seed42 (1790974728104139450\_p02\_v7\_f000\_seed42\_9a84aad6): running, step=3750/3750, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-03 01:40:16 UTC — p04\_v7\_capacity (1790981433321562451\_p04\_v7\_capacity\_febb28e3): running, step=0/72, alerts=[]; p02\_v7\_f000\_seed42 (1790974728104139450\_p02\_v7\_f000\_seed42\_9a84aad6): succeeded, step=3750/3750, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-03 01:42:01 UTC — p02\_v7\_f001\_seed42 (1790974728116054138\_p02\_v7\_f001\_seed42\_e0e1afbe): succeeded, step=3750/3750, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-03 01:43:09 UTC — p04\_v7\_capacity (1790981433321562451\_p04\_v7\_capacity\_febb28e3): failed, step=0/72, alerts=[&#x27;runner\_failure&#x27;]; p02\_v7\_cv\_late\_fusion\_seed42 (1790974728135266312\_p02\_v7\_cv\_late\_fusion\_seed42\_7b904983): succeeded, step=3750/3750, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-03 01:43:43 UTC — p02\_v7\_cv\_segment\_global\_seed42 (1790974728125458819\_p02\_v7\_cv\_segment\_global\_seed42\_fb1caf55): succeeded, step=3750/3750, alerts=[&#x27;source\_changed&#x27;]; p02\_v7\_f000\_seed43 (1790974728147557201\_p02\_v7\_f000\_seed43\_e4b3bc10): succeeded, step=3750/3750, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-03 01:44:18 UTC — p02\_v7\_cv\_segment\_global\_seed43 (1790974728166373677\_p02\_v7\_cv\_segment\_global\_seed43\_8938fe7e): succeeded, step=3750/3750, alerts=[&#x27;source\_changed&#x27;]; p02\_v7\_f001\_seed43 (1790974728156871579\_p02\_v7\_f001\_seed43\_e810d85a): succeeded, step=3750/3750, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-03 01:45:28 UTC — p02\_v7\_cv\_late\_fusion\_seed43 (1790974728176231879\_p02\_v7\_cv\_late\_fusion\_seed43\_a68e3b52): succeeded, step=3750/3750, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-03 02:05:21 UTC — p05\_v7\_capacity\_v2 (1790992416352344471\_p05\_v7\_capacity\_v2\_a4854542): running, step=0/36, alerts=[]
- 2026-10-03 02:27:57 UTC — p05\_v7\_capacity\_v2 (1790992416352344471\_p05\_v7\_capacity\_v2\_a4854542): succeeded, step=0/36, alerts=[&#x27;inconsistent\_completion&#x27;]
- 2026-10-03 03:09:55 UTC — p04\_v7\_cv\_late\_fusion\_seed42 (1790995583011921799\_p04\_v7\_cv\_late\_fusion\_seed42\_905230bb): pending, step=0/3750, alerts=[]; p04\_v7\_cv\_segment\_global\_seed42 (1790995583011879919\_p04\_v7\_cv\_segment\_global\_seed42\_ffdd2332): pending, step=0/3750, alerts=[]; p04\_v7\_f101\_seed42 (1790995583011757747\_p04\_v7\_f101\_seed42\_850e3092): pending, step=0/3750, alerts=[]; p04\_v7\_fixed\_gate\_control\_seed43 (1790995583011991824\_p04\_v7\_fixed\_gate\_control\_seed43\_7ec4120b): pending, step=0/3750, alerts=[]; p04\_v7\_f100\_seed43 (1790995583011737505\_p04\_v7\_f100\_seed43\_87f53303): pending, step=0/3750, alerts=[]; p04\_v7\_f110\_seed43 (1790995583011818395\_p04\_v7\_f110\_seed43\_2f6da59d): pending, step=0/3750, alerts=[]; p04\_v7\_f110\_seed42 (1790995583011798231\_p04\_v7\_f110\_seed42\_adafd438): pending, step=0/3750, alerts=[]; p04\_v7\_cv\_segment\_global\_seed43 (1790995583011900971\_p04\_v7\_cv\_segment\_global\_seed43\_d5fe067c): pending, step=0/3750, alerts=[]; p04\_v7\_f101\_seed43 (1790995583011778233\_p04\_v7\_f101\_seed43\_e8a9c4d4): pending, step=0/3750, alerts=[]; p04\_v7\_cv\_late\_fusion\_seed43 (1790995583011948104\_p04\_v7\_cv\_late\_fusion\_seed43\_e268efa5): pending, step=0/3750, alerts=[]; p04\_v7\_fixed\_gate\_control\_seed42 (1790995583011969884\_p04\_v7\_fixed\_gate\_control\_seed42\_f6421658): pending, step=0/3750, alerts=[]; p04\_v7\_f111\_seed42 (1790995583011837977\_p04\_v7\_f111\_seed42\_0f7dc638): pending, step=0/3750, alerts=[]; p04\_v7\_f100\_seed42 (1790995583011717074\_p04\_v7\_f100\_seed42\_ca6382b3): pending, step=0/3750, alerts=[]; p04\_v7\_f111\_seed43 (1790995583011859514\_p04\_v7\_f111\_seed43\_0a175776): pending, step=0/3750, alerts=[]; p04\_v7\_f010\_seed42 (1790995583011584963\_p04\_v7\_f010\_seed42\_93cb62b2): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]; p04\_v7\_f001\_seed43 (1790995583011651525\_p04\_v7\_f001\_seed43\_6374caab): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]; p04\_v7\_f011\_seed42 (1790995583011673811\_p04\_v7\_f011\_seed42\_4863ec8f): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]; p04\_v7\_f011\_seed43 (1790995583011694915\_p04\_v7\_f011\_seed43\_3ff5b7c7): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]; p04\_v7\_f000\_seed42 (1790995583011457554\_p04\_v7\_f000\_seed42\_539537fe): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]; p04\_v7\_f000\_seed43 (1790995583011557084\_p04\_v7\_f000\_seed43\_34d8ec5b): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]; p04\_v7\_f001\_seed42 (1790995583011630260\_p04\_v7\_f001\_seed42\_40c762d3): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]; p04\_v7\_f010\_seed43 (1790995583011607766\_p04\_v7\_f010\_seed43\_5b45614b): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-03 08:33:31 UTC — p04\_v7\_f101\_seed42 (1790995583011757747\_p04\_v7\_f101\_seed42\_850e3092): running, step=0/3750, alerts=[]; p04\_v7\_f100\_seed43 (1790995583011737505\_p04\_v7\_f100\_seed43\_87f53303): running, step=0/3750, alerts=[]; p04\_v7\_f100\_seed42 (1790995583011717074\_p04\_v7\_f100\_seed42\_ca6382b3): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]; p04\_v7\_f010\_seed42 (1790995583011584963\_p04\_v7\_f010\_seed42\_93cb62b2): failed, step=0/3750, alerts=[&#x27;source\_changed&#x27;, &#x27;runner\_failure&#x27;]; p04\_v7\_f000\_seed42 (1790995583011457554\_p04\_v7\_f000\_seed42\_539537fe): failed, step=0/3750, alerts=[&#x27;source\_changed&#x27;, &#x27;runner\_failure&#x27;]; p04\_v7\_f010\_seed43 (1790995583011607766\_p04\_v7\_f010\_seed43\_5b45614b): failed, step=0/3750, alerts=[&#x27;source\_changed&#x27;, &#x27;runner\_failure&#x27;]
- 2026-10-03 08:34:05 UTC — p04\_v7\_f101\_seed42 (1790995583011757747\_p04\_v7\_f101\_seed42\_850e3092): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]; p04\_v7\_f100\_seed43 (1790995583011737505\_p04\_v7\_f100\_seed43\_87f53303): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-03 09:05:53 UTC — p04\_v7\_f101\_seed42 (1790995583011757747\_p04\_v7\_f101\_seed42\_850e3092): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]; p04\_v7\_f100\_seed43 (1790995583011737505\_p04\_v7\_f100\_seed43\_87f53303): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]; p04\_v7\_f001\_seed43 (1790995583011651525\_p04\_v7\_f001\_seed43\_6374caab): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]; p04\_v7\_f011\_seed42 (1790995583011673811\_p04\_v7\_f011\_seed42\_4863ec8f): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]; p04\_v7\_f011\_seed43 (1790995583011694915\_p04\_v7\_f011\_seed43\_3ff5b7c7): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]; p04\_v7\_f000\_seed43 (1790995583011557084\_p04\_v7\_f000\_seed43\_34d8ec5b): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]; p04\_v7\_f001\_seed42 (1790995583011630260\_p04\_v7\_f001\_seed42\_40c762d3): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]; p04\_v7\_f100\_seed42 (1790995583011717074\_p04\_v7\_f100\_seed42\_ca6382b3): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]; p02\_v7\_cv\_segment\_global\_seed42 (1790974728125458819\_p02\_v7\_cv\_segment\_global\_seed42\_fb1caf55): succeeded, step=3750/3750, alerts=[&#x27;source\_changed&#x27;]; p02\_v7\_f000\_seed43 (1790974728147557201\_p02\_v7\_f000\_seed43\_e4b3bc10): succeeded, step=3750/3750, alerts=[&#x27;source\_changed&#x27;]; p04\_v7\_f010\_seed42 (1790995583011584963\_p04\_v7\_f010\_seed42\_93cb62b2): failed, step=0/3750, alerts=[&#x27;source\_changed&#x27;, &#x27;runner\_failure&#x27;]; p02\_v7\_f001\_seed42 (1790974728116054138\_p02\_v7\_f001\_seed42\_e0e1afbe): succeeded, step=3750/3750, alerts=[&#x27;source\_changed&#x27;]; p02\_v7\_cv\_segment\_global\_seed43 (1790974728166373677\_p02\_v7\_cv\_segment\_global\_seed43\_8938fe7e): succeeded, step=3750/3750, alerts=[&#x27;source\_changed&#x27;]; p02\_v7\_cv\_late\_fusion\_seed43 (1790974728176231879\_p02\_v7\_cv\_late\_fusion\_seed43\_a68e3b52): succeeded, step=3750/3750, alerts=[&#x27;source\_changed&#x27;]; p04\_v7\_f000\_seed42 (1790995583011457554\_p04\_v7\_f000\_seed42\_539537fe): failed, step=0/3750, alerts=[&#x27;source\_changed&#x27;, &#x27;runner\_failure&#x27;]; p04\_v7\_f010\_seed43 (1790995583011607766\_p04\_v7\_f010\_seed43\_5b45614b): failed, step=0/3750, alerts=[&#x27;source\_changed&#x27;, &#x27;runner\_failure&#x27;]; p02\_v7\_cv\_late\_fusion\_seed42 (1790974728135266312\_p02\_v7\_cv\_late\_fusion\_seed42\_7b904983): succeeded, step=3750/3750, alerts=[&#x27;source\_changed&#x27;]; p02\_v7\_f000\_seed42 (1790974728104139450\_p02\_v7\_f000\_seed42\_9a84aad6): succeeded, step=3750/3750, alerts=[&#x27;source\_changed&#x27;]; p02\_v7\_f001\_seed43 (1790974728156871579\_p02\_v7\_f001\_seed43\_e810d85a): succeeded, step=3750/3750, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-03 12:05:18 UTC — p04\_v7\_f101\_seed43 (1790995583011778233\_p04\_v7\_f101\_seed43\_e8a9c4d4): running, step=0/3750, alerts=[]; p04\_v7\_f011\_seed42 (1790995583011673811\_p04\_v7\_f011\_seed42\_4863ec8f): succeeded, step=3750/3750, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-03 12:05:51 UTC — p04\_v7\_f110\_seed42 (1790995583011798231\_p04\_v7\_f110\_seed42\_adafd438): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]; p04\_v7\_f101\_seed43 (1790995583011778233\_p04\_v7\_f101\_seed43\_e8a9c4d4): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]; p04\_v7\_f001\_seed42 (1790995583011630260\_p04\_v7\_f001\_seed42\_40c762d3): succeeded, step=3750/3750, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-03 12:09:14 UTC — p04\_v7\_f110\_seed43 (1790995583011818395\_p04\_v7\_f110\_seed43\_2f6da59d): running, step=0/3750, alerts=[]; p04\_v7\_f000\_seed43 (1790995583011557084\_p04\_v7\_f000\_seed43\_34d8ec5b): succeeded, step=3750/3750, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-03 12:09:48 UTC — p04\_v7\_f110\_seed43 (1790995583011818395\_p04\_v7\_f110\_seed43\_2f6da59d): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]; p04\_v7\_f111\_seed42 (1790995583011837977\_p04\_v7\_f111\_seed42\_0f7dc638): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]; p04\_v7\_f111\_seed43 (1790995583011859514\_p04\_v7\_f111\_seed43\_0a175776): running, step=0/3750, alerts=[]; p04\_v7\_f001\_seed43 (1790995583011651525\_p04\_v7\_f001\_seed43\_6374caab): succeeded, step=3750/3750, alerts=[&#x27;source\_changed&#x27;]; p04\_v7\_f011\_seed43 (1790995583011694915\_p04\_v7\_f011\_seed43\_3ff5b7c7): succeeded, step=3750/3750, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-03 12:10:22 UTC — p04\_v7\_f111\_seed43 (1790995583011859514\_p04\_v7\_f111\_seed43\_0a175776): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-03 17:29:02 UTC — p04\_v7\_cv\_segment\_global\_seed42 (1790995583011879919\_p04\_v7\_cv\_segment\_global\_seed42\_ffdd2332): running, step=0/3750, alerts=[]; p04\_v7\_f100\_seed42 (1790995583011717074\_p04\_v7\_f100\_seed42\_ca6382b3): succeeded, step=3750/3750, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-03 17:29:36 UTC — p04\_v7\_cv\_segment\_global\_seed42 (1790995583011879919\_p04\_v7\_cv\_segment\_global\_seed42\_ffdd2332): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]; p04\_v7\_cv\_segment\_global\_seed43 (1790995583011900971\_p04\_v7\_cv\_segment\_global\_seed43\_d5fe067c): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]; p04\_v7\_f101\_seed42 (1790995583011757747\_p04\_v7\_f101\_seed42\_850e3092): succeeded, step=3750/3750, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-03 17:33:34 UTC — p04\_v7\_cv\_late\_fusion\_seed42 (1790995583011921799\_p04\_v7\_cv\_late\_fusion\_seed42\_905230bb): running, step=0/3750, alerts=[]; p04\_v7\_f100\_seed43 (1790995583011737505\_p04\_v7\_f100\_seed43\_87f53303): succeeded, step=3750/3750, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-03 17:34:08 UTC — p04\_v7\_cv\_late\_fusion\_seed42 (1790995583011921799\_p04\_v7\_cv\_late\_fusion\_seed42\_905230bb): running, step=0/3750, alerts=[&#x27;source\_changed&#x27;]
- 2026-10-03 19:24:32 UTC — p04\_v7\_cv\_late\_fusion\_seed42 (1790995583011921799\_p04\_v7\_cv\_late\_fusion\_seed42\_905230bb): failed, step=0/3750, alerts=[&#x27;source\_changed&#x27;, &#x27;runner\_failure&#x27;]; p04\_v7\_cv\_segment\_global\_seed42 (1790995583011879919\_p04\_v7\_cv\_segment\_global\_seed42\_ffdd2332): failed, step=0/3750, alerts=[&#x27;source\_changed&#x27;, &#x27;runner\_failure&#x27;]; p04\_v7\_f110\_seed43 (1790995583011818395\_p04\_v7\_f110\_seed43\_2f6da59d): failed, step=0/3750, alerts=[&#x27;source\_changed&#x27;, &#x27;runner\_failure&#x27;]; p04\_v7\_f110\_seed42 (1790995583011798231\_p04\_v7\_f110\_seed42\_adafd438): failed, step=0/3750, alerts=[&#x27;source\_changed&#x27;, &#x27;runner\_failure&#x27;]; p04\_v7\_cv\_segment\_global\_seed43 (1790995583011900971\_p04\_v7\_cv\_segment\_global\_seed43\_d5fe067c): failed, step=0/3750, alerts=[&#x27;source\_changed&#x27;, &#x27;runner\_failure&#x27;]; p04\_v7\_f101\_seed43 (1790995583011778233\_p04\_v7\_f101\_seed43\_e8a9c4d4): failed, step=0/3750, alerts=[&#x27;source\_changed&#x27;, &#x27;runner\_failure&#x27;]; p04\_v7\_f111\_seed42 (1790995583011837977\_p04\_v7\_f111\_seed42\_0f7dc638): failed, step=0/3750, alerts=[&#x27;source\_changed&#x27;, &#x27;runner\_failure&#x27;]; p04\_v7\_f111\_seed43 (1790995583011859514\_p04\_v7\_f111\_seed43\_0a175776): failed, step=0/3750, alerts=[&#x27;source\_changed&#x27;, &#x27;runner\_failure&#x27;]

证据目录：/vepfs-mlp2/c20250601/251105016/project/dllm\_test/output/immune\_research\_pool\_c20250601\_20260930。完整监督快照：/vepfs-mlp2/c20250601/251105016/project/dllm\_test/output/immune\_research\_pool\_c20250601\_20260930/monitor/supervision.json。
