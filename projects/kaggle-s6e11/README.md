# Kaggle Playground S6E11 四模型训练代码

本目录基于同一份数据划分和同一套评价指标，分别训练：

- CatBoost
- LightGBM
- XGBoost
- Random Forest

训练集内部按 `id` 排序后取最后 20% 作为验证集，避免随机切分与比赛测试集分布不一致。验证集用于早停和模型比较；参数确定后，各模型在完整训练集上重新训练，最后只对 `test.csv` 做预测。

## 目录

```text
kanggle/
├── playground-series-s6e11/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── config.py
└── outputs/
    ├── models/
    ├── submissions/
    └── run_*/

```

## 安装环境

项目已经包含一个独立虚拟环境 `.venv`。在 `kanggle` 目录打开 PowerShell 后，优先使用：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果以后重新创建环境，也可以先运行：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果下载速度较慢，可使用镜像：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 先快速验证流程

快速模式会减少树的数量和迭代轮数，适合检查路径、字段和提交格式，不代表最终成绩：

```powershell
.\.venv\Scripts\python.exe train_models.py --model all --quick
```

## 单独训练某个模型

```powershell
.\.venv\Scripts\python.exe train_models.py --model catboost
.\.venv\Scripts\python.exe train_models.py --model lightgbm
.\.venv\Scripts\python.exe train_models.py --model xgboost
.\.venv\Scripts\python.exe train_models.py --model random_forest
```

默认 `--feature-set v1`，与第一次正式运行完全一致。检测新增特征是否有效时，
先用 LightGBM 在相同时间切分上训练：

```powershell
.\.venv\Scripts\python.exe train_models.py --model lightgbm --feature-set v2
```

只有 v2 验证 AUC 相对第一次 LightGBM 的 `0.948607` 有稳定提高时，再扩展到
CatBoost、XGBoost 和随机森林。

当前特征实验结果：

| 特征集 | 特征数 | 内容 | LightGBM Validation AUC |
| --- | ---: | --- | ---: |
| v1 | 28 | 第一次运行使用的原始特征工程 | **0.948607** |
| v2 | 46 | 全部候选特征与 3 个类别组合 | 0.946023 |
| v3 | 34 | 原特征加 6 个连续交互 | 0.948097 |
| v4 | 25 | 删除 3 个低贡献类别特征 | 0.948415 |

三个新方案都没有超过 v1，因此默认仍使用 `--feature-set v1`。v2、v3、v4
仅保留为复现实验，不用于最终提交。

## 依次训练四个模型

```powershell
.\.venv\Scripts\python.exe train_models.py --model all
```

也可以直接运行：

```powershell
.\run_all.ps1
```

运行完成后：

- 四个模型的验证指标位于新建的 `outputs/run_时间/` 目录；
- 模型文件位于 `outputs/models/`；
- 每个模型各有一份提交文件，位于 `outputs/submissions/`；
- 每个模型会保存 `valid_predictions.csv`，用于画 ROC、PR 和混淆矩阵；
- 每份提交文件都使用 `id,is_fraud` 两列，`is_fraud` 为连续概率。

## 生成项目报告图表

完整训练结束后运行：

```powershell
.\.venv\Scripts\python.exe make_report_assets.py --run-id 你的运行编号
```

省略 `--run-id` 时自动使用最近一次运行。输出位于：

```text
outputs/report_assets/
```

如果只需要补充验证集预测，不重新训练最终模型和生成提交文件：

```powershell
.\.venv\Scripts\python.exe train_models.py --model all --validation-only
```

## OOF 多模型融合实验

该实验保留第一次运行结果，使用 5 折交叉验证生成 LightGBM 和 XGBoost
的 OOF 预测，然后搜索概率平均或排序平均的最优权重：

```powershell
.\.venv\Scripts\python.exe train_oof_ensemble.py
```

实验结果和对比表位于：

```text
outputs/experiments/exp01_oof_ensemble_时间/
outputs/submissions/submission_oof_ensemble_时间.csv
```

在完全相同的时间顺序验证集上，对比 OOF 融合权重与第一次运行结果：

```powershell
.\.venv\Scripts\python.exe evaluate_ensemble_holdout.py
```

## CatBoost 参数搜索与多种子平均

脚本先在小规模分层样本上筛选参数，再把最优候选放到完整 80/20 时间验证集复核。
只有验证 AUC 超过第一次 CatBoost 至少 `0.0003`，才会训练全量模型并生成提交：

```powershell
.\.venv\Scripts\python.exe optimize_catboost.py
```

快速检查脚本流程：

```powershell
.\.venv\Scripts\python.exe optimize_catboost.py --quick
```

## CatBoost 变体与多种子平均

在相同时间验证集上比较 SymmetricTree、Depthwise、Lossguide 等变体，并对
最佳变体进行 3 个随机种子的概率平均：

```powershell
.\.venv\Scripts\python.exe train_catboost_variants.py
```

只有种子平均超过第一次 CatBoost 基线至少 `0.0002` 时才会生成新提交。

第一次正式实验结果：

| 方案 | Validation AUC |
| --- | ---: |
| 第一次 CatBoost 基线 | **0.948834** |
| SymmetricTree depth 6 复跑 | 0.948516 |
| SymmetricTree depth 7 | 0.948316 |
| Depthwise depth 6 | 0.948263 |
| Lossguide 64 | 0.947821 |
| 最佳变体三种子平均 | 0.948701 |

三种子平均没有超过第一次单模型基线，因此没有创建新提交。第一次基线与同参数
复跑之间存在约 `0.0003` 的差异，说明当前验证集上的微小模型差异包含明显的
随机种子波动。

如需仍然生成一份探索性提交用于线上试验，可复用已有实验中的最佳迭代轮数：

```powershell
.\.venv\Scripts\python.exe build_catboost_variant_submission.py `
  --experiment-dir .\outputs\experiments\exp04_catboost_variants_20261109_185159
```

已生成的探索性提交文件：

```text
outputs/submissions/submission_catboost_variant_seedavg_exploratory_20261109_192255.csv
```

该文件来自最佳 CatBoost 变体的三个完整训练种子平均，验证 AUC 为
`0.948701`，低于第一次基线 `0.948834`。它仅用于线上试榜，不应覆盖当前最佳
CatBoost 提交。

第一次正式搜索的最优候选为 `depth=6, learning_rate=0.03`。它在小样本上的
AUC 为 `0.949074`，但完整时间验证集复核只有 `0.948517`，低于基线
`0.948834`，因此脚本没有生成提交文件。该保护逻辑用于避免仅凭小样本调参结果
覆盖当前最佳模型。

### 第一次融合实验结果

第一次运行结果保存在：

```text
outputs/run_20261104_221722/
```

同一时间顺序验证集上的核心结果：

| 模型 | Validation AUC |
| --- | ---: |
| 第一次 CatBoost | 0.948834 |
| LightGBM + XGBoost 概率融合 | 0.948985 |
| LightGBM + XGBoost 等权排序融合 | 0.949013 |

目前验证 AUC 最高的提交文件为：

```text
outputs/submissions/submission_ensemble_holdout_rank_20261106_222741.csv
```

该文件仍然是 `id,is_fraud` 两列。融合使用排序值，只用于 ROC-AUC；
如果需要保留概率解释，应使用 `submission_ensemble_holdout_probability_20261106_222741.csv`。

公开榜结果显示该融合没有超过 CatBoost：

| 提交 | Public AUC |
| --- | ---: |
| CatBoost 第一次运行 | 0.94630 |
| LightGBM + XGBoost 排序融合 | 0.94586 |

因此当前正式最佳模型仍是 CatBoost。v2、v3、v4 特征实验均没有超过 v1，
后续不再继续在该验证切分上搜索融合权重，也不默认启用新增特征。

## 防止数据泄露

- `id` 不作为模型特征，也不作为普通数值特征输入。
- `is_fraud` 只存在于训练集，训练前与特征矩阵分离。
- 验证集来自训练集最后 20%，不读取测试集标签。
- 所有统计型预处理只在训练部分拟合。
- LightGBM 和 XGBoost 的类别字典只从训练部分生成。
- 随机森林的 One-Hot 编码器只在训练部分拟合。
- 测试集只在模型训练完成后执行 `predict_proba`。
- `sample_submission.csv` 只用于核对 ID 和提交格式，不作为特征。

## 说明

完整训练四个模型的耗时取决于 CPU、内存和线程数。若内存有限，建议先单独运行 LightGBM，再运行 CatBoost、XGBoost 和随机森林。
