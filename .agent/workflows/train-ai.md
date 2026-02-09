---
description: Train ai-v0 model using Optuna hyperparameter tuning
---

# AI-v0 学習ワークフロー

// turbo-all

Optunaを使用してai-v0モデルをハイパーパラメータチューニングします。

## 必要条件

- conda環境 `ai` がアクティブ
- 教師データCSVが存在すること

## 基本実行

```powershell
conda run -n ai python src/ai-v0/train_optuna.py
```

## オプション指定

```powershell
conda run -n ai python src/ai-v0/train_optuna.py --data src/ai-v0/merged_samples.csv --n-trials 50
```

## 利用可能なオプション

| オプション    | デフォルト                  | 説明                   |
| ------------- | --------------------------- | ---------------------- |
| `--data`      | `proto3-hybrid_samples.csv` | 教師データCSVパス      |
| `--n-trials`  | 50                          | Optunaトライアル数     |
| `--artifacts` | `artifacts_optuna/`         | 成果物出力ディレクトリ |
| `--seed`      | 42                          | 乱数シード             |

## 出力成果物

学習完了後、`artifacts_optuna/`に以下が保存:

- `model.joblib` - 学習済みMLPモデル
- `scaler.joblib` - 特徴量スケーラー
- `metrics.json` - 評価指標（Accuracy, F1, ROC-AUC）
- `best_params.json` - 最適ハイパーパラメータ

## 典型的なワークフロー

1. 新規データ生成: `/generate-samples`
2. データマージ:
   ```powershell
   conda run -n ai python src/ai-v0/merge_samples.py --inputs src/ai-v0/proto3-hybrid_samples.csv src/ai-v0/new_samples.csv --output src/ai-v0/merged_samples.csv
   ```
3. 学習実行:
   ```powershell
   conda run -n ai python src/ai-v0/train_optuna.py --data src/ai-v0/merged_samples.csv --n-trials 50
   ```
