---
description: Generate training data for ai-v0 using proto3-hybrid via FreeCAD
---

# データ生成ワークフロー

// turbo-all

proto3-hybridを使用してai-v0の教師データを生成します。

## 必要条件

- FreeCAD 1.0がインストールされていること
- `ref/TH1-ref.FCStd` と `input/dims_deg.csv` が存在すること

## 基本実行

```powershell
& 'C:\Program Files\FreeCAD 1.0\bin\freecadcmd.exe' 'c:\github_repo\cad_automaton\scripts\generate_training_data.py'
```

## オプション指定

```powershell
& 'C:\Program Files\FreeCAD 1.0\bin\freecadcmd.exe' 'c:\github_repo\cad_automaton\scripts\generate_training_data.py' --samples 10000 --output 'c:\github_repo\cad_automaton\src\ai-v0\new_samples.csv'
```

## 利用可能なオプション

| オプション    | デフォルト                                | 説明                 |
| ------------- | ----------------------------------------- | -------------------- |
| `--samples`   | 5000                                      | 生成サンプル数       |
| `--output`    | `src/ai-v0/proto3-hybrid_samples_new.csv` | 出力CSVパス          |
| `--ratio-min` | 0.78                                      | サンプリング最小比率 |
| `--ratio-max` | 1.22                                      | サンプリング最大比率 |
| `--seed`      | 42                                        | 乱数シード           |

## 生成後のマージ

新しいデータを既存データとマージ:

```powershell
conda run -n ai python src/ai-v0/merge_samples.py --inputs src/ai-v0/proto3-hybrid_samples.csv src/ai-v0/new_samples.csv --output src/ai-v0/merged_samples.csv
```

## 注意事項

- FreeCAD内蔵Pythonで実行されるため、Active Learningにはsklearn不要のフォールバック動作
- 生成には5000サンプルで約5-10分かかる（マシン性能による）
