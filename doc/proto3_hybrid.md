# proto3-hybrid アルゴリズムと実装詳細

本書は `src/proto3-hybrid/hybrid_solver.py` の設計・動作を、  
**抽象レイヤーのアルゴリズム**と**具体レイヤーの実装**に分けて説明する。

## 1. 抽象レイヤー（アルゴリズム）

### 1.1 目的
- FreeCAD スケッチの寸法空間における **破綻/安全の境界**を高効率で推定する。
- 単変数探索ではなく **多変数同時探索**を前提とする。
- 評価コスト（FreeCAD リコンピュート）は高いので、**境界付近に評価を集中**する。

### 1.2 問題の捉え方
- 各寸法ベクトルを入力 `x`、出力を `safe(1)/fail(0)` とする **2値分類問題**。
- 境界の推定 = `P(safe|x) ≈ 0.5` 近傍を重点探索。

### 1.3 ハイブリッド探索の全体像

1) **初期サンプリング（広域探索）**  
   - LHS（Latin Hypercube Sampling）で全空間を均等カバー。  
   - 目的: 大域的な安全/破綻の概形を掴む。

2) **分類器学習**  
   - ラベル付きデータ `(X, y)` を使って RandomForest を学習。  
   - `predict_proba` で `P(safe)` を推定。

3) **境界集中サンプリング（Active Learning）**  
   - 候補点を Sobol で多数生成。  
   - `|P(safe)-0.5|` が小さい点を優先し評価。

4) **探索と活用のバランス**  
   - `explore_frac` により **ランダム探索**も混ぜる。  
   - 境界推定が偏るのを防ぐ。

5) **フォールバック探索（安全域再取得）**  
   - 1イテレーションで safe がゼロなら  
     **基準値近傍 (ratio 1±narrow_ratio)** で再サンプル。

6) **安全域の抽出**  
   - `safe` サンプルから各寸法の `min/max` を抽出。  
   - CSV に `feasible_min` / `feasible_max` を追記。

### 1.4 アルゴリズムの特徴
- **境界に強い**: Active Learningで境界付近に評価を集中。
- **安定性**: 各評価前に基準寸法へリセットして破綻状態の伝播を防止。
- **安全域が小さい場合にも対応**: fallback で基準近傍を再探索。

---

## 2. 具体レイヤー（実装）

### 2.1 エントリポイント
ファイル: `src/proto3-hybrid/hybrid_solver.py`

CLI 例:
```
python src/proto3-hybrid/hybrid_solver.py ^
  ref/TH1-ref.FCStd ^
  input/dims_deg.csv ^
  --template temp/constraints.json ^
  --samples 5000 ^
  --ratio-min 0.85 --ratio-max 1.15 ^
  --init-samples 500 --batch-size 450 --iters 10 ^
  --explore-frac 0.6 --narrow-ratio 0.05
```

### 2.2 入力
- `fcstd`: FreeCAD のテンプレート `.FCStd`
- `csv`: 寸法値 CSV（`dims_deg.csv`）
- `template`: proto2で抽出した拘束テンプレート（`temp/constraints.json`）
  - **拘束 index / 型 / 値** を正しく保持

### 2.3 データ構造
- `ConstraintSpec`  
  - `index`: Sketcher内の拘束 index  
  - `name`: 拘束名  
  - `ctype`: Distance / DistanceX / DistanceY / Angle  
  - `base_value`: 基準値  
  - `angle_unit`: rad or deg（テンプレート値から推定）

### 2.4 FreeCAD 操作
- `_apply_sample(...)`
  - **各サンプル適用前にベース値でリセット**
  - `doc.recompute()` → 形状チェック
  - `safe(1)/fail(0)` を返す

**破綻判定**
- `doc.recompute()` 後の `Invalid` / `RecomputeError`
- Shape: `isNull`, `!isValid`, `check()` 失敗, `Area<=0`

### 2.5 サンプリング

#### 初期（LHS）
- `_lhs_samples(...)`  
  - `scipy.stats.qmc.LatinHypercube`

#### 候補生成（Sobol）
- `_sobol_samples(...)`  
  - `scipy.stats.qmc.Sobol`
  - サンプル数は内部で **2の冪**に丸める

#### Fallback
- `safe=0` の反復で `1±narrow_ratio` 近傍を再探索

### 2.6 機械学習モデル
- `RandomForestClassifier`
  - `class_weight="balanced"`
  - `predict_proba` を使って境界近傍を推定

### 2.7 ログ
```
[init] 50/500 evals, safe=...
[iter 1/10] training model on ...
[iter 1] 50/200 evals, safe=...
[iter 1 fallback] ...
```

### 2.8 出力
- CSV (`dims_deg.csv`) に
  - `feasible_min`
  - `feasible_max`
  を追記
- `temp/proto3-hybrid_samples.csv` に
  - 全サンプル + safe/fail

### 2.9 依存
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`

### 2.10 注意点
- **探索範囲が狭すぎると全て safe**  
  → Active Learning 不成立 → 広げる必要あり
- **探索範囲が広すぎると安全域が希少**  
  → fallback を活かす

---

## 3. 推奨パラメータ例

### 安全域が広い場合
```
--ratio-min 0.9 --ratio-max 1.1
```

### 境界を精密に狙う場合
```
--ratio-min 0.85 --ratio-max 1.15
--init-samples 500
--batch-size 450 --iters 10
--explore-frac 0.6 --narrow-ratio 0.05
```

---

## 4. まとめ
- **広域 LHS → Active Learning → fallback 探索**という三段構成。
- FreeCADの再計算/面検証により **安全/破綻**を判定。
- 計算予算が多いほど、境界をより精密に抽出できる。
