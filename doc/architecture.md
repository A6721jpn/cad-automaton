# CAD Automaton - アーキテクチャ設計書

## 1. コンセプト

### 1.1 背景と課題

CATIAの拘束スケッチ寸法をPyCATIA（COM）で変更して形状生成する従来手法には以下の問題がある：

- 寸法組合せによってスケッチが自己交差・縮退して破綻
- COM経由では確実にエラー検知・例外化できない
- 破綻モデルを後工程へ流してしまうリスク

### 1.2 設計方針

1. **拘束スケッチ中心の生成をやめる**
   - 2D拘束スケッチ → ソルバ解の枠組みは地雷を生みやすい
   - 閉じた2D領域を直接構成する方がロバスト

2. **品質ゲートを強制する**
   - 検証 → 修復 → 補正 → 再検証のパイプライン
   - B-Rep valid のみSTEP出力

3. **自動補正の三段構え**
   - パラメータ射影（制約内へ最近傍）
   - 形状正則化（量子化・スリバー除去）
   - 段階フォールバック（R縮小など）

---

## 2. アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py (CLI)                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────────────────────────┐  │
│  │ -r Mode │  │ -e Mode │  │ Random Mode (default)       │  │
│  └────┬────┘  └────┬────┘  └──────────────┬──────────────┘  │
└───────┼────────────┼───────────────────────┼────────────────┘
        │            │                       │
        ▼            ▼                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     config.py                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ PARAM_RANGES│  │ Config      │  │ FileConfig          │  │
│  │ (min/max)   │  │ (YAML R/W)  │  │ (per-file params)   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   step_generator.py                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ generate_step() Pipeline:                           │    │
│  │  1. Parameter Validation → auto_correction.py       │    │
│  │  2. Geometry Build → geometry.py                    │    │
│  │  3. Shape Validation → quality_gate.py              │    │
│  │  4. STEP Export → build123d.export_step()           │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐  ┌────────────────────┐  ┌──────────────────┐
│ geometry.py   │  │ quality_gate.py    │  │auto_correction.py│
│               │  │                    │  │                  │
│ ProfileParams │  │ validate_shape()   │  │ project_params() │
│ compute_      │  │ fix_shape()        │  │ regularize_      │
│   vertices()  │  │ check_min_         │  │   geometry()     │
│ build_profile │  │   thickness()      │  │ fallback_        │
│   _face()     │  │                    │  │   simplify()     │
└───────────────┘  └────────────────────┘  └──────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   build123d     │
                    │   (OCP/OCCT)    │
                    └─────────────────┘
```

---

## 3. モジュール詳細

### 3.1 geometry.py

**責務**: パラメータから幾何形状を構築

```python
@dataclass
class ProfileParams:
    L1: float = 0.775   # 直線長さ (10個)
    ...
    R1: float = 0.25    # フィレット半径 (2個)
    R2: float = 0.24
    min_thickness: float = 0.2  # 最小肉厚制約
```

**処理フロー**:
1. `compute_vertices()`: パラメータ → 16頂点座標計算
2. `build_profile_face()`: 頂点 → BuildSketch → Polygon → Face

### 3.2 config.py

**責務**: YAML設定ファイルの読み書き

```yaml
version: '1.0'
files:
  TH1-ref.stp:
    parameters:
      L1:
        description: top_inner_y
        value: 0.775
        min: 0.1
        max: 2.0
    constraints:
      min_thickness: 0.2
```

**主要クラス**:
- `PARAM_RANGES`: 破綻しないパラメータ範囲定義
- `Config`: 複数ファイルを管理するコンテナ
- `FileConfig`: ファイル毎のパラメータ設定

### 3.3 quality_gate.py

**責務**: 形状の妥当性検証と修復

| 関数                    | 説明                   |
| ----------------------- | ---------------------- |
| `validate_shape()`      | B-Rep妥当性チェック    |
| `fix_shape()`           | ShapeFixによる自動修復 |
| `check_min_thickness()` | 最小肉厚制約チェック   |

### 3.4 auto_correction.py

**責務**: 無効パラメータの自動補正

| 段階 | 関数                    | 処理                     |
| ---- | ----------------------- | ------------------------ |
| 1    | `project_params()`      | 範囲外を境界に射影       |
| 2    | `regularize_geometry()` | 0.01mmグリッドに量子化   |
| 3    | `fallback_simplify()`   | フィレットを安全値に縮小 |

### 3.5 step_generator.py

**責務**: 生成パイプラインの統合

```
┌────────────┐     ┌─────────────┐     ┌────────────┐
│ Parameters │────▶│ Validation  │────▶│ Correction │
└────────────┘     └─────────────┘     └────────────┘
                          │                   │
                          ▼                   ▼
┌────────────┐     ┌─────────────┐     ┌────────────┐
│ STEP Export│◀────│ Shape Valid │◀────│ Geometry   │
└────────────┘     └─────────────┘     └────────────┘
```

---

## 4. パラメータ定義

### 4.1 形状パラメータ（12個）

| ID  | 名称          | 説明            | 範囲(mm)   |
| --- | ------------- | --------------- | ---------- |
| L1  | top_inner_y   | 内側上部幅      | 0.1 - 2.0  |
| L2  | top_outer_y   | 外側上部幅      | 0.1 - 2.0  |
| L3  | top_z         | 上部垂直長      | 0.1 - 2.0  |
| L4  | upper_step_y  | 上段ステップ幅  | 0.1 - 1.0  |
| L5  | outer_y       | 外側幅          | 0.1 - 2.0  |
| L6  | outer_z       | 外側高さ        | 0.1 - 2.0  |
| L7  | lower_step_z  | 下段ステップ高  | 0.1 - 2.0  |
| L8  | inner_shelf_y | 内側棚幅        | 0.1 - 2.0  |
| L9  | inner_shelf_z | 内側棚高        | 0.1 - 1.5  |
| L10 | bottom_z      | 底部高さ        | 0.3 - 3.0  |
| R1  | fillet_inner  | 内側フィレットR | 0.05 - 0.5 |
| R2  | fillet_bottom | 底部フィレットR | 0.05 - 0.5 |

### 4.2 制約パラメータ

| ID            | 説明           | デフォルト |
| ------------- | -------------- | ---------- |
| min_thickness | 最小肉厚（mm） | 0.2        |

---

## 5. 使用方法

### 5.1 環境構築

```powershell
conda create -n b123d python=3.13
conda activate b123d
conda install build123d
pip install pyyaml
```

### 5.2 実行

```powershell
# 1. テンプレートSTEPをinputに配置
copy ref\*.stp input\

# 2. 認識モード（config.yaml生成）
python -m src.main -r

# 3. 編集モード（config.yamlに従いSTEP生成）
python -m src.main -e

# 4. ランダムモード（範囲内でランダム生成）
python -m src.main
```

---

## 6. 今後の拡張

1. **STEP解析機能**: 実STEPファイルからパラメータ自動抽出
2. **フィレット改善**: build123d fillet APIの活用
3. **バッチ最適化**: CAE連携用の夜間バッチ対応
4. **ログ強化**: 失敗ケース保存、補正量KPI化
