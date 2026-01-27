# proto2-codex (FreeCAD sketch-driven prototype)

ユーザーが用意した **完全拘束済みの2Dスケッチ**（FreeCAD）を入力として、
拘束名をそのままキーにしたテンプレートを生成するプロトタイプです。

## 目的
- STEP直接編集ではなく **Sketcherの拘束ソルバで形状を確定**する
- ユーザーが付けた拘束名をそのまま使い、外部から自動化できるようにする

## 前提
- FreeCADがインストール済み
- FreeCADのPythonで実行（`FreeCADCmd` または `conda env fcad-codex`）

## スケッチ側の準備
- Sketcher で **DOF=0（完全拘束）**
- 寸法拘束に **名前（Name）を設定**  
  - Sketcherで拘束を選択 → 名前を編集  
  - 例: `L1`, `L2`, `R1`, `outer_right_y` など

## 使い方

### 1) 拘束値の抽出（テンプレート生成）
```
python src/proto2-codex/extract_constraints.py path\to\model.FCStd --sketch Sketch
```
出力ファイルを作る場合:
```
python src/proto2-codex/extract_constraints.py path\to\model.FCStd --sketch Sketch --out constraints.yaml
```

### 2) 拘束値を反映して保存
```
python src/proto2-codex/apply_constraints.py path\to\model.FCStd --sketch Sketch --in constraints.yaml --out updated.FCStd
```

## 出力フォーマット（概要）
```
schema_version: 1
source:
  fcstd: path/to/model.FCStd
  sketch: Sketch
constraints:
  L1:
    index: 3
    type: Distance
    value: 12.3
    source_name: L1
constraint_order:
  - L1
  - L2
```

## 備考
- unnamed拘束は既定では除外されます（`--include-unnamed` で含める）
- まだSTEP出力までは実装していません（必要なら拡張します）
