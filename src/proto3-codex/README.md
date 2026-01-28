# proto3-codex (CSV -> Sketch -> Surface)

FreeCADのスケッチ拘束をCSVで更新し、SURFACE(Part::Face)の形状が破綻していないか検証する簡易ツール。

CSVの寸法拘束リストは不変で、値だけ変更される前提。サーフェス名も固定（`Face` / `SURFACE`）。
`input/dims.csv` を固定で参照します。

## 使い方

```
"C:\Program Files\FreeCAD 1.0\bin\freecadcmd.exe" -c "c:\github_repo\cad_automaton\src\proto3-codex\run_proto3.py"
```

`src/proto3-codex/proto3_args.json` は固定です:

```json
[
  "c:\\github_repo\\cad_automaton\\ref\\TH1-ref.FCStd",
  "c:\\github_repo\\cad_automaton\\input\\dims.csv",
  "--out",
  "c:\\github_repo\\cad_automaton\\output\\TH1-ref-updated.FCStd"
]
```

- CSVは列 `index,type,name,value` を想定（`name` と `value` を使用）。
- デフォルトのスケッチは最初の Sketcher::SketchObject。
- デフォルトのサーフェスは `Name=Face` または `Label=SURFACE`。

## オプション

- `--sketch` : SketchのName/Label指定
- `--surface-name` : サーフェスのName指定（既定: Face）
- `--surface-label` : サーフェスのLabel指定（既定: SURFACE）
- `--dry-run` : 変更適用せず検証のみ
- `--allow-missing` : CSV/スケッチの未一致を許容

## 実行方法の補足

FreeCADCmdが引数の解釈を誤るケースがあるため、`run_proto3.py`経由で実行してください。
（直接 `apply_csv_to_surface.py` を実行する場合は `proto3_args.json` が必要です。）

## サーフェス検証

- shape is null/invalid のチェック
- shape.check(True) の問題件数
- 面積が 0 以下でないこと
