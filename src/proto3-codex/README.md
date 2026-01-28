# proto3-codex (CSV -> Sketch -> Surface)

FreeCADのスケッチ拘束をCSVで更新し、SURFACE(Part::Face)の形状が破綻していないか検証する簡易ツール。

## 使い方

```
"C:\Program Files\FreeCAD 1.0\bin\freecadcmd.exe" -c "c:\github_repo\cad_automaton\src\proto3-codex\apply_csv_to_surface.py" -- c:\github_repo\cad_automaton\ref\TH1-ref.FCStd c:\github_repo\cad_automaton\input\dims.csv --out c:\github_repo\cad_automaton\output\TH1-ref-updated.FCStd
```

- CSVは列 `index,type,name,value` を想定（`name` と `value` を使用）。
- デフォルトのスケッチは最初の Sketcher::SketchObject。
- デフォルトのサーフェスは `Name=Face` または `Label=SURFACE`。

## オプション

- `--sketch` : SketchのName/Label指定
- `--surface-name` : サーフェスのName指定（既定: Face）
- `--surface-label` : サーフェスのLabel指定（既定: SURFACE）
- `--dry-run` : 変更適用せず検証のみ
- `--strict` : CSV側/スケッチ側の未一致をエラーにする

## サーフェス検証

- shape is null/invalid のチェック
- shape.check(True) の問題件数
- 面積が 0 以下でないこと
