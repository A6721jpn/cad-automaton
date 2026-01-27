## プロジェクト概要
参照STEP (TH-1ref.step)から、動的に寸法テンプレートを生成し、それにユーザーが入力した寸法に従ってSTEPファイルを生成するpythonプログラム

## アルゴリズム
1. 入力として FreeCAD の .FCStd と対象スケッチ（Label/Name）を受け取る。スケッチは DOF=0 の完全拘束で、寸法拘束には名前が付いていることを前提とする。
2. スケッチ内の拘束情報（index, type, name, value）を抽出する。`--list` 指定時は一覧表示のみで終了する。
3. 抽出結果をテンプレートとして YAML/JSON に出力する。拘束名をキーにし、`schema_version`/`source`/`constraints`/`constraint_order` を含む構成で保存する。未命名拘束は既定で除外し、`--include-unnamed` 指定時のみ含める。
4. 既存テンプレート（YAML/JSON）を読み込み、拘束名に対応する値を抽出する。
5. テンプレートの拘束名に一致するスケッチ拘束へ値を反映する（必要なら `sketch_order` で複数スケッチ対応）。`--dry-run` 指定時は検証のみ行う。
6. 反映後に再計算し、指定した出力 .FCStd として保存する。



## パッケージマネージャー
npm

## 仮想環境
conda activate b123d
conda activate fcad
conda activate fcad-codex

## 実行コマンド
python -m src.main.py -r
python -m src.main.py -e