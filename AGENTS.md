## プロジェクト概要
参照STEP (TH-1ref.step)から、動的に寸法テンプレートを生成し、それにユーザーが入力した寸法に従ってSTEPファイルを生成するpythonプログラム

## アルゴリズム
1. FreeCADでinputフォルダにあるジオメトリをロードする。
2. このジオメトリ形状のすべてが一意に決まる必要十分な寸法と外形エッジを参照STEPから抽出する。
3. 寸法をconfig.yamlへリストアップ、各寸法の最大・最小価を自動判定し寸法テンプレートをする。
4. 抽出したエッジと寸法をプロット用ライブラリでプロットとして再現、PNG画像で出力する。-r オプションで実行された場合はここまでの処理だけを行う。
5. ユーザーが寸法テンプレートに記入した寸法どおりにSTEPの各部の寸法を変更し保存する。-eオプションで実行された場合これ以降の処理だけを行う。
6. 生成したSTEPのファイル名を{参照STEP}_generated.stepとし、保存する。



## パッケージマネージャー
npm

## 仮想環境
conda activate b123d
conda activate fcad
conda activate fcad-codex

## 実行コマンド
python -m src.main.py -r
python -m src.main.py -e