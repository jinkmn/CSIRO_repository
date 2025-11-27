# 🏆 [コンペ名: CSIRO Biomass Prediction] - Team Codebase

チーム用のコード管理リポジトリです。 Hydra + uv を使用し、ローカルでの高速な実験と Kaggle Notebook での再現性を両立させる構成になっています。

## 📂 ディレクトリ構成

```Plaintext
.
├── bin/                  # 実行用スクリプト (train.py, run_ml.py)
├── conf/                 # Hydra設定ファイル (実験パラメータの管理)
│   ├── dir/              # パス設定 (local.yaml / kaggle.yaml)
│   ├── feature/          # 特徴抽出器の設定 (DINOv2, SigLIP...)
│   ├── model/            # 予測モデルの設定 (Lasso, ResNet...)
│   ├── training/         # 学習パラメータ (Epoch, Fold数...)
│   └── experiment/       # 実験レシピ (複数の設定を組み合わせたもの)
├── src/                  # ソースコード
│   ├── data/             # 前処理、Dataset定義
│   ├── features/         # 特徴抽出ロジック
│   └──  models/           # モデル定義
├── data/                 # ローカル用データ置き場 (Git管理外)
├── output/               # 実験結果の保存先 (Git管理外)
├── uv.lock               # ライブラリのバージョン固定ファイル
└── pyproject.toml        # プロジェクト設定
```

# 🛠️ 環境構築 (Local Setup)

パッケージマネージャーには uv を採用しています。pip や conda よりも圧倒的に高速で、チーム全員の環境を完全に一致させることができます。

## 1. uv のインストール

まだインストールしていない場合のみ実行してください。

```Bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Mac / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

※ インストール後、ターミナルを再起動してください。

## 2. リポジトリのクローンと同期

```Bash
git clone https://github.com/jinkmn/CSIRO_repository.git
cd https://github.com/jinkmn/CSIRO_repository.git

# 依存ライブラリを一括インストール (これで環境構築完了です)
# uvへのpathが通っていない場合は設定してください。
uv sync
```

## 3. コンペデータの配置

Kaggle からデータをダウンロードし、data/ フォルダに配置してください。

```Plaintext
data/
├── train.csv
├── test.csv
└── ...
```

# 🏃‍♂️ ローカルでの実行 (Local Execution)

uv run を頭に付けることで、仮想環境内でスクリプトを実行できます。

基本的な実行 (DINOv2 + Lasso)

```Bash

# デフォルト設定で実行 (conf/config.yaml の内容)
uv run python bin/run_ml.py
```

パラメータを変更して実行 (Hydra)
設定ファイルを書き換えなくても、コマンドライン引数で上書き可能です。

```Bash

# 実験名を指定 (output/my_test_run に保存される)
uv run python bin/run_ml.py exp_name=my_test_run

# データ読み込み数を制限してデバッグ (CPUでも動きやすい)
uv run python bin/run_ml.py dir.data_limit=10

# モデルのパラメータを変更
uv run python bin/run_ml.py model.alpha=0.5
```

本格的な実験 (Experiment)
conf/experiment/ にあるレシピファイルを使用する場合：

```Bash
uv run python bin/run_ml.py experiment=exp001_best_lasso
```

# ☁️ Kaggle Notebook での実行 (Kaggle Execution)

GitHub Actions により、main ブランチに Push されたコードは自動的に Kaggle Dataset としてアップロードされます。

実行手順

1. Kaggle Notebook を作成する。

2. Input に Code Dataset ([あなたのデータセット名]) と コンペデータ を追加する。

3. Internet Access を ON にする (Settings パネル)。

4. 以下のコードをセルに貼り付けて実行する。

```Python
import sys
import os

# =================================================
# 1. パスの自動特定 & 環境セットアップ
# =================================================
input_dirs = os.listdir('/kaggle/input')

# コード置き場を探す
code_dir_candidates = [d for d in input_dirs if 'code' in d.lower()]
if code_dir_candidates:
    CODE_DIR = f"/kaggle/input/{code_dir_candidates[0]}"
else:
    CODE_DIR = "/kaggle/input/csiro-code-repository"

# コンペデータ置き場を探す
data_dir_candidates = [d for d in input_dirs if 'csiro' in d.lower() and 'code' not in d.lower()]
if data_dir_candidates:
    DATA_DIR = f"/kaggle/input/{data_dir_candidates[0]}"
else:
    DATA_DIR = "/kaggle/input/csiro-biomass-data"

print(f"✅ Code Dir: {CODE_DIR}")
print(f"✅ Data Dir: {DATA_DIR}")

# ライブラリのインストール (NumPyバージョン対策含む)
# uvをインストール
!curl -LsSf https://astral.sh/uv/install.sh | sh

# NumPyのバージョン対策 (uv経由でインストール)
!/root/.cargo/bin/uv pip install "numpy<2.0" --system

# requirements.txt のインストール
import os
# ... (パス特定のロジックは同じ) ...

if os.path.exists(f"{CODE_DIR}/requirements.txt"):
    print("Installing requirements with uv...")
    # pip install の代わりに uv pip install を使う (爆速です)
    !/root/.cargo/bin/uv pip install -r {CODE_DIR}/requirements.txt --system

# ソースコードをimport可能にする
sys.path.append(CODE_DIR)

# =================================================
# 2. 実験実行
# =================================================
# Session Restartが必要な場合があるので、エラーが出たらRestart Sessionしてください
print("🚀 Starting Experiment...")

#例
!python {CODE_DIR}/bin/run_ml.py \
    dir=kaggle \
    dir.code_dir={CODE_DIR} \
    dir.data_dir={DATA_DIR} \
    exp_name=kaggle_run_001 \
    feature=dino_giant \
    model=lasso

```

# 🔄 開発ワークフロー (Development)

コード編集: ローカルで src/ や conf/ を編集。

ライブラリ追加: 新しいライブラリが必要な場合は uv add [ライブラリ名] を実行し、requirements.txt を更新。

```Bash
# Kaggle用に requirements.txt を書き出し (torch系は除外推奨)
uv export --format requirements-txt --no-emit-package torch --no-emit-package torchvision --output-file requirements.txt
```

Push: GitHub に Push すると、自動で Kaggle 上のコードも更新されます。

Kaggle: Notebook の Input を更新 ("Check for updates") し、Session を再起動して実行。

# Tips

Hydra の使い方がわからない？

conf/config.yaml がベースの設定です。

conf/model/ などを書き換えるか、コマンドライン引数 key=value で上書きします。

ローカルで CPU 実行したい

dir.data_limit=10 を付けると、最初の 10 件だけで動くのでデバッグに便利です。
