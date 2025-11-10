## 環境作成

## uv のインストール
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

## 環境の同期
`lstm-vae/` 下で
```bash
uv sync
```

## ライブラリの追加
```bash
uv add [package-name]
```

## コードの実行

### 学習
`main` の `params` の編集でパラメータ変更
```bash
uv run python3 main.py --mode train
```
`outputs/models/yyyymmdd-index/model.pt` にモデルができる

### 推論
```bash
uv run python3 main.py --mode pred --model {path/to/model} --input {path/to/input}
```
`outputs/pred/from_yyyymmdd-index/` に出力
