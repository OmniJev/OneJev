<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="ja/banner-dark.svg">
    <img alt="OneJev、マルチモーダル System One 意思決定モデル" src="ja/banner.svg" width="100%">
  </picture>
</p>

<p align="center"><a href="https://github.com/OmniJev/OneJev">English</a> · <a href="README_zh.md">简体中文</a> · <b>日本語</b></p>

<p align="center">
  <a href="https://huggingface.co/collections/OmniJev/onejev15"><img alt="Hugging Face" src="https://img.shields.io/badge/Hugging_Face-OneJev1.5-FFD21E?style=flat-square&logo=huggingface&logoColor=black"></a>
  <a href="https://huggingface.co/spaces/BradNLP/OneJev"><img alt="Demo" src="https://img.shields.io/badge/Demo-Try_it-F83800?style=flat-square&logo=gradio&logoColor=white"></a>
  <a href="https://github.com/OmniJev/OneJev"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-OmniJev%2FOneJev-181717?style=flat-square&logo=github&logoColor=white"></a>
  <a href="https://omnijev.github.io/OneJev/"><img alt="Website" src="https://img.shields.io/badge/Website-OneJev-0A84FF?style=flat-square&logo=googlechrome&logoColor=white"></a>
  <a href="../LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-D22128?style=flat-square&logo=apache&logoColor=white"></a>
</p>
<p align="center">
  <a href="https://huggingface.co/datasets/OmniJev/OneJev-Data"><img alt="Data" src="https://img.shields.io/badge/Data-OneJev--Data-FF9D00?style=flat-square&logo=huggingface&logoColor=white"></a>
  <a href="../examples/request.json"><img alt="API" src="https://img.shields.io/badge/API-System_One-009688?style=flat-square&logo=fastapi&logoColor=white"></a>
  <a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white"></a>
  <a href="https://pytorch.org"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.6%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"></a>
  <a href="https://github.com/huggingface/transformers"><img alt="Transformers" src="https://img.shields.io/badge/Transformers-5%2B-FFD21E?style=flat-square&logo=huggingface&logoColor=black"></a>
</p>

OneJev は [Qwen](https://huggingface.co/Qwen) をベースにしたマルチモーダルな System One 意思決定モデルです。スクリーンショット、写真、動画、テキストに関する型付きの質問に、1 回の順伝播で較正済みの確率を返します。サイズは 0.8B、4B、9B、27B の 4 種類です。

## 🔥 ニュース

- 🔥 **2026-10-13** [OneJev 1.5](https://huggingface.co/collections/OmniJev/onejev15) を公開しました。ぜひお試しください。
- **2026-09-28** オンラインデモを公開しました。👉 [こちらから](https://huggingface.co/spaces/BradNLP/OneJev)
- **2026-09-27** [OneJev](https://huggingface.co/collections/OmniJev/onejev) と [OneJev データセット](https://huggingface.co/datasets/OmniJev/OneJev-Data) を公開しました。

## Decision Index（公開部分）

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="ja/progress-dark.svg">
    <img alt="ブロック縦棒グラフ：Decision Index 0.3 公開部分のスコア、OneJev 1.5 と OneJev 1.0 の比較。0.8B 42.71 対 22.23（+20.5）、4B 54.89 対 40.84（+14.0）、9B 59.03 対 45.71（+13.3）、27B 63.64 対 55.56（+8.1）" src="ja/progress.svg" width="100%">
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="ja/leaderboard-dark.svg">
    <img alt="モデルサイズ別の Decision Index 0.3 公開部分ランキング 4 つ、各上位 10 件、OneJev1.5 を強調：OneJev1.5-0.8B はパラメータ 2.5B 以下の 48 モデル中 1 位、OneJev1.5-4B は 3B〜6B の 34 モデル中 2 位、OneJev1.5-9B は 6B〜12B の 24 モデル中 1 位、OneJev1.5-27B は 25B〜36B の 44 モデル中 5 位" src="ja/leaderboard.svg" width="100%">
  </picture>
</p>

## その他の評価

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="ja/results-dark.svg">
    <img alt="ブロック棒グラフ：4 サイズの OneJev1.5 と Jev 1.13、Jev-Omni 12B、Qwen3.8-27B 思考モードを OneJev テストセット、DecisionBench 難、TypeSafe、MMStar で比較" src="ja/results.svg" width="100%">
  </picture>
</p>

正解率（%）。OneJev テストセットは私たちの非公開評価セット、その他は公開ベンチマークです。

## 速度

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="ja/speed-dark.svg">
    <img alt="H200 1 枚、1280x720 のスクリーンショット 1 枚での遅延：0.8B 質問 1 つ 31 ms、質問 10 個 51 ms、質問あたり 5.1 ms；4B 質問 1 つ 64 ms、質問 10 個 104 ms、質問あたり 10.4 ms；9B 質問 1 つ 81 ms、質問 10 個 131 ms、質問あたり 13.1 ms；27B 質問 1 つ 189 ms、質問 10 個 324 ms、質問あたり 32.4 ms" src="ja/speed.svg" width="100%">
  </picture>
</p>

## クイックスタート

どちらかのバックエンドを起動し、下の Python サンプルを実行してください。

### 選択肢 A：PyTorch

NVIDIA GPU 向け。テキスト、画像、動画に対応しています。

```bash
pip install "qev[torch] @ git+https://github.com/OmniJev/OneJev.git"
qev serve --model OmniJev/OneJev1.5-4B --port 8000
```

### 選択肢 B：llama.cpp

GGUF モデル向け。テキストと画像に対応し、動画には PyTorch を使います。まず [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md) をインストールしてください（macOS では `brew install llama.cpp`）。

```bash
pip install git+https://github.com/OmniJev/OneJev.git
qev serve --gguf prithivMLmods/OneJev1.5-9B-GGUF:Q8_0 --port 8000
```

### 選択肢 C：Docker

リポジトリには、ハードウェアを検出してバックエンドを 1 つ起動する Compose スタックが含まれています。llama.cpp の profile は初回起動時に `prithivMLmods/OneJev1.5-9B-GGUF:Q8_0` をダウンロードします。別のサイズや量子化は `docker/.env` の `QEV_GGUF` で指定してください。

```bash
cp docker/.env.example docker/.env   # 非公開リポジトリでは HF_TOKEN を設定
./docker/start.sh                    # 自動検出（Windows：.\docker\start.ps1）
./docker/start.sh logs               # ログを追う
./docker/start.sh down               # すべての profile を停止
```

`./docker/start.sh <torch|rocm|vulkan|gguf|gguf-rocm|cpu>` でバックエンドを強制できます（`--build` で再ビルド）。どの profile も :8000 で API と Playground UI を提供します。Windows では `torch`（NVIDIA）と `cpu` のみ利用できます。

### リクエストを送る

どちらのバックエンドも `http://localhost:8000` で同じ API を提供します。別のターミナルで、自分の `screenshot.png` を使ってサンプルを実行してください：

```python
from qev import Client, Choice, Noul, Score
from qev.media import data_uri

r = Client("http://localhost:8000").system_one(
    state={"task": "Pay the open invoice from ACME", "screen": "<image:1>"},
    media=[{"type": "image", "data": data_uri("screenshot.png")}],
    questions={"done": Noul("The invoice has been paid"),
               "next": Choice("What should the agent do next?",
                              {"click": "click an element", "type": "type text", "scroll": "scroll", "stop": "stop"}),
               "progress": Score("How far along is the task?", ["not started", "halfway", "almost done", "done"])},
)
r.answers["done"].noul                  # 「はい」の確率
r.answers["next"].probabilities         # 選択肢ごとの確率
r.answers["progress"].score             # 期待レベル
```

API は TypeSafe System One と互換です。その他の例：[動画](../examples/video.py)、[curl](../examples/curl.sh)、[公式 TypeSafe SDK](../examples/official_sdk.py)。

## 学習

```bash
git clone https://github.com/OmniJev/OneJev.git
cd OneJev
pip install -e ".[train]"

# A・B のどちらかを選択。以下では A が有効です。
# A. デモ：画像付き 100 件（23 MB）
hf download OmniJev/OneJev-Data sample/sample-100.parquet --repo-type dataset --local-dir data/onejev
python -m train.unpack data/onejev --sample

# B. 全データ：94,707 件（17.7 GB）。B を使う場合は次の 2 行のコメントを外し、A を省いてください。
# hf download OmniJev/OneJev-Data --repo-type dataset --include "data/*.parquet" --local-dir data/onejev
# python -m train.unpack data/onejev
```

どちらを選んでも `data/onejev/train.jsonl` が作成され、画像と動画フレームは `data/onejev/media/` に展開されます。

[OneJev-Data](https://huggingface.co/datasets/OmniJev/OneJev-Data) は元の学習データ 99,193 問のうち 94,707 問を公開しています。残りは元データの再配布が許可されていません。

### ファインチューニング

4 モデルはそれぞれ Qwen3.5-0.8B、Qwen3.5-4B、Qwen3.5-9B、Qwen3.8-27B をベースに、ビジョンタワーを固定して 1 エポック学習します。設定は学習率 `5e-6`、最大 16,384 トークンで、回答の確率に対する交差エントロピーと Brier 損失を使います。4B を GPU 4 枚で学習するには：

```bash
torchrun --nproc-per-node 4 -m train.sft --config train/configs/onejev_4b_full.yaml
```

設定：[0.8B](../train/configs/onejev_08b_full.yaml) · [4B](../train/configs/onejev_4b_full.yaml) · [9B](../train/configs/onejev_9b_full.yaml) · [27B](../train/configs/onejev_27b_full.yaml)。`--nproc-per-node` を GPU の枚数に合わせてください。9B と 27B は FSDP でモデルとオプティマイザーの状態を分割します。独自のデータを使う場合は、設定の `train` と `media_root` を変更します。

4B の最終モデルは `train/runs/onejev_4b/final` に保存されます。次のコマンドで起動できます：

```bash
qev serve --model train/runs/onejev_4b/final --port 8000
```

## Friendly Links

- [LINUX DO](https://linux.do)
- [Jev](https://typesafe.ai)
- [Awesome JEV](https://github.com/OmniJev/awesome-jev-gallery)
- [Awesome JEV Website](https://omnijev.github.io/awesome-jev-gallery/)
- [PlayJev](https://github.com/OmniJev/PlayJev)

## ライセンス

[Apache 2.0](../LICENSE)

## スターとダウンロード数

<p>
  <a href="https://star-history.com/#OmniJev/OneJev&Date"><img src="https://api.star-history.com/svg?repos=OmniJev/OneJev&type=Date" alt="GitHub での OneJev のスター数" width="49%"></a>
  <a href="https://huggingface.co/OmniJev"><img src="downloads.svg" alt="Hugging Face での OneJev のダウンロード数" width="49%"></a>
</p>

## 引用

```bibtex
@misc{onejev2026,
  title        = {{OneJev}: A Multimodal System One Decision Model},
  author       = {{OmniJev Team}},
  year         = {2026},
  howpublished = {\url{https://github.com/OmniJev/OneJev}}
}
```
