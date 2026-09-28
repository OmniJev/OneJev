<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="ja/banner-dark.svg">
    <img alt="OneJev、マルチモーダル System One 意思決定モデル" src="ja/banner.svg" width="100%">
  </picture>
</p>

<p align="center"><a href="https://github.com/OmniJev/OneJev">English</a> · <a href="README_zh.md">简体中文</a> · <b>日本語</b></p>

<p align="center">
  <a href="https://huggingface.co/collections/OmniJev/onejev"><img alt="Hugging Face" src="https://img.shields.io/badge/Hugging_Face-OneJev-FFD21E?style=flat-square&logo=huggingface&logoColor=black"></a>
  <a href="https://huggingface.co/spaces/BradNLP/OneJev"><img alt="Demo" src="https://img.shields.io/badge/Demo-Try_it-D45BB6?style=flat-square&logo=gradio&logoColor=white"></a>
  <a href="https://github.com/OmniJev/OneJev"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-OmniJev%2FOneJev-181717?style=flat-square&logo=github&logoColor=white"></a>
  <a href="https://omnijev.github.io/OneJev/"><img alt="Website" src="https://img.shields.io/badge/Website-OneJev-0A84FF?style=flat-square&logo=googlechrome&logoColor=white"></a>
  <a href="../LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-D22128?style=flat-square&logo=apache&logoColor=white"></a>
</p>
<p align="center">
  <a href="https://huggingface.co/datasets/OmniJev/OneJev-Data"><img alt="Data" src="https://img.shields.io/badge/Data-OneJev--Data-FF9D00?style=flat-square&logo=huggingface&logoColor=white"></a>
  <a href="#api"><img alt="API" src="https://img.shields.io/badge/API-System_One-009688?style=flat-square&logo=fastapi&logoColor=white"></a>
  <a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white"></a>
  <a href="https://pytorch.org"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.6%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"></a>
  <a href="https://github.com/huggingface/transformers"><img alt="Transformers" src="https://img.shields.io/badge/Transformers-5%2B-FFD21E?style=flat-square&logo=huggingface&logoColor=black"></a>
</p>

OneJev を提案します。マルチモーダルな System One 意思決定モデルです。スクリーンショット、写真、動画、テキストのいずれかと型付きの質問をいくつか渡すと、1 回の順伝播で各質問のすべての選択肢に較正済みの確率を返します。TypeSafe の System One API に対応し、画像と動画に拡張しています。サイズは 4 種類で、どれも実際のエージェント実行、動画、画像から集めた同じ 99,193 問で学習しています。

## 結果

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="ja/results-dark.svg">
    <img alt="ブロック棒グラフ：4 サイズの OneJev と Jev 1.13、Jev-Omni 12B、Qwen3.8-27B 思考モードを OneJev テストセット、DecisionBench 難、TypeSafe、MMStar で比較" src="ja/results.svg" width="100%">
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="ja/table-dark.svg">
    <img alt="結果表。各行の最高値はマゼンタ、2 位は薄いピンク" src="ja/table.svg" width="100%">
  </picture>
</p>

スコアは正解率（%）です。OneJev テストセットは学習データと同じ種類の問題で、どのモデルも学習中に一度も見ていません。Jev 1.13 のスコアは公開値で、テキストのみを読みます。Jev-Omni と Qwen3.8-27B 思考モードは同じ問題で私たちが評価しました。思考モデルは毎回数千語の推論を書いてから答え、OneJev はそのまま答えます。

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="ja/speed-dark.svg">
    <img alt="H200 1 枚、1280x720 のスクリーンショット 1 枚での遅延：OneJev-0.8B 質問 1 つ 31 ms、質問 10 個 51 ms、質問あたり 5.1 ms；OneJev-4B 質問 1 つ 64 ms、質問 10 個 104 ms、質問あたり 10.4 ms；OneJev-9B 質問 1 つ 81 ms、質問 10 個 131 ms、質問あたり 13.1 ms；OneJev-27B 質問 1 つ 189 ms、質問 10 個 324 ms、質問あたり 32.4 ms" src="ja/speed.svg" width="100%">
  </picture>
</p>

## クイックスタート

```bash
pip install git+https://github.com/OmniJev/OneJev.git
qev serve --model OmniJev/OneJev-4B --port 8000
```

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

テキストのみのリクエストは通常の System One リクエストです。公式の `typesafe-sdk` も`TYPESAFE_BASE_URL=http://localhost:8000` を設定すればそのまま使えます。そのほかの例は [examples/](../examples)にあります（スクリーンショット、動画、curl、公式 SDK）。[benchmarks/latency.py](../benchmarks/latency.py) で速度グラフの数値を手元の GPU で測れます。

## 仕組み

状態は画像や動画フレームごと 1 回だけ読み込み、各質問はその読み込みから分岐する短い枝として処理します。答えは最後の位置での各選択肢の文字の確率です。同じ画面について 10 問聞いても、コストは 1 問とほとんど変わりません。

## API

`POST /v1/systemone` は TypeSafe のリクエストに加えて、任意の `media` リストを受け取ります。state の中では各項目を`<image:N>` または `<video:N>` で参照します。

```text
image   {"type": "image", "url": ...}   {"type": "image", "data": <base64>}   {"type": "image", "path": ...}
video   {"type": "video", "frames": [<image>, ...], "fps": 2.0}
```

## 学習

Qwen3.5-0.8B、Qwen3.5-4B、Qwen3.5-9B、Qwen3.8-27B を 1 エポック、全パラメータでファインチューニングしました。ビジョンタワーは固定しています。

データは [Hugging Face](https://huggingface.co/datasets/OmniJev/OneJev-Data) にあります。

```bash
git clone https://github.com/OmniJev/OneJev.git && cd OneJev
pip install -e ".[train]"
hf download OmniJev/OneJev-Data --repo-type dataset --local-dir data/onejev
python -m train.unpack data/onejev
torchrun --nproc-per-node 4 -m train.sft --config train/configs/onejev_4b_full.yaml
```

## 引用

```bibtex
@misc{onejev2026,
  title        = {{OneJev}: A Multimodal System One Decision Model},
  author       = {{OmniJev Team}},
  year         = {2026},
  howpublished = {\url{https://github.com/OmniJev/OneJev}}
}
```

## ライセンス

[Apache 2.0](../LICENSE)
