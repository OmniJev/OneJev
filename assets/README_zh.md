<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="zh/banner-dark.svg">
    <img alt="OneJev，多模态 System One 决策模型" src="zh/banner.svg" width="100%">
  </picture>
</p>

<p align="center"><a href="../README.md">English</a> · <b>简体中文</b> · <a href="README_ja.md">日本語</a></p>

<p align="center">
  <a href="https://huggingface.co/OmniJev/OneJev"><img alt="Hugging Face" src="https://img.shields.io/badge/Hugging_Face-OneJev-FFD21E?style=flat-square&logo=huggingface&logoColor=black"></a>
  <a href="https://github.com/OmniJev/OneJev"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-OmniJev%2FOneJev-181717?style=flat-square&logo=github&logoColor=white"></a>
  <a href="https://omnijev.github.io/OneJev/"><img alt="Website" src="https://img.shields.io/badge/Website-OneJev-0A84FF?style=flat-square&logo=googlechrome&logoColor=white"></a>
  <a href="#api"><img alt="API" src="https://img.shields.io/badge/API-System_One-009688?style=flat-square&logo=fastapi&logoColor=white"></a>
  <a href="../LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-D22128?style=flat-square&logo=apache&logoColor=white"></a>
</p>
<p align="center">
  <a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white"></a>
  <a href="https://pytorch.org"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.6%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"></a>
  <a href="https://github.com/huggingface/transformers"><img alt="Transformers" src="https://img.shields.io/badge/Transformers-5%2B-FFD21E?style=flat-square&logo=huggingface&logoColor=black"></a>
</p>

我们提出 OneJev，一个多模态 System One 决策模型。给它一张截图、一张照片、一段视频或一段文本，再加上几个带类型的问题，它在一次前向传播里为每个问题的每个选项给出校准后的概率。它兼容 TypeSafe 的 System One API，并扩展了图片和视频，提供四种尺寸，全部在同一批 99,193 个问题上训练，这些问题来自真实的智能体运行记录、视频和图片。

## 结果

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="zh/results-dark.svg">
    <img alt="像素柱状图：四个尺寸的 OneJev 与 Jev 1.13、Jev-Omni 12B、Qwen3.8-27B 思考模式在 OneJev 测试集、DecisionBench 困难、TypeSafe 和 MMStar 上的对比" src="zh/results.svg" width="100%">
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="zh/table-dark.svg">
    <img alt="结果表，每行最高分为洋红色，第二名为浅粉色" src="zh/table.svg" width="100%">
  </picture>
</p>

分数是百分制准确率。OneJev 测试集的问题和训练数据同类，但所有模型在训练中都没见过。Jev 1.13 的分数取自公开结果，它只读文本。Jev-Omni 和 Qwen3.8-27B 思考模式由我们在同一批问题上测试；思考模型每次作答前要写几千字的推理，OneJev 直接给出答案。

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="zh/speed-dark.svg">
    <img alt="一块 H200、一张 1280x720 截图的延迟：OneJev-0.8B 1 个问题 31 ms，10 个问题 51 ms，每个问题 5.1 ms；OneJev-4B 1 个问题 64 ms，10 个问题 104 ms，每个问题 10.4 ms；OneJev-9B 1 个问题 81 ms，10 个问题 131 ms，每个问题 13.1 ms；OneJev-27B 1 个问题 189 ms，10 个问题 324 ms，每个问题 32.4 ms" src="zh/speed.svg" width="100%">
  </picture>
</p>

## 快速开始

```bash
pip install git+https://github.com/OmniJev/OneJev.git
qev serve --model OmniJev/OneJev/4B --port 8000
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
r.answers["done"].noul                  # “是”的概率
r.answers["next"].probabilities         # 每个选项一个概率
r.answers["progress"].score             # 期望等级
```

纯文本请求就是普通的 System One 请求；官方 `typesafe-sdk` 设置 `TYPESAFE_BASE_URL=http://localhost:8000` 即可使用。更多示例见 [examples/](../examples)：截图、视频、curl 和官方 SDK。[benchmarks/latency.py](../benchmarks/latency.py)可以在你自己的 GPU 上测出速度图里的数字。

## 工作原理

状态连同其中的图片和视频帧只读一次，每个问题都是从这次读取分出的一小段分支。答案是最后一个位置上每个选项字母的概率。对同一个屏幕问十个问题，花的时间和问一个差不多。

## API

`POST /v1/systemone` 接收 TypeSafe 的请求，外加一个可选的 `media` 列表；state 里用 `<image:N>` 或 `<video:N>`指向其中每一项。

```text
image   {"type": "image", "url": ...}   {"type": "image", "data": <base64>}   {"type": "image", "path": ...}
video   {"type": "video", "frames": [<image>, ...], "fps": 2.0}
```

## 训练

对 Qwen3.5-0.8B、Qwen3.5-4B、Qwen3.5-9B 和 Qwen3.8-27B 做一轮全参数微调，视觉塔冻结。

## 引用

```bibtex
@misc{onejev2026,
  title        = {{OneJev}: A Multimodal System One Decision Model},
  author       = {{OmniJev Team}},
  year         = {2026},
  howpublished = {\url{https://github.com/OmniJev/OneJev}}
}
```

## 许可证

[Apache 2.0](../LICENSE)
