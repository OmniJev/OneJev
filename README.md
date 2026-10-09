<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/banner-dark.svg">
    <img alt="OneJev, a Multimodal System One Decision Model" src="assets/banner.svg" width="100%">
  </picture>
</p>

<p align="center"><b>English</b> · <a href="assets/README_zh.md">简体中文</a> · <a href="assets/README_ja.md">日本語</a></p>

<p align="center">
  <a href="https://huggingface.co/collections/OmniJev/onejev"><img alt="Hugging Face" src="https://img.shields.io/badge/Hugging_Face-OneJev-FFD21E?style=flat-square&logo=huggingface&logoColor=black"></a>
  <a href="https://huggingface.co/spaces/BradNLP/OneJev"><img alt="Demo" src="https://img.shields.io/badge/Demo-Try_it-D45BB6?style=flat-square&logo=gradio&logoColor=white"></a>
  <a href="https://github.com/OmniJev/OneJev"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-OmniJev%2FOneJev-181717?style=flat-square&logo=github&logoColor=white"></a>
  <a href="https://omnijev.github.io/OneJev/"><img alt="Website" src="https://img.shields.io/badge/Website-OneJev-0A84FF?style=flat-square&logo=googlechrome&logoColor=white"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-D22128?style=flat-square&logo=apache&logoColor=white"></a>
</p>
<p align="center">
  <a href="https://huggingface.co/datasets/OmniJev/OneJev-Data"><img alt="Data" src="https://img.shields.io/badge/Data-OneJev--Data-FF9D00?style=flat-square&logo=huggingface&logoColor=white"></a>
  <a href="examples/request.json"><img alt="API" src="https://img.shields.io/badge/API-System_One-009688?style=flat-square&logo=fastapi&logoColor=white"></a>
  <a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white"></a>
  <a href="https://pytorch.org"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.6%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"></a>
  <a href="https://github.com/huggingface/transformers"><img alt="Transformers" src="https://img.shields.io/badge/Transformers-5%2B-FFD21E?style=flat-square&logo=huggingface&logoColor=black"></a>
</p>

OneJev is a multimodal System One decision model. It returns calibrated probabilities for typed questions about
screenshots, photos, videos and text in a single forward pass. Available in 0.8B, 4B, 9B and 27B.

## Results

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/results-dark.svg">
    <img alt="Block bar charts: the four OneJev sizes against Jev 1.13, Jev-Omni 12B and Qwen3.8-27B thinking on the OneJev test set, DecisionBench hard, TypeSafe and MMStar" src="assets/results.svg" width="100%">
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/table-dark.svg">
    <img alt="Results table, best score in each row in magenta, second best in light pink" src="assets/table.svg" width="100%">
  </picture>
</p>

Accuracy (%). The OneJev test set is held out from OneJev training. Jev 1.13 uses published text-only scores;
Jev-Omni and Qwen3.8-27B thinking were evaluated by us.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/speed-dark.svg">
    <img alt="Latency on one H200 for one 1280x720 screenshot: OneJev-0.8B 31 ms for 1 question, 51 ms for 10, 5.1 ms per question; OneJev-4B 64 ms for 1 question, 104 ms for 10, 10.4 ms per question; OneJev-9B 81 ms for 1 question, 131 ms for 10, 13.1 ms per question; OneJev-27B 189 ms for 1 question, 324 ms for 10, 32.4 ms per question" src="assets/speed.svg" width="100%">
  </picture>
</p>

## Quick start

Choose one backend, then run the Python example below.

### Option A: PyTorch

For NVIDIA GPUs. Supports text, images and video.

```bash
pip install "qev[torch] @ git+https://github.com/OmniJev/OneJev.git"
qev serve --model OmniJev/OneJev-4B --port 8000
```

### Option B: llama.cpp

For GGUF models. Supports text and images; use PyTorch for video. Install
[llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md) first (`brew install llama.cpp` on macOS).

```bash
pip install git+https://github.com/OmniJev/OneJev.git
qev serve --gguf mradermacher/OneJev-4B-GGUF:Q8_0 --port 8000
```

### Option C: Docker

The repo ships a Compose stack that detects your hardware and starts one backend.
The llama.cpp profiles download `mradermacher/OneJev-4B-GGUF:Q8_0` on first start; set `QEV_GGUF` in `.env` for
another size or quant.

```bash
cp .env.example .env                 # set HF_TOKEN for gated/private repos
./scripts/start.sh                   # auto-detect (Windows: .\scripts\start.ps1)
./scripts/start.sh logs              # follow logs
./scripts/start.sh down              # stop every profile
```

Force a backend with `./scripts/start.sh <torch|rocm|vulkan|gguf|gguf-rocm|cpu>`
(add `--build` to rebuild). Every profile serves the API and the playground on :8000.
On Windows only `torch` (NVIDIA) and `cpu` are available.

### Send a request

Both backends serve the same API at `http://localhost:8000`. In another terminal, run this example with your own
`screenshot.png`:

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
r.answers["done"].noul                  # probability of yes
r.answers["next"].probabilities         # one probability per option
r.answers["progress"].score             # expected level
```

The API is compatible with TypeSafe System One. More examples: [video](examples/video.py),
[curl](examples/curl.sh), [official TypeSafe SDK](examples/official_sdk.py).

## Training

```bash
git clone https://github.com/OmniJev/OneJev.git
cd OneJev
pip install -e ".[train]"

# Choose A or B. A is enabled below.
# A. Demo: 100 examples with images (23 MB)
hf download OmniJev/OneJev-Data sample/sample-100.parquet --repo-type dataset --local-dir data/onejev
python -m train.unpack data/onejev --sample

# B. Full dataset: 94,707 examples (17.7 GB). Uncomment these two lines instead of A.
# hf download OmniJev/OneJev-Data --repo-type dataset --include "data/*.parquet" --local-dir data/onejev
# python -m train.unpack data/onejev
```

Either option creates `data/onejev/train.jsonl` and extracts images and video frames into `data/onejev/media/`.

[OneJev-Data](https://huggingface.co/datasets/OmniJev/OneJev-Data) releases 94,707 of the original 99,193
training questions; the remaining sources do not permit redistribution.

### Fine-tune

The four models fine-tune Qwen3.5-0.8B, Qwen3.5-4B, Qwen3.5-9B and Qwen3.8-27B for one epoch with the vision tower
frozen. The configs use a learning rate of `5e-6`, a 16,384-token limit, and cross-entropy plus Brier loss over answer
probabilities. To train the 4B model on four GPUs:

```bash
torchrun --nproc-per-node 4 -m train.sft --config train/configs/onejev_4b_full.yaml
```

Configs: [0.8B](train/configs/onejev_08b_full.yaml) · [4B](train/configs/onejev_4b_full.yaml) ·
[9B](train/configs/onejev_9b_full.yaml) · [27B](train/configs/onejev_27b_full.yaml).
Set `--nproc-per-node` to your GPU count; the 9B and 27B configs enable FSDP to shard model and optimizer state.
For your own data, change `train` and `media_root` in the config.

The final 4B checkpoint is saved to `train/runs/onejev_4b/final`. Start it with:

```bash
qev serve --model train/runs/onejev_4b/final --port 8000
```

## Citation

```bibtex
@misc{onejev2026,
  title        = {{OneJev}: A Multimodal System One Decision Model},
  author       = {{OmniJev Team}},
  year         = {2026},
  howpublished = {\url{https://github.com/OmniJev/OneJev}}
}
```

## Friendly Links

- [LINUX DO](https://linux.do)
- [Jev](https://typesafe.ai)
- [Awesome JEV](https://github.com/OmniJev/awesome-jev-gallery)
- [Awesome JEV Website](https://omnijev.github.io/awesome-jev-gallery/)
- [PlayJev](https://github.com/OmniJev/PlayJev)

## License

[Apache 2.0](LICENSE)
