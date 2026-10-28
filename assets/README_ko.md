<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="ko/banner-dark.svg">
    <img alt="OneJev, 멀티모달 System One 의사결정 모델" src="ko/banner.svg" width="100%">
  </picture>
</p>

<p align="center"><a href="https://github.com/OmniJev/OneJev">English</a> · <a href="README_zh.md">简体中文</a> · <a href="README_zh-Hant.md">繁體中文</a> · <a href="README_ja.md">日本語</a> · <b>한국어</b> · <a href="README_th.md">ภาษาไทย</a> · <a href="README_tr.md">Türkçe</a></p>

<p align="center">
  <a href="https://huggingface.co/collections/OmniJev/onejev16"><img alt="Hugging Face" src="https://img.shields.io/badge/Hugging_Face-OneJev1.6-FFD21E?style=flat-square&logo=huggingface&logoColor=black"></a>
  <a href="https://modelscope.ai/organization/OmniJev"><img alt="ModelScope" src="https://img.shields.io/badge/ModelScope-OneJev1.6-624AFF?style=flat-square&logo=modelscope&logoColor=white"></a>
  <a href="https://huggingface.co/spaces/BradNLP/OneJev"><img alt="Demo" src="https://img.shields.io/badge/Demo-Try_it-F83800?style=flat-square&logo=gradio&logoColor=white"></a>
  <a href="https://github.com/OmniJev/OneJev"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-OmniJev%2FOneJev-181717?style=flat-square&logo=github&logoColor=white"></a>
  <a href="https://omnijev.github.io/OneJev/"><img alt="Website" src="https://img.shields.io/badge/Website-OneJev-0A84FF?style=flat-square&logo=googlechrome&logoColor=white"></a>
</p>
<p align="center">
  <a href="https://huggingface.co/datasets/OmniJev/OneJev-Data"><img alt="Data" src="https://img.shields.io/badge/Data-OneJev--Data-FF9D00?style=flat-square&logo=huggingface&logoColor=white"></a>
  <a href="../LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-D22128?style=flat-square&logo=apache&logoColor=white"></a>
  <a href="https://github.com/OmniJev/Flide"><img alt="Flide" src="https://img.shields.io/badge/Engine-Flide-00A890?style=flat-square&logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjIgMS41IDE3IDE3IiBzaGFwZS1yZW5kZXJpbmc9ImNyaXNwRWRnZXMiPjxwYXRoIGZpbGw9IiMzQTg2RkYiIGQ9Ik0yIDhoN3YxaC03ek0yIDloN3YxaC03ek0yIDEwaDd2MWgtN3pNMiAxMWg3djFoLTd6Ii8+PHBhdGggZmlsbD0iI0VGNDc2RiIgZD0iTTEyIDNoNHYxaC00ek0xNyAzaDJ2MWgtMnpNMTEgNGg1djFoLTV6TTE3IDRoMnYxaC0yek0xMCA1aDJ2MWgtMnpNMTAgNmgydjFoLTJ6TTkgN2gydjFoLTJ6TTkgOGgydjFoLTJ6Ii8+PHBhdGggZmlsbD0iI0YyOEUyQiIgZD0iTTkgOWg3djFoLTd6TTE3IDloMnYxaC0yek05IDEwaDd2MWgtN3pNMTcgMTBoMnYxaC0yeiIvPjxwYXRoIGZpbGw9IiMwMEE4OTAiIGQ9Ik05IDExaDJ2MWgtMnpNOSAxMmgydjFoLTJ6TTEwIDEzaDJ2MWgtMnpNMTAgMTRoMnYxaC0yek0xMSAxNWg1djFoLTV6TTE3IDE1aDJ2MWgtMnpNMTIgMTZoNHYxaC00ek0xNyAxNmgydjFoLTJ6Ii8+PC9zdmc+Cg=="></a>
  <a href="https://pytorch.org"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.6%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"></a>
  <a href="https://github.com/huggingface/transformers"><img alt="Transformers" src="https://img.shields.io/badge/Transformers-5%2B-FFD21E?style=flat-square&logo=huggingface&logoColor=black"></a>
</p>

OneJev는 [Qwen](https://huggingface.co/Qwen) 기반의 멀티모달 System One 의사결정 모델입니다. 스크린샷, 사진, 동영상, 텍스트에 대한 타입이 지정된 질문에 순전파 한 번으로 보정된 확률을 돌려줍니다. 0.8B, 4B, 9B, 27B 네 가지 크기로 제공됩니다.

## 🔥 소식

- 🔥 **2026-10-17** [Hugging Face](https://huggingface.co/collections/OmniJev/onejev16)와 [ModelScope](https://modelscope.ai/organization/OmniJev)에 OneJev 1.6을 공개했습니다. 사용해 보세요.
- **2026-10-13** [OneJev 1.5](https://huggingface.co/collections/OmniJev/onejev15)를 공개했습니다. 사용해 보세요.
- **2026-09-28** 온라인 데모를 공개했습니다. 👉 [바로 가기](https://huggingface.co/spaces/BradNLP/OneJev)
- **2026-09-27** [OneJev](https://huggingface.co/collections/OmniJev/onejev)와 [OneJev 데이터셋](https://huggingface.co/datasets/OmniJev/OneJev-Data)을 공개했습니다.

## Decision Index (공개 부분)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="ko/progress-dark.svg">
    <img alt="블록 세로 막대 그래프: Decision Index 0.3 공개 부분 점수, OneJev 1.6과 OneJev 1.5 비교. 0.8B 48.82 대 42.71 (+6.1), 4B 62.01 대 54.89 (+7.1), 9B 65.31 대 59.03 (+6.3), 27B 69.49 대 63.64 (+5.8)" src="ko/progress.svg" width="100%">
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="ko/leaderboard-dark.svg">
    <img alt="모델 크기별 Decision Index 0.3 공개 부분 리더보드 네 개, 각 상위 10개, OneJev1.6 강조: OneJev1.6-0.8B는 파라미터 2.5B 이하 51개 모델 중 1위, OneJev1.6-4B는 3B~6B 38개 모델 중 1위, OneJev1.6-9B는 6B~12B 26개 모델 중 1위, OneJev1.6-27B는 25B~36B 59개 모델 중 2위" src="ko/leaderboard.svg" width="100%">
  </picture>
</p>

## 기타 벤치마크

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="ko/results-dark.svg">
    <img alt="블록 막대 그래프: 네 가지 크기의 OneJev1.6과 Jev 1.13, Jev-Omni 12B, Qwen3.8-27B 사고 모드를 OneJev 테스트셋, DecisionBench 어려움, TypeSafe, MMStar에서 비교" src="ko/results.svg" width="100%">
  </picture>
</p>

정확도 (%). OneJev 테스트셋은 학습에 쓰지 않은 자체 평가셋이며, 나머지는 공개 벤치마크입니다.

## 속도

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="ko/speed-dark.svg">
    <img alt="H200 1장, 1280x720 스크린샷 1장 기준 지연 시간: 0.8B 질문 1개 31 ms, 질문 10개 51 ms, 질문당 5.1 ms; 4B 질문 1개 64 ms, 질문 10개 104 ms, 질문당 10.4 ms; 9B 질문 1개 81 ms, 질문 10개 131 ms, 질문당 13.1 ms; 27B 질문 1개 189 ms, 질문 10개 324 ms, 질문당 32.4 ms" src="ko/speed.svg" width="100%">
  </picture>
</p>

## 빠른 시작

백엔드를 하나 골라 실행한 다음, 아래 예제를 실행하세요.

### 옵션 A: Flide

NVIDIA GPU용입니다. 텍스트, 이미지, 동영상을 지원합니다. [Flide](https://github.com/OmniJev/Flide)는 의사결정 모델을 위한 저희 추론 엔진입니다.

```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install flide torchvision
flide serve --model OmniJev/OneJev1.6-0.8B --port 8000
```

### 옵션 B: llama.cpp

GGUF 모델용입니다. 텍스트와 이미지를 지원하며, 동영상은 Flide를 사용하세요. 먼저 [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md)를 설치합니다 (macOS에서는 `brew install llama.cpp`).

```bash
pip install git+https://github.com/OmniJev/OneJev.git
qev serve --gguf mradermacher/OneJev1.6-0.8B-GGUF:Q8_0 --port 8000
```

### 옵션 C: Docker

저장소에 포함된 Compose 스택이 하드웨어를 감지해 백엔드 하나를 시작합니다. llama.cpp 프로필은 처음 시작할 때 `mradermacher/OneJev1.6-0.8B-GGUF:Q8_0`을 내려받습니다. 다른 크기나 양자화를 쓰려면 `docker/.env`에서 `QEV_GGUF`를 설정하세요.

```bash
cp docker/.env.example docker/.env   # gated/비공개 저장소는 HF_TOKEN 설정
./docker/start.sh                    # 자동 감지 (Windows: .\docker\start.ps1)
./docker/start.sh logs               # 로그 보기
./docker/start.sh down               # 모든 프로필 중지
```

`./docker/start.sh <torch|rocm|vulkan|gguf|gguf-rocm|cpu>`로 백엔드를 직접 지정할 수 있습니다 (다시 빌드하려면 `--build` 추가). 모든 프로필이 :8000에서 API와 Playground를 제공합니다. Windows에서는 `torch` (NVIDIA)와 `cpu`만 사용할 수 있습니다.

### 요청 보내기

서버 주소는 `http://localhost:8000`입니다. 다른 터미널에서 직접 준비한 `screenshot.png`로 이 예제를 실행하세요.

```bash
IMG=$(base64 -w0 screenshot.png)
curl http://localhost:8000/v1/systemone -H "Content-Type: application/json" -d @- <<EOF
{
  "state": {"task": "Pay the open invoice from ACME", "screen": "<image:1>"},
  "media": [{"type": "image", "data": "data:image/png;base64,$IMG"}],
  "questions": {
    "done": {"type": "noul", "instructions": "The invoice has been paid"},
    "next": {"type": "choice", "instructions": "What should the agent do next?",
             "criteria": {"click": "click an element", "type": "type text", "scroll": "scroll", "stop": "stop"}},
    "progress": {"type": "score", "instructions": "How far along is the task?",
                 "criteria": ["not started", "halfway", "almost done", "done"]}
  }
}
EOF
```

`answers.done.noul`은 "예"일 확률, `answers.next.probabilities`는 선택지별 확률, `answers.progress.score`는 기대 단계입니다.

API는 TypeSafe System One과 호환됩니다. 더 많은 예제: [동영상](../examples/video.py), [curl](../examples/curl.sh), [공식 TypeSafe SDK](../examples/official_sdk.py).

## 학습

```bash
git clone https://github.com/OmniJev/OneJev.git
cd OneJev
pip install -e ".[train]"

# A와 B 중 하나를 고르세요. 아래에서는 A가 켜져 있습니다.
# A. 데모: 이미지가 포함된 예제 100개 (23 MB)
hf download OmniJev/OneJev-Data sample/sample-100.parquet --repo-type dataset --local-dir data/onejev
python -m train.unpack data/onejev --sample

# B. 전체 데이터셋: 예제 94,707개 (17.7 GB). B를 쓰려면 A 대신 아래 두 줄의 주석을 해제하세요.
# hf download OmniJev/OneJev-Data --repo-type dataset --include "data/*.parquet" --local-dir data/onejev
# python -m train.unpack data/onejev
```

어느 쪽이든 `data/onejev/train.jsonl`이 만들어지고, 이미지와 동영상 프레임은 `data/onejev/media/`에 풀립니다.

[OneJev-Data](https://huggingface.co/datasets/OmniJev/OneJev-Data)는 원래 학습 문항 99,193개 중 94,707개를 공개합니다. 나머지 출처는 재배포를 허용하지 않습니다.

### 파인튜닝

네 모델은 각각 Qwen3.5-0.8B, Qwen3.5-4B, Qwen3.5-9B, Qwen3.8-27B를 비전 타워를 고정한 채 1 에폭 파인튜닝한 것입니다. 설정은 학습률 `5e-6`, 최대 16,384 토큰이며, 답변 확률에 대해 교차 엔트로피와 Brier 손실을 함께 씁니다. GPU 4장으로 4B 모델을 학습하려면:

```bash
torchrun --nproc-per-node 4 -m train.sft --config train/configs/onejev_4b_full.yaml
```

설정: [0.8B](../train/configs/onejev_08b_full.yaml) · [4B](../train/configs/onejev_4b_full.yaml) · [9B](../train/configs/onejev_9b_full.yaml) · [27B](../train/configs/onejev_27b_full.yaml). `--nproc-per-node`를 GPU 수에 맞추세요. 9B와 27B 설정은 FSDP로 모델과 옵티마이저 상태를 분산합니다. 직접 만든 데이터를 쓰려면 설정의 `train`과 `media_root`를 바꾸세요.

4B 최종 체크포인트는 `train/runs/onejev_4b/final`에 저장됩니다. 다음 명령으로 실행합니다.

```bash
flide serve --model train/runs/onejev_4b/final --format onejev --port 8000
```

## Friendly Links

- [LINUX DO](https://linux.do)
- [Jev](https://typesafe.ai)
- [Awesome JEV](https://github.com/OmniJev/awesome-jev-gallery)
- [Awesome JEV Website](https://omnijev.github.io/awesome-jev-gallery/)
- [PlayJev](https://github.com/OmniJev/PlayJev)

## 라이선스

[Apache 2.0](../LICENSE)

## 스타와 다운로드

<p>
  <a href="https://star-history.com/#OmniJev/OneJev&Date"><img src="https://api.star-history.com/svg?repos=OmniJev/OneJev&type=Date" alt="GitHub에서 OneJev가 받은 스타" width="49%"></a>
  <a href="https://huggingface.co/OmniJev"><img src="downloads.svg" alt="Hugging Face에서 OneJev 다운로드 수" width="49%"></a>
</p>

## 인용

```bibtex
@misc{onejev2026,
  title        = {{OneJev}: A Multimodal System One Decision Model},
  author       = {{OmniJev Team}},
  year         = {2026},
  howpublished = {\url{https://github.com/OmniJev/OneJev}}
}
```
