<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="tr/banner-dark.svg">
    <img alt="OneJev, Çok Modlu System One Karar Modeli" src="tr/banner.svg" width="100%">
  </picture>
</p>

<p align="center"><a href="https://github.com/OmniJev/OneJev">English</a> · <a href="README_zh.md">简体中文</a> · <a href="README_zh-Hant.md">繁體中文</a> · <a href="README_ja.md">日本語</a> · <a href="README_ko.md">한국어</a> · <a href="README_th.md">ภาษาไทย</a> · <b>Türkçe</b></p>

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
  <a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white"></a>
  <a href="https://pytorch.org"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.6%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"></a>
  <a href="https://github.com/huggingface/transformers"><img alt="Transformers" src="https://img.shields.io/badge/Transformers-5%2B-FFD21E?style=flat-square&logo=huggingface&logoColor=black"></a>
</p>

OneJev, [Qwen](https://huggingface.co/Qwen) tabanlı çok modlu bir System One karar modelidir. Ekran görüntüleri, fotoğraflar, videolar ve metinler hakkındaki tipli sorulara tek bir ileri geçişte kalibre edilmiş olasılıklar döndürür. 0.8B, 4B, 9B ve 27B boyutlarında sunulur.

## 🔥 Haberler

- 🔥 **2026-10-17** OneJev 1.6'yı [Hugging Face](https://huggingface.co/collections/OmniJev/onejev16) ve [ModelScope](https://modelscope.ai/organization/OmniJev) üzerinde yayınladık. Denemenizi bekliyoruz.
- **2026-10-13** [OneJev 1.5](https://huggingface.co/collections/OmniJev/onejev15)'i yayınladık. Denemenizi bekliyoruz.
- **2026-09-28** Çevrimiçi demoyu yayınladık, 👉 [Buraya tıklayın](https://huggingface.co/spaces/BradNLP/OneJev).
- **2026-09-27** [OneJev](https://huggingface.co/collections/OmniJev/onejev)'i ve [OneJev veri setini](https://huggingface.co/datasets/OmniJev/OneJev-Data) yayınladık.

## Decision Index (herkese açık kısım)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="tr/progress-dark.svg">
    <img alt="Blok sütun grafiği, Decision Index 0.3 herkese açık kısım puanları, OneJev 1.6 ile OneJev 1.5 karşılaştırması: 0.8B 48.82 ve 42.71 (+6.1), 4B 62.01 ve 54.89 (+7.1), 9B 65.31 ve 59.03 (+6.3), 27B 69.49 ve 63.64 (+5.8)" src="tr/progress.svg" width="100%">
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="tr/leaderboard-dark.svg">
    <img alt="Model boyutuna göre dört Decision Index 0.3 herkese açık kısım sıralaması, her birinde ilk on, OneJev1.6 vurgulu: OneJev1.6-0.8B 2.5B parametreye kadar 51 model arasında 1., OneJev1.6-4B 3B ile 6B arası 38 model arasında 1., OneJev1.6-9B 6B ile 12B arası 26 model arasında 1., OneJev1.6-27B 25B ile 36B arası 59 model arasında 2." src="tr/leaderboard.svg" width="100%">
  </picture>
</p>

## Diğer karşılaştırmalar

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="tr/results-dark.svg">
    <img alt="Blok çubuk grafikleri: dört OneJev1.6 boyutu ile Jev 1.13, Jev-Omni 12B ve Qwen3.8-27B düşünme modu; OneJev test seti, DecisionBench zor, TypeSafe ve MMStar üzerinde" src="tr/results.svg" width="100%">
  </picture>
</p>

Doğruluk (%). OneJev test seti eğitimde kullanılmayan kendi değerlendirme setimizdir; diğerleri herkese açıktır.

## Hız

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="tr/speed-dark.svg">
    <img alt="Tek H200, tek 1280x720 ekran görüntüsü ile gecikme: 0.8B 1 soru için 31 ms, 10 soru için 51 ms, soru başına 5.1 ms; 4B 1 soru için 64 ms, 10 soru için 104 ms, soru başına 10.4 ms; 9B 1 soru için 81 ms, 10 soru için 131 ms, soru başına 13.1 ms; 27B 1 soru için 189 ms, 10 soru için 324 ms, soru başına 32.4 ms" src="tr/speed.svg" width="100%">
  </picture>
</p>

## Hızlı başlangıç

Bir arka uç seçip başlatın, ardından aşağıdaki örneği çalıştırın.

### Seçenek A: Flide

NVIDIA GPU'lar içindir. Metin, görsel ve video destekler. [Flide](https://github.com/OmniJev/Flide), karar modelleri için geliştirdiğimiz çıkarım motorudur.

```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install flide torchvision
flide serve --model OmniJev/OneJev1.6-0.8B --port 8000
```

### Seçenek B: llama.cpp

GGUF modelleri içindir. Metin ve görsel destekler; video için Flide kullanın. Önce [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md)'yi kurun (macOS'ta `brew install llama.cpp`).

```bash
pip install git+https://github.com/OmniJev/OneJev.git
qev serve --gguf mradermacher/OneJev1.6-0.8B-GGUF:Q8_0 --port 8000
```

### Seçenek C: Docker

Depoda, donanımınızı algılayıp tek bir arka uç başlatan bir Compose yığını bulunur. llama.cpp profilleri ilk açılışta `mradermacher/OneJev1.6-0.8B-GGUF:Q8_0` dosyasını indirir; başka bir boyut ya da nicemleme için `docker/.env` içinde `QEV_GGUF` değerini ayarlayın.

```bash
cp docker/.env.example docker/.env   # erişimi kısıtlı/özel depolar için HF_TOKEN ayarlayın
./docker/start.sh                    # otomatik algılama (Windows: .\docker\start.ps1)
./docker/start.sh logs               # günlükleri izle
./docker/start.sh down               # tüm profilleri durdur
```

Arka ucu `./docker/start.sh <torch|rocm|vulkan|gguf|gguf-rocm|cpu>` ile zorlayabilirsiniz (yeniden derlemek için `--build` ekleyin). Her profil API'yi ve Playground'u :8000 üzerinde sunar. Windows'ta yalnızca `torch` (NVIDIA) ve `cpu` kullanılabilir.

### İstek gönderme

Sunucu `http://localhost:8000` adresinden yanıt verir. Başka bir terminalde, kendi `screenshot.png` dosyanızla bu örneği çalıştırın:

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

`answers.done.noul` "evet" olasılığıdır, `answers.next.probabilities` her seçenek için bir olasılık verir ve `answers.progress.score` beklenen düzeydir.

API, TypeSafe System One ile uyumludur. Daha fazla örnek: [video](../examples/video.py), [curl](../examples/curl.sh), [resmi TypeSafe SDK](../examples/official_sdk.py).

## Eğitim

```bash
git clone https://github.com/OmniJev/OneJev.git
cd OneJev
pip install -e ".[train]"

# A veya B'yi seçin. Aşağıda A etkin.
# A. Demo: görselli 100 örnek (23 MB)
hf download OmniJev/OneJev-Data sample/sample-100.parquet --repo-type dataset --local-dir data/onejev
python -m train.unpack data/onejev --sample

# B. Tam veri seti: 94,707 örnek (17.7 GB). A yerine bu iki satırın yorumunu kaldırın.
# hf download OmniJev/OneJev-Data --repo-type dataset --include "data/*.parquet" --local-dir data/onejev
# python -m train.unpack data/onejev
```

İki seçenek de `data/onejev/train.jsonl` dosyasını oluşturur ve görselleri ile video karelerini `data/onejev/media/` altına çıkarır.

[OneJev-Data](https://huggingface.co/datasets/OmniJev/OneJev-Data), özgün 99,193 eğitim sorusunun 94,707'sini yayınlar; kalan kaynaklar yeniden dağıtıma izin vermez.

### İnce ayar

Dört model; Qwen3.5-0.8B, Qwen3.5-4B, Qwen3.5-9B ve Qwen3.8-27B'nin görsel kulesi dondurularak bir epoch boyunca ince ayarlanmasıyla elde edilir. Yapılandırmalar `5e-6` öğrenme oranı, 16,384 token sınırı ve yanıt olasılıkları üzerinde çapraz entropi ile Brier kaybı kullanır. 4B modeli dört GPU'da eğitmek için:

```bash
torchrun --nproc-per-node 4 -m train.sft --config train/configs/onejev_4b_full.yaml
```

Yapılandırmalar: [0.8B](../train/configs/onejev_08b_full.yaml) · [4B](../train/configs/onejev_4b_full.yaml) · [9B](../train/configs/onejev_9b_full.yaml) · [27B](../train/configs/onejev_27b_full.yaml). `--nproc-per-node` değerini GPU sayınıza göre ayarlayın; 9B ve 27B yapılandırmaları model ve optimizer durumunu bölmek için FSDP'yi etkinleştirir. Kendi verileriniz için yapılandırmadaki `train` ve `media_root` alanlarını değiştirin.

Son 4B checkpoint'i `train/runs/onejev_4b/final` konumuna kaydedilir. Şu komutla başlatın:

```bash
flide serve --model train/runs/onejev_4b/final --format onejev --port 8000
```

## Friendly Links

- [LINUX DO](https://linux.do)
- [Jev](https://typesafe.ai)
- [Awesome JEV](https://github.com/OmniJev/awesome-jev-gallery)
- [Awesome JEV Website](https://omnijev.github.io/awesome-jev-gallery/)
- [PlayJev](https://github.com/OmniJev/PlayJev)

## Lisans

[Apache 2.0](../LICENSE)

## Yıldızlar ve indirmeler

<p>
  <a href="https://star-history.com/#OmniJev/OneJev&Date"><img src="https://api.star-history.com/svg?repos=OmniJev/OneJev&type=Date" alt="OneJev'in GitHub yıldızları" width="49%"></a>
  <a href="https://huggingface.co/OmniJev"><img src="downloads.svg" alt="OneJev'in Hugging Face indirmeleri" width="49%"></a>
</p>

## Atıf

```bibtex
@misc{onejev2026,
  title        = {{OneJev}: A Multimodal System One Decision Model},
  author       = {{OmniJev Team}},
  year         = {2026},
  howpublished = {\url{https://github.com/OmniJev/OneJev}}
}
```
