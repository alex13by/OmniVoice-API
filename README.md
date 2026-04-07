# OmniVoice 🌍

<p align="center">
  <img width="200" height="200" alt="OmniVoice" src="https://zhu-han.github.io/omnivoice/pics/omnivoice.jpg" />
</p>

<p align="center">
  <a href="https://huggingface.co/k2-fsa/OmniVoice"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E" alt="Hugging Face Model"></a>
  &nbsp;
  <a href="https://huggingface.co/spaces/k2-fsa/OmniVoice"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue" alt="Hugging Face Space"></a>
  &nbsp;
  <a href="https://arxiv.org/abs/2604.00688"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a>
  &nbsp;
  <a href="https://zhu-han.github.io/omnivoice"><img src="https://img.shields.io/badge/GitHub.io-Demo_Page-blue?logo=GitHub&style=flat-square"></a>
</p>

OmniVoice is a state-of-the-art massive multilingual zero-shot text-to-speech (TTS) model supporting over 600 languages. Built on a novel diffusion language model-style architecture, it generates high-quality speech with superior inference speed, supporting voice cloning and voice design.

**Contents**: [Key Features](#key-features) | [Installation](#installation) | [Quick Start](#quick-start) | [Python API](#python-api) | [Command-Line Tools](#command-line-tools) | [Training & Evaluation](#training--evaluation) | [Discussion](#discussion--communication) | [Citation](#citation)

## Key Features

- **600+ Languages Supported**: The broadest language coverage among zero-shot TTS models ([full list](docs/languages.md)).
- **Voice Cloning**: State-of-the-art voice cloning quality.
- **Voice Design**: Control voices via assigned speaker attributes (gender, age, pitch, dialect/accent, whisper, etc.).
- **Fine-grained Control**: Non-verbal symbols (e.g., `[laughter]`) and pronunciation correction via pinyin or phonemes.
- **Fast Inference**: RTF as low as 0.025 (40x faster than real-time).
- **Diffusion Language Model-Style Architecture**: A clean, streamlined, and scalable design that delivers both quality and speed.
---
# 分支说明

完整功能版 OmniVoice 服务器，保留官方所有 Web UI 功能，同时添加 OpenAI 兼容 API。

## 特点

- **完整 Web UI**：声音克隆、声音设计、自动声音三个模式
- **OpenAI API**：兼容 AnythingLLM 等工具,可以选择克隆音色作为API的默认音色
- **模型共享**：Web UI 和 API 共用模型，节省内存
---
## AnythingLLM 配置

1. 设置 → Voice & Speech → TTS Provider: **Generic OpenAI**
2. 配置：
   - **API Endpoint**: `http://localhost:5050/v1`
   - **API Key**: `omnivoice`
   - **Model**: `tts-1`
   - **Voice**: `alloy` 或描述如 `female, low pitch`
---
uv run python omnivoice_full_server.py --web_port 8080 --api_port 5051

## Installation

### uv

Clone the repository and sync dependencies:

```bash
git clone https://github.com/alex13by/OmniVoice-API.git
cd OmniVoice-API
uv sync
```
---

## Quick Start

```bash
# 自定义端口
uv run python omnivoice_full_server.py --web_port 8080 --api_port 5051

# 创建公开链接（Gradio）
uv run python omnivoice_full_server.py --share
```
---

## Citation

```bibtex
@article{zhu2026omnivoice,
      title={OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models},
      author={Zhu, Han and Ye, Lingxuan and Kang, Wei and Yao, Zengwei and Guo, Liyong and Kuang, Fangjun and Han, Zhifeng and Zhuang, Weiji and Lin, Long and Povey, Daniel},
      journal={arXiv preprint arXiv:2604.00688},
      year={2026}
}
```
