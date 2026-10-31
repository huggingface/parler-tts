# Parler-TTS

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

Parler-TTS 是一个轻量级文本转语音（Text-to-Speech, TTS）模型，能够根据指定说话人的风格（性别、音高、说话风格等）生成高质量、自然逼真的语音。该项目是对 Dan Lyth 和 Simon King（分别来自 Stability AI 和爱丁堡大学）发表的研究论文《[Natural language guidance of high-fidelity text-to-speech with synthetic annotations](https://www.text-description-to-speech.com)》的完整复现。

与其他 TTS 模型不同，Parler-TTS 采用**完全开源**的方式发布。所有数据集、预处理流程、训练代码和模型权重均在宽松的开源许可证下公开，使社区能够在我们的成果之上继续构建并开发自己的强大 TTS 模型。

本仓库包含 Parler-TTS 的推理与训练代码，并专门配合用于数据集标注的 [Data-Speech](https://github.com/huggingface/dataspeech) 仓库使用。

> [!IMPORTANT]
> **2024/08/08：** 我们很荣幸发布两个全新的 Parler-TTS 检查点：
> 1. [Parler-TTS Mini](https://huggingface.co/parler-tts/parler-tts-mini-v1)，拥有 8.8 亿（880M）参数的模型。
> 2. [Parler-TTS Large](https://huggingface.co/parler-tts/parler-tts-large-v1)，拥有 23 亿（2.3B）参数的模型。
>
> 这些检查点均基于 4.5 万小时的有声读物数据训练而成。
>
> 此外，代码经过了深度优化以实现更快的生成速度：我们增加了与 SDPA 和 Flash Attention 2 的兼容性，以及对模型编译（torch.compile）的支持。

## 📖 快速索引
* [安装指南](#安装指南)
* [使用方法](#使用方法)
  - [🎲 使用随机音色](#-随机音色)
  - [🎯 使用指定说话人](#-使用指定说话人)
* [模型训练](#模型训练)
* [在线体验 Demo](https://huggingface.co/spaces/parler-tts/parler_tts)
* [模型权重与数据集](https://huggingface.co/parler-tts)
* [优化推理速度](#-优化推理速度)

## 安装指南

Parler-TTS 依赖轻量，仅需一行命令即可完成安装：

```sh
pip install git+https://github.com/huggingface/parler-tts.git
```

Apple Silicon（Mac）用户需要运行后续命令以使用 PyTorch 每夜版（2.4）以获得 bfloat16 支持：

```sh
pip3 install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

## 使用方法

> [!TIP]
> 您可以在[交互式在线 Demo](https://huggingface.co/spaces/parler-tts/parler_tts)中直接体验！

使用 Parler-TTS 就像说“bonjour”一样简单。只需安装一次库：

```sh
pip install git+https://github.com/huggingface/parler-tts.git
```

### 🎲 随机音色


**Parler-TTS** 经过专门训练，可以通过简单的文本提示词直接控制所生成语音的各项特征，例如：

```py
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

prompt = "Hey, how are you doing today?"
description = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)
```

### 🎯 使用指定说话人

为确保跨次生成的音色一致性，该检查点还在 34 位具名说话人数据上进行了训练。全部可用说话人列表包括：
Laura, Gary, Jon, Lea, Karen, Rick, Brenda, David, Eileen, Jordan, Mike, Yann, Joy, James, Eric, Lauren, Rose, Will, Jason, Aaron, Naomie, Alisa, Patrick, Jerry, Tina, Jenna, Bill, Tom, Carol, Barbara, Rebecca, Anna, Bruce, Emily。

只需在文本描述中指定所使用的说话人即可使用该功能：`Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise.`

您可以将 "Jon" 替换为上述列表中的任何姓名，以使用不同的说话人特征。每位说话人都具有独特的嗓音质感，可根据您的具体需求灵活选用。关于音色一致性的说话人表现更详细的信息，请参阅[推理指南](INFERENCE.md#speaker-consistency)。

```py
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

prompt = "Hey, how are you doing today?"
description = "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)
```

**实用提示**：
* 包含 "very clear audio"（极清晰音频）短语可生成最高质量的音频，而 "very noisy audio"（高噪声音频）则会引入明显的背景噪点
* 可以使用标点符号来控制生成的韵律，例如使用逗号在语音中添加短暂的停顿
* 其余语音特征（性别、语速、音高和混响）均可通过提示词直接控制

### ✨ 优化推理速度

我们编写了专门的[推理指南](INFERENCE.md)以加快生成速度。包含 SDPA、torch.compile 以及流式输出（Streaming）！


https://github.com/huggingface/parler-tts/assets/52246514/251e2488-fe6e-42c1-81cd-814c5b7795b0

## 模型训练

<a target="_blank" href="https://github.com/ylacombe/scripts_and_notebooks/blob/main/Finetuning_Parler_TTS_v1_on_a_single_speaker_dataset.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

[training 目录](/training/) 包含训练或微调您自己的 Parler-TTS 模型的全部信息，包括：
- [1. Parler-TTS 架构介绍](/training/README.md#1-architecture)
- [2. 快速上手的第一步](/training/README.md#2-getting-started)
- [3. 训练完整指南](/training/README.md#3-training)

> [!IMPORTANT]
> **简明摘要（TL;DR）：** 按照[安装步骤](/training/README.md#requirements)完成依赖安装后，您可以通过以下命令行复现 Parler-TTS Mini v1 的训练配方：

```sh
accelerate launch ./training/run_parler_tts_training.py ./helpers/training_configs/starting_point_v1.json
```

> [!IMPORTANT]
> 您还可以参考[这份微调指南](https://github.com/ylacombe/scripts_and_notebooks/blob/main/Finetuning_Parler_TTS_v1_on_a_single_speaker_dataset.ipynb)在单说话人数据集示例上进行微调。

## 致谢

本项目构建于众多杰出的开源基石之上，我们向提供这些工具的开发者致以最诚挚的感谢！

特别鸣谢：
- Dan Lyth 与 Simon King（分别来自 Stability AI 和爱丁堡大学），感谢他们发表了如此富有前景且条理清晰的研究论文：[Natural language guidance of high-fidelity text-to-speech with synthetic annotations](https://arxiv.org/abs/2402.01912)。
- 所使用的众多开源库，特别是 [🤗 datasets](https://huggingface.co/docs/datasets/v2.17.0/en/index)、[🤗 accelerate](https://huggingface.co/docs/accelerate/en/index)、[jiwer](https://github.com/jitsi/jiwer)、[wandb](https://wandb.ai/) 以及 [🤗 transformers](https://huggingface.co/docs/transformers/index)。
- Descript 提供的 [DAC 编解码模型](https://github.com/descriptinc/descript-audio-codec)。
- Hugging Face 🤗 提供的算力资源以及宝贵的探索时间！


## 引用本项目

如果您发现本仓库对您的研究或工作有所帮助，请考虑引用本项目以及 Stability AI 的原始论文：

```
@misc{lacombe-etal-2024-parler-tts,
  author = {Yoach Lacombe and Vaibhav Srivastav and Sanchit Gandhi},
  title = {Parler-TTS},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/parler-tts}}
}
```

```
@misc{lyth2024natural,
      title={Natural language guidance of high-fidelity text-to-speech with synthetic annotations},
      author={Dan Lyth and Simon King},
      year={2024},
      eprint={2402.01912},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```

## 参与贡献

欢迎社区参与贡献！本项目拥有广阔的改进与探索空间。

具体而言，我们正在寻求提升音质与速度的方案：
- 数据集：
    - 在更大规模数据上进行训练
    - 增加更多语音特征（如口音/方言）
- 模型训练：
    - 添加 PEFT 兼容性以支持 LoRA 微调
    - 支持在无描述列（description column）的数据集上训练
    - 支持在 Jupyter Notebook 中进行训练
    - 探索多语言训练
    - 探索单说话人微调
    - 探索更多模型架构
- 性能优化：
    - 模型编译与静态缓存（Static Cache）
    - 支持 Flash Attention 2 (FA2) 与 SDPA
- 效果评估：
    - 增加更多评估指标

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月31日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
