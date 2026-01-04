# FinTune: Financial Sentiment Analysis with Llama-3

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Model: Llama-3](https://img.shields.io/badge/Model-Llama--3--8B-blue)](https://huggingface.co/unsloth/llama-3-8b-bnb-4bit)
[![Fine-Tuning: QLoRA](https://img.shields.io/badge/Fine--Tuning-QLoRA%20%2B%20SFT-orange)](https://arxiv.org/abs/2305.14314)
[![Library: Unsloth](https://img.shields.io/badge/Library-Unsloth-green)](https://github.com/unslothai/unsloth)

## Project Overview
**FinTune** is a specialized Large Language Model (LLM) designed to understand the nuance of financial news, earnings reports, and market headlines. 

Built by fine-tuning **Llama-3-8B** using **QLoRA (Quantized Low-Rank Adaptation)** and **Supervised Fine-Tuning (SFT)**, this model outperforms generic base models in categorizing financial sentiment (Positive, Negative, Neutral) and extracting key insights from complex economic text.

This project demonstrates the end-to-end pipeline of preparing financial datasets, training with memory-efficient techniques, and deploying for inference.

## Key Features
* **Domain Specificity:** Fine-tuned specifically on the `finance-alpaca` dataset to understand terms like "EBITDA", "bearish", "bullish", and "fiscal year".
* **Efficient Training:** Utilized **Unsloth** and **QLoRA** to fine-tune a 8B parameter model on a single free T4 GPU (Google Colab), reducing VRAM usage by ~60%.
* **4-Bit Quantization:** Deployed using `bitsandbytes` 4-bit loading for low-latency inference on consumer hardware.
* **Instruction Tuned:** Follows the Alpaca prompt format for clear Instruction-Input-Response structured outputs.

## 🛠️ Technical Stack
* **Base Model:** `unsloth/llama-3-8b-bnb-4bit`
* **Fine-Tuning Library:** Unsloth (for 2x faster training), Hugging Face TRL (`SFTTrainer`)
* **Peft Method:** LoRA (Rank=16, Alpha=16)
* **Dataset:** [gbharti/finance-alpaca](https://huggingface.co/datasets/gbharti/finance-alpaca)
* **Hardware:** NVIDIA T4 GPU (via Google Colab)

## Model Evaluation & Results
The model was tested on unseen financial headlines to evaluate its reasoning capabilities.

### Example 1: Nuanced Negative
**Input:** *"Although Q3 revenue grew 5% year-over-year, the company missed analyst estimates by $0.10 per share."*
> **Base Llama-3:** (Often focuses only on "revenue grew")
> **FinTune Output:** *"The sentiment is negative. While revenue grew, missing analyst estimates is a key indicator of underperformance."*

### Example 2: Market Movement
**Input:** *"Shares of BioPharma soared 15% in pre-market trading following FDA approval of their new drug."*
> **FinTune Output:** *"The sentiment is positive. The double-digit stock rise and FDA approval are strong bullish signals."*

