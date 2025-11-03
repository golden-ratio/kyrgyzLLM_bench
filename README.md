# Kyrgyz LLM Evaluation Dataset

Welcome to the **KyrgyzLLM-Bench - kyrgyz LLM Evaluation Dataset**, your one-stop solution for evaluating Large Language Models (LLMs) in Kyrgyz. This toolkit helps you measure model performance across diverse domains and question types specific to the Kyrgyz language, so your models can be more accurate, robust, and helpful for Kyrgyz-speaking users. Whether you're a researcher, developer, or practitioner—this dataset is tailored to help your Kyrgyz-capable LLM thrive.

[![Paper](https://img.shields.io/badge/IEEE%20Xplore-Paper-blue)](https://ieeexplore.ieee.org/document/11206960)  
[![Model](https://img.shields.io/badge/HuggingFace-Hub-yellow)](https://huggingface.co/TTimur)


![Kyrgyz LLM Evaluation in Tech Landscape](Kyrgyz%20LLM%20Evaluation%20in%20Tech%20Landscape.png "Kyrgyz LLM Evaluation")


Quick facts:
- Language Support: Kyrgyz (ky)
- Audience: Researchers, developers, and the Kyrgyz NLP community
- A native kyrgyz language datasets:
• MMLU
• Reading comprehension

- Translated Benchmarks
• Commonsense reasoning & understanding: HellaSwag and WinoGrande
• Reading comprehension: BoolQ
• Mathematics: GSM8K
• Robustness & factuality: TruthfulQA 
- Tooling: First-class support with Lighteval (see `lighteval/` scripts) and LM_harness (see `lm_harness/` scripts)

## 🔍 What's Inside?

KyrgyzLLM-Bench is a comprehensive suite purpose-built to evaluate LLMs’ deep understanding and reasoning in **Kyrgyz**. It combines natively authored benchmarks with carefully translated and post-edited international tasks to provide broad and culturally grounded coverage.

- **Language**: Kyrgyz (ky)
- **Components**:
  - KyrgyzMMLU (native, multiple-choice, 7,977 questions)
  - KyrgyzRC (native, reading comprehension, 400 questions)
  - Translated benchmarks: HellaSwag, WinoGrande, BoolQ, GSM8K, TruthfulQA (manually post-edited)

### 🧠 Diverse and Deep Evaluation Domains

KyrgyzLLM-Bench spans foundational sciences, humanities, and applied domains relevant to the Kyrgyz national curriculum and public knowledge.

#### KyrgyzMMLU (native, multiple-choice)
- Total: 7,977 questions written by curriculum experts
- Subjects and counts:
  - Math: 1,169
  - Physics: 1,228
  - Geography: 640
  - Biology: 1,550
  - Kyrgyz Language: 360
  - Kyrgyz Literature: 1,169
  - Kyrgyz History: 440
  - Medicine: 216
  - Chemistry: 1,205

#### KyrgyzRC (native, reading comprehension)
- Total: 400 multiple-choice questions (4 options, 1 correct)
- Sources: Kyrgyz Wikipedia, national news, literature, and school-style math word problems
- Skills evaluated: factual understanding, inference, vocabulary-in-context, multi-sentence reasoning

#### Translated Benchmarks (with manual post-editing)
- Commonsense reasoning: HellaSwag, WinoGrande
- Reading comprehension: BoolQ
- Mathematics: GSM8K
- Robustness/factuality: TruthfulQA

Translation pipeline: dual-model machine translation (Claude 4 Sonnet, Gemini 2.5 Flash), ensemble comparison, expert post-editing, and quality checks (incl. back-translation sampling).

## ⚡ Turbocharge Your Evaluations with Lighteval 🚀
If you want to evaluate models with Lighteval, please see `README_lighteval.md` — all installation steps, supported Kyrgyz tasks, example commands (HF and local), and leaderboard task files are documented there.

- Guide: [README_lighteval.md](./README_lighteval.md)

## 📊 Results

Below are the benchmark results for **Kyrgyz** in both zero-shot and few-shot settings.  
Higher scores indicate better performance (accuracy for most tasks, QEM for GSM8K).

---

### 🏔️ Kyrgyz Zero-Shot Evaluation Results

| Model                 | KyrgyzMMLU | KyrgyzRC | WinoGrande | BoolQ | HellaSwag | GSM8K | TruthfulQA | **Average** |
| :-------------------- | :--------: | :------: | :--------: | :---: | :-------: | :---: | :--------: | :---------: |
| **Qwen**              |            |          |            |       |           |       |            |             |
| Qwen2.5-0.5B-Instruct |    27.4    |   53.2   |    51.5    |  37.9 |    14.6   |  0.7  |    33.5    |     31.3    |
| Qwen2.5-1.5B-Instruct |    27.9    |   60.5   |    50.1    |  38.6 |    22.9   |  0.7  |    32.5    |     33.3    |
| Qwen2.5-3B-Instruct   |    28.6    |   66.0   |    50.5    |  59.4 |    22.0   |  0.7  |    34.2    |     37.3    |
| Qwen2.5-7B-Instruct   |    31.5    |   70.0   |    48.7    |  56.3 |    10.0   |  1.1  |    34.1    |     36.0    |
| Qwen3-0.6B            |    26.0    |   61.8   |    49.8    |  38.0 |    11.1   |  0.7  |    29.9    |     31.0    |
| Qwen3-1.7B            |    27.9    |   61.8   |    48.9    |  40.4 |    24.6   |  0.7  |    29.6    |     33.4    |
| Qwen3-4B              |    30.3    |   68.2   |    49.0    |  38.3 |    24.5   |  0.7  |    32.9    |     34.8    |
| Qwen3-8B              |    32.1    |   71.8   |    51.0    |  39.2 |    24.6   |  0.7  |    34.7    |     36.3    |
| **Gemma**             |            |          |            |       |           |       |            |             |
| gemma-3-1b-it         |    26.7    |   58.2   |    50.0    |  37.9 |    24.4   |  0.7  |    34.0    |     33.1    |
| gemma-3-270m          |    27.5    |   56.8   |    48.3    |  37.9 |    17.4   |  0.7  |    34.7    |     31.9    |
| gemma-3-4b-it         |    30.3    |   70.2   |    50.6    |  58.3 |    24.6   |  0.7  |    34.7    |   **38.5**  |
| **Meta-Llama**        |            |          |            |       |           |       |            |             |
| Llama-3.1-8B-Instruct |    31.0    |   75.2   |    50.6    |  50.3 |    26.6   |  0.7  |    33.7    |     38.3    |
| Llama-3.2-1B-Instruct |    26.3    |   58.2   |    49.4    |  38.3 |    0.2    |  0.7  |    30.1    |     29.0    |
| Llama-3.2-3B-Instruct |    27.8    |   64.2   |    49.1    |  43.1 |    24.5   |  0.7  |    31.5    |     34.4    |


*Zero-shot evaluation results on Kyrgyz benchmarks (%). The metric is accuracy, except for GSM8K which uses QEM. Higher is better.*

---

### 🏔️ Kyrgyz Few-Shot Evaluation Results

| Model                 | KyrgyzMMLU | KyrgyzRC | WinoGrande | BoolQ | HellaSwag | GSM8K | TruthfulQA | **Average** |
| :-------------------- | :--------: | :------: | :--------: | :---: | :-------: | :---: | :--------: | :---------: |
| **Qwen**              |            |          |            |       |           |       |            |             |
| Qwen2.5-0.5B-Instruct |    25.4    |   54.0   |    49.7    |  61.0 |    25.9   |  2.2  |    33.4    |     35.9    |
| Qwen2.5-1.5B-Instruct |    28.7    |   67.5   |    50.1    |  58.0 |    26.5   |  6.1  |    32.9    |     38.5    |
| Qwen2.5-3B-Instruct   |    34.0    |   73.2   |    51.3    |  57.4 |    23.7   |  9.5  |    34.4    |     40.5    |
| Qwen2.5-7B-Instruct   |    38.5    |   74.8   |    50.4    |  64.6 |    17.8   |  32.1 |    36.2    |     44.9    |
| Qwen3-0.6B            |    26.8    |   59.5   |    50.1    |  60.1 |    26.4   |  4.3  |    30.0    |     36.8    |
| Qwen3-1.7B            |    30.8    |   71.2   |    48.6    |  62.0 |    25.2   |  18.5 |    30.3    |     41.0    |
| Qwen3-4B              |    38.5    |   77.2   |    48.1    |  74.0 |    24.7   |  51.5 |    32.5    |     49.4    |
| Qwen3-8B              |    44.5    |   81.8   |    50.6    |  76.9 |    26.4   |  60.0 |    35.8    |   **53.7**  |
| **Gemma**             |            |          |            |       |           |       |            |             |
| gemma-3-1b-it         |    26.5    |   38.0   |    48.9    |  62.8 |    23.5   |  3.2  |    31.3    |     33.5    |
| gemma-3-270m          |    27.0    |   53.2   |    48.7    |  61.5 |    27.6   |  1.4  |    36.6    |     36.6    |
| gemma-3-4b-it         |    29.5    |   25.0   |    49.6    |  62.1 |    24.6   |  0.0  |    50.0    |     34.5    |
| **Meta-Llama**        |            |          |            |       |           |       |            |             |
| Llama-3.1-8B-Instruct |    38.1    |   80.5   |    51.6    |  75.5 |    21.9   |  37.0 |    34.4    |     48.5    |
| Llama-3.2-1B-Instruct |    26.1    |   45.8   |    49.7    |  62.0 |    25.8   |  2.7  |    30.3    |     34.7    |
| Llama-3.2-3B-Instruct |    29.4    |   64.8   |    48.9    |  62.3 |    25.3   |  12.9 |    32.9    |     39.6    |


*Few-shot evaluation results on Kyrgyz benchmarks (%). All tasks are 5-shot, except for HellaSwag (10-shot). The metric is accuracy, except for GSM8K which uses QEM. Higher is better.*


## 💡 Contributions Welcome!

Have ideas, bug fixes, or want to add a custom task? We'd love for you to be part of the journey! Contributions help grow and enhance the capabilities of the **KyrgyzLLM-Bench**.

## 📜 Citation

Thanks for using **KyrgyzLLM-Bench** — where language learning models meet Serbian precision and creativity! Let's build smarter models together. 🚀�

If you find this dataset useful in your research, please cite it as follows:

```bibtex
@article{KyrgyzLLM-Bench,
  title={Bridging the Gap in Less-Resourced Languages: Building a Benchmark for Kyrgyz Language Models},
  author={Timur Turatali, Aida Turdubaeva, Islam Zhenishbekov, Zhoomart Suranbaev, Anton Alekseev, Rustem Izmailov},
  year={2025},
  url={https://huggingface.co/datasets/TTimur/kyrgyzMMLU, 
  https://huggingface.co/datasets/TTimur/kyrgyzRC, 
  https://huggingface.co/datasets/TTimur/winogrande_kg, 
  https://huggingface.co/datasets/TTimur/boolq_kg, 
  https://huggingface.co/datasets/TTimur/truthfulqa_kg, 
  https://huggingface.co/datasets/TTimur/gsm8k_kg, 
  https://huggingface.co/datasets/TTimur/hellaswag_kg}
}
```
