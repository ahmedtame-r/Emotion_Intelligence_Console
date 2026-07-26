<div align="center">

# Pulsewave — Emotion Intelligence Console

**A fine-tuned DistilBERT model that reads six core emotions from raw text, served through a custom-built Gradio interface.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-DistilBERT-yellow?style=flat)](https://huggingface.co/docs/transformers)
[![Gradio](https://img.shields.io/badge/Gradio-UI-FF7C00?style=flat&logo=gradio&logoColor=white)](https://www.gradio.app/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Inference-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-informational)](#license)

[Overview](#overview) • [Demo](#demo) • [How It Works](#how-it-works) • [Getting Started](#getting-started) • [Project Structure](#project-structure) • [Results](#results)

</div>

---

## Overview

Pulsewave is a production-style deployment of a fine-tuned **DistilBERT** sequence classifier that scores any sentence across six core emotions:

`Joy` · `Sadness` · `Fear` · `Anger` · `Surprise` · `Disgust`

The model itself was trained separately on a six-class collapse of the **GoEmotions** dataset. This repository is focused entirely on serving it well: a fast inference pipeline, a clear prediction breakdown, and an interface designed to feel like a real product rather than a notebook demo.

## Demo

<div align="center">
<img src="docs/demo.gif" alt="Pulsewave demo" width="800"/>
<br/>
<sub>Replace with an actual screen recording or screenshot of the running app.</sub>
</div>

## Features

- **Six-class emotion detection** with a full confidence breakdown, not just a top label
- **Real-time inference** — sub-100ms predictions on CPU
- **Confidence gauge and probability bars** for interpretable, at-a-glance results
- **Ten curated example inputs** for instant exploration
- **Copy-to-clipboard** for sharing a prediction result
- **Fully responsive** custom interface, built from scratch on top of Gradio Blocks
- **Local-only inference** — no external API calls, no internet dependency at runtime

## How It Works

```
                ┌──────────────────────┐
   Raw text  →  │ DistilBERT Tokenizer │
                └──────────┬───────────┘
                           │
                ┌──────────▼───────────┐
                │ Fine-tuned DistilBERT │
                │ Classification Head  │
                └──────────┬───────────┘
                           │
                ┌──────────▼───────────┐
                │  Softmax over 6       │
                │  emotion classes      │
                └──────────┬───────────┘
                           │
                ┌──────────▼───────────┐
                │  Gradio UI rendering  │
                │  (gauge + bars)       │
                └───────────────────────┘
```

1. Input text is tokenized and truncated to a fixed max sequence length.
2. The fine-tuned DistilBERT model produces logits over six emotion classes.
3. A softmax converts logits into probabilities.
4. The interface renders the top prediction as a confidence gauge, plus a sorted bar chart for every class.

## Getting Started

### Prerequisites

- Python 3.10+
- A trained model checkpoint at `distilbert_best/` containing:
  ```
  config.json
  model.safetensors
  tokenizer.json
  tokenizer_config.json
  training_args.bin
  ```

### Installation

```bash
git clone https://github.com/ahmedtame-r/pulsewave-emotion-classifier.git
cd pulsewave-emotion-classifier
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

The app launches locally at `http://127.0.0.1:7860`.

## Project Structure

```
.
├── app.py                # Full deployment: model loading, inference, UI
├── requirements.txt      # Runtime dependencies
├── README.md             # This file
└── distilbert_best/      # Trained model checkpoint (not included in repo)
```

## Results

| Metric | Value |
|---|---|
| Model | DistilBERT (fine-tuned) |
| Dataset | GoEmotions (6-class collapse) |
| Classes | 6 |
| Accuracy | _fill in from evaluation_ |
| Macro F1 | _fill in from evaluation_ |
| Avg. inference time (CPU) | < 100 ms |

## Tech Stack

Python · PyTorch · HuggingFace Transformers · DistilBERT · Gradio

## Roadmap

- [ ] Add batch prediction / CSV upload support
- [ ] Add model explainability (attention or SHAP visualization)
- [ ] Containerize with Docker for one-command deployment
- [ ] Deploy a public demo on Hugging Face Spaces

## Author

**Ahmed Tamer (Shouman)**
Electronics and Communication Engineering student focused on AI, data science, and business intelligence.

- GitHub: [@ahmedtame-r](https://github.com/ahmedtame-r)
- Kaggle: [ahmedtamer047](https://www.kaggle.com/ahmedtamer047)

## License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
<sub>If this project was useful to you, consider giving it a star</sub>
</div>
