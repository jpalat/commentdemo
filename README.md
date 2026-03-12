CBRE Customer Service Analysis Tool
---

## Description

CBRE surveys internal customers for open-ended feedback on software service offerings. This project provides a multi-label text classifier that categorizes incoming comments into four categories:

- **Technical** — integration issues, configuration complexity, bugs, update breakage
- **Performance** — slowness, latency, timeouts, resource usage
- **UX** — interface design, navigation, usability, mobile experience
- **Data/Security** — compliance, audit trails, data handling, security concerns

A single comment may belong to multiple categories.

## Model

The classifier uses **SetFit** (Sentence Transformers Fine-tuning), a Hugging Face framework designed for few-shot classification. Training proceeds in two phases:

1. **Contrastive fine-tuning** — the `all-MiniLM-L6-v2` sentence transformer is fine-tuned on automatically generated sentence pairs, adapting the encoder to the feedback domain.
2. **Classifier head** — four independent logistic regression classifiers (one-vs-rest) are fit on the domain-adapted embeddings.

The model is trained on 149 hand-labeled examples from `data/labeled_data.csv`.

## Web Application

A Flask web application is provided with three pages:

| Route | Description |
|---|---|
| `/` | Submit a comment and see its predicted labels with confidence scores |
| `/report` | Baseline model training methodology and results |
| `/report/setfit` | SetFit model architecture, training details, and evaluation results |

## Usage

```bash
# Install dependencies
uv venv && uv pip install -e .

# Train the model
python main.py train

# Start the web server (auto-trains if no model exists)
python main.py serve
# → http://localhost:2273
```

## Project Structure

```
cbre/
├── data/
│   └── labeled_data.csv        # 149 hand-labeled training examples
├── data_analysis/
│   └── labelPrompt.txt         # Prompt used to generate labels
├── models/
│   └── setfit/                 # Saved fine-tuned model artifacts
├── src/
│   ├── classifier.py           # CommentClassifier (SetFit)
│   └── train.py                # Training + evaluation script
├── templates/
│   ├── index.html              # Classify UI
│   ├── report.html             # Baseline model report
│   └── report_setfit.html      # SetFit model report
├── app.py                      # Flask routes
├── main.py                     # CLI entry point
└── pyproject.toml
```

## Results (SetFit, held-out 23 examples)

| Label | Accuracy | F1 (Yes) |
|---|---|---|
| Technical | 91% | 0.50 |
| Performance | 100% | 1.00 |
| UX | 87% | 0.92 |
| Data/Security | 100% | 1.00 |
