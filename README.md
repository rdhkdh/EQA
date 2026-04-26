# Extractive Question Answering (EQA)

Fine-tuning and evaluation pipeline for extractive QA on domain-specific documents, using a **RoBERTa-based transformer** fine-tuned on a custom ~3,700-example dataset in SQuAD 2.0 format.

---

## What it does

Given a context passage and a question, the model extracts the exact answer span from the text:

```
Context : "Authorized Representative shall mean any person authorized by either
           of the parties. Bidder means any firm offering the solution."

Question: "Who is an Authorized Representative?"

Answer  : "any person authorized by either of the parties"
```

The dataset targets procurement and contract documents where precise span extraction matters.

---

## Results

| Metric | Score |
|---|---|
| Exact Match (EM) | **55.50** |
| F1 Score | **75.54** |
| F1 (answerable only) | **80.52** |

Evaluated on a held-out test set of 200 examples (187 answerable, 13 unanswerable).

---

## Model Architecture

| Property | Value |
|---|---|
| Architecture | `RobertaForQuestionAnswering` |
| Transformer layers | 6 |
| Hidden size | 768 |
| Attention heads | 12 |
| Max sequence length | 384 tokens |
| Doc stride | 128 tokens |

**Training config:** 3 epochs · AdamW (lr=3e-4) · linear warmup · batch size 8 · seed 42

The 6-layer RoBERTa variant was chosen to balance accuracy and compute — it fits on a free Colab GPU while maintaining strong F1.

---

## Tech Stack

`Python 3.9` · `PyTorch 1.13` · `HuggingFace Transformers 4.27` · `SQuAD 2.0` · `Google Colab`

---

## Repository Structure

| File | Purpose |
|---|---|
| `FinetuneWorkingColab.ipynb` | Full fine-tuning pipeline (Colab, recommended) |
| `Finetune_Model.ipynb` | Local fine-tuning workflow |
| `EvaluateQA_Github.ipynb` | Evaluation — computes EM and F1 on test set |
| `SingleContextColab.ipynb` | Single-context inference demo |
| `testoriginalwithEQA.ipynb` | Baseline comparison experiments |
| `predict.json` | Sample prediction input |
| `requirements_minimal.txt` | Minimal environment (start here) |
| `requirements*.txt` | Full Colab environment snapshots for exact reproducibility |

---

## Setup

**Minimal install (recommended):**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_minimal.txt
jupyter notebook
```

**Reproduce a specific Colab environment:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirementsQA.txt   # or requirementsFinetuneWorkingColab.txt
```

> The full `requirements*.txt` files are Colab environment exports — they install 400+ packages and take time. Use `requirements_minimal.txt` unless you need exact snapshot reproduction.

---

## Running on Google Colab

1. Open `FinetuneWorkingColab.ipynb` in Colab.
2. Mount your Google Drive when prompted.
3. Update the path variables to match your own Drive directory (look for `./eqa/` path references).
4. Place `train.json` and `test.json` in your `eqa/` folder on Drive.
5. Run all cells. The best checkpoint is saved automatically.

After training, run `EvaluateQA_Github.ipynb` against the saved checkpoint to reproduce the EM/F1 results.

---

## Data Format

SQuAD 2.0-style JSON — each example contains:

```json
{
  "context": "...",
  "question": "...",
  "answers": { "text": ["..."], "answer_start": [42] }
}
```

Unanswerable questions (no answer span in context) are included as in the original SQuAD 2.0 benchmark. The format is natively compatible with HuggingFace's QA training pipelines.

---

## Model Checkpoint

The fine-tuned checkpoint is not stored in this repository due to file size. To replicate:

1. Run `FinetuneWorkingColab.ipynb` with your own copy of the dataset.
2. The best checkpoint (`trob55007534best/`) will be saved to your Google Drive.
3. Point `SingleContextColab.ipynb` or `EvaluateQA_Github.ipynb` at that checkpoint path.

---

## Roadmap

- Hyperparameter search for improved EM/F1
- TensorRT inference optimization
- Triton Inference Server deployment

---

