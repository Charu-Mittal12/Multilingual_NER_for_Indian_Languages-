# Multilingual Named Entity Recognition for Indian Languages

Named Entity Recognition (NER) for **Hindi, Urdu, Odia, and Telugu** using a single multilingual model trained on 39K+ annotated sentences. The project compares BiLSTM-CRF taggers on frozen **MuRIL** and **XLM-RoBERTa** embeddings, and tests whether **LoRA** fine-tuning of MuRIL improves on frozen features.

## Highlights

- Trained BiLSTM-CRF on frozen MuRIL and XLM-RoBERTa embeddings using 39K+ annotated sentences (train + dev) across four languages.
- Best entity-level F1 with MuRIL: **83.7%** (Hindi), **82.5%** (Urdu), **78.1%** (Odia), **81.6%** (Telugu).
- XLM-RoBERTa reached 82.6%, 80.5%, 71.6%, and 75.8% on the same languages.
- Fine-tuned MuRIL with LoRA, achieving results comparable to frozen MuRIL + BiLSTM-CRF.
- One model is trained on the pooled data and evaluated separately on each language's test set.

## Dataset

**IL-NER**: human-annotated CoNLL-format NER corpus for Hindi, Urdu, Odia, and Telugu.
Source: [github.com/ltrc/IL-NER](https://github.com/ltrc/IL-NER)

Labels use BIO format over six entity types (`NEP` person, `NEL` location, `NEO` organization, `NEAR` artefact, `NEN` number, `NETI` time), giving 13 labels in total.

| Language | Train | Dev | Test |
|---|---:|---:|---:|
| Hindi | 11,076 | 1,389 | 1,388 |
| Urdu | 8,720 | 1,094 | 1,096 |
| Odia | 12,109 | 1,517 | 1,519 |
| Telugu | 2,993 | 384 | 384 |
| **Total** | **34,898** | **4,384** | **4,387** |

Train + dev = 39,282 sentences (the "39K+"). Test sets are kept separate and never used for training or model selection.

## Experiments

| # | Model | Trained parts |
|---|---|---|
| 1 | Frozen MuRIL → BiLSTM-CRF | BiLSTM, LayerNorm, linear layer, CRF |
| 2 | Frozen XLM-RoBERTa → BiLSTM-CRF | BiLSTM, LayerNorm, linear layer, CRF |
| 3 | MuRIL + LoRA → CRF | LoRA adapters, classification head, CRF |

```text
Exp. 1 / 2:  Words → frozen encoder (avg of last 4 layers)
             → sub-word mean-pool (1 vector per word)
             → 2-layer BiLSTM (hidden 320) → LayerNorm → Linear → CRF → BIO tags

Exp. 3:      Words → MuRIL tokenizer → MuRIL (frozen) + LoRA adapters
             → token-classification head → CRF → BIO tags
```

**Setup shared by all experiments**
- Telugu training sentences oversampled 3× to reduce language imbalance (40,884-sentence train pool).
- Shared 13-label tag map across languages.
- Loss: CRF negative log-likelihood. Decoding: Viterbi.
- Early stopping on combined dev F1.

**Experiments 1 and 2:** Adam (lr 1e-3, weight decay 1e-5), batch size 32, up to 25 epochs.

**Experiment 3:** LoRA `r=32`, `alpha=64`, dropout 0.10 on attention and feed-forward layers (about 0.56% trainable parameters). AdamW (lr 1.5e-4) with cosine schedule and 10% warmup, batch size 8, max length 128, up to 20 epochs.

Models: [`google/muril-base-cased`](https://huggingface.co/google/muril-base-cased), [`xlm-roberta-base`](https://huggingface.co/xlm-roberta-base).

## Results

Test entity-level F1 (`seqeval`), reported per language.

| Model | Hindi | Urdu | Odia | Telugu |
|---|---:|---:|---:|---:|
| Frozen MuRIL + BiLSTM-CRF | **83.7%** | **82.5%** | **78.1%** | **81.6%** |
| Frozen XLM-RoBERTa + BiLSTM-CRF | 82.6% | 80.5% | 71.6% | 75.8% |
| MuRIL + LoRA + CRF | comparable to frozen MuRIL | | | |

**Takeaways**
- Frozen MuRIL is best in every language. The gap to XLM-RoBERTa is small for Hindi and Urdu but large for Odia (6.5 points) and Telugu (5.8 points).
- LoRA fine-tuning of MuRIL gave no clear gain over frozen features with a BiLSTM-CRF.

## Repository

- `Multilingual_lstmcrf.ipynb`: data parsing, embedding extraction, BiLSTM-CRF training, LoRA training, and per-language evaluation.

## Run

```bash
pip install torch transformers peft pytorch-crf seqeval numpy
```

Download IL-NER and set `BASE_PATH` in the notebook (a CUDA GPU is recommended):

```python
BASE_PATH = "/kaggle/input/datasets/charumittalma25m008/ner-models-for-indian-languages/Datasets"
```

```text
Datasets/
├── Hindi/   (Hindi-train.txt, Hindi-dev.txt, Hindi-test.txt)
├── Urdu/    (Urdu-train.txt, Urdu-dev.txt, Urdu-test.txt)
├── Odia/    (Odia-train.txt, Odia-dev.txt, Odia-test.txt)
└── Telugu/  (Telugu-train.txt, Telugu-dev.txt, Telugu-test.txt)
```

## Limitations

- Telugu has far fewer training sentences than the other languages, and its test set is small (384 sentences).
- Results are from single runs, so small differences between models may fall within run-to-run variation.
- Rare entity types (`NEAR`, `NEN`, `NEO`) are harder; per-tag error analysis is future work.

## Tech Stack

Python · PyTorch · Hugging Face Transformers · PEFT (LoRA) · pytorch-crf · seqeval · Jupyter

## Author

**Charu Mittal**

## Reference

Bahad, S., Mishra, P., Arora, K., Balabantaray, R. C., Sharma, D. M., and Krishnamurthy, P. *Fine-tuning Pre-trained Named Entity Recognition Models For Indian Languages.* NAACL 2024 Student Research Workshop. [arXiv:2405.04829](https://arxiv.org/abs/2405.04829)
