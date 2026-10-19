# GPT from Scratch & Fine-Tuning Experiments

A PyTorch implementation of a GPT-style decoder-only Transformer, with end-to-end training, autoregressive inference, and controlled comparisons of parameter-efficient fine-tuning and preference optimization.

The core Transformer is implemented directly with PyTorch tensor operations and modules. Hugging Face Transformers is used to load a fixed OpenAI GPT-2 checkpoint and verify weight conversion against a reference model. Small-model training starts from random initialization; the GPT-2 fine-tuning experiments start from pretrained weights.

## Implemented from scratch

- **GPT architecture:** token and learned positional embeddings, multi-head causal self-attention, explicit causal masking, pre-layer normalization, residual connections, GELU feed-forward blocks, and optional embedding/output weight tying.
- **Autoregressive inference:** per-layer KV caching, greedy decoding, temperature scaling, and top-k sampling.
- **Training:** next-token language modeling, classification, and instruction fine-tuning; response-only loss masking, padding exclusion, gradient clipping, checkpoint resumption, and validation-based checkpoint selection.
- **Native adapters:** LoRA, rsLoRA, and DoRA implemented around linear layers. NF4 quantization uses bitsandbytes.
- **Preference objectives:** DPO, clipped PPO, and group-relative policy optimization in a deliberately small, single-step sentiment environment.

The implementation makes the attention computation, adapter updates, loss masking, and optimization objectives directly inspectable.

## Fine-tuning comparison

Experiments were run on a **Colab Tesla T4** with GPT-2 124M, seeds **123, 456, and 789**, and the same instruction-data split: **913 training / 54 validation / 108 test examples**.

All methods used two epochs and batch size 4. Adapters used rank 8 and alpha 16 across 48 Transformer linear layers. Learning rates were `5e-5` for full fine-tuning and `5e-4` for adapters. Checkpoints were selected on validation loss; the test set was not used for selection.

| Method | Trainable parameters | Test response PPL ↓ | Peak allocated GPU memory (GiB) ↓ | Training time, two epochs (s) ↓ |
| --- | ---: | ---: | ---: | ---: |
| Full fine-tuning | 124,439,808 | 4.46 ± 0.12 | 2.35 | 47.04 ± 0.42 |
| LoRA | 1,179,648 | 4.22 ± 0.06 | 0.82 | 22.00 ± 0.22 |
| rsLoRA | 1,179,648 | 4.28 ± 0.09 | 0.82 | 21.69 ± 0.07 |
| DoRA | 1,262,592 | 4.22 ± 0.07 | 1.45 | 36.66 ± 0.05 |
| NF4 LoRA | 1,179,648 | 4.29 ± 0.04 | 0.55 | 31.56 ± 0.57 |

PPL and time are mean ± sample standard deviation over three seeds; memory values are rounded. These are fixed-configuration exploratory comparisons, not independently tuned method rankings. Training time excludes data preparation, model loading, evaluation, and generation.

LoRA trained approximately **0.95%** of the base model's parameters and used approximately **65% less peak allocated GPU memory** than full fine-tuning in this setup.

The checkpoint with the lowest validation PPL across the 15 SFT runs was **rsLoRA, seed 456, epoch 2**: validation PPL **3.368**, test PPL **4.269**. This individual checkpoint result is distinct from the three-seed method averages above.

### What these results do—and do not—show

- Similar reference-answer likelihood does not guarantee correct or helpful generation. Fixed qualitative examples included failed instruction following and incorrect factual answers.
- rsLoRA and LoRA use different effective scaling at the same rank and alpha; this experiment does not isolate scaling from learning-rate effects.
- DoRA explicitly constructs the effective dense weight. Its memory and runtime costs should not be generalized to optimized implementations.
- NF4 LoRA quantizes Transformer linear weights, keeps embeddings and the tied output head in FP32, and uses FP32 computation and double quantization. It does not use a paged optimizer or reproduce the complete original QLoRA recipe.

## Preference optimization comparison

The preference experiment uses SST-2 with **512 train / 128 validation / 256 test examples**. Validation and test are disjoint subsets of the original validation split. The action space contains only the single-token labels ` negative` and ` positive`; labels provide rewards and preference pairs.

Each seed shares a two-epoch LoRA SFT warm start. Subsequent methods run for 80 updates with learning rate `5e-4`; PPO and GRPO sample four actions per prompt. Selection uses validation accuracy, with NLL as a tie-breaker, and permits retaining the warm start.

| Method | Test accuracy (%) ↑ | Test binary NLL ↓ |
| --- | ---: | ---: |
| Warm-start SFT | 88.54 ± 1.48 | 0.271 ± 0.024 |
| Continued SFT | 86.98 ± 0.23 | 0.406 ± 0.077 |
| DPO | 88.54 ± 1.58 | 3.519 ± 0.944 |
| PPO | 88.02 ± 2.77 | 0.441 ± 0.157 |
| GRPO | 89.32 ± 0.81 | 0.539 ± 0.070 |

These three-seed results illustrate why reward, accuracy, and confidence must be evaluated separately. They do not establish a universal winner. This is a **single-step contextual bandit**, not a long-form RLHF or reasoning benchmark. PPO uses a scalar value head on a shared backbone; DPO uses offline preference pairs.

The tables summarize locally retained experiments from 2026-10-19. Raw run artifacts and model weights are not distributed in this repository.

## Quick start

Run commands from the repository root in an activated Python environment. CPU supports the native model and unquantized adapters; the NF4 experiment requires CUDA. The recorded Colab experiments used Python 3.13.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-pretrained.txt
```

For NF4 comparison, install the additional dependency in a CUDA-enabled environment:

```bash
python -m pip install -r requirements-comparison.txt
```

### Prepare data and convert pretrained GPT-2 weights

```bash
python -m minigpt.prepare_instructions --out data/instructions
python -m minigpt.pretrained --out runs/gpt2-pretrained --device cpu
```

The converter uses `openai-community/gpt2` at revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, checks logits against the reference implementation, and records provenance and a checkpoint hash. Data preparation records split hashes and source provenance. Use fresh output directories.

### Run an instruction fine-tuning experiment

```bash
python -m minigpt.compare_methods --init runs/gpt2-pretrained/model.pt --data data/instructions --method rslora --seed 456 --rank 8 --epochs 2 --batch-size 4 --out runs/sft/rslora-456
```

To reproduce the comparison matrix, run each method (`full`, `lora`, `rslora`, `dora`, `qlora`) with seeds `123`, `456`, and `789`, using a separate output directory for each run. The entry point selects CUDA when available. Each run saves metrics, metadata, selected trainable parameters, and generation examples.

### Generate with a trained adapter

```bash
python -m minigpt.adapter_generate --base runs/gpt2-pretrained/model.pt --adapter runs/sft/rslora-456/selected-trainable.pt --method rslora --rank 8 --instruction "Explain gravity in one sentence."
```

### Merge and use a standalone local checkpoint

```bash
python -m minigpt.merge_adapter --base runs/gpt2-pretrained/model.pt --adapter runs/sft/rslora-456/selected-trainable.pt --method rslora --rank 8 --alpha 16 --out runs/merged-rslora/model.pt
python -m minigpt.generate --checkpoint runs/merged-rslora/model.pt --instruction --prompt "Explain gravity in one sentence."
```

Use the original pretrained base for merging. The merge tool checks adapter keys, shapes, finite values, and pre/post-merge logits, and writes SHA256 provenance. Existing output files are not overwritten. These commands describe the export workflow; the selected experimental checkpoint's full merge and generation verification are still pending.

### Run preference experiments

```bash
python -m minigpt.prepare_sentiment
python -m minigpt.compare_preferences --init runs/gpt2-pretrained/model.pt --data data/sentiment-control --seed 123 --steps 80 --out runs/preferences/seed-123
```

Repeat with seeds `456` and `789` for the three-seed comparison.

### Tests

```bash
python -m unittest discover -s tests
```

The current suite contains **18 tests**, covering core model behavior, caching, training/resumption, data preparation, pretrained-weight conversion, adapters and merge equivalence, and preference objectives. Some tests need dependencies above and an available GPT-2 tokenizer cache or network access on first use.

## Code map

| Path | Purpose |
| --- | --- |
| `minigpt/model.py` | Native GPT Transformer and classification head |
| `minigpt/data.py`, `minigpt/train.py` | Task datasets, loss masking, training, and checkpoint selection |
| `minigpt/generate.py`, `minigpt/benchmark.py` | Generation and cache benchmarking |
| `minigpt/pretrained.py` | GPT-2 weight conversion and reference-equivalence checks |
| `minigpt/adapters.py`, `minigpt/compare_methods.py` | Native adapters and SFT comparisons |
| `minigpt/preference_objectives.py`, `minigpt/compare_preferences.py` | Preference objectives and controlled experiments |
| `minigpt/merge_adapter.py` | Local unquantized adapter merging |
| `configs/`, `tests/` | Model configurations and automated checks |

## Data, attribution, and local artifacts

This learning project builds on concepts explored in Sebastian Raschka's *Build a Large Language Model (From Scratch)*. The instruction-data preparation code uses the [LLMs-from-scratch dataset](https://github.com/rasbt/LLMs-from-scratch/tree/cdbd33e6d57a71e20ea31434841095fac50cecd8/ch07/01_main-chapter-code) and preserves its source license. GPT-2 pretrained weights originate from [OpenAI GPT-2](https://github.com/openai/gpt-2) and are loaded from its Hugging Face distribution. Sentiment experiments use SST-2; consult the preparation code for source and split provenance.

**Model weights remain local.** The repository ignores `runs/`, `data/`, `*.pt`, and `*.pth`. Store base weights, adapters, merged checkpoints, and generated experiment artifacts under these ignored directories. Only source code, configuration, dependencies, tests, and this README are published.
