<div align="center">

<img src="https://img.shields.io/static/v1?label=University&message=Sapienza%20University%20of%20Rome&color=8B0000&style=for-the-badge&logoColor=white"/>
<img src="https://img.shields.io/badge/MSc-Computer%20Science-1a1a2e?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Academic%20Year-2025%2F2026-4a4e69?style=for-the-badge"/>

<br/><br/>

# 🧠 Evaluating Pre-Trained Vision Transformers Under Continual Learning

### *Can transformers learn forever without forgetting?*

<br/>

**Candidate:** Swati Aggrawal &nbsp;·&nbsp; **Matricola:** 2118031  

**Thesis Advisor:** Prof. Maria De Marsico

**Thesis Co-Advisor:** Prof. Marco Raoul Marini

**Faculty:** Ingegneria dell'Informazione, Informatica e Statistica

<br/>

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![timm](https://img.shields.io/badge/timm-0.9+-blue?style=flat-square)](https://github.com/huggingface/pytorch-image-models)
[![CIFAR-100](https://img.shields.io/badge/Benchmark-Split%20CIFAR--100-green?style=flat-square)](https://www.cs.toronto.edu/~kriz/cifar.html)
[![Colab](https://img.shields.io/badge/Trained%20on-Google%20Colab%20T4-F9AB00?style=flat-square&logo=googlecolab&logoColor=black)](https://colab.research.google.com)
[![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-red?style=flat-square)](#license)

</div>

---

## 📖 Overview

Humans learn continuously — new knowledge builds on old without erasing what came before. For neural networks, this is a profound challenge known as **catastrophic forgetting**.

This thesis investigates whether **pre-trained Vision Transformers (ViTs)**, augmented with **bottleneck adapter modules** and **Elastic Weight Consolidation (EWC)** regularization, can achieve strong sequential task performance with minimal forgetting — evaluated on a 5-task image classification benchmark.

> **Key finding:** Combining a frozen ViT backbone + lightweight adapters + EWC reduces catastrophic forgetting by ~97% relative to a CNN baseline, achieving **90.59% average accuracy** with only **2.01% average forgetting**.

---

## 🏆 Results at a Glance

| Model | Avg Accuracy ↑ | Avg Forgetting ↓ | BWT |
|:------|:--------------:|:----------------:|:---:|
| 🔴 ResNet-18 (naive FT) | 24.22% | 69.17% | -69.17% |
| 🟡 ViT-Tiny (naive FT) | 41.91% | 63.26% | -63.26% |
| 🟢 ViT + Adapters | 76.93% | 19.91% | -19.91% |
| ✅ **ViT + Adapters + EWC** | **90.59%** | **2.01%** | **-1.85%** |

> Evaluated on **Split CIFAR-100** · 5 sequential tasks · 20 classes each · Task-Incremental Learning (TIL)  
> Cross-validation (3 seeds): **86.93% ± 1.78%** accuracy · **5.91% ± 1.61%** forgetting

---

## 🏗️ Method

The approach combines three complementary mechanisms operating at different levels of the network:

```
┌─────────────────────────────────────────────────────┐
│              Frozen ViT-Tiny Backbone               │  ← Preserves shared
│         (5.85M params, ImageNet pre-trained)        │    visual knowledge
│  ┌───────────────────────────────────────────────┐  │
│  │  Transformer Block × 12                       │  │
│  │  ┌─────────────────┐   ┌──────────────────┐   │  │
│  │  │  Self-Attention │   │  Feed-Forward    │   │  │
│  │  └────────┬────────┘   └────────┬─────────┘   │  │
│  │           └──────────┬──────────┘             │  │
│  │                      ▼                        │  │
│  │  ┌─────────────────────────────────────────┐  │  │
│  │  │     Bottleneck Adapter (192→64→192)     │  │  │  ← Task-specific
│  │  │   LayerNorm → Linear → GELU → Linear    │  │  │    learning (~5.5%
│  │  │              + Residual                 │  │  │    of all params)
│  │  └─────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
                          │
              EWC Regularization (λ=1000)
         Protects critical adapter parameters
         via Fisher Information importance scores
```

| Component | Role | Params |
|:---|:---|:---:|
| 🧊 Frozen ViT backbone | Stable shared representations | 5.53M (frozen) |
| 🔌 Bottleneck adapters | Task-specific adaptation | ~322K trainable |
| 📐 EWC regularization | Shields adapter weights across tasks | 0 extra |
| 🏷️ Task-specific heads | Per-task 20-class classifiers | ~4K each |

---

## 📂 Repository Structure

```
thesis-vit-continual-learning/
│
├── 📄 README.md
├── 📦 requirements.txt
├── 🚫 .gitignore
│
├── 📝 thesis/
│   ├── Thesis_Updated_Draft_Latex_2118031.pdf
│   └── (LaTeX source files)
│
├── 🧪 experiments/ 
│   ├── VITCL_baseline.ipynb           # ResNet-18 and ViT-Tiny naive fine-tuning
│   ├── VITCL_adapters.ipynb           # ViT + frozen backbone + adapters
│   ├── VITCL_adapters_ewc.ipynb           # ViT + adapters + EWC ⭐
│   ├── VITCL_adapters_ewc_crossval.ipynb  # ViT + adapters + EWC with cross-validation⭐
│   └── VITCL_ablation_study_graphs.ipynb        # Ablation study
│
└──  📊 results/
      ├── figures/                      # Accuracy curves, heatmaps, ablation charts
      └── tables/                       # Numerical results 

```

---

## ⚙️ Experimental Setup

| Parameter | Value |
|:---|:---|
| Dataset | CIFAR-100 split into 5 tasks × 20 classes |
| CL Scenario | Task-Incremental Learning (TIL) |
| Backbone | ViT-Tiny (5.85M params, ImageNet-1K pre-trained) |
| Input resolution | 32×32 → upsampled to 224×224 (bilinear) |
| Adapter bottleneck | 192 → 64 → 192 per transformer block |
| Trainable params | ~322K (5.5% of total) |
| Optimizer | Adam (β₁=0.9, β₂=0.999) |
| Learning rate | 1×10⁻³ (adapters) · 3×10⁻⁵ (ViT naive FT) |
| Epochs per task | 10 |
| Batch size | 64 |
| EWC λ | 1000 |
| Fisher samples | 500 per task |
| Seeds tested | 42, 123, 7 |
| Hardware | Google Colab (NVIDIA T4 GPU) |

---

## 🚀 Quick Start

```markdown
This project is implemented using five Google Colab notebooks. To reproduce the experiments, open the notebooks in the `notebooks/` folder and run them in the following order:

1. `VITCL_baseline.ipynb`
2. `VITCL_adapters.ipynb`
3. `VITCL_adapters_ewc.ipynb`
4. `VITCL_adapter_ewc_crossval.ipynb`
5. `VITCL_ablation_study_graphs.ipynb`

Each notebook contains the required setup and dependency installation commands. For best results, run the notebooks in Google Colab with GPU enabled:

**Runtime → Change runtime type → GPU**

The notebooks reproduce the four main configurations used in the thesis:

- ResNet-18 baseline
- ViT-Tiny baseline
- ViT-Tiny with adapters
- ViT-Tiny with adapters and EWC
```
---

## 📊 Ablation Chain

Each component contributes measurably — no single piece alone achieves the result:

```
ResNet-18          ████░░░░░░░░░░░░░░░░  24.22%  forgetting: 69.17%
ViT-Tiny           ████████░░░░░░░░░░░░  41.91%  forgetting: 63.26%
ViT + Adapters     ███████████████░░░░░  76.93%  forgetting: 19.91%
ViT + Adapt + EWC  ██████████████████░░  90.59%  forgetting:  2.01% ✅
```

---

## 📖 Related Work

| Paper | Method | Notes |
|:---|:---|:---|
| [EWC (Kirkpatrick et al., 2017)](https://doi.org/10.1073/pnas.1611835114) | Fisher regularization | Foundational CL method used here |
| [ViT (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929) | Vision Transformer | Backbone architecture |
| [Adapters (Houlsby et al., 2019)](https://arxiv.org/abs/1902.00751) | Bottleneck adapters | Adapter design used here |
| [L2P (Wang et al., 2022)](https://arxiv.org/abs/2112.08654) | Prompt-based CL | Prompt approach (different paradigm) |
| [DualPrompt (Wang et al., 2022)](https://arxiv.org/abs/2204.04799) | Task/general prompts | Larger backbone (ViT-B/16, ImageNet-21K) |

> ⚠️ **Note on comparisons:** L2P and DualPrompt use ViT-B/16 (86M params, ImageNet-21K) under CIL — direct numerical comparison with this work is not appropriate due to differences in backbone, scenario, and pre-training data.

---

## 📜 License

© 2026 Swati Aggrawal. All rights reserved.  
This thesis has been typeset by LaTeX and the Sapthesis class.  
Contact: [swatiagg357@gmail.com](mailto:swatiagg357@gmail.com)
