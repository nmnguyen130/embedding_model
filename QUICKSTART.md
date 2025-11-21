# Quick Start Guide - Modern T4 Optimized

## ⚡ Quick Test (30-60 min)

```bash
cd d:/Coding/AI/Embedding_Model
python -m src.training.train --quick-start
```

This will:
- ✅ Train WordPiece tokenizer (10K samples)
- ✅ Generate training triplets  
- ✅ Train for 2 epochs (~1000 steps)
- ✅ Save to `./outputs`

## 🚀 Full Training (8-12 hours на T4)

**AdamW (default)**:
```bash
pythonм src.training.train \
  --batch-size 32 \
  --grad-accum-steps 8 \
  --num-epochs 10 \
  --use-wikipedia \
  --use-snli
```

**Muon (faster, experimental)**:
```bash
python -m src.training.train \
  --optimizer muon \
  --muon-lr 0.02 \
  --batch-size 32 \
  --grad-accum-steps 8 \
  --num-epochs 10 \
  --use-wikipedia \
  --use-snli
```

## 📊 Inference

```python
from src.inference import load_model

model = load_model(
    "outputs/best_model/checkpoint.pt",
    "data/tokenizer/tokenizer.json"
)

# Encode
embedding = model.encode("Hello world")  # (384,)

# Similarity
score = model.similarity("I love AI", "AI is amazing")  # 0.85

# Search
results = model.find_similar(
    query="machine learning",
    candidates=["AI algorithms", "Cooking", "Neural networks"],
    top_k=2
)
```

## 🔧 Evaluation

```bash
python -m src.evaluation.sts_evaluation \
  --checkpoint outputs/best_model/checkpoint.pt \
  --tokenizer data/tokenizer/tokenizer.json
```

## ⚙️ Advanced Options

### Custom Architecture

```bash
python -m src.training.train \
  --num-layers 8 \
  --num-heads 8 \
  --num-kv-heads 2 \
  --output-dim 512
```

### Optimizer Selection

```bash
# AdamW
python -m src.training.train --optimizer adamw --learning-rate 2e-4

# Muon
python -m src.training.train --optimizer muon --muon-lr 0.02
```

### Resume Training

```bash
python -m src.training.train \
  --resume-from outputs/checkpoint-5000/checkpoint.pt \
  --num-epochs 15
```

## 📈 Expected Performance

| Training | Epochs | Time | STS-B Spearman |
|----------|--------|------|----------------|
| Quick (10K) | 2 | 30-60 min | 0.20-0.30 |
| Full (100K) | 10 | 8-12 hours | 0.62-0.72 |

## 🛠️ Troubleshooting

**Out of Memory**:
```bash
python -m src.training.train --batch-size 16 --grad-accum-steps 16
```

**Slow Training**:
```bash
python -m src.training.train --optimizer muon  # Try Muon optimizer
```

**Colab Disconnection**:
```python
from google.colab import drive
drive.mount('/content/drive')

# Save to Drive
!python -m src.training.train --output-dir /content/drive/MyDrive/outputs
```

## 📚 Documentation

- [README.md](file:///d:/Coding/AI/Embedding_Model/README.md) - Full guide
- [ARCHITECTURE_UPGRADES.md](file:///d:/Coding/AI/Embedding_Model/ARCHITECTURE_UPGRADES.md) - Technical details

---

**Ready to train!** 🚀
