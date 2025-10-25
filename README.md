# CASSL
Context-Aware Self-Support Learning for Weakly-Supervised Open-World Visual Grounding.


---
## Getting Started
- torch 1.12.1
- Python 3.8
- Transformer 4.46.3


---
## Data Preparation

- You will need the following datasets:
- Visual Genome (train/val) for training and validation
- COCO (train) for training
- Flickr30k (val) for validation
- ReferIt (val) for validation

---

## Quick Start: Training

Visual Genome training with Flickr30k validation:

```markdown
python train_grounding.py -bs 32 -nW 8 -nW_eval 1 -task vg_train -data_path ../data/vg/ -val_path ../data/f30k
```

COCO training with Flickr30k validation:
```markdown
python train_grounding.py -bs 32 -nW 8 -nW_eval 1 -task coco -data_path ../data/coco -val_path ../data/f30k
```
---

## Acknowledgements
- [WWbL](https://github.com/talshaharabany/what-is-where-by-looking)
- Visual Genome, COCO, Flickr30k and ReferIt datasets and their maintainers
- Open-source deep learning community

## Contact Us
We welcome contributions to improve this code! Feel free to:
- Email us directly at [luzhaoxuan@smail.fjut.edu.cn].

---
