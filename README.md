# Commends
<p align="center">
  <img src="figures/teaser.png" width="700">
</p>

Usage:
```python
python main_supcon.py --batch_size 512 \
  --learning_rate 0.5 \
  --temp 0.5 \
  --cosine \
  --method SimCLR \
  --epochs 1000 \
  \
  --model resnet50 \
  --dataset cifar10 \
  --imbalance_order ascent \
  --imbalance_ratio 1 \
  --total_data_num 10000\
```

model: [resnet50, VGG]
dataset: [cifar100, cifar10, SVHN]
imbalance order: [ascent, descent]
imbalance ratio: float
total data num: int

