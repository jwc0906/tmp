# Commends

step1 Usage:
```python
CUDA_VISIBLE_DEVICES=0,1 \
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
  --imbalance_ratio 150 \
  --total_data_num 10000\
```

make weight:
```python
CUDA_VISIBLE_DEVICES=0,1 \
python make_weight.py --model resnet50 \
  --dataset tiny-imagenet-200 \
  --dir_path ./save/SupCon/cifar10_models/SimCLR_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.5_trial_0_cosine_warm_ir_150.0_i_order_ascent__total_data_10000 \
```


step2 Usage:
```python
CUDA_VISIBLE_DEVICES=2,3 \
python main_supcon_second_step.py --batch_size 512 \
  --learning_rate 0.5 \
  --temp 0.5 \
  --cosine \
  --method SimCLR \
  --epochs 1000 \
  \
  --model resnet50 \
  --dataset cifar10 \
  --dir_path ./save/SupCon/cifar10_models/SimCLR_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.5_trial_0_cosine_warm_ir_150.0_i_order_ascent__total_data_10000 \
  --scale 10 \
```
  
model: [resnet50, VGG19]  
dataset: [cifar100, cifar10, SVHN]  
imbalance order: [ascent, descent]  
imbalance ratio: float (1~150)  
total data num: int

dirpath: str  
scale: int  

