# Commends

require:  
```
# CUDA 10.1
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
```


tiny-imagenet-200 downloader:  
run this code at this dir ./datasets/  
https://gist.github.com/moskomule/2e6a9a463f50447beca4e64ab4699ac4  




step1:
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
  --dataset tiny-imagenet \
  --imbalance_order ascent \
  --imbalance_ratio 150 \
  --total_data_num 10000\
```

make weight:
```python
CUDA_VISIBLE_DEVICES=0,1 \
python make_weight.py --model resnet50 \
  --dataset tiny-imagenet-200 \
  --dir_path ./save/SupCon/tiny-imagenet-200_models/SimCLR_tiny-imagenet-200_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.5_trial_0_cosine_ir_150.0_i_order_ascent__total_data_10000 \
```


step2:
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
  --dataset tiny-imagenet-200 \
  --dir_path ./save/SupCon/tiny-imagenet-200_models/SimCLR_tiny-imagenet-200_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.5_trial_0_cosine_ir_150.0_i_order_ascent__total_data_10000 \
  --scale 10 \
```
  
model: [resnet50, VGG19]  
dataset: [cifar100, cifar10, SVHN, tiny-imagenet-200]  
imbalance order: [ascent, descent]  
imbalance ratio: float (1~150)  
total data num: int

dirpath: str  
scale: int  

