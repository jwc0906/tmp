# Commends

require:  
```
# CUDA 10.1
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
```



step1 * 4:
```
CUDA_VISIBLE_DEVICES=0,1 \
python main_supcon.py --batch_size 512 \
  --learning_rate 0.5 \
  --temp 0.5 \
  --cosine \
  --method SimCLR \
  --epochs 1000 \
  \
  --total_data_num 10000\
  --model resnet50 \
  --dataset cifar10 \
  --imbalance_ratio 150 \
  --imbalance_order 8012964753   \
;\
CUDA_VISIBLE_DEVICES=0,1 \
python main_supcon.py --batch_size 512 \
  --learning_rate 0.5 \
  --temp 0.5 \
  --cosine \
  --method SimCLR \
  --epochs 1000 \
  \
  --total_data_num 10000\
  --model resnet50 \
  --dataset cifar10 \
  --imbalance_ratio 100 \
  --imbalance_order 8012964753   \
;\
CUDA_VISIBLE_DEVICES=0,1 \
python main_supcon.py --batch_size 512 \
  --learning_rate 0.5 \
  --temp 0.5 \
  --cosine \
  --method SimCLR \
  --epochs 1000 \
  \
  --total_data_num 10000\
  --model resnet50 \
  --dataset SVHN \
  --imbalance_ratio 150 \
  --imbalance_order 1724593680 \
;\
CUDA_VISIBLE_DEVICES=0,1 \
python main_supcon.py --batch_size 512 \
  --learning_rate 0.5 \
  --temp 0.5 \
  --cosine \
  --method SimCLR \
  --epochs 1000 \
  \
  --total_data_num 10000\
  --model resnet50 \
  --dataset SVHN \
  --imbalance_ratio 100 \
  --imbalance_order 1724593680 \
```

make weight * 4:
```
CUDA_VISIBLE_DEVICES=0,1 \
python make_weight.py \
--model resnet50 \
--dataset cifar10 \
--dir_path ./save/SupCon/cifar10_models/SimCLR_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.5_trial_0_cosine_warm_ir_150.0_i_order_8012964753__total_data_10000 \
;\
CUDA_VISIBLE_DEVICES=0,1 \
python make_weight.py \
--model resnet50 \
--dataset cifar10 \
--dir_path ./save/SupCon/cifar10_models/SimCLR_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.5_trial_0_cosine_warm_ir_100.0_i_order_8012964753__total_data_10000 \
;\
CUDA_VISIBLE_DEVICES=0,1 \
python make_weight.py \
--model resnet50 \
--dataset SVHN \
--dir_path ./save/SupCon/SVHN_models/SimCLR_SVHN_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.5_trial_0_cosine_warm_ir_150.0_i_order_1724593680__total_data_10000 \
;\
CUDA_VISIBLE_DEVICES=0,1 \
python make_weight.py \
--model resnet50 \
--dataset SVHN \
--dir_path ./save/SupCon/SVHN_models/SimCLR_SVHN_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.5_trial_0_cosine_warm_ir_100.0_i_order_1724593680__total_data_10000 \
```


step2 * 8:
```
CUDA_VISIBLE_DEVICES=0,1 \
python main_supcon_second_step.py --batch_size 512 \
  --learning_rate 0.5 \
  --temp 0.5 \
  --cosine \
  --method SimCLR \
  --epochs 1000 \
  \
  --model resnet50 \
  --dataset cifar10 \
  --dir_path ./save/SupCon/cifar10_models/SimCLR_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.5_trial_0_cosine_warm_ir_150.0_i_order_8012964753__total_data_10000 \
  --step2_method 3 \
  --scale 100 \
;\
CUDA_VISIBLE_DEVICES=0,1 \
python main_supcon_second_step.py --batch_size 512 \
  --learning_rate 0.5 \
  --temp 0.5 \
  --cosine \
  --method SimCLR \
  --epochs 1000 \
  \
  --model resnet50 \
  --dataset cifar10 \
  --dir_path ./save/SupCon/cifar10_models/SimCLR_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.5_trial_0_cosine_warm_ir_100.0_i_order_8012964753__total_data_10000 \
  --step2_method 3 \
  --scale 100 \
;\
CUDA_VISIBLE_DEVICES=0,1 \
python main_supcon_second_step.py --batch_size 512 \
  --learning_rate 0.5 \
  --temp 0.5 \
  --cosine \
  --method SimCLR \
  --epochs 1000 \
  \
  --model resnet50 \
  --dataset SVHN \
  --dir_path ./save/SupCon/SVHN_models/SimCLR_SVHN_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.5_trial_0_cosine_warm_ir_150.0_i_order_1724593680__total_data_10000 \
  --step2_method 3 \
  --scale 100 \
;\
CUDA_VISIBLE_DEVICES=0,1 \
python main_supcon_second_step.py --batch_size 512 \
  --learning_rate 0.5 \
  --temp 0.5 \
  --cosine \
  --method SimCLR \
  --epochs 1000 \
  \
  --model resnet50 \
  --dataset SVHN \
  --dir_path ./save/SupCon/SVHN_models/SimCLR_SVHN_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.5_trial_0_cosine_warm_ir_100.0_i_order_1724593680__total_data_10000 \
  --step2_method 3 \
  --scale 100 \
;\
\
\
\
\
\
\
CUDA_VISIBLE_DEVICES=0,1 \
python main_supcon_second_step.py --batch_size 512 \
  --learning_rate 0.5 \
  --temp 0.5 \
  --cosine \
  --method SimCLR \
  --epochs 1000 \
  \
  --model resnet50 \
  --dataset cifar10 \
  --dir_path ./save/SupCon/cifar10_models/SimCLR_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.5_trial_0_cosine_warm_ir_150.0_i_order_8012964753__total_data_10000 \
  --step2_method 1 \
  --scale 0 \
;\
CUDA_VISIBLE_DEVICES=0,1 \
python main_supcon_second_step.py --batch_size 512 \
  --learning_rate 0.5 \
  --temp 0.5 \
  --cosine \
  --method SimCLR \
  --epochs 1000 \
  \
  --model resnet50 \
  --dataset cifar10 \
  --dir_path ./save/SupCon/cifar10_models/SimCLR_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.5_trial_0_cosine_warm_ir_100.0_i_order_8012964753__total_data_10000 \
  --step2_method 1 \
  --scale 0 \
;\
CUDA_VISIBLE_DEVICES=0,1 \
python main_supcon_second_step.py --batch_size 512 \
  --learning_rate 0.5 \
  --temp 0.5 \
  --cosine \
  --method SimCLR \
  --epochs 1000 \
  \
  --model resnet50 \
  --dataset SVHN \
  --dir_path ./save/SupCon/SVHN_models/SimCLR_SVHN_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.5_trial_0_cosine_warm_ir_150.0_i_order_1724593680__total_data_10000 \
  --step2_method 1 \
  --scale 0 \
;\
CUDA_VISIBLE_DEVICES=0,1 \
python main_supcon_second_step.py --batch_size 512 \
  --learning_rate 0.5 \
  --temp 0.5 \
  --cosine \
  --method SimCLR \
  --epochs 1000 \
  \
  --model resnet50 \
  --dataset SVHN \
  --dir_path ./save/SupCon/SVHN_models/SimCLR_SVHN_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.5_trial_0_cosine_warm_ir_100.0_i_order_1724593680__total_data_10000 \
  --step2_method 1 \
  --scale 0 \
```
  
model: [resnet50, VGG19]  
dataset: [cifar100, cifar10, SVHN, tiny-imagenet-200]  
imbalance order: [ascent, descent]  
imbalance ratio: float (1~150)  
total data num: int

dirpath: str  
scale: int  

