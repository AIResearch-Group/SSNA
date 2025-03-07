# SSNA PAM
# ======================================================================================================================
# CIFAR 10
CUDA_VISIBLE_DEVICES=0 python SSNA.py /data/SSL/cifar10 -d CIFAR10 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.4912 0.4824 0.4467 --norm-std 0.2471 0.2435 0.2616 --num-samples-per-class 4 -a resnet18 \
  --lr 0.03 --finetune --seed 0 --log logs/ --trade-off-SSNA-training 1 --epochs 20 --PAM

# ======================================================================================================================
# CIFAR 100
CUDA_VISIBLE_DEVICES=0 python SSNA.py /data/SSL/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --num-samples-per-class 4 -a resnet18 \
  --lr 0.01 --finetune --seed 0 --log logs/ --trade-off-SSNA-training 10 --epochs 20 --PAM

# ======================================================================================================================
# DTD
CUDA_VISIBLE_DEVICES=0 python SSNA.py /data/SSL/DTD -d DTD --num-samples-per-class 4 -a resnet18 \
  --lr 0.03 --finetune --seed 0 --log logs/ --trade-off-SSNA-training 1 --epochs 20 --PAM

# ======================================================================================================================
# Caltech-101
CUDA_VISIBLE_DEVICES=0 python SSNA.py /data/SSL/caltech101 -d Caltech101 --num-samples-per-class 4 -a resnet18 \
  --lr 0.003 --finetune --seed 0 --log logs/ --trade-off-SSNA-training 10 --epochs 20 --PAM


# SSNA PCM
# ======================================================================================================================
# CIFAR 10
CUDA_VISIBLE_DEVICES=0 python SSNA.py /data/SSL/cifar10 -d CIFAR10 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.4912 0.4824 0.4467 --norm-std 0.2471 0.2435 0.2616 --num-samples-per-class 4 -a resnet18 \
  --lr 0.03 --finetune --seed 0 --log logs/ --trade-off-SSNA-training 1 --epochs 20

# ======================================================================================================================
# CIFAR 100
CUDA_VISIBLE_DEVICES=0 python SSNA.py /data/SSL/cifar100 -d CIFAR100 --train-resizing 'cifar' --val-resizing 'cifar' \
  --norm-mean 0.5071 0.4867 0.4408 --norm-std 0.2675 0.2565 0.2761 --num-samples-per-class 4 -a resnet18 \
  --lr 0.01 --finetune --seed 0 --log logs/ --trade-off-SSNA-training 10 --epochs 20

# ======================================================================================================================
# DTD
CUDA_VISIBLE_DEVICES=0 python SSNA.py /data/SSL/DTD -d DTD --num-samples-per-class 4 -a resnet18 \
  --lr 0.03 --finetune --seed 0 --log logs/ --trade-off-SSNA-training 1 --epochs 20

# ======================================================================================================================
# Caltech-101
CUDA_VISIBLE_DEVICES=0 python SSNA.py /data/SSL/caltech101 -d Caltech101 --num-samples-per-class 4 -a resnet18 \
  --lr 0.003 --finetune --seed 0 --log logs/ --trade-off-SSNA-training 10 --epochs 20