# Semi-Supervised Noise Adaptation: Transferring Knowledge from a Noise Domain

This is the official implementation codes for Semi-Supervised Noise Adaptation (SSNA).

This study introduces the Semi-Supervised Noise Adaptation (SSNA) problem, which exploits information from a noise domain to improve learning performance in the target domain. To tackle this challenge, we propose the NA framework, which simultaneously minimizes the empirical risks of the noise and target domains while mitigating the distributional discrepancy between them.

## Getting Started

### Datasets
Datasets will be download automatically. This repository supports experiments on 4 various datasets: CIFAR-10, CIFAR-100, DTD and Caltech-101.

The dataset folder structure should look like:
```
data
└── SSL
    ├── DTD
    │   ├── image_list
    │   ├── image_list.zip
    │   ├── test
    │   ├── test.tgz
    │   ├── train
    │   ├── train.tgz
    │   ├── validation
    │   └── validation.tgz
    ├── caltech101
    │   ├── image_list
    │   ├── image_list.zip
    │   ├── test
    │   ├── test.tgz
    │   ├── train
    │   └── train.tgz
    ├── cifar10
    │   ├── cifar-10-batches-py
    │   └── cifar-10-python.tar.gz
    └── cifar100
        ├── cifar-100-python
        └── cifar-100-python.tar.gz
```

### Dependencies
You can implement the environment by using the instruction below:
```bash
pip install -r requirements.txt
```

## Evaluations
You can use the following instruction to reproduce experiments with SSNA.

```bash
bash SSNA.sh
```

This script will automatically train ResNet-18 with NAF on CIFAR-10, CIFAR-100, DTD and Caltech-101. Detailed training logs will be saved in 'logs/'.
