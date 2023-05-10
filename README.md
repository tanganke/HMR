# Improving Heterogeneous Model Reuse by Density Estimation

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [Improving Heterogeneous Model Reuse by Density Estimation](#improving-heterogeneous-model-reuse-by-density-estimation)
  * [1. Toy Example](#1-toy-example)
  * [2. Benchmark Experiments on Fashion-MNIST](#2-benchmark-experiments-on-fashion-mnist)
    + [prepare dataset](#prepare-dataset)
    + [2.1 Ours](#21-ours)
    + [2.2 Centralized Baseline](#22-centralized-baseline)
    + [2.3 HMR (compared)](#23-hmr-compared)
    + [2.4 RKME (compared)](#24-rkme-compared)

<!-- TOC end -->

my enviroment:

- Python 3
- Linux

insatll package dependency:

```bash
pip install -r ./requirements.txt
```

project layout:

```conf
# code for toy example experiment
toy example/

# benchmark experiments
data/
    fashion_mnist/          # Fashion-MNIST datasets under multiparty settings
        A/
            ${party_name}/
                ${class_name}/
                    ${image_name}.png
        B/
        C/
        D/
        train/              # global train dataset
        test/               # global test dataset
        fashion_mnist.py    # code to load multiparty fashion datasets
fashion_mnist.ipynb         # reproduce all figures in the paper
fashion_mnist.conv/         # train classifiers on local dataset and train centralized baseline model
    data/                   # symbolic link to ../data/
    output/                 # symbolic link to ../output/
fashion_mnist.realnvp/      # train density estimators on locally
fashion_mnist.global/       # global model (Ours)
    deploy_global_model.py  # deploy the global model
    prepare_global_model.py # calibration the global model from raw local models (random initialized)
fashion_mnist.RKME/         # deploy the global model (RKME)
output/                     # training logs, model checkpoints 
```

## 1. Toy Example

- [Ours](toy_example/HMR_Ours.ipynb)
- [HMR (ICML 2019)](toy_example/HMR_ICML2019.ipynb)

## 2. Benchmark Experiments on Fashion-MNIST

[download tensorboard logs from github releases](https://github.com/tanganke/HMR/releases/download/material/output.zip)

### prepare dataset

[download datasets from github releases](https://github.com/tanganke/HMR/releases/download/material/datasets.zip)

```bash
cd data
unzip fashion_mnist.zip
```

### 2.1 Ours

train classifiers on Fashion-MNIST:

```bash
cd fashion_mnist.conv
python3 prepare_conv.py # log dir: output/fashion_mnist/conv/log
```

train density estimators on Fashion-MNIST:

```bash
cd fashion_mnist.realnvp
python3 prepare_realnvp.py # log dir: output/fashion_mnist/realnvp/log
```

evaluate global model on global test set (10k images, 10 classes):

```bash
cd fashion_mnist.global
python3 deploy_global_model.py 
# zero-shot accuracy: output/fashion_mnist/global/deploy
# calibration log: output/fashion_mnist/global/calibration/log
```

train global model from raw model on global train set:

```bash
cd fashion_mnist.global
python3 prepare_global_model.py
# raw accuracy: output/fashion_mnist/global/raw
# log dir: output/fashion_mnist/global/raw/log
```

*NOTE*: structure of log directories:

```bash
.
├── A
│   ├── party_0
│   │   ├──  version_0
│   │   │... version_XX
│   └── party_1
├── B
│   ├── party_0
│   ├── party_1
│   └── party_2
├── C
│   ├── party_0
│   ├── party_1
│   └── party_2
└── D
    ├── party_0
    ├── party_1
    ├── party_2
    ├── party_3
    ├── party_4
    ├── party_5
    └── party_6
```

### 2.2 Centralized Baseline

```bash
cd fashion_mnist.conv
python3 prepare_baseline.py # log dir: output/fashion_mnist/conv/baseline/log
```

### 2.3 HMR (compared)

> Wu, Xi Zhu, Song Liu, and Zhi Hua Zhou. 2019. “Heterogeneous Model Reuse via Optimizing Multiparty Multiclass Margin.” 36th International Conference on Machine Learning, ICML 2019 2019-June: 11862–71.

see [GitHub](https://github.com/YuriWu/HMR).  
pre-run results - [output/fashion_mnist.HMR/result.csv](./output/fashion_mnist.HMR/result.csv)

### 2.4 RKME (compared)

> X. Wu, W. Xu, S. Liu, and Z. Zhou. Model reuse with reduced kernel mean embedding specification. IEEE Transactions on Knowledge and Data Engineering, 35(01):699–710, jan 2023.


```bash
cd fashion_mnist_RKME
```

Kernel methods usually cannot work directly on the raw-pixel level or raw-document level due to the high input dimension.
We exact features as the outputs from the penultimate layer of pre-trained ResNet-110.

```bash
python3 prepare_features.py # save features to: output/fashion_mnist.RKME/features.resnet101
```

fit reduced kernel mean embedding, find optimal betas and reduced points (M = 10).

```bash
python3 prepare_rkme.py # log dir: output/fashion_mnist.RKME/features.resnet101.RKME.M=10
```

deploy RKME on global test set.

```bash
python3 deploy_rkme.py # log dir: output/fashion_mnist.RKME/features.resnet101.RKME.M=10/deploy
```
