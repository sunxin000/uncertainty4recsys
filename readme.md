# Uncertainty Calibration for Counterfactual Propensity Estimation in Recommendation

This repository contains the implementation code for the paper "Uncertainty Calibration for Counterfactual Propensity Estimation in Recommendation".

## Overview

This work addresses the challenge of miscalibrated propensity scores in recommendation systems. We propose using calibration as a quality metric for evaluating these scores and demonstrate that implementing propensity score calibration leads to significant improvements in recommendation results.

## Installation

Create a conda environment using the provided configuration:

```shell
conda env create -f environment.yml
```

## Usage

### 1. Train the Propensity Model

First, train a propensity estimation model:

```shell
python train_propensity.py --dataset coat
```

The trained model checkpoint will be saved in `propensity/saved_model/`.

### 2. Generate Propensity Scores

Generate propensity scores using the trained model:

```shell
python gen_propensity.py --dataset coat
```

### 3. Calibrate Scores (Optional)

For Platt scaling calibration:

```shell
python platt_scale.py
```

### 4. Train Recommendation Model

Train a recommendation model using various methods (IPS, DR, MRDR):

```shell
python train_rec_with_ips.py --dir Platt_Scaling --dataset coat --path propensity/Platt_Scaling/coat.pt
```

## Citation

If you use this code in your research, please cite our paper:
```
@article{hu2023uncertainty,
  title={Uncertainty calibration for counterfactual propensity estimation in recommendation},
  author={Hu, Wenbo and Sun, Xin and Wu, Le and Wang, Liang and others},
  journal={arXiv preprint arXiv:2303.12973},
  year={2023}
}
```
