# Pytorch implementation of Mahendran & Vedaldi, CVPR, 2015, with automated hyperparameter optimization

This repository contains a Pytorch-based implementation of [Mahendran & Vedaldi, CVPR, 2015](https://arxiv.org/abs/1412.0035). The code here is desigined for inverting a VGG16-like network trained on CIFAR10 dataset, but you can modify slightely to work for other network architectures and datasets.

## Additional features
In addition to the inversion procedures described in the original paper, this repository contains some additional features:

* "Change-of-variable" technique, as introduced in [Carlini & Wagner, 2017](https://arxiv.org/abs/1608.04644). Instead of optimizing the raw input pixel values `x`, this technique
optimizes another variable `w`, where `x_i = 0.5 * (tanh(w_i) + 1)`, such that `0 <= x_i <= 1` is guaranteed.  
* Automated hyperparameter tuning powered by Optuna [[Akiba+, KDD, 2019](https://arxiv.org/abs/1907.10902)]

## Tested environment
* Python 3.6.6
* Pytorch 1.0.1

## Usage
1. run `ff_training.py` to train the feed-forward network on CIFAR10 (input --> target).
2. run `get_rpr.py` to obtain the internal representation at the specified layer (input --> hidden).
3. run `invert_MV.py` to invert the internal representation back to the input space (hidden --> input).

Incorporate change-of-variable technique: `python invert_MV.py --variable_change`  
Perform hyperparameter tuning: `python invert_MV.py --tune_hyperparams`

## References

Understanding Deep Image Representations by Inverting Them https://arxiv.org/abs/1412.0035

Towards Evaluating the Robustness of Neural Networks https://arxiv.org/abs/1608.04644

Optuna: A Next-generation Hyperparameter Optimization Framework https://arxiv.org/abs/1907.10902
