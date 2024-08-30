# UniSOD
This repository provides the source code and results for the paper entilted "Unified-modal Salient Object Detection via Adaptive Prompt Learning".

arXiv version: https://arxiv.org/abs/2311.16835.

Thank you for your attention.
## Citing our work

If you think our work is helpful, please cite

```
@article{wang2023unified,
  title={Unified-modal Salient Object Detection via Adaptive Prompt Learning},
  author={Wang, Kunpeng and Li, Chenglong and Tu, Zhengzheng and Luo, Bin},
  journal={arXiv preprint arXiv:2311.16835},
  year={2023}
}
```

## Overview
### Framework
[![avatar](https://github.com/Angknpng/UniSOD/raw/main/figures/framework.png)](https://github.com/Angknpng/UniSOD/blob/main/figures/framework.png)
### Baseline SOD framework
[![avatar](https://github.com/Angknpng/UniSOD/raw/main/figures/framework_base.png)](https://github.com/Angknpng/UniSOD/blob/main/figures/framework_base.png)
### RGB SOD Performance
[![avatar](https://github.com/Angknpng/UniSOD/raw/main/figures/performance_RGB.png)](https://github.com/Angknpng/UniSOD/blob/main/figures/performance_RGB.png)
### RGB-D SOD Performance
[![avatar](https://github.com/Angknpng/UniSOD/raw/main/figures/performance_RGBD.png)](https://github.com/Angknpng/UniSOD/blob/main/figures/performance_RGBD.png)
### RGB-T SOD Performance
[![avatar](https://github.com/Angknpng/UniSOD/raw/main/figures/performance_RGBT.png)](https://github.com/Angknpng/UniSOD/blob/main/figures/performance_RGBT.png)

## Predictions

The predicted RGB, RGB-D, and RGB-T saliency maps can be found here. [[baidu pan](https://pan.baidu.com/s/1zBqZAChDCJfkmC_Pj_xHXQ?pwd=vpvt) fetch code: vpvt]

## Usage

### Requirement

0. Download the datasets for training and testing.
1. Download the pretrained parameters of the backbone.
2. Organize dataset directories for pre-training and fine-tuning.
3. Create directories for the experiment and parameter files.
4. Please use `conda` to install `torch` (1.12.0) and `torchvision` (0.13.0).
5. Install other packages: `pip install -r requirements.txt`.
6. Set your path of all datasets in `./options.py`.

### Pre-train

```
python -m torch.distributed.launch --nproc_per_node=2 --master_port=2024 train_parallel.py
```

### Fine-tuning

```
python -m torch.distributed.launch --nproc_per_node=2 --master_port=2024 train_parallel_multi.py
```

### Test

```
python test_produce_maps.py
```

## Acknowledgement

The implement of this project is based on the following link.

- [SOD Literature Tracking](https://github.com/jiwei0921/SOD-CNNs-based-code-summary-)
- [PR Curve](https://github.com/lartpang/PySODEvalToolkit)

## Contact

If you have any questions, please contact us (kp.wang@foxmail.com).
