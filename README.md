# UniSOD
This repository provides the source code and results for the paper entilted "Unified-modal Salient Object Detection via Adaptive Prompt Learning".

arXiv version: https://arxiv.org/abs/2311.16835.

Thank you for your attention.

## :tada: **News** :tada:  (July, 2025)

We are pleased to announce that our paper has been accepted to **[TCSVT 2025](https://ieeexplore.ieee.org/document/11082344)**! 🙏Thank you for your continued interest and support! 

## Citing our work

If you think our work is helpful, please cite

```
@article{wang2025unified,
  title={Unified-modal salient object detection via adaptive prompt learning},
  author={Wang, Kunpeng and Tu, Zhengzheng and Li, Chenglong and Liu, Zhengyi and Luo, Bin},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025},
  publisher={IEEE}
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

## Pretrained Models
The pretrained parameters of our models can be found here. [[baidu pan](https://pan.baidu.com/s/1IX4Ejz4eBP6J3mmp_k3KrQ?pwd=o8yx) fetch code: o8yx]

## Usage

### Requirement

0. Download the datasets for training and testing from here. [[baidu pan](https://pan.baidu.com/s/1auw5rbBzEQ2hsrxUtQyvzg?pwd=2sfr) fetch code: 2sfr]
1. Download the pretrained parameters of the backbone from here. [[baidu pan](https://pan.baidu.com/s/14xGtKVSs53zRNZVKK-x4HA?pwd=mad3) fetch code: mad3]
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
