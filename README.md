# MPCViT-Evaluation
This is the source code of model evalution/inference for [MPCViT: Searching for Accurate and Efficient MPC-friendly Vision Transformer with Heterogeneous Attention](https://arxiv.org/pdf/2211.13955.pdf).

## Model inference of loaded MPCViT checkpoints
**Command examples:**

Below is an example to evaluate MPCViT with $\mu=0.5$ **w/o** knowledge distillation (KD) on CIFAR-10.
```shell
python inference.py --config configs/datasets/cifar10.yml --model vit_7_4_32 /path/to/cifar10 --model-checkpoint mpcvit_checkpoints/mpcvit_cifar10-0.5.pth.tar
```

Below is an example to evaluate MPCViT with $\mu=0.5$ **w/** knowledge distillation (KD) on CIFAR-10.
```shell
python inference.py --config configs/datasets/cifar10.yml --model vit_7_4_32 /path/to/cifar10 --model-checkpoint mpcvit_checkpoints/mpcvit_cifar10-0.5-kd.pth.tar
```

**Datasets:**

We provide model checkpoints on three widely used datasets, i.e., CIFAR-10, CIFAR-100 and Tiny-ImageNet.

Usage: just simply modify the command above including `config, model, data_dir, model-checkpoint`.


## Citation
```bibtex
@article{zeng2022mpcvit,
  title={MPCViT: Searching for MPC-friendly Vision Transformer with Heterogeneous Attention},
  author={Zeng, Wenxuan and Li, Meng and Xiong, Wenjie and Lu, Wenjie and Tan, Jin and Wang, Runsheng and Huang, Ru},
  journal={arXiv preprint arXiv:2211.13955},
  year={2022}
}
```
