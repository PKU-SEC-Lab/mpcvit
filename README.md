# MPCViT
This is the source code of [MPCViT: Searching for Accurate and Efficient MPC-friendly Vision Transformer with Heterogeneous Attention](https://arxiv.org/pdf/2211.13955.pdf).

## Inference of loaded ViT checkpoint on Tiny-ImageNet dataset
Here is an example to evaluate mpcvit with mu=0.5 and knowledge distillation.
```shell
python inference.py --config configs/datasets/tiny_imagenet.yml --model vit_9_12_64 /path/to/tiny-imagenet-200 --model-checkpoint mpcvit_checkpoints/mpcvit_tinyimagenet-0.5-kd.pth.tar
```

## Citation
```bibtex
@article{zeng2022mpcvit,
  title={MPCViT: Searching for MPC-friendly Vision Transformer with Heterogeneous Attention},
  author={Zeng, Wenxuan and Li, Meng and Xiong, Wenjie and Lu, Wenjie and Tan, Jin and Wang, Runsheng and Huang, Ru},
  journal={arXiv preprint arXiv:2211.13955},
  year={2022}
}
```
