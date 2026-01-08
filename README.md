# DCNv4

A library-focused fork of [OpenGVLab/DCNv4](https://github.com/OpenGVLab/DCNv4) — Deformable Convolution v4 for PyTorch.

This fork focuses on making DCNv4 easier to use as a standalone library. The original FlashInternImage models and downstream task configurations are not modified; for those, please refer to the [upstream repository](https://github.com/OpenGVLab/DCNv4).

## Installation

```bash
pip install git+https://github.com/nusu-github/DCNv4.git
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- CUDA toolkit (for CUDA backend)

## Usage

```python
import torch
from dcnv4 import dcnv4

# Create DCNv4 layer
layer = dcnv4(channels=64, kernel_size=3, group=4).cuda()

# Input shape: (batch, height * width, channels)
x = torch.randn(2, 64 * 64, 64).cuda()

# Forward pass (provide spatial shape)
output = layer(x, shape=(64, 64))
```

### Triton Backend

To use the Triton backend instead of CUDA:

```bash
export DCNV4_USE_TRITON=1
```

The Triton backend offers better portability across GPU architectures without recompilation.

## API Reference

### `dcnv4`

Main DCNv4 module for use as a drop-in replacement for convolutions.

| Parameter              | Type        | Default | Description                              |
| ---------------------- | ----------- | ------- | ---------------------------------------- |
| `channels`             | int         | —       | Number of input/output channels          |
| `kernel_size`          | int         | 3       | Size of the deformable kernel            |
| `stride`               | int         | 1       | Stride of the convolution                |
| `pad`                  | int         | 1       | Padding size                             |
| `dilation`             | int         | 1       | Dilation rate                            |
| `group`                | int         | 4       | Number of groups for grouped convolution |
| `offset_scale`         | float       | 1.0     | Scaling factor for offsets               |
| `dw_kernel_size`       | int \| None | None    | Depthwise convolution kernel size        |
| `center_feature_scale` | bool        | False   | Enable center feature scaling            |

### `FlashDeformAttn`

Multi-scale deformable attention module with optimized kernels.

## About DCNv4

DCNv4 introduces two key improvements over DCNv3:

1. **Unbounded aggregation weights**: Removes softmax normalization, allowing dynamic weights similar to standard convolutions
2. **Optimized memory access**: Eliminates redundant operations, achieving 3x+ speedup over DCNv3

For details, see the paper: [Efficient Deformable ConvNets: Rethinking Dynamic and Sparse Operator for Vision Applications](https://arxiv.org/abs/2401.06197)

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use DCNv4 in your research, please cite the original paper:

```bibtex
@article{xiong2024efficient,
  title={Efficient Deformable ConvNets: Rethinking Dynamic and Sparse Operator for Vision Applications},
  author={Yuwen Xiong and Zhiqi Li and Yuntao Chen and Feng Wang and Xizhou Zhu and Jiapeng Luo and Wenhai Wang and Tong Lu and Hongsheng Li and Yu Qiao and Lewei Lu and Jie Zhou and Jifeng Dai},
  journal={arXiv preprint arXiv:2401.06197},
  year={2024}
}
```

## Acknowledgments

This fork is based on the official [DCNv4 implementation](https://github.com/OpenGVLab/DCNv4) by OpenGVLab.
