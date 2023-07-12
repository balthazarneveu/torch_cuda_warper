# Torch cuda warper
A simple Cuda Kernel to warp torch tensors using an homogrpahy
- Cuda version
- CPU version fallback (not optimized)
Seems like the custom cuda version is much faster that the native torch combination of `torch.nn.functional.affine_grid` & `torch.nn.functional.grid_sample`.



This code sample was initiated during a discussion on  [Pytorch github discussion](https://github.com/pytorch/pytorch/issues/104296).


## Important notes
This draft is far less generic than what pytorch provides regarding the warping feature.
- **Tensor format**:
    - Supports channel first NCHW tensors only.
    - 2D images only, no 3D grid warps.
- **Interpolation mode**:
    - Bilinear interpolation only
    - no nearest neighbor, no bicubic.
- **No backward operator supported**


# Compile
Compile as a python library. Requires the right Cuda / Torch compiling environment, follow the Torch C++ / Cuda extention [tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html) for more info.

`cd torch_cuda_warper`

`python3 setup.py install --user`

## Test

```
pytest-3 test_warp.py
```

## Bench
```
python3 test_warp.py
```
Using **pytorch 2.1.0.dev20230711+cu121** on a  **NVIDIA Geforce RTX2060**
```
Warp forward custom cuda kernel 0.0027 s, bandwidth 446GB/s
Warp forward torch 0.0110 s, bandwidth 109GB/s
Speed up 4.0x
```

-----------------------------------------------------

Using **pytorch 1.11.0+cu113** on a **Nvidia A100 & RTX3090**.
Nvidia A100 80GB PCIe

```
Warp forward custom cuda kernel 0.0004 s, bandwidth 3158GB/s
Warp forward torch 0.0030 s, bandwidth 403GB/s
Speed up 7.7x
```

Nvidia RTX3090
```
Warp forward custom cuda kernel 0.0011 s, bandwidth 1065GB/s
Warp forward torch 0.0047 s, bandwidth 254GB/s
Speed up 4.1x
```

