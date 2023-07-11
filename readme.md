# Torch cuda warper
A simple Cuda Kernel to warp torch tensors using an homogrpahy
- Cuda version
- CPU version fallback (not optimized)
Seems like the custom cuda version is much faster that the native torch combination of `torch.nn.functional.affine_grid` & `torch.nn.functional.grid_sample`.

At least using **pytorch 1.11.0+cu113** on a **A100** machine.

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
python3 test_warper.py
```

```
Warp forward custom cuda kernel 0.0003 s, bandwidth 3576.8264GB/s
Warp forward torch 0.0036 s, bandwidth 332.6989GB/s
Speed up 10.6x
```